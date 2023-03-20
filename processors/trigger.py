import numpy as np
import awkward as ak
from distributed import Client
import matplotlib.pyplot as plt
import mplhep
import pandas as pd
import coffea.util
import re
from coffea.nanoevents import NanoEventsFactory, ScoutingNanoAODSchema
from coffea import processor
import hist
from hist import Hist
from coffea.analysis_tools import PackedSelection

def PackedSelection_any(self, *names):
    consider = 0
    for name in names:
        idx = self._names.index(name)
        consider |= 1 << idx
    return (self._data & consider) != 0

class TriggerProcessor(processor.ProcessorABC):
    def __init__(self, year="Run3Summer22EE"):
        self._year = year
        # here you should add your signal triggers (can be more than 1)
        self._sigtriggers = {
            'Run3Summer22EE': [
                'Run3_PFScoutingPixelTracking'
            ]
        }
        # here you should add your reference trigger
        self._reftriggers = {
            'Run3Summer22EE': [
               'Mu50'
            ]
        }
        
        # to start with, we are interested in jet pt and mass, however you can use any jet variable
        commonaxes = (
            hist.axis.StrCategory([], name="dataset", label="Dataset name", growth=True),
            hist.axis.StrCategory([], name="trigger", label="Trigger name", growth=True),
            hist.axis.Regular(100, 0, 1000, name="pt", label="Leading jet $p_T$"),
            hist.axis.Regular(30, 0, 300, name="mass", label="Leading jet mass"),
        )
        
        self._output = {
                "nevents": 0,
                "ak8": Hist(
                    *commonaxes
                ),
                "ak4": Hist(
                    *commonaxes
                ),
            }

    def process(self, events):
        
        dataset = events.metadata['dataset']
        self._output["nevents"] = len(events)
        
        # here we keep track of events that passed our signal triggers
        triggers = PackedSelection()
        trigger_names = self._sigtriggers[self._year]
        for tname in trigger_names:
            if tname in events.DST.fields:
                triggers.add(tname, events.DST[tname])
            else:
                triggers.add(tname, np.zeros(len(events), dtype=bool))
        
        # here we keep track of events passed the reference trigger
        reftrigger = np.zeros(len(events), dtype=bool)
        for tname in self._reftriggers[self._year]:
            if tname in events.HLT.fields:
                reftrigger |= ak.to_numpy(events.HLT[tname])
                
        # you might want to remove events with muons close to your jet
        muons = events.ScoutingMuon[
            (events.ScoutingMuon.pt > 25)
            & (abs(events.ScoutingMuon.eta) < 2.4)
            & (abs(events.ScoutingMuon.trk_dxy) < 0.2)
            & (abs(events.ScoutingMuon.trackIso) < 0.15)
            & (abs(events.ScoutingMuon.trk_dz) < 0.5)
            #& (events.ScoutingMuon["type"] == 2)
            & (events.ScoutingMuon.normchi2 < 10)
            & (events.ScoutingMuon.nValidRecoMuonHits > 0)
            & (events.ScoutingMuon.nRecoMuonMatchedStations > 1)
            & (events.ScoutingMuon.nValidPixelHits > 0)
            & (events.ScoutingMuon.nTrackerLayersWithMeasurement > 5)

        ]
        
        fatjets = events.ScoutingFatJet
        fatjets = fatjets[
            (fatjets.neHEF < 0.9)
            & (fatjets.neEmEF < 0.9)
            & (fatjets.muEmEF < 0.8)
            & (fatjets.chHEF > 0.01)
            & (fatjets.nCh > 0)
            & (fatjets.chEmEF < 0.8)
        ]

        jets = events.ScoutingJet
        jets = jets[
            (jets.neHEF < 0.9)
            & (jets.neEmEF < 0.9)
            & (jets.muEmEF < 0.8)
            & (jets.chHEF > 0.01)
            & (jets.nCh > 0)
            & (jets.chEmEF < 0.8)
        ]

        # for each event we only keep the leading jet
        fatjet = ak.firsts(fatjets[
            (fatjets.pt > 50)
            & (abs(fatjets.eta) < 2.5)
            & ak.all(fatjets.metric_table(muons) > 0.8, axis=-1)  # default metric: delta_r
        ])

        jet = ak.firsts(jets[
            (jets.pt > 0)
            & (abs(jets.eta) < 2.5)
            & ak.all(jets.metric_table(muons) > 0.4, axis=-1)  # default metric: delta_r
        ])
        
        # this is the minimum requirement
        # 1. the jet exist
        # 2. the event passed our reference trigger
        fatjet_exists = ~ak.is_none(fatjet) & reftrigger
        jet_exists = ~ak.is_none(jet) & reftrigger

        for jet_type in ["ak4", "ak8"]:

            if jet_type == "ak4":
                   tmpjet = jet
                   tmp_exists = jet_exists
                   mass_type = "mass"
            else:
                   tmpjet = fatjet
                   tmp_exists = fatjet_exists
                   mass_type = "msoftdrop"

            # now we start filling the histograms which we will use to calculate the scale factors
            # the first one only contains the events that passed the minimum requirement (jet exist and event passed the reference trigger)
            self._output[jet_type].fill(
                dataset = dataset,
                pt = tmpjet[tmp_exists].pt,
                mass = tmpjet[tmp_exists][mass_type],
                trigger="none",
            )
            
            # the next requires the minimum AND that the jet passed ANY of the signal triggers
            cut = tmp_exists & PackedSelection_any(triggers, *set(trigger_names))
            self._output[jet_type].fill(
                dataset=dataset,
                pt=tmpjet[cut].pt,
                mass = tmpjet[cut][mass_type],
                trigger="any",
            )
            
            # this is already enough to compute the trigger efficiency. However as mentioned above, we also keep track of
            # a histogram containing the events that passed each individual signal trigger

            # loop over all signal triggers
            for tname in trigger_names:
                # require the minimum AND that the jet passed the selected signal trigger
                cut = tmp_exists & triggers.all(tname)
                self._output[jet_type].fill(
                    dataset=dataset,
                    pt=tmpjet[cut].pt,
                    mass = tmpjet[cut][mass_type],
                    trigger=tname,
                )

        return self._output

    def postprocess(self, accumulator):
        pass



