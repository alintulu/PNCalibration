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
    def __init__(self, year="2022", isScouting=True):
        self._year = year
        self._isScouting = isScouting

        # here you should add your signal triggers (can be more than 1)
        self._scouting_sigtriggers = {
            '2022': [
                'L1_HTT200er',
                'L1_HTT225er',
                'L1_HTT280er',
                'L1_HTT320er',
                'L1_HTT360er',
                'L1_HTT400er',
                'L1_HTT450er',
                'L1_SingleJet180',
                'L1_SingleJet200',
            ]
        }
        self._offline_sigtriggers = {
            '2022': [
                "HLT_PFHT180",
                "HLT_PFHT250",
                "HLT_PFHT370",
                "HLT_PFHT430",
                "HLT_PFHT510",
                "HLT_PFHT590",
                "HLT_PFHT680",
                "HLT_PFHT780",
                "HLT_PFHT890",
                "HLT_PFHT1050",
                "HLT_PFJet40",
                "HLT_PFJet60",
                "HLT_PFJet60",
                "HLT_PFJet110",
                "HLT_PFJet140",
                "HLT_PFJet200",
                "HLT_PFJet260",
                "HLT_PFJet320",
                "HLT_PFJet400",
                "HLT_PFJet450",
                "HLT_PFJet500",
                "HLT_PFJet550",
            ]
        }

        if isScouting:
            self._sigtriggers = self._scouting_sigtriggers
        else:
            self._sigtriggers = self._offline_sigtriggers

        # here you should add your reference trigger
        self._reftriggers = {
            '2022': [
               'HLT_IsoMu27',
               'HLT_Mu50',
            ]
        }
        
        # to start with, we are interested in jet pt and mass, however you can use any jet variable
        commonaxes = (
            hist.axis.StrCategory([], name="dataset", label="Dataset name", growth=True),
            hist.axis.StrCategory([], name="trigger", label="Trigger name", growth=True),
            hist.axis.Regular(100, 0, 1000, name="pt", label="Leading jet $p_T$"),
            hist.axis.Regular(30, 0, 300, name="mass", label="Leading jet mass"),
            hist.axis.Regular(300, 0, 3000, name="ht", label="Event HT"),
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
            split = tname.split("_")
            start = split[0]
            rest = "_".join(split[1:])
            if rest in events[start].fields:
                triggers.add(tname, events[start][rest])
            else:
                triggers.add(tname, np.zeros(len(events), dtype=bool))
        
        # here we keep track of events passed the reference trigger
        reftrigger = np.zeros(len(events), dtype=bool)
        for tname in self._reftriggers[self._year]:
            split = tname.split("_")
            start = split[0]
            rest = "_".join(split[1:])
            if rest in events[start].fields:
                reftrigger |= ak.to_numpy(events[start][rest])
        
        if self._isScouting: 
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
        else:
            muons = events.Muon[
                (events.Muon.pt > 10)
                & (abs(events.Muon.eta) < 2.4)
                & (events.Muon.pfRelIso04_all < 0.25)
                & (events.Muon.looseId)
            ]
        
        if self._isScouting: 
            fatjets = events.ScoutingFatJet
            fatjets = fatjets[
                (fatjets.neHEF < 0.9)
                & (fatjets.neEmEF < 0.9)
                & (fatjets.muEmEF < 0.8)
                & (fatjets.chHEF > 0.01)
                & (fatjets.nCh > 0)
                & (fatjets.chEmEF < 0.8)
            ]
        else:
            fatjets = events.FatJet
            fatjets = fatjets[
                (abs(fatjets.eta) < 2.5)
                #& fatjets.isTight
            ]

        if self._isScouting: 
            jets = events.ScoutingJet
            jets = jets[
                (jets.neHEF < 0.9)
                & (jets.neEmEF < 0.9)
                & (jets.muEmEF < 0.8)
                & (jets.chHEF > 0.01)
                & (jets.nCh > 0)
                & (jets.chEmEF < 0.8)
            ]
        else:
            jets = events.Jet
            jets = jets[
                (abs(jets.eta) < 2.5)
                #& jets.isTight
            ]

        # for each event we only keep the leading jet
        fatjets = fatjets[
            (fatjets.pt > 170)
            & (abs(fatjets.eta) < 2.5)
            & ak.all(fatjets.metric_table(muons) > 0.8, axis=-1)  # default metric: delta_r
        ]
        fatjet = ak.firsts(fatjets)

        jets = jets[
            (jets.pt > 30)
            & (abs(jets.eta) < 2.5)
            & ak.all(jets.metric_table(muons) > 0.4, axis=-1)  # default metric: delta_r
        ]
        jet = ak.firsts(jets)
        
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
                   tmpjet["ht"] = ak.sum(jets.pt, axis=-1)
            else:
                   tmpjet = fatjet
                   tmp_exists = fatjet_exists
                   mass_type = "msoftdrop"
                   tmpjet["ht"] = ak.sum(fatjets.pt, axis=-1)

            # now we start filling the histograms which we will use to calculate the scale factors
            # the first one only contains the events that passed the minimum requirement (jet exist and event passed the reference trigger)
            self._output[jet_type].fill(
                dataset = dataset,
                pt = tmpjet[tmp_exists].pt,
                mass = tmpjet[tmp_exists][mass_type],
                ht = tmpjet[tmp_exists].ht,
                trigger="none",
            )
            
            # the next requires the minimum AND that the jet passed ANY of the signal triggers
            cut = tmp_exists & PackedSelection_any(triggers, *set(trigger_names))
            self._output[jet_type].fill(
                dataset=dataset,
                pt=tmpjet[cut].pt,
                mass = tmpjet[cut][mass_type],
                ht = tmpjet[cut].ht,
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
                    ht = tmpjet[cut].ht,
                    trigger=tname,
                )

        return self._output

    def postprocess(self, accumulator):
        pass



