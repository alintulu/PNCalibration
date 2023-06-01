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
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty

def PackedSelection_any(self, *names):
    consider = 0
    for name in names:
        idx = self._names.index(name)
        consider |= 1 << idx
    return (self._data & consider) != 0

class TriggerDijetProcessor(processor.ProcessorABC):
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
                'HLT_PFHT1050',
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
            hist.axis.Regular(100, 0, 1000, name="pt", label="Average jet $p_T$"),
            hist.axis.Variable([0, 1.3, 2.5], name="eta1", label=r"|$\eta_{1}$|"),
            hist.axis.Variable([0, 1.3, 2.5], name="eta2", label=r"|$\eta_{2}$|"),
        )
        
        self._output = {
                "nevents": 0,
                "ak4": Hist(
                    *commonaxes
                ),
            }
        
        # scouting JEC
        ext = extractor()
        ext.add_weight_sets([
            "* * data/jec/Run3Summer21_V2_MC_L1FastJet_AK4PFchsHLT.txt",
            "* * data/jec/Run3Summer21_V2_MC_L2Relative_AK4PFchsHLT.txt",
            "* * data/jec/Run3Summer21_V2_MC_L3Absolute_AK4PFchsHLT.txt",
        ])
        ext.finalize()
        evaluator = ext.make_evaluator()

        jec_stack_names = evaluator.keys()
        jec_inputs = {name: evaluator[name] for name in jec_stack_names}
        jec_stack = JECStack(jec_inputs)
        
        name_map = jec_stack.blank_name_map
        name_map['JetPt'] = 'pt'
        name_map['JetMass'] = 'mass'
        name_map['JetEta'] = 'eta'
        name_map['JetA'] = 'area'
        name_map['ptRaw'] = 'pt_raw'
        name_map['massRaw'] = 'mass_raw'
        name_map['Rho'] = 'rho'
        
        self._jet_factory_scouting = CorrectedJetsFactory(name_map, jec_stack)

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
        # all events need to pass the scouting dataset
        reftrigger *= ak.to_numpy(events.DST["Run3_PFScoutingPixelTracking"])
        
        def apply_jec(jets, rho_name):
            
            if "Scouting" in rho_name:
                
                jets["pt_raw"] = jets["pt"]
                jets["mass_raw"] = jets["mass"]
                jets['rho'] = ak.broadcast_arrays(events[rho_name], jets.pt)[0]
                corrected_jets = self._jet_factory_scouting.build(jets, lazy_cache=events.caches[0])
                
            else:
                
                jets['pt_raw'] = (1 - jets['rawFactor']) * jets['pt']
                jets['mass_raw'] = (1 - jets['rawFactor']) * jets['mass']
                jets['rho'] = ak.broadcast_arrays(events.Rho[rho_name], jets.pt)[0]
                
                corrected_jets = self._jet_factory_offline.build(jets, lazy_cache=events.caches[0])
                
            return corrected_jets
        
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
            jets = apply_jec(events.ScoutingJet, "ScoutingRho")
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

        jets = jets[
            (ak.num(jets) > 0)
            & (jets.pt > 30)
            & (abs(jets.eta) < 2.5)
            & ak.all(jets.metric_table(muons) > 0.4, axis=-1)  # default metric: delta_r
        ]
        
        def require_back2back(obj1, obj2, phi=2.7):

            return (abs(obj1.delta_phi(obj2)) > phi)

        def require_3rd_jet(jets, pt_type="pt"):

            jet = jets[:, 2]
            pt_ave = (jets[:, 0][pt_type] + jets[:, 1][pt_type]) / 2

            return ~((jet[pt_type] > 30) & ((jet[pt_type] / pt_ave) > 0.2))

        def require_n(jets, two=True):

            if two:
                req = (ak.num(jets) == 2)
            else:
                req = (ak.num(jets) > 2)

            return req
        
        def require_eta(jets):
    
            return ((abs(jets[:, 0].eta) < 1.3) | (abs(jets[:, 1].eta) < 1.3))

        def criteria_one(jet, phi=2.7):

            b2b = require_back2back(jet[:, 0], jet[:, 1], phi)
            eta = require_eta(jet)

            return (b2b) # & (eta)]

        def criteria_n(jets, pt_type="pt", phi=2.7):

            third_jet = require_3rd_jet(jets, pt_type)

            b2b = require_back2back(jet[:, 0], jet[:, 1], phi)
            eta = require_eta(jet)

            return (third_jet) & (b2b) # & (eta)]
        
        pt_type = "pt_jec"       
        
        req_2 = require_n(jets, two=True)
        req_n = require_n(jets, two=False)

        for n, req in [("2", req_2), ("n", req_n)]:
        
            jet = jets[req]
            req_reftrigger = reftrigger[req]
            any_trigger = PackedSelection_any(triggers, *set(trigger_names))[req]
        
            if n == "2":
                criteria = criteria_one(jet, phi=2.7)
            else:
                criteria = criteria_n(jet, pt_type=pt_type, phi=2.7)
        
            # this is the minimum requirement
            # 1. the jet exist
            # 2. the event passed our reference trigger
            # 3. dijet criteria
            jet_exists = ~ak.is_none(jet) & req_reftrigger & criteria
            
            pt_ave = (jet[:,0][pt_type] + jet[:,1][pt_type]) / 2
            eta1 = np.abs(jet[:,0].eta)
            eta2 = np.abs(jet[:,1].eta)
            
            # loop over all signal triggers
            for tname in trigger_names:
                all_triggers = triggers.all(tname)[req]
                # require the minimum AND that the jet passed the selected signal trigger
                cut = jet_exists & all_triggers
                self._output["ak4"].fill(
                    dataset=dataset,
                    pt=pt_ave[cut],
                    eta1=eta1[cut],
                    eta2=eta2[cut],
                    trigger=tname,
                )

            # now we start filling the histograms which we will use to calculate the scale factors
            # the first one only contains the events that passed the minimum requirement (jet exist and event passed the reference trigger)
            self._output["ak4"].fill(
                dataset = dataset,
                pt = pt_ave[jet_exists],
                eta1=eta1[jet_exists],
                eta2=eta2[jet_exists],
                trigger="none",
            )

            # the next requires the minimum AND that the jet passed ANY of the signal triggers
            cut = jet_exists & any_trigger
            self._output["ak4"].fill(
                dataset=dataset,
                pt=pt_ave[cut],
                eta1=eta1[cut],
                eta2=eta2[cut],
                trigger="any",
            )

        return self._output

    def postprocess(self, accumulator):
        pass