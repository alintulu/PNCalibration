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
import random
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty

class JERDijetProcessor(processor.ProcessorABC):
    def __init__(self, triggers=[], isGen=False, lowPt=False):
        
        commonaxes = (
            hist.axis.Regular(40, 0, 2000, name="pt_ave", label=r"$p_{T}^{RECO,tag}$"),
            hist.axis.Variable([0, 1.3, 2.5], name="eta1", label=r"|$\eta_{1}$|"),
            hist.axis.Variable([0, 1.3, 2.5], name="eta2", label=r"|$\eta_{2}$|"),
            hist.axis.StrCategory([], name="dataset", label="Dataset name", growth=True),
        )
        
        self._triggers = triggers
        self._isGen = isGen
        self._lowPt = lowPt
        self._recs = ["gen", "scouting", "offline"] if isGen else ["scouting", "offline"]
        
        self._output = {
                "nevents": 0,
                "scouting" : Hist(
                    *commonaxes,
                    hist.axis.Regular(100, -0.5, 0.5, name="asymmetry", label="Asymmetry")
                ),
                "scouting_mean": Hist(
                    *commonaxes,
                    storage=hist.storage.Mean()
                ),
                "offline" : Hist(
                    *commonaxes,
                    hist.axis.Regular(100, -0.5, 0.5, name="asymmetry", label="Asymmetry")
                ),
                "offline_mean": Hist(
                    *commonaxes,
                    storage=hist.storage.Mean()
                ),
                "gen" : Hist(
                    *commonaxes,
                    hist.axis.Regular(100, -0.5, 0.5, name="asymmetry", label="Asymmetry")
                ),
                "gen_mean": Hist(
                    *commonaxes,
                    storage=hist.storage.Mean()
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
        
        # offline JEC
        ext = extractor()
        ext.add_weight_sets([
            "* * data/jec/Winter22Run3_V2_MC_L1FastJet_AK4PFPuppi.txt",
            "* * data/jec/Winter22Run3_V2_MC_L2Relative_AK4PFPuppi.txt",
            "* * data/jec/Winter22Run3_V2_MC_L3Absolute_AK4PFPuppi.txt",
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
        
        self._jet_factory_offline = CorrectedJetsFactory(name_map, jec_stack)

    def process(self, events):
        
        dataset = events.metadata['dataset']
        self._output["nevents"] = len(events)
        
        def cond(old, new):
            if old == "": return new
            elif new == "": return old

            return old + " AND " + new
        
        if self._triggers:
            paths = {}
            reftrigger = np.zeros(len(events), dtype=bool)
            expression = []

            for trigger in self._triggers:
                split = trigger.split("_")
                start = split[0]
                rest = "_".join(split[1:])

                if start not in paths.keys():
                    paths[start] = [rest]
                else:
                    paths[start].append(rest)

            for key, values in paths.items():
                for value in values:
                    if value in events[key].fields:
                        reftrigger |= ak.to_numpy(events[key][value])

            if self._lowPt:
                reftrigger *= ~ak.to_numpy(events.HLT["PFHT1050"])
                        
            events = events[reftrigger]

        scoutingJets = events.ScoutingJet
        offlineJets = events.Jet #events.JetCHS

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
            
        muons_s = events.ScoutingMuon[
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

        muons_o = muons = events.Muon[
            (events.Muon.pt > 10)
            & (abs(events.Muon.eta) < 2.4)
            & (events.Muon.pfRelIso04_all < 0.25)
            & (events.Muon.looseId)
        ]

        jets_s = apply_jec(scoutingJets, "ScoutingRho") #events.ScoutingJet
        jets_s = jets_s[
            (abs(jets_s.eta) < 2.5)
            & (jets_s.pt > 12)
            & (jets_s.neHEF < 0.9)
            & (jets_s.neEmEF < 0.9)
            & (jets_s.muEmEF < 0.8)
            & (jets_s.chHEF > 0.01)
            & (jets_s.chEmEF < 0.8)
            & ak.all(jets_s.metric_table(muons_s) > 0.4, axis=-1)
        ]

        jets_o = apply_jec(offlineJets, "fixedGridRhoFastjetAll") #events.JetCHS
        jets_o = jets_o[
            (abs(jets_o.eta) < 2.5)
            & (jets_o.pt > 12)
            & (jets_o.neHEF < 0.9)
            & (jets_o.neEmEF < 0.9)
            & (jets_o.muEF < 0.8)
            & (jets_o.chHEF > 0.01)
            & (jets_o.chEmEF < 0.8)
            & ak.all(jets_o.metric_table(muons_o) > 0.4, axis=-1)
        ]

        jet_s = jets_s[
            (ak.num(jets_s) > 0)
        #     & (ak.num(jets_o) > 0)
        ]
        jet_o = jets_o[
            (ak.num(jets_o) > 0)
        #     & (ak.num(jets_s) > 0)
        ]
        
        if self._isGen:
            
            jets_gen = events.GenJet
            jet_gen = jets_gen[
                (ak.num(jets_gen) > 1)
            ]
            
        def require_back2back(obj1, obj2, phi=2.7):

            return (abs(obj1.delta_phi(obj2)) > phi)

        def require_3rd_jet(jets, pt_type="pt"):

            jet = jets[:, 2]
            pt_ave = (jets[:, 0][pt_type] + jets[:, 1][pt_type]) / 2

            return ~((jet[pt_type] > 30) & ((jet[pt_type] / pt_ave) > 0.2))

        def require_n(jets, two=True):

            if two:
                jet = jets[(ak.num(jets) == 2)][:, :2]
            else:
                jet = jets[(ak.num(jets) > 2)]

            return jet
        
        def require_eta(jets):
    
            return ((abs(jets[:, 0].eta) < 1.3) | (abs(jets[:, 1].eta) < 1.3))

        def criteria_one(jet, phi=2.7):

            b2b = require_back2back(jet[:, 0], jet[:, 1], phi)
            eta = require_eta(jet)

            return jet[(b2b)] # & (eta)]

        def criteria_n(jets, pt_type="pt", phi=2.7):

            third_jet = require_3rd_jet(jets, pt_type)

            jet = jets[third_jet][:, :2]

            b2b = require_back2back(jet[:, 0], jet[:, 1], phi)
            eta = require_eta(jet)

            return jet[(b2b)] # & (eta)]

        def compute_asymmetry(jets, pt_type="pt"):
            
            shuffle = random.choices([-1, 1], k=len(jets[:,0]))
            shuffle_opp = [s * -1 for s in shuffle]
            
            asymmetry = (shuffle * jets[:,0][pt_type] + shuffle_opp * jets[:,1][pt_type]) / (jets[:,0][pt_type] + jets[:,1][pt_type])
            
            asymmetry_cut = (np.abs(asymmetry) < 0.5)
            
            return asymmetry[asymmetry_cut], asymmetry_cut

        for rec in self._recs:
        
            if (rec == "scouting"):
                jets = jet_s
                pt_type = "pt_jec"
            elif (rec == "offline"):
                jets = jet_o
                pt_type = "pt_jec"
            else:
                jets = jet_gen
                pt_type = "pt"

            jet_2 = require_n(jets, two=True)
            jet_n = require_n(jets, two=False)

            jet_2 = criteria_one(jet_2, phi=2.7)
            jet_n = criteria_n(jet_n, pt_type=pt_type, phi=2.7)
            
            asymmetry_2, asymmetry_2_cut = compute_asymmetry(jet_2)
            jet_2_cut = jet_2[asymmetry_2_cut]
            asymmetry_n, asymmetry_n_cut = compute_asymmetry(jet_n)
            jet_n_cut = jet_n[asymmetry_n_cut]

            self._output[rec].fill(
                asymmetry = asymmetry_2,
                pt_ave = (jet_2_cut[:,0][pt_type] + jet_2_cut[:,1][pt_type]) / 2,
                eta1 = np.abs(jet_2_cut[:,0].eta),
                eta2 = np.abs(jet_2_cut[:,1].eta),
                dataset = dataset,
            )
            
            self._output[rec].fill(
                asymmetry = asymmetry_n,
                pt_ave = (jet_n_cut[:,0][pt_type] + jet_n_cut[:,1][pt_type]) / 2,
                eta1 = np.abs(jet_n_cut[:,0].eta),
                eta2 = np.abs(jet_n_cut[:,1].eta),
                dataset = dataset,
            )
            
            self._output[rec + "_mean"].fill(
                sample = asymmetry_2,
                pt_ave = (jet_2_cut[:,0][pt_type] + jet_2_cut[:,1][pt_type]) / 2,
                eta1 = np.abs(jet_2_cut[:,0].eta),
                eta2 = np.abs(jet_2_cut[:,1].eta),
                dataset = dataset,
            )
            
            self._output[rec + "_mean"].fill(
                sample = asymmetry_n,
                pt_ave = (jet_n_cut[:,0][pt_type] + jet_n_cut[:,1][pt_type]) / 2,
                eta1 = np.abs(jet_n_cut[:,0].eta),
                eta2 = np.abs(jet_n_cut[:,1].eta),
                dataset = dataset,
            )
            
        return self._output

    def postprocess(self, accumulator):
        pass
