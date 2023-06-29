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
from collections import defaultdict

from processors.helper import (
    run_deltar_matching,
    require_n,
    criteria_one,
    compute_asymmetry,
    select_eta,
)

class JERDijetV2Processor(processor.ProcessorABC):
    def __init__(self, isGen=False):
        
        self._commonaxes = (
            hist.axis.Variable([0, 85, 115, 145, 165, 210, 230, 295, 360, 445, 495, 550, 600] + list(np.arange(650, 2050, 50)), name="pt_ave", label=r"$p_{T}^{ave}$"),
            hist.axis.Regular(30, 0, 300, name="pt_third", label=r"$p_{T}^{third}$"),
            hist.axis.Variable([0, 0.01, 0.05, 0.1, 0.15, 0.2], name="alpha", label=r"$\alpha$"),
            hist.axis.StrCategory([], name="eta", label=r"|$\eta$|", growth=True),
            hist.axis.StrCategory([], name="dataset", label="Dataset name", growth=True),
        )
        
        self._triggers = {
            85 : ['L1_SingleJet60', 'HLT_PFJet60', 'HLT_PFJet80'],
            115 : ['L1_SingleJet90'],
            145 : ['L1_SingleJet120'],
            165 : ['HLT_PFJet140'],
            210 : ['L1_SingleJet180'],
            230 : ['L1_SingleJet200', 'HLT_PFJet200'],
            295 : ['HLT_PFJet260'],
            360 : ['HLT_PFJet320'],
            445 : ['HLT_PFJet400'], 
            495 : ['HLT_PFJet450'],
            550 : ['HLT_PFJet500'],
            600 : ['HLT_PFJet550', 'L1_SingleJet60', 'HLT_PFJet60', 'HLT_PFJet80', 'L1_SingleJet90', 'L1_SingleJet120', 'HLT_PFJet140', 'L1_SingleJet180', 'L1_SingleJet200', 'HLT_PFJet200', 'HLT_PFJet260', 'HLT_PFJet320', 'HLT_PFJet400', 'HLT_PFJet450', 'HLT_PFJet500'],
        }
        
        self._isGen = isGen
        self._recs = ["gen", "scouting", "offline"] if isGen else ["scouting", "offline"]
        
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
            "* * data/jec/Winter22Run3_V1_MC_L1FastJet_AK4PFchs.txt",
            "* * data/jec/Winter22Run3_V1_MC_L2Relative_AK4PFchs.txt",
            "* * data/jec/Winter22Run3_V1_MC_L3Absolute_AK4PFchs.txt",
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
        output = defaultdict()
        output["nevents"] = len(events)
        
        for X, triggers in self._triggers.items():
            
            h = {
                "scouting" : Hist(
                    *self._commonaxes,
                    hist.axis.Regular(200, -0.5, 0.5, name="asymmetry", label="Asymmetry")
                ),
                "scouting_mean": Hist(
                    *self._commonaxes,
                    storage=hist.storage.Mean()
                ),
                "offline" : Hist(
                    *self._commonaxes,
                    hist.axis.Regular(200, -0.5, 0.5, name="asymmetry", label="Asymmetry")
                ),
                "offline_mean": Hist(
                    *self._commonaxes,
                    storage=hist.storage.Mean()
                ),
                "gen" : Hist(
                    *self._commonaxes,
                    hist.axis.Regular(100, -0.5, 0.5, name="asymmetry", label="Asymmetry")
                ),
                "gen_mean": Hist(
                    *self._commonaxes,
                    storage=hist.storage.Mean()
                ),
                "scouting_gen" : Hist(
                    *self._commonaxes,
                    hist.axis.Regular(100, 0.5, 1.5, name="ratio", label=r"Leading scouting $p_T$/Leading gen $p_T$")
                ),
                "offline_gen" : Hist(
                    *self._commonaxes,
                    hist.axis.Regular(100, 0.5, 1.5, name="ratio", label="Leading offline $p_T$/Leading gen $p_T$")
                ),
            }

            paths = {}
            reftrigger = np.zeros(len(events), dtype=bool)

            for trigger in triggers:
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
            
            events_trigger = events[reftrigger]
            trigger = triggers[0]

            def apply_jec(jets, rho_name, events):

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

            muons_s = events_trigger.ScoutingMuon
            muons_s = muons_s[
                (muons_s.pt > 25)
                & (abs(muons_s.eta) < 2.4)
                & (abs(muons_s.trk_dxy) < 0.2)
                & (abs(muons_s.trackIso) < 0.15)
                & (abs(muons_s.trk_dz) < 0.5)
                #& (muons_s["type"] == 2)
                & (muons_s.normchi2 < 10)
                & (muons_s.nValidRecoMuonHits > 0)
                & (muons_s.nRecoMuonMatchedStations > 1)
                & (muons_s.nValidPixelHits > 0)
                & (muons_s.nTrackerLayersWithMeasurement > 5)

            ]

            muons_o = events_trigger.Muon
            muons_o = muons_o[
                (muons_o.pt > 10)
                & (abs(muons_o.eta) < 2.4)
                & (muons_o.pfRelIso04_all < 0.25)
                & (muons_o.looseId)
            ]

            jets_s = apply_jec(events_trigger.ScoutingJet, "ScoutingRho", events_trigger)
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

            jets_o = apply_jec(events_trigger.JetCHS, "fixedGridRhoFastjetAll", events_trigger)
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
                (ak.num(jets_s) > 1)
            ]
            jet_o = jets_o[
                (ak.num(jets_o) > 1)
            ]

            if self._isGen:
                jets_gen = events_trigger.GenJet
                jet_gen = jets_gen[
                    (ak.num(jets_gen) > 1)
                ]

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
                jet_n = criteria_one(jet_n, phi=2.7)

                asymmetry_2, asymmetry_2_cut = compute_asymmetry(jet_2)
                jet_2_cut = jet_2[asymmetry_2_cut]
                asymmetry_n, asymmetry_n_cut = compute_asymmetry(jet_n)
                jet_n_cut = jet_n[asymmetry_n_cut]
                
                for eta_region in ["barrel", "endcap"]:
                    for two in [True, False]:

                        if (two):
                            jet = jet_2_cut
                            asymmetry = asymmetry_2
                        else:
                            jet = jet_n_cut
                            asymmetry = asymmetry_n

                        eta_cut = select_eta(eta_region, jet)
                        jet = jet[eta_cut]
                        asymmetry = asymmetry[eta_cut]
            
                        h[rec].fill(
                            asymmetry = asymmetry,
                            pt_ave = (jet[:,0][pt_type] + jet[:,1][pt_type]) / 2,
                            eta = eta_region,
                            pt_third = 0.0 if two else jet[:,2][pt_type],
                            alpha = 0.0 if two else (jet[:,2][pt_type] / ((jet[:,0][pt_type] + jet[:,1][pt_type]) / 2)),
                            dataset = dataset,
                        )

                        h[rec + "_mean"].fill(
                            sample = asymmetry,
                            pt_ave = (jet[:,0][pt_type] + jet[:,1][pt_type]) / 2,
                            eta = eta_region,
                            pt_third = 0.0 if two else jet[:,2][pt_type],
                            alpha = 0.0 if two else (jet[:,2][pt_type] / ((jet[:,0][pt_type] + jet[:,1][pt_type]) / 2)),
                            dataset = dataset,
                        )
                        
                if self._isGen:
                    gens_orig = jets_gen
                    
                    for eta_region in ["barrel", "endcap"]:
                        for two in [True, False]:
                            for rec in ["scouting", "offline"]:

                                if (rec == "scouting"):
                                    jets_orig = jets_s
                                    pt_type = "pt_jec"
                                elif (rec == "offline"):
                                    jets_orig = jets_o
                                    pt_type = "pt_jec"
                                    
                                jets_cut = jets_orig[
                                    (ak.num(jets_orig) > 1)
                                    & (ak.num(gens_orig) > 1)
                                ]
                                gens_cut = gens_orig[
                                    (ak.num(jets_orig) > 1)
                                    & (ak.num(gens_orig) > 1)
                                ]

                                jets_cut, gens_cut = require_n(jets_cut, gens_cut, two=two)
                                jets_cut, gens_cut = criteria_one(jets_cut, gens_cut, phi=2.7)
                                eta_cut = select_eta(eta_region, jets_cut, gens_cut)
                                jets_cut = jets_cut[eta_cut]
                                gens_cut = gens_cut[eta_cut]

                                match = ak.flatten(run_deltar_matching(jets_cut, gens_cut, 0.2), axis=2)
                                jets_match = jets_cut[(ak.num(match) > 1)]
                                gens_match = gens_cut[(ak.num(match) > 1)]
                                
                                h[f"{rec}_gen"].fill(
                                    pt_ave = (jets_match[:,0][pt_type] + jets_match[:,1][pt_type]) / 2,
                                    eta = eta_region,
                                    pt_third = 0.0 if two else jets_match[:,2][pt_type],
                                    alpha = 0.0 if two else (jets_match[:,2][pt_type] / ((jets_match[:,0][pt_type] + jets_match[:,1][pt_type]) / 2)),
                                    dataset = dataset,
                                    ratio = jets_match[:,0][pt_type] / gens_match[:,0].pt
                                )
                            
            output[trigger] = h 
        
        return output

    def postprocess(self, accumulator):
        pass
