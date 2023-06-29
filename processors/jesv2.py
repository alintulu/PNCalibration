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
    cut_ratio,
    select_eta,
)

class JESV2Processor(processor.ProcessorABC):
    def __init__(self):

        self._commonaxes = (
            hist.axis.Variable([0, 85, 115, 145, 165, 210, 230, 295, 360, 445, 495, 550, 600] + list(np.arange(650, 2050, 50)), name="pt", label=r"$p_{T}^{ave}$"),
            hist.axis.StrCategory([], name="dataset", label="Dataset name", growth=True),
            hist.axis.StrCategory([], name="eta", label=r"|$\eta$|", growth=True),
            hist.axis.Variable([0, 0.01, 0.05, 0.1, 0.15, 0.2], name="alpha", label=r"$\alpha$"),
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
        output = defaultdict()
        output["nevents"] = len(events)
        
        for X, triggers in self._triggers.items():
            
            h = {
                "h1": Hist(
                    *self._commonaxes,
                    hist.axis.Regular(100, 0, 2, name="ratio", label=r"$p_{T}^{HLT,probe}$ / $p_{T}^{RECO,tag}$")
                ),
                "h1_mean": Hist(
                    *self._commonaxes,
                    storage=hist.storage.Mean()
                ),
                "h2": Hist(
                    *self._commonaxes,
                    hist.axis.Regular(100, 0, 2, name="ratio", label=r"$p_{T}^{RECO,probe}$ / $p_{T}^{RECO,tag}$"),
                ),
                "h2_mean": Hist(
                    *self._commonaxes,
                    storage=hist.storage.Mean()
                ),
                "h3": Hist(
                    *self._commonaxes,
                    hist.axis.Regular(100, 0, 2, name="ratio", label=r"$p_{T}^{HLT,probe}$ / $p_{T}^{RECO,probe}$"),
                ),
                "h3_mean": Hist(
                    *self._commonaxes,
                    storage=hist.storage.Mean()
                ),
                "h4": Hist(
                    *self._commonaxes,
                    hist.axis.Variable([0, 85, 115, 145, 165, 210, 230, 295, 360, 445, 495, 550, 600] + list(np.arange(650, 2050, 50)), name="ratio", label=r"$p_{T}^{HLT,probe}$"),
                ),
                "h4_mean": Hist(
                    *self._commonaxes,
                    storage=hist.storage.Mean()
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

            jets_s = apply_jec(events_trigger.ScoutingJet, "ScoutingRho", events_trigger)
            pt_type = "pt_jec"
            jets_s = jets_s[
                (abs(jets_s.eta) < 2.5)
                & (jets_s.neHEF < 0.9)
                & (jets_s.neEmEF < 0.9)
                & (jets_s.muEmEF < 0.8)
                & (jets_s.chHEF > 0.01)
                #& (jets_s.nCh > 0)
                & (jets_s.chEmEF < 0.8)
            ]

            jets_o = apply_jec(events_trigger.Jet, "fixedGridRhoFastjetAll", events_trigger)
            jets_o = jets_o[
                (abs(jets_o.eta) < 2.5)
                & (jets_o.neHEF < 0.9)
                & (jets_o.neEmEF < 0.9)
                & (jets_o.muEF < 0.8)
                & (jets_o.chHEF > 0.01)
                #& (jets_o.nCh > 0)
                & (jets_o.chEmEF < 0.8)
            ]

            jet_s = jets_s[
                (ak.num(jets_s) > 1)
                & (ak.num(jets_o) > 1)
            ]
            jet_o = jets_o[
                (ak.num(jets_s) > 1)
                & (ak.num(jets_o) > 1)
            ]

            jets_s_2, jets_o_2 = require_n(jet_s, jet_o, two=True)
            jets_s_n, jets_o_n = require_n(jet_s, jet_o, two=False)

            dijet_s_2, dijet_o_2 = criteria_one(jets_s_2, jets_o_2, phi=2.7)
            dijet_s_n, dijet_o_n = criteria_one(jets_s_n, jets_o_n, phi=2.7)

            dijet_2_dr = ak.flatten(run_deltar_matching(dijet_s_2, dijet_o_2, 0.2), axis=2)
            dijet_s_2_dr = dijet_s_2[(ak.num(dijet_2_dr) > 1)]
            dijet_o_2_dr = dijet_o_2[(ak.num(dijet_2_dr) > 1)]

            dijet_n_dr = ak.flatten(run_deltar_matching(dijet_s_n, dijet_o_n, 0.2), axis=2)
            dijet_s_n_dr = dijet_s_n[(ak.num(dijet_n_dr) > 1)]
            dijet_o_n_dr = dijet_o_n[(ak.num(dijet_n_dr) > 1)]
            
            for eta_region in ["barrel", "endcap"]:
                for two in [True, False]:

                    if (two):
                        dijet_s_dr = dijet_s_2_dr
                        dijet_o_dr = dijet_o_2_dr
                    else:
                        dijet_s_dr = dijet_s_n_dr
                        dijet_o_dr = dijet_o_n_dr
                        
                    eta_cut = select_eta(eta_region, dijet_s_dr, dijet_o_dr)
                    dijet_s_dr = dijet_s_dr[eta_cut]
                    dijet_o_dr = dijet_o_dr[eta_cut]

                    for ijet, jjet in [(0, 1), (1, 0)]:

                        ratios = {
                               "h1" : dijet_s_dr[:, jjet][pt_type] / dijet_o_dr[:, ijet][pt_type],
                               "h2" : dijet_o_dr[:, jjet][pt_type] / dijet_o_dr[:, ijet][pt_type],
                               "h3" : dijet_s_dr[:, jjet][pt_type] / dijet_o_dr[:, jjet][pt_type],
                               "h4" : dijet_s_dr[:, jjet][pt_type],
                        }

                        for key, value in ratios.items():

                            value_cut, cut = cut_ratio(value, h4=True if key == "h4" else False)
                            dijet_o_dr_cut = dijet_o_dr[cut]
                            dijet_s_dr_cut = dijet_s_dr[cut]

                            h[key].fill(
                                ratio = value_cut,
                                pt = dijet_o_dr_cut[:, ijet][pt_type],
                                dataset = dataset,
                                eta = eta_region,
                                alpha = 0.0 if two else (dijet_o_dr_cut[:,2][pt_type] / ((dijet_o_dr_cut[:,0][pt_type] + dijet_o_dr_cut[:,1][pt_type]) / 2)),
                            )

                            h[f"{key}_mean"].fill(
                                sample = value_cut,
                                pt = dijet_o_dr_cut[:, ijet][pt_type],
                                dataset = dataset,
                                eta = eta_region,
                                alpha = 0.0 if two else (dijet_o_dr_cut[:,2][pt_type] / ((dijet_o_dr_cut[:,0][pt_type] + dijet_o_dr_cut[:,1][pt_type]) / 2)),
                            )
                        
            output[trigger] = h

        return output

    def postprocess(self, accumulator):
        pass
