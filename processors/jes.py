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

class JESProcessor(processor.ProcessorABC):
    def __init__(self, triggers=[]):
        commonaxes = (
            hist.axis.Regular(20, 0, 1000, name="pt", label=r"$p_{T}^{RECO,tag}$"),
        )
        
        self._triggers = triggers
        
        self._output = {
                "nevents": 0,
                "h1": Hist(
                    *commonaxes,
                    hist.axis.Regular(100, 0, 2, name="ratio", label=r"$p_{T}^{HLT,probe}$ / $p_{T}^{RECO,tag}$")
                ),
                "h1_mean": Hist(
                    *commonaxes,
                    storage=hist.storage.Mean()
                ),
                "h2": Hist(
                    *commonaxes,
                    hist.axis.Regular(100, 0, 2, name="ratio", label=r"$p_{T}^{RECO,probe}$ / $p_{T}^{RECO,tag}$"),
                ),
                "h2_mean": Hist(
                    *commonaxes,
                    storage=hist.storage.Mean()
                ),
                "h3": Hist(
                    *commonaxes,
                    hist.axis.Regular(100, 0, 2, name="ratio", label=r"$p_{T}^{HLT,probe}$ / $p_{T}^{RECO,probe}$"),
                ),
                "h3_mean": Hist(
                    *commonaxes,
                    storage=hist.storage.Mean()
                ),
                "h4": Hist(
                    hist.axis.Regular(20, 0, 1000, name="pt_reco", label=r"$p_{T}^{RECO,tag}$"),
                    hist.axis.Regular(50, 0, 1000, name="pt_hlt", label=r"$p_{T}^{HLT,probe}$"),
                ),
                "h4_mean": Hist(
                    hist.axis.Regular(20, 0, 1000, name="pt_reco", label=r"$p_{T}^{RECO,tag}$"),
                    storage=hist.storage.Mean()
                ),
#                 "h5": Hist(
#                     hist.axis.Regular(20, 0, 1000, name="pt", label=r"$p_{T}^{HLT,probe}$"),
#                     hist.axis.Regular(100, 0, 2, name="ratio", label=r"$p_{T}^{HLT,probe}$ / $p_{T}^{RECO,probe}$"),
#                 ),
#                 "h5_mean": Hist(
#                     hist.axis.Regular(20, 0, 1000, name="pt", label=r"$p_{T}^{HLT,probe}$"),
#                     storage=hist.storage.Mean()
#                 ),
#                 "hlt_probe_mean": Hist(
#                     hist.axis.Regular(20, 0, 1000, name="pt", label=r"$p_{T}^{HLT,probe}$"),
#                     storage=hist.storage.Mean()
#                 ),
#                 "reco_probe_mean": Hist(
#                     hist.axis.Regular(20, 0, 1000, name="pt", label=r"$p_{T}^{HLT,probe}$"),
#                     storage=hist.storage.Mean()
#                 ),
  
            }
        
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
        
        self._jet_factory = CorrectedJetsFactory(name_map, jec_stack)

    def process(self, events):
        
        dataset = events.metadata['dataset']
        
        if self._triggers:
            paths = {}
            reftrigger = np.zeros(len(events), dtype=bool)

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

            events = events[reftrigger]
        
        self._output["nevents"] = len(events)

        def apply_jec(jets, rho_name):
            
            jets["pt_raw"] = jets["pt"]
            jets["mass_raw"] = jets["mass"]
            jets['rho'] = ak.broadcast_arrays(events[rho_name], jets.pt)[0]
            
            corrected_jets = self._jet_factory.build(jets, lazy_cache=events.caches[0])
            return corrected_jets

        # scouting        
        jets_s = apply_jec(events.ScoutingJet, "ScoutingRho")
        pt_type = "pt_jec"
        #jets_s = events.ScoutingJet
        jets_s = jets_s[
            (abs(jets_s.eta) < 2.5)
            #& (jets_s.pt > 20)
            & (jets_s.neHEF < 0.9)
            & (jets_s.neEmEF < 0.9)
            & (jets_s.muEmEF < 0.8)
            & (jets_s.chHEF > 0.01)
            #& (jets_s.nCh > 0)
            & (jets_s.chEmEF < 0.8)
        ]

        jets_o = events.JetCHS
        jets_o = jets_o[
            (abs(jets_o.eta) < 2.5)
            #& (jets_o.pt > 20)
            & (jets_o.neHEF < 0.9)
            & (jets_o.neEmEF < 0.9)
            & (jets_o.muEF < 0.8)
            & (jets_o.chHEF > 0.01)
            #& (jets_o.nCh > 0)
            & (jets_o.chEmEF < 0.8)
        ]

        jets_ss = jets_s[
            (ak.num(jets_s) > 1)
            & (ak.num(jets_o) > 1)
        ]
        jets_oo = jets_o[
            (ak.num(jets_s) > 1)
            & (ak.num(jets_o) > 1)
        ]

        def require_dijets(jets, pt_type="pt"):

            dijets = (
                (abs(jets[:, 0].delta_phi(jets[:, 1])) > 2.7)
                & (
                  (ak.num(jets) == 2)
                  | ((ak.num(jets) > 2) & (jets[:,2][pt_type] < 0.1 * (jets[:,0][pt_type] + jets[:,1][pt_type]) / 2)))
            )

            return dijets
        
        def run_deltar_matching(obj1, obj2, radius=0.4): # NxM , NxG arrays
            _, obj2 = ak.unzip(ak.cartesian([obj1, obj2], nested=True)) # Obj2 is now NxMxG
            obj2['dR'] = obj1.delta_r(obj2)  # Calculating delta R
            t_index = ak.argmin(obj2.dR, axis=-2) # Finding the smallest dR (NxG array)
            s_index = ak.local_index(obj1.eta, axis=-1) #  NxM array
            _, t_index = ak.unzip(ak.cartesian([s_index, t_index], nested=True)) 
            obj2 = obj2[s_index == t_index] # Pairwise comparison to keep smallest delta R

            # Cutting on delta R
            obj2 = obj2[obj2.dR < radius] # Additional cut on delta R, now a NxMxG' array 
            return obj2
    
        req_dijets_s = require_dijets(jets_ss, "pt_jec")
        req_dijets_o = require_dijets(jets_oo)

        dijets_s = jets_ss[(req_dijets_s) & (req_dijets_o)][:,:2]
        dijets_o = jets_oo[(req_dijets_s) & (req_dijets_o)][:,:2]
        
        dijets_o_dr = ak.flatten(run_deltar_matching(dijets_s, dijets_o, 0.2), axis=2)
        dijet_s_dr = dijets_s[(ak.num(dijets_o_dr) > 1)]
        dijet_o_dr = dijets_o_dr[(ak.num(dijets_o_dr) > 1)]
        
        self._output["h1"].fill(
            ratio = dijet_s_dr[:, 1][pt_type] / dijet_o_dr[:, 0].pt,
            pt = dijet_o_dr[:, 0].pt,
        )
        
        self._output["h1"].fill(
            ratio = dijet_s_dr[:, 0][pt_type] / dijet_o_dr[:, 1].pt,
            pt = dijet_o_dr[:, 1].pt,
        )
        
        self._output["h1_mean"].fill(
            sample = dijet_s_dr[:, 1][pt_type] / dijet_o_dr[:, 0].pt,
            pt = dijet_o_dr[:, 0].pt,
        )
        
        self._output["h1_mean"].fill(
            sample = dijet_s_dr[:, 0][pt_type] / dijet_o_dr[:, 1].pt,
            pt = dijet_o_dr[:, 1].pt,
        )
        
        self._output["h2"].fill(
            ratio = dijet_o_dr[:, 1].pt / dijet_o_dr[:, 0].pt,
            pt = dijet_o_dr[:, 0].pt,
        )
        
        self._output["h2"].fill(
            ratio = dijet_o_dr[:, 0].pt / dijet_o_dr[:, 1].pt,
            pt = dijet_o_dr[:, 1].pt,
        )
        
        self._output["h2_mean"].fill(
            sample = dijet_o_dr[:, 1].pt / dijet_o_dr[:, 0].pt,
            pt = dijet_o_dr[:, 0].pt,
        )
        
        self._output["h2_mean"].fill(
            sample = dijet_o_dr[:, 0].pt / dijet_o_dr[:, 1].pt,
            pt = dijet_o_dr[:, 1].pt,
        )
        
        self._output["h3"].fill(
            ratio = dijet_s_dr[:, 1][pt_type] / dijet_o_dr[:, 1].pt,
            pt = dijet_o_dr[:, 0].pt,
        )
        
        self._output["h3"].fill(
            ratio = dijet_s_dr[:, 0][pt_type] / dijet_o_dr[:, 0].pt,
            pt = dijet_o_dr[:, 1].pt,
        )
        
        self._output["h3_mean"].fill(
            sample = dijet_s_dr[:, 1][pt_type] / dijet_o_dr[:, 1].pt,
            pt = dijet_o_dr[:, 0].pt,
        )
        
        self._output["h3_mean"].fill(
            sample = dijet_s_dr[:, 0][pt_type] / dijet_o_dr[:, 0].pt,
            pt = dijet_o_dr[:, 1].pt,
        )
        
        self._output["h4"].fill(
            pt_hlt = dijet_s_dr[:, 1][pt_type],
            pt_reco = dijet_o_dr[:, 0].pt,
        )
        
        self._output["h4"].fill(
            pt_hlt = dijet_s_dr[:, 0][pt_type],
            pt_reco = dijet_o_dr[:, 1].pt,
        )
        
        self._output["h4_mean"].fill(
            sample = dijet_s_dr[:, 1][pt_type],
            pt_reco = dijet_o_dr[:, 0].pt,
        )
        
        self._output["h4_mean"].fill(
            sample = dijet_s_dr[:, 0][pt_type],
            pt_reco = dijet_o_dr[:, 1].pt,
        )

#         self._output["h5"].fill(
#             ratio = dijet_s_dr[:, 1][pt_type] / dijet_o_dr[:, 1].pt,
#             pt = dijet_s_dr[:, 1].pt,
#         )
        
#         self._output["h5"].fill(
#             ratio = dijet_s_dr[:, 0][pt_type] / dijet_o_dr[:, 0].pt,
#             pt = dijet_s_dr[:, 0].pt,
#         )
        
#         self._output["h5_mean"].fill(
#             sample = dijet_s_dr[:, 1][pt_type] / dijet_o_dr[:, 1].pt,
#             pt = dijet_s_dr[:, 1].pt,
#         )
        
#         self._output["h5_mean"].fill(
#             sample = dijet_s_dr[:, 0][pt_type] / dijet_o_dr[:, 0].pt,
#             pt = dijet_s_dr[:, 0].pt,
#         )
        
#         self._output["hlt_probe_mean"].fill(
#             sample = dijet_s_dr[:, 0][pt_type],
#             pt = dijet_s_dr[:, 0].pt,
#         )
        
#         self._output["hlt_probe_mean"].fill(
#             sample = dijet_s_dr[:, 1][pt_type],
#             pt = dijet_s_dr[:, 1].pt,
#         )
        
#         self._output["reco_probe_mean"].fill(
#             sample = dijet_o_dr[:, 0].pt,
#             pt = dijet_s_dr[:, 0].pt,
#         )
        
#         self._output["reco_probe_mean"].fill(
#             sample = dijet_o_dr[:, 1].pt,
#             pt = dijet_s_dr[:, 1].pt,
#         )
        
        return self._output

    def postprocess(self, accumulator):
        pass
