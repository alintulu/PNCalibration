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

class JERProcessor(processor.ProcessorABC):
    def __init__(self, triggers=[]):
        commonaxes = (
            hist.axis.Regular(18, 100, 1000, name="pt_ave", label=r"Average $p_T$"),
            #hist.axis.Regular(20, 0, 1, name="pt_third", label=r"(2 $\times$ $p_{T,3}$) / ($p_{T,1}$ + $p_{T,2}$)"),
        )

        self._triggers = triggers
        
        self._output = {
                "nevents": 0,
                "scouting": Hist(
                    *commonaxes,
                    hist.axis.Regular(100, -0.5, 0.5, name="asymmetry", label="Asymmetry")
                ),
                "scouting_mean": Hist(
                    *commonaxes,
                    storage=hist.storage.Mean()
                ),
                "offline_mean": Hist(
                    *commonaxes,
                    storage=hist.storage.Mean()
                ),
                "offline": Hist(
                    *commonaxes,
                    hist.axis.Regular(100, -0.5, 0.5, name="asymmetry", label="Asymmetry")
                ),
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
        self._output["nevents"] = len(events)

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

            exactly_two = (
                (ak.num(jets) == 2)
                & (abs(jets[:, 0].delta_phi(jets[:, 1])) > 2.7)
            )

            more_than_two = (ak.num(jets) > 2)
            jets_more = jets[more_than_two]
            more_than_two = (
                (abs(jets_more[:, 0].delta_phi(jets_more[:, 1])) > 2.7)
                & (jets_more[:,2][pt_type] < 0.1 * (jets_more[:,0][pt_type] + jets_more[:,1][pt_type]) / 2)
            )

            #dijets = (
            #    (abs(jets[:, 0].delta_phi(jets[:, 1])) > 2.7)
            #    & (jets[:,2][pt_type] < 0.1 * (jets[:,0][pt_type] + jets[:,1][pt_type]) / 2)
            #)

            return (exactly_two | more_than_two)
        
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

        dijets_s = jets_ss[(req_dijets_s) & (req_dijets_o)][:,:3]
        dijets_o = jets_oo[(req_dijets_s) & (req_dijets_o)][:,:3]
        
#         dijets_o_dr = ak.flatten(run_deltar_matching(dijets_s, dijets_o, 0.2), axis=2)
#         dijet_s_dr = dijets_s[(ak.num(dijets_o_dr) > 1)]
#         dijet_o_dr = dijets_o_dr[(ak.num(dijets_o_dr) > 1)]
        
#         def get_asymmetry(jets, pt_type="pt"):
            
#             shuffle = random.choices([-1, 1], k=len(jets[:,0]))
#             shuffle_opp = [s * -1 for s in shuffle]
            
#             asymmetry = (shuffle * jets[:,0][pt_type] + shuffle_opp * jets[:,1][pt_type]) / (jets[:,0][pt_type] + jets[:,1][pt_type])
#             pt_ave = (jets[:,0][pt_type] + jets[:,1][pt_type]) / 2
#             pt_third = (2 * jets[:,2][pt_type]) / (jets[:,0][pt_type] + jets[:,1][pt_type])

#             return asymmetry, pt_ave, pt_third

#         a_s, pt_s = get_asymmetry(dijet_s_dr, "pt_jec")
#         a_o, pt_o = get_asymmetry(dijet_o_dr)

        
        def fill_asymmetry(jets, rec, pt_type="pt"):
            
            for i, j in [(0, 1), (1, 0)]:
                
                asymmetry = (jets[:, i][pt_type] - jets[:, j][pt_type]) / (jets[:, i][pt_type] + jets[:, j][pt_type])
                pt_ave = (jets[:, i][pt_type] + jets[:, j][pt_type]) / 2
        
                self._output[rec + "_mean"].fill(
                    sample = asymmetry,
                    pt_ave = pt_ave,
                )

                self._output[rec].fill(
                    asymmetry = asymmetry,
                    pt_ave = pt_ave,
                )
                
        fill_asymmetry(dijets_s, "scouting", "pt_jec")
        fill_asymmetry(dijets_o, "offline")
        
        return self._output

    def postprocess(self, accumulator):
        pass
