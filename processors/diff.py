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

class DiffProcessor(processor.ProcessorABC):
    def __init__(self, triggers):

        self._triggers = triggers
        
        self._output = {
                "nevents": 0,
                "pt": Hist(
                    hist.axis.Regular(100, 100, 1000, name="hlt", label=r"$p_{T}^{HLT}$"),
                    hist.axis.Regular(100, 100, 1000, name="reco", label=r"$p_{T}^{RECO}$"),
                ),
                "pt_mean": Hist(
                    hist.axis.Regular(18, 100, 1000, name="hlt", label=r"$p_{T}^{HLT}$"),
                    storage=hist.storage.Mean()
                ),
                "mass": Hist(
                    hist.axis.Regular(100, 0, 300, name="hlt", label=r"$mass^{HLT}$"),
                    hist.axis.Regular(100, 0, 300, name="reco", label=r"$mass^{RECO}$"),
                ),
                "mass_mean": Hist(
                    hist.axis.Regular(18, 100, 1000, name="hlt", label=r"$p_{T}^{HLT}$"),
                    storage=hist.storage.Mean()
                ),
                "eta": Hist(
                    hist.axis.Regular(50, 0, 3, name="hlt", label=r"$\eta^{HLT}$"),
                    hist.axis.Regular(50, 0, 3, name="reco", label=r"$\eta^{RECO}$"),
                ),
                "eta_mean": Hist(
                    hist.axis.Regular(18, 100, 1000, name="hlt", label=r"$p_{T}^{HLT}$"),
                    storage=hist.storage.Mean()
                ),
                "phi": Hist(
                    hist.axis.Regular(50, 0, 4, name="hlt", label=r"$\phi^{HLT}$"),
                    hist.axis.Regular(50, 0, 4, name="reco", label=r"$\phi^{RECO}$"),
                ),
                "phi_mean": Hist(
                    hist.axis.Regular(18, 100, 1000, name="hlt", label=r"$p_{T}^{HLT}$"),
                    storage=hist.storage.Mean()
                ),

            }
        
        self._pt_type = "pt_jec"
        self._mass_type = "mass_jec"
        
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
            (ak.num(jets_s) > 2)
            & (ak.num(jets_o) > 2)
        ][:,:3]
        jets_oo = jets_o[
            (ak.num(jets_s) > 2)
            & (ak.num(jets_o) > 2)
        ][:,:3]

        def require_dijets(jets, pt_type="pt"):

            dijets = (
                (abs(jets[:, 0].delta_phi(jets[:, 1])) > 2.7)
                & (jets[:,2][pt_type] < 0.1 * (jets[:,0][pt_type] + jets[:,1][pt_type]) / 2)
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
            obj2 = obj2[obj2.dR < radius] #Additional cut on delta R, now a NxMxG' array 
            return obj2
    
        req_dijets_s = require_dijets(jets_ss, self._pt_type)
        req_dijets_o = require_dijets(jets_oo)

        dijets_s = jets_ss[(req_dijets_s) & (req_dijets_o)][:,:2]
        dijets_o = jets_oo[(req_dijets_s) & (req_dijets_o)][:,:2]
        
        dijets_o_dr = ak.flatten(run_deltar_matching(dijets_s, dijets_o, 0.2), axis=2)
        dijet_s_dr = dijets_s[(ak.num(dijets_o_dr) > 1)]
        dijet_o_dr = dijets_o_dr[(ak.num(dijets_o_dr) > 1)]
        
        def fill(hlt_jets, reco_jets, var):

            var_tmp = var
            if var == "pt":
                var_tmp = self._pt_type
            if var == "mass":
                var_tmp = self._mass_type
            
            for i in [0, 1]:

                self._output[var].fill(
                    hlt = abs(hlt_jets[:, i][var_tmp]),
                    reco = abs(reco_jets[:, i][var]),
                )

                self._output[var + "_mean"].fill(
                    sample = abs(hlt_jets[:, i][var_tmp]) / abs(reco_jets[:, i][var]),
                    hlt = hlt_jets[:, i][self._pt_type],
                )

        for var in ["pt", "mass", "eta", "phi"]:

            fill(dijet_s_dr, dijet_o_dr, var) 
        
        return self._output

    def postprocess(self, accumulator):
        pass
