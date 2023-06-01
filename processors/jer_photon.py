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

class JERPhotonProcessor(processor.ProcessorABC):
    def __init__(self, triggers=[]):
        
        commonaxes = (
            hist.axis.Variable(np.append(np.arange(0, 200, 20), np.arange(200, 1050, 50)), name="pt_ave", label=r"Average $p_T$"),
            hist.axis.StrCategory([], name="dataset", label="Dataset name", growth=True),
        )
        # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 23, 27, 30, 35, 40, 45, 57, 72, 90, 120, 150, 200, 300, 400, 550, 750, 1000
 
        self._triggers = triggers
        
        self._output = {
                "nevents": 0,
                "scouting" : Hist(
                    *commonaxes,
                    hist.axis.Regular(100, 0, 2, name="ratio", label=r"$p_T^{jet} / p_T^{\gamma}$")
                ),
                "scouting_mean": Hist(
                    *commonaxes,
                    storage=hist.storage.Mean()
                ),
                "offline" : Hist(
                    *commonaxes,
                    hist.axis.Regular(100, 0, 2, name="ratio", label=r"$p_T^{jet} / p_T^{\gamma}$")
                ),
                "offline_mean": Hist(
                    *commonaxes,
                    storage=hist.storage.Mean()
                ),
            }
        
        self._pt_type = "pt"
        
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
   
        jets_s = apply_jec(events.ScoutingJet, "ScoutingRho") #events.ScoutingJet
        jets_s = jets_s[
            (abs(jets_s.eta) < 1.3)
            & (jets_s.pt > 12)
            & (jets_s.neHEF < 0.9)
            & (jets_s.neEmEF < 0.9)
            & (jets_s.muEmEF < 0.8)
            & (jets_s.chHEF > 0.01)
            & (jets_s.chEmEF < 0.8)
        ]

        jets_o = events.JetCHS
        jets_o = jets_o[
            (abs(jets_o.eta) < 1.3)
            & (jets_o.pt > 12)
            & (jets_o.neHEF < 0.9)
            & (jets_o.neEmEF < 0.9)
            & (jets_o.muEF < 0.8)
            & (jets_o.chHEF > 0.01)
            & (jets_o.chEmEF < 0.8)
        ]

        phos_s = events.ScoutingPhoton
        phos_s = phos_s[
            (abs(phos_s.eta) < 1.3)
            & (phos_s.pt > 40)
            & (phos_s.sigmaIetaIeta < 0.0417588) #0.01
            & (phos_s.hOverE < 0.00999299) #0.03
            & (phos_s.ecalIso < 0.14189 + 0.000652035 * phos_s.pt)
            & (phos_s.hcalIso < 0.39057 + 0.0100547 * phos_s.pt + 5.78332e-05 * phos_s.pt**2)
        ]

        phos_o = events.Photon
        phos_o = phos_o[
            (abs(phos_o.eta) < 1.3)
            & (phos_o.pt > 40)
            & (phos_o.sieie < 0.0417588) #0.01
            & (phos_o.hoe < 0.00999299) #0.03
            & (phos_o.cutBased >= 3)
        ]

        jet_s = jets_s[
            (ak.num(jets_s) > 0)
        #     & (ak.num(jets_o) > 0)
            & (ak.num(phos_s) == 1)
        #     & (ak.num(phos_o) == 1)
        ]
        jet_o = jets_o[
            (ak.num(jets_o) > 0)
        #     & (ak.num(jets_s) > 0)
        #     & (ak.num(phos_s) == 1)
            & (ak.num(phos_o) == 1)
        ]
        pho_s = phos_s[
            (ak.num(jets_s) > 0)
        #     & (ak.num(jets_o) > 0)
            & (ak.num(phos_s) == 1)
        #     & (ak.num(phos_o) == 1)   
        ][:, 0]
        pho_o = phos_o[
            (ak.num(jets_o) > 0)
        #     & (ak.num(jets_s) > 0)
        #     & (ak.num(phos_s) == 1)
            & (ak.num(phos_o) == 1)   
        ][:, 0]

        def require_back2back(obj1, obj2, phi=2.7):

            return (abs(obj1.delta_phi(obj2)) > phi)

        def require_2nd_jet(jets, pho, pt_type="pt"):

            jet = jets[:, 1]

            return ~((jet[pt_type] > 5) & ((jet[pt_type] / pho.pt) > 0.3))

        def require_n(jets, pho, one=True):

            if one:
                jet = jets[(ak.num(jets) == 1)][:, 0]
                pho = pho[(ak.num(jets) == 1)]
            else:
                jet = jets[(ak.num(jets) > 1)]
                pho = pho[(ak.num(jets) > 1)]

            return jet, pho

        def criteria_one(jet, pho, phi=2.7):

            b2b = require_back2back(jet, pho, phi)

            return jet[b2b], pho[b2b]

        def criteria_n(jets, pho, pt_type="pt", phi=2.7,):

            sec_jet = require_2nd_jet(jets, pho, pt_type)

            jet = jets[sec_jet][:, 0]
            pho = pho[sec_jet]

            b2b = require_back2back(jet, pho, phi)

            return jet[b2b], pho[b2b]

        for rec in ["scouting", "offline"]:

            jets = jet_s if rec == "scouting" else jet_o
            pho = pho_s if rec == "scouting" else pho_o
            pt_type = self._pt_type if rec == "scouting" else "pt"

            jet_1, pho_1 =  require_n(jets, pho, one=True)
            jet_n, pho_n =  require_n(jets, pho, one=False)

            jet_1, pho_1 = criteria_one(jet_1, pho_1, phi=2.8)
            jet_n, pho_n = criteria_n(jet_n, pho_n, pt_type=pt_type, phi=2.8)

            self._output[rec].fill(
                ratio = jet_1[pt_type] / pho_1.pt,
                pt_ave = (jet_1[pt_type] + pho_1.pt) / 2,
                dataset = dataset,
            )
            
            self._output[rec].fill(
                ratio = jet_n[pt_type] / pho_n.pt,
                pt_ave = (jet_n[pt_type] + pho_n.pt) / 2,
                dataset = dataset,
            )
            
            self._output[rec + "_mean"].fill(
                sample = jet_1[pt_type] / pho_1.pt,
                pt_ave = (jet_1[pt_type] + pho_1.pt) / 2,
                dataset = dataset,
            )
            
            self._output[rec + "_mean"].fill(
                sample = jet_n[pt_type] / pho_n.pt,
                pt_ave = (jet_n[pt_type] + pho_n.pt) / 2,
                dataset = dataset,
            )
            
        return self._output


    def postprocess(self, accumulator):
        pass
