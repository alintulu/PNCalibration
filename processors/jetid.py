import awkward as ak
import matplotlib.pyplot as plt
import os, sys
import subprocess
import json
import uproot
from coffea.nanoevents import NanoEventsFactory, ScoutingNanoAODSchema
from coffea.lookup_tools.lookup_base import lookup_base
import numpy as np
from coffea import processor, util
from hist import Hist
import hist
from coffea.analysis_tools import Weights, PackedSelection
from collections import defaultdict

class JetIDProcessor(processor.ProcessorABC):
    def __init__(self):
        commonaxes = (
            hist.axis.Variable([0, 0.5, 1, 1.5, 2, 2.5], name="eta", label="$\eta$"),
            hist.axis.Regular(30, 0, 1.2, name="chf", label="Charged Hadron Fraction"),
            hist.axis.Regular(30, 0, 1.2, name="nhf", label="Neutral Hadron Fraction"),
            hist.axis.Regular(30, 0, 1.2, name="cemf", label="Charged Electromagnetic Fration"),
            hist.axis.Regular(30, 0, 1.2, name="nemf", label="Neutral Electromagnetic Fraction"),
#             hist.axis.Regular(30, 0, 1.2, name="muf", label="Muon Fraction"),
#             hist.axis.Regular(30, 0, 1.2, name="met", label="MET/SumET"),
        )
        
        self._output = {
                "nevents": 0,
                "scouting": Hist(
                    *commonaxes
                ),
                "offline": Hist(
                    *commonaxes
                ),
                "scouting_jetid": Hist(
                    *commonaxes
                ),
                "offline_jetid": Hist(
                    *commonaxes
                ),
            }

    def process(self, events):
        
        dataset = events.metadata['dataset']
        self._output["nevents"] = len(events)

        def require_njets(jets):

            return ak.num(jets) > 1

        def require_dijets(jets):

            return abs(jets[:, 0].delta_phi(jets[:, 1])) > 2.7

        def jet_id(jet):

            if "muEmEF" in jet.fields:
                muf = (jet.muEmEF < 0.8)
            else:
                muf = (jet.muEF < 0.8)

            return (muf
                    & (jet.neHEF < 0.9)
                    & (jet.neEmEF < 0.9)
                    & (jet.chHEF > 0.01)
                    #& (jet.nCh > 0)
                    & (jet.chEmEF < 0.8)
                   )
        
        def normalize(val):
            return ak.to_numpy(ak.fill_none(val, np.nan))

        # scouting        
        jets_s = events.ScoutingJet[
            require_njets(events.ScoutingJet)
        ]
        met_s = events.ScoutingMET.pt[
            require_njets(events.ScoutingJet)
        ]

        # offline
        jets_o = events.Jet[
            require_njets(events.Jet)
        ]
        met_o = events.MET.pt[
            require_njets(events.Jet)
        ]
        
        for rec in ["scouting", "offline", "scouting_jetid", "offline_jetid"]:
            
            if "offline" in rec:
                jets = jets_o
                met = met_o
            else:
                jets = jets_s
                met = met_s
            
            jets = jets[
                require_dijets(jets)
            ]
            met = met[
                require_dijets(jets)
            ]
            leadingjet = ak.firsts(jets)

            if "jetid" in rec:
                jetid = jet_id(jets)
                jets = jets[jetid]
                leadingjet = ak.firsts(jets)
                met = met[
                    ak.any(jetid) if len(jetid) != 0 else []
                ]
                
            self._output[rec].fill(
                eta = normalize(abs(leadingjet.eta)),
                chf = normalize(leadingjet.chHEF),
                nhf = normalize(leadingjet.neHEF),
                cemf = normalize(leadingjet.chEmEF),
                nemf = normalize(leadingjet.neEmEF),
#                 muf = normalize(leadingjet["muEmEF" if "scouting" in rec else "muEF"]),
#                 met = normalize(met / (met + ak.sum(jets.pt, axis=-1))),
            )

        return self._output

    def postprocess(self, accumulator):
        pass
