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

class JERProcessor(processor.ProcessorABC):
    def __init__(self):
        commonaxes = (
            hist.axis.StrCategory([], name="dataset", label="Dataset name", growth=True),
            hist.axis.Regular(100, -1, 1, name="asymmetry", label="Asymmetry"),
            hist.axis.Regular(20, 0, 1000, name="pt_ave", label=r"Average $p_T$"),
        )
        
        self._output = {
                "nevents": 0,
                "scouting": Hist(
                    *commonaxes
                ),
                "offline": Hist(
                    *commonaxes
                ),
            }

    def process(self, events):
        
        dataset = events.metadata['dataset']
        self._output["nevents"] = len(events)

        # scouting        
        jets_s = events.ScoutingJet
        jets_s = jets_s[
            (abs(jets_s.eta) < 2.5)
            & (jets_s.pt > 20)
            & (jets_s.neHEF < 0.9)
            & (jets_s.neEmEF < 0.9)
            & (jets_s.muEmEF < 0.8)
            & (jets_s.chHEF > 0.01)
            & (jets_s.nCh > 0)
            & (jets_s.chEmEF < 0.8)
        ]
        jets_s = jets_s[ak.num(jets_s) > 1][:,:2]

        # offline
        jets_o = events.Jet
        jets_o = jets_o[
            (abs(jets_o.eta) < 2.5)
            & (jets_o.pt > 20)
            #& (jets_o.jetId == 4)
        ]
        jets_o = jets_o[ak.num(jets_o) > 1][:,:2]

        def get_asymmetry(jets):

            dijets = jets[
                abs(jets[:, 0].delta_phi(jets[:, 1])) > 2.7
            ]
            
            shuffle = random.choices([-1, 1], k=len(dijets[:,0]))
            shuffle_opp = [s * -1 for s in shuffle]
            
            asymmetry = (shuffle * dijets[:,0].pt + shuffle_opp * dijets[:,1].pt) / (dijets[:,0].pt + dijets[:,1].pt)

            pt_ave = (dijets[:,0].pt + dijets[:,1].pt) / 2

            return asymmetry, pt_ave

        a_s, pt_s = get_asymmetry(jets_s)
        a_o, pt_o = get_asymmetry(jets_o)
        
        self._output["scouting"].fill(
            dataset=dataset,
            asymmetry = a_s,
            pt_ave = pt_s,
        )
        
        self._output["offline"].fill(
            dataset=dataset,
            asymmetry = a_o,
            pt_ave = pt_o,
        )
        
        return self._output

    def postprocess(self, accumulator):
        pass