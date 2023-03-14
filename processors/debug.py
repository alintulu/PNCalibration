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
from coffea.lumi_tools import LumiList

class DebugProcessor(processor.ProcessorABC):
    def __init__(self, wp_btag=0.2, do_jetid=False, do_isomuon=False):
        self._wp_btag = wp_btag
        self._do_jetid = do_jetid
        self._do_isomuon = do_isomuon
        
    @property
    def accumulator(self):
        return {
            "events": defaultdict(int),
            "passtrig": defaultdict(int),
            "lumi": processor.value_accumulator(LumiList),
        }
           
        
    def process(self, events):
        
        output = self.accumulator
        dataset = events.metadata['dataset']
        
        isRealData = not hasattr(events, "genWeight")
         
        if len(events) == 0:
            return output

        output['events'][dataset] += len(events)
        output['passtrig'][dataset] += ak.sum(events.HLT["Mu50"])
        output["lumi"] = LumiList(events.run, events.luminosityBlock)
        
        return output
    
    def postprocess(self, accumulator):
        return accumulator
