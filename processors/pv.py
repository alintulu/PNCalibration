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

class PVProcessor(processor.ProcessorABC):
    def __init__(self, wp_btag=0.2):
        self._wp_btag = wp_btag
        
    @property
    def accumulator(self):
        return {
            "sumw": defaultdict(float),
            "templates": (
                            Hist.new.StrCategory(
                                [], name="dataset", label="Dataset", growth=True
                            ).Reg(
                                30, 40, 220, name="msoftdrop", label=r"msoftdrop"
                            ).Reg(
                                30, 150, 1000, name="pt", label=r"$p_T$"
                            ).Reg(
                                30, 0, 10, name="nPV", label=r"Number of PV"
                            ).Double()
                        ),
        }
           
        
    def process(self, events):
        
        output = self.accumulator
        dataset = events.metadata['dataset']
        
        isRealData = not hasattr(events, "genWeight")
        
        selection = PackedSelection()
        
        if not isRealData:
            output['sumw'][dataset] += ak.sum(events.genWeight)
            
        if len(events) == 0:
            return output
        
        pvs = events.ScoutingPrimaryVertex
        pvs = pvs[
            (pvs.isValidVtx)
            & (pvs.ndof > 4)
            & (np.abs(pvs["z"]) < 24)
        ]
        
        fatjets = events.ScoutingFatJet
        fatjets = fatjets[
            (fatjets.neHEF < 0.9)
            & (fatjets.neEmEF < 0.9)
            & (fatjets.muEmEF < 0.8)
            & (fatjets.chHEF > 0.01)
            & (fatjets.nCh > 0)
            & (fatjets.chEmEF < 0.8)
        ]
        fatjets = fatjets[ak.argsort(fatjets.pt, axis=1)]
        
        # trigger
        selection.add("trigger", events.L1["SingleJet180"])
        
        # large radius jet
        selection.add("fatjetpt", ak.firsts(fatjets).pt > 150)
            
        regions = {
            'all': ['trigger', 'fatjetpt'],
            #'noselection' : []
        }
        
        def normalize(val, cut):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
                return ar
            
        def fill(region):
            selections = regions[region]
            cut = selection.all(*selections)
            
            output['templates'].fill(
                dataset=dataset,
                msoftdrop=normalize(ak.firsts(fatjets).msoftdrop, cut),
                pt=normalize(ak.firsts(fatjets).pt, cut),
                nPV=normalize(ak.num(pvs), cut)
            )
            
        for region in regions:
            fill(region)
            
        return output
    
    def postprocess(self, accumulator):
        return accumulator
