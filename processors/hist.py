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

class HistProcessor(processor.ProcessorABC):
    def __init__(self, wp_btag=0):
        self._wp_btag = wp_btag
        
    @property
    def accumulator(self):
        return {
            "sumw": defaultdict(float),
            "templates": (
                            Hist.new.StrCategory(
                                [], name="dataset", label="Dataset", growth=True
                            ).StrCategory(
                                [], name="region", label="Region", growth=True
                            ).Reg(
                                50, 200, 700, name="pt", label=r"$p_T$"
                            ).Reg(
                                50, 40, 220, name="msoftdrop", label=r"msoftdrop"
                            ).Reg(
                                50, 40, 220, name="mreg", label=r"mreg"
                            ).Reg(
                                50, 0, 1, name="ddb", label=r"ddb"
                            ).Double()
                        ),
        }
           
        
    def process(self, events):
        
        output = self.accumulator
        dataset = events.metadata['dataset']
        
        isRealData = not hasattr(events, "genWeight")
        isQCDMC = 'QCD' in dataset
        
        selection = PackedSelection()
        
        if not isRealData:
            output['sumw'][dataset] += ak.sum(events.genWeight)
            
        if len(events) == 0:
            return output
        
        fatjets = events.ScoutingFatJet
        fatjets["pn_Hbb"] = ak.where((fatjets.particleNet_prob_Hbb + fatjets.particleNet_prob_QCD) == 0, 0, (fatjets.particleNet_prob_Hbb / (fatjets.particleNet_prob_Hbb + fatjets.particleNet_prob_QCD)))
        jets = events.ScoutingJet
        jets["pn_b"] = ak.where((jets.particleNet_prob_b + jets.particleNet_prob_g) == 0, 0, (jets.particleNet_prob_b / (jets.particleNet_prob_b + jets.particleNet_prob_g)))
        
        # trigger
        selection.add("trigger", events.HLT["Mu50"])
        
        # require MET
        met = events.ScoutingMET
        selection.add('met', met.pt > 50)
        
        # require at least one good muon
        goodmuon = (
            (events.ScoutingMuon.pt > 55)
            & (abs(events.ScoutingMuon.eta) < 2.4)
            & (abs(events.ScoutingMuon.trk_dxy) < 0.2)
            #& (abs(events.ScoutingMuon.trk_dz) < 0.5)
            #& (events.ScoutingMuon["type"] == 2)
            & (events.ScoutingMuon.normchi2 < 10)
            & (events.ScoutingMuon.nValidRecoMuonHits > 0)
            & (events.ScoutingMuon.nRecoMuonMatchedStations > 1)
            & (events.ScoutingMuon.nValidPixelHits > 0)
            & (events.ScoutingMuon.nTrackerLayersWithMeasurement > 5)            
        )
        
        nmuons = ak.sum(goodmuon, axis=1)
        selection.add('onemuon', (nmuons > 0))
        
        # require good leptonic W 
        leadingmuon = ak.firsts(events.ScoutingMuon[goodmuon])
        leptonicW = met + leadingmuon
        selection.add('leptonicW', leptonicW.pt > 150)
        
        # require at least one b-jet in the same hemisphere of the leading muon
        dphi = abs(jets.delta_phi(leadingmuon))
        jetsamehemisp = jets[dphi < 2]
        bjets = (jetsamehemisp.pn_b > self._wp_btag)
        nbjets = ak.sum(bjets, axis=1)
        selection.add('onebjet', (nbjets > 0))

        # require fatjet away from the leading muon
        dphi = abs(fatjets.delta_phi(leadingmuon))
        is_away = (dphi > 2)
        nfatjets = ak.sum(is_away, axis=1)
        selection.add('onefatjet', (nfatjets > 0))
        
        proxy = ak.firsts(fatjets[(is_away) & (fatjets.pt > 200)])
            
        regions = {
            'all': ['trigger','met','onemuon','leptonicW','onebjet','onefatjet'],
            #'noselection': [],
        }
        
        def normalize(val, cut):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
                return ar
            
        def fill(region, jet, _cut=None):
            selections = regions[region]
            cut = selection.all(*selections)
            
            output['templates'].fill(
                dataset=dataset,
                region=region,
                pt=normalize(jet.pt, cut),
                msoftdrop=normalize(jet.msoftdrop, cut),
                mreg=normalize(jet.particleNet_mass, cut),
                ddb=normalize(jet.pn_Hbb, cut),
            )
            
        for region, cuts in regions.items():
            fill(region, proxy)
            
        return output
    
    def postprocess(self, accumulator):
        return accumulator
