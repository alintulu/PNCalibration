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

class ZbbProcessor(processor.ProcessorABC):
    def __init__(self, jet_arbitration='pt', tagger="v1", systematics=False):
        self._jet_arbitration = jet_arbitration
        self._tagger = tagger
        self._wp_btag = 0.2
        self._tightMatch = False
        self._systematics = systematics
        
    @property
    def accumulator(self):
        return {
            "sumw": defaultdict(float),
            "cutflow": (
                            Hist.new.StrCategory(
                                [], name="dataset", label="Dataset", growth=True
                            ).StrCategory(
                                [], name="region", label="Region", growth=True
                            ).StrCategory(
                                [], name="systematic", label="Systematic", growth=True
                            ).IntCategory(
                                [], name="genflavour", label="Genflavour", growth=True
                            ).Reg(
                                50, 100, 700, name="pt", label=r"Leading jet $p_T$"
                            ).Reg(
                                50, 0, 150, name="msoftdrop", label=r"Leading jet mass"
                            ).Reg(
                                50, 0, 1, name="pn_Hbb", label=r"H(bb) vs QCD score"
                            ).IntCategory(
                                [], name="cut", label="Cut Idx", growth=True
                            ).Weight()
                        ),
        }
           
        
    def process(self, events):
        return self.process_shift(events, None)
    
    def process_shift(self, events, shift_name):
        
        output = self.accumulator
        
        dataset = events.metadata['dataset']
        
        isRealData = not hasattr(events, "genWeight")
        isQCDMC = 'QCD' in dataset
        
        selection = PackedSelection()
        weights = Weights(len(events), storeIndividual=True)
        
        if shift_name is None and not isRealData:
            output['sumw'][dataset] += ak.sum(events.genWeight)
            
        if len(events) == 0:
            return output
        
        fatjets = events.ScoutingFatJet
        fatjets['qcdrho'] = 2 * np.log(fatjets.msoftdrop / fatjets.pt)
        fatjets["pn_Hbb"] = ak.where(
            (fatjets.particleNet_prob_Hbb + fatjets.particleNet_prob_QCD) == 0, 
            0, 
            (fatjets.particleNet_prob_Hbb / (fatjets.particleNet_prob_Hbb + fatjets.particleNet_prob_QCD))
        )
        
        jets = events.ScoutingJet
        jets["pn_b"] = ak.where(
            (jets.particleNet_prob_b + jets.particleNet_prob_g) == 0, 
            0, 
            (jets.particleNet_prob_b / (jets.particleNet_prob_b + jets.particleNet_prob_g))
        )
        
        candidatejet = fatjets[
            (fatjets.pt > 200)
            & (abs(fatjets.eta) < 2.5)
        ]
        
        candidatejet = candidatejet[:, :2]
        if self._jet_arbitration == 'pt':
            candidatejet = ak.firsts(candidatejet)
        else:
            raise RuntimeError("Unknown candidate jet arbitration")
            
        if self._tagger == 'v1':
            bvq = candidatejet.pn_Hbb
        else:
            raise RuntimeError("Unknown jet tag version")
            
        selection.add("trigger", events.L1["SingleJet180"])
            
        selection.add('minjetkin',
            (candidatejet.pt >= 350)
            & (candidatejet.pt < 1200)
            & (candidatejet.qcdrho < -1.7)
            & (candidatejet.qcdrho > -6.0)
            & (abs(candidatejet.eta) < 2.5)
        )
        
        selection.add('jetid', 
            (candidatejet.neHEF < 0.9)
            & (candidatejet.neEmEF < 0.9)
            & (candidatejet.muEmEF < 0.8)
            & (candidatejet.chHEF > 0.01)
            & (candidatejet.nCh > 0)
            & (candidatejet.chEmEF < 0.8)
        )
        
        jets = jets[
            (jets.pt > 30.)
            & (abs(jets.eta) < 5.0)
        ]
        
        jets = jets[:, :4]
        dphi = abs(jets.delta_phi(candidatejet))
        selection.add('ak4btagOppHem', ak.max(jets[dphi > np.pi / 2].pn_b, axis=1, mask_identity=False) < self._wp_btag) 

        met = events.ScoutingMET
        selection.add('met', met.pt < 140.)
        
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
        leadingmuon = ak.firsts(events.ScoutingMuon[goodmuon])

        goodelectron = (
            (events.ScoutingElectron.pt > 10)
            & (abs(events.ScoutingElectron.eta) < 2.5)
        )
        nelectrons = ak.sum(goodelectron, axis=1)
        
        selection.add('noleptons', (nmuons == 0) & (nelectrons == 0))
        
        if isRealData :
            genflavour = ak.zeros_like(candidatejet.pt)
        else:
            weights.add('genweight', events.genWeight)

            bosons = self.getBosons(events.GenPart)
            matchedBoson = candidatejet.nearest(bosons, axis=None, threshold=0.8)
            if self._tightMatch:
                match_mask = ((candidatejet.pt - matchedBoson.pt)/matchedBoson.pt < 0.5) & ((candidatejet.msoftdrop - matchedBoson.mass)/matchedBoson.mass < 0.3)
                selmatchedBoson = ak.mask(matchedBoson, match_mask)
                genflavour = self.bosonFlavour(selmatchedBoson)
            else:
                genflavour = self.bosonFlavour(matchedBoson)
            genBosonPt = ak.fill_none(ak.firsts(bosons.pt), 0)
            
        msoftdrop_matched = candidatejet.msoftdrop * (genflavour > 0) + candidatejet.msoftdrop * (genflavour == 0)
            
        regions = {
            'signal': ['trigger','minjetkin','jetid','ak4btagOppHem','met','noleptons'],
        }
        
        def normalize(val, cut):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
                return ar
            
        if shift_name is None:
            systematics = [None] + list(weights.variations)
        else:
            systematics = [shift_name]
            
        for region, cuts in regions.items():
            if region == "noselection":
                continue
            allcuts = set([])
            cut = selection.all(*allcuts)
            
            output['cutflow'].fill(
                dataset=dataset,
                region=region,
                systematic="snominal",
                genflavour=normalize(genflavour, cut),
                pt=normalize(candidatejet.pt, cut),
                msoftdrop=normalize(msoftdrop_matched, cut),
                pn_Hbb=normalize(bvq, cut),
                weight=weights.weight(),
                cut=0,
            )
            
            for i, cut in enumerate(cuts):
                allcuts.add(cut)
                cut = selection.all(*allcuts)
                
                output['cutflow'].fill(
                    dataset=dataset,
                    region=region,
                    systematic="snominal",
                    genflavour=normalize(genflavour, cut),
                    pt=normalize(candidatejet.pt, cut),
                    msoftdrop=normalize(msoftdrop_matched, cut),
                    pn_Hbb=normalize(bvq, cut),
                    weight=weights.weight()[cut],
                    cut=i+1,
            )
            
#         def fill(region, systematic):
#             selections = regions[region]
#             cut = selection.all(*selections)
#             sname = 'nominal' if systematic is None else systematic
#             if systematic in weights.variations:
#                 weight = weights.weight(modifier=systematic)[cut]
#             else:
#                 weight = weights.weight()[cut]

#             output['templates'].fill(
#                 dataset=dataset,
#                 region=region,
#                 systematic=sname,
#                 genflavour=normalize(genflavour, cut),
#                 pt=normalize(candidatejet.pt, cut),
#                 msoftdrop=normalize(msoftdrop_matched, cut),
#                 pn_Hbb=normalize(bvq, cut),
#                 weight=weight,
#             )
            
#         for region in regions:
#             for systematic in systematics:
#                 if isRealData and systematic is not None:
#                     continue
#                 fill(region, systematic)
                
        if shift_name is None:
            output["weightStats"] = weights.weightStatistics
        
        return output
    
    def getBosons(self, genparticles):
        absid = abs(genparticles.pdgId)
        return genparticles[
            (absid >= 22) # no gluons
            & (absid <= 25)
            & genparticles.hasFlags(['fromHardProcess', 'isLastCopy'])
        ]

    def bosonFlavour(self, bosons):
        childid = abs(bosons.children.pdgId)
        genflavour = ak.any(childid == 5, axis=-1) * 5 + ak.any(childid == 4, axis=-1) * 4 + ak.all(childid < 4, axis=-1) * 1        
        return ak.fill_none(genflavour, 0)
    
    def postprocess(self, accumulator):
        return accumulator
