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

class BTagProcessor(processor.ProcessorABC):
    def __init__(self, wp_btag=0.2, do_jetid=False, do_isomuon=False):
        self._wp_btag = wp_btag
        self._do_jetid = do_jetid
        self._do_isomuon = do_isomuon
        
    @property
    def accumulator(self):
        return {
            "sumw": defaultdict(float),
            "events": defaultdict(int),
            "passtrig": defaultdict(int),
            "cutflow": (
                            Hist.new.IntCategory(
                                [], name="cat", label="Category", growth=True
                            ).Reg(
                                30, 40, 220, name="msoftdrop", label=r"msoftdrop"
                            ).Reg(
                                30, 0, 1, name="pn_Hbb", label=r"H(bb) vs QCD score"
                            ).StrCategory(
                                 [], name="region", label="Region", growth=True
#                            ).Variable(
#                                [0, 0.5, 0.15, 0.2, 0.25], name="trackIso", label=r"trackIso"
#                            ).Variable(
#                                [0, 0.25, 0.5, 0.75, 1], name="trk_dz", label=r"Track dz"
                            ).IntCategory(
                                [], name="cut", label="Cut Idx", growth=True
                            ).Weight()
                        ),
#             "met": (
#                             Hist.new.StrCategory(
#                                 [], name="dataset", label="Dataset", growth=True
#                             ).StrCategory(
#                                 [], name="region", label="Region", growth=True
#                             ).Reg(
#                                 30, 0, 400, name="pt", label=r"MET $p_T$"
#                             ).IntCategory(
#                                 [], name="cat", label="Category", growth=True
#                             ).Weight()
#                         ),
#             "goodmuon": (
#                             Hist.new.StrCategory(
#                                 [], name="dataset", label="Dataset", growth=True
#                             ).StrCategory(
#                                 [], name="region", label="Region", growth=True
#                             ).Reg(
#                                 30, 0, 300, name="pt", label=r"Leading muon $p_T$"
#                             ).Reg(
#                                 30, 0, 2, name="trk_dz", label=r"Leading muon track dz"
#                             ).Reg(
#                                 30, 0, 2, name="trk_dxy", label=r"Leading muon track dxy"
#                             ).IntCategory(
#                                 [], name="cat", label="Category", growth=True
#                             ).Weight()
#                         ),
#             "leptonicW": (
#                             Hist.new.StrCategory(
#                                 [], name="dataset", label="Dataset", growth=True
#                             ).StrCategory(
#                                 [], name="region", label="Region", growth=True
#                             ).Reg(
#                                 30, 0, 400, name="pt", label=r"Leptonic W $p_T$"
#                             ).IntCategory(
#                                 [], name="cat", label="Category", growth=True
#                             ).Weight()
#                         ),
#             "mujetiso": (
#                             Hist.new.StrCategory(
#                                 [], name="dataset", label="Dataset", growth=True
#                             ).StrCategory(
#                                 [], name="region", label="Region", growth=True
#                             ).Reg(
#                                 30, 0, 2, name="dr", label=r"Min. $\Delta$R(muon, AK4 jet)"
#                             ).Reg(
#                                 30, 0, 100, name="pt", label=r"Perp. p$_T$ (muon, nearest AK4 jet)"
#                             ).IntCategory(
#                                 [], name="cat", label="Category", growth=True
#                             ).Weight()
#                         ),
             "ak4bjet": (
                             Hist.new.StrCategory(
                                 [], name="region", label="Region", growth=True
                             ).Reg(
                                 30, 0, 50, name="njets", label=r"Number of AK4 b-jet in same hem. as muon"
                             ).Reg(
                                 30, 0, 1.1, name="pn_b_1", label=r"Leading AK4 b vs g score"
                             ).IntCategory(
                                 [], name="cat", label="Category", growth=True
                             ).Weight()
                         ),
             "ak8jet": (
                             Hist.new.StrCategory(
                                 [], name="region", label="Region", growth=True
                             ).Reg(
                                 30, 0, 4, name="dr", label=r"dR(proxy, leading muon)"
                             ).Reg(
                                 30, 0, 1, name="pn_Hbb", label=r"H(bb) vs QCD score"
                             ).IntCategory(
                                 [], name="cat", label="Category", growth=True
                             ).Weight()
                         ),
        }
           
        
    def process(self, events):
        
        output = self.accumulator
        dataset = events.metadata['dataset']
        
        isRealData = not hasattr(events, "genWeight")
        
        selection = PackedSelection()
        weights = Weights(len(events), storeIndividual=True)
        
        output['events'][dataset] += len(events)
        
        if not isRealData:
            output['sumw'][dataset] += ak.sum(events.genWeight)
            weights.add('genweight', events.genWeight)
            
        if len(events) == 0:
            return output
        
        fatjets = events.ScoutingFatJet
        if self._do_jetid:
            fatjets = fatjets[
                (fatjets.neHEF < 0.9)
                & (fatjets.neEmEF < 0.9)
                & (fatjets.muEmEF < 0.8)
                & (fatjets.chHEF > 0.01)
                & (fatjets.nCh > 0)
                & (fatjets.chEmEF < 0.8)
            ]
        fatjets["pn_Hbb"] = ak.where(
            (fatjets.particleNet_prob_Hbb + fatjets.particleNet_prob_QCD) == 0, 
            0, 
            (fatjets.particleNet_prob_Hbb / (fatjets.particleNet_prob_Hbb + fatjets.particleNet_prob_QCD))
        )
        
        jets = events.ScoutingJet
        if self._do_jetid:
            jets = jets[
                (jets.neHEF < 0.9)
                & (jets.neEmEF < 0.9)
                & (jets.muEmEF < 0.8)
                & (jets.chHEF > 0.01)
                & (jets.nCh > 0)
                & (jets.chEmEF < 0.8)
            ]
        jets["pn_b"] = ak.where(
            (jets.particleNet_prob_b + jets.particleNet_prob_g) == 0, 
            0, 
            (jets.particleNet_prob_b / (jets.particleNet_prob_b + jets.particleNet_prob_g))
        )
        
        # trigger
        selection.add("trigger", events.HLT["Mu50"])
        
        output['passtrig'][dataset] += ak.sum(events.HLT["Mu50"])
        
        # large radius jet
        selection.add("fatjetpt", ak.firsts(fatjets).pt > 200)
        
        # MET
        met = events.ScoutingMET
        selection.add('met', met.pt > 50)
        
        # Good muon
        goodmuon = (
            (events.ScoutingMuon.pt > 55)
            & (abs(events.ScoutingMuon.eta) < 2.4)
            & (abs(events.ScoutingMuon.trk_dxy) < 0.2)
            #& (abs(events.ScoutingMuon.trackIso) < 0.15)
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
        
        goodmuons = events.ScoutingMuon[goodmuon]
        
        # Muon-jet isolation
        if (self._do_isomuon):
            near_jet, near_jet_dr = goodmuons.nearest(jets, axis=1, return_metric=True)
            muon_pt_rel = np.sqrt(goodmuons.rho**2 - goodmuons.dot(near_jet.unit)**2)
            goodmuons = goodmuons[(
                (near_jet_dr > 0.4) | (muon_pt_rel > 25)
            )]
        
        # Leptonic W 
        leadingmuon = ak.firsts(goodmuons)
        leptonicW = met + leadingmuon
        selection.add('leptonicW', leptonicW.pt > 150)
       
        # Trk_dz requirement
        selection.add('trk_dz', (leadingmuon.trk_dz < 0.5))

        # trackIso requirement
        selection.add('trackIso', (leadingmuon.trackIso < 0.15))
 
        # Same. hem. AK4 b-jet
        dphi = abs(jets.delta_phi(leadingmuon))
        jetsamehemisp = jets[dphi < 2]
        bjets = (jetsamehemisp.pn_b > self._wp_btag)
        nbjets = ak.sum(bjets, axis=1)
        selection.add('onebjet', (nbjets > 0))

        # Opp. hem. AK8 jet
        dphi = abs(fatjets.delta_phi(leadingmuon))
        is_away = (dphi > 2)
        nfatjets = ak.sum(is_away, axis=1)
        selection.add('onefatjet', (nfatjets > 0))

        proxy = ak.firsts(fatjets[(is_away)])
        
        if not isRealData:
            proxy = self.category(events, proxy)
            
        regions = {
            'trk_dz': ['trigger','fatjetpt','met','onemuon','leptonicW','trk_dz','trackIso','onebjet','onefatjet'],
            'onebjet': ['trigger','fatjetpt','met','onemuon','leptonicW','onebjet','onefatjet','trk_dz','trackIso'],
            'noselection' : [],
        }
        
        def normalize(val, cut):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
                return ar
            
        for region, cuts in regions.items():
            if region == "noselection":
                continue
            allcuts = set([])
            cut = selection.all(*allcuts)
            weight = weights.weight()[cut]
            
            output['cutflow'].fill(
                region=region,
#                dataset=dataset,
                cat=normalize(proxy.cat, cut) if not isRealData else -1,
                msoftdrop=normalize(proxy.msoftdrop, cut),
#                 pt=normalize(proxy.pt, cut),
                pn_Hbb=normalize(proxy.pn_Hbb, cut),
                cut=0,
                weight=weight,
            )
            
            for i, cut in enumerate(cuts):
                allcuts.add(cut)
                cut = selection.all(*allcuts)
                weight = weights.weight()[cut]
                
                output['cutflow'].fill(
                    region=region,
#                    dataset=dataset,
                    cat=normalize(proxy.cat, cut) if not isRealData else -1,
                    msoftdrop=normalize(proxy.msoftdrop, cut),
#                     pt=normalize(proxy.pt, cut),
                    pn_Hbb=normalize(proxy.pn_Hbb, cut),
                    cut=i+1,
                    weight=weight,
                )
                
        def fill(region):
            selections = regions[region]
            cut = selection.all(*selections)
            weight = weights.weight()[cut]
            
#             output['met'].fill(
#                 dataset=dataset,
#                 region=region,
#                 pt=normalize(met.pt, cut),
#                 cat=normalize(proxy.cat, cut) if True else -1,
#                 weight=weight,
#             )
            
#             output['goodmuon'].fill(
#                 dataset=dataset,
#                 region=region,
#                 pt=normalize(ak.firsts(events.ScoutingMuon).pt, cut),
#                 trk_dz=normalize(ak.firsts(events.ScoutingMuon).trk_dz, cut),
#                 trk_dxy=normalize(ak.firsts(events.ScoutingMuon).trk_dxy, cut),
#                 cat=normalize(proxy.cat, cut) if True else -1,
#                 weight=weights.weight()[cut],
#             )
            
#             output['leptonicW'].fill(
#                 dataset=dataset,
#                 region=region,
#                 pt=normalize(leptonicW.pt, cut),
#                 cat=normalize(proxy.cat, cut) if True else -1,
#                 weight=weights.weight()[cut],
#             )
            
#             output['mujetiso'].fill(
#                 dataset=dataset,
#                 region=region,
#                 dr=normalize(near_jet_dr, cut),
#                 pt=normalize(muon_pt_rel, cut),
#                 cat=normalize(proxy.cat, cut) if True else -1,
#                 weight=weights.weight()[cut],
#             )

#             output['ak4bjet'].fill(
#                 dataset=dataset,
#                 region=region,
#                 njets=normalize(ak.num(jetsamehemisp), cut),
#                 pn_b_1=normalize(ak.firsts(jetsamehemisp)["pn_b"], cut),
#                 cat=normalize(proxy.cat, cut) if True else -1,
#                 weight=weights.weight()[cut],
#             )
            
            output['ak8jet_neu'].fill(
                dataset=dataset,
                region=region,
                msoftdrop=normalize(proxy.msoftdrop, cut),
                neHEF=normalize(proxy.neHEF, cut),
                neEmEF=normalize(proxy.neEmEF, cut),
                cat=normalize(proxy.cat, cut) if not isRealData else -1,
                weight=weights.weight()[cut],
            )
        
            output['ak8jet_mu'].fill(
                dataset=dataset,
                region=region,
                msoftdrop=normalize(proxy.msoftdrop, cut),
                muEmEF=normalize(proxy.muEmEF, cut),
                chHEF=normalize(proxy.chHEF, cut),
                cat=normalize(proxy.cat, cut) if not isRealData else -1,
                weight=weights.weight()[cut],
            )
            
            output['ak8jet_ch'].fill(
                dataset=dataset,
                region=region,
                msoftdrop=normalize(proxy.msoftdrop, cut),
                nCh=normalize(proxy.nCh, cut),
                chEmEF=normalize(proxy.chEmEF, cut),
                cat=normalize(proxy.cat, cut) if not isRealData else -1,
                weight=weights.weight()[cut],
            )
            
        #for region in regions:
        #    fill(region)

        return output
    
    def postprocess(self, accumulator):
        return accumulator
    
    def category(self, events, jet):

        # get hadronic W
        w = events.GenPart[
            (abs(events.GenPart.pdgId) == 24)
            & (events.GenPart.hasFlags(['fromHardProcess', 'isLastCopy']))
        ]

        w_is_had = ak.any(
            (abs(w.distinctChildren.pdgId) < 7)
            & (w.distinctChildren.hasFlags(['isLastCopy']))
        , axis=2)
        had_w = w[w_is_had]

        near_W, near_W_dr = jet.nearest(had_w, axis=None, threshold=0.8, return_metric=True)
        near_W_dr = ak.to_numpy(ak.fill_none(near_W_dr, 99))

        q_W = near_W.distinctChildren
        q_W_dr = jet.delta_r(q_W)
        index_descend_q_W = ak.argsort(q_W_dr, axis=1, ascending=False)
        q_W_dr_descend = q_W_dr[index_descend_q_W]

        # get hadronic top
        top = events.GenPart[
            (abs(events.GenPart.pdgId) == 6)
            & (events.GenPart.hasFlags(['fromHardProcess', 'isLastCopy']))
        ]

        w_top = top.distinctChildren[
            (abs(top.distinctChildren.pdgId) == 24)
            & (top.distinctChildren.hasFlags(['isLastCopy']))
        ]
        w_top = ak.flatten(w_top, axis=2)

        w_top_is_had = ak.any(
            (abs(w_top.distinctChildren.pdgId) < 7)
            & (w_top.distinctChildren.hasFlags(['isLastCopy']))
        , axis=2)

        had_top = w_top[w_top_is_had].distinctParent

        near_top, near_top_dr = jet.nearest(had_top, axis=None, threshold=0.8, return_metric=True)
        near_top_dr = ak.to_numpy(ak.fill_none(near_top_dr, 99))

        b_near_top = near_top.distinctChildren[abs(near_top.distinctChildren.pdgId) == 5]
        b_near_top_matched, b_near_top_dr = jet.nearest(b_near_top, axis=None, threshold=0.8, return_metric=True)
        b_near_top_dr = ak.to_numpy(ak.fill_none(b_near_top_dr, 99))

        W_near_top = near_top.distinctChildren[(
            (abs(near_top.distinctChildren.pdgId) == 24)
            & (ak.any(abs(near_top.distinctChildren.distinctChildren.pdgId) < 7))
        )]

        q_W_near_top = ak.flatten(W_near_top.distinctChildren, axis=2)
        q_W_near_top_dr = jet.delta_r(q_W_near_top)

        index_ascend = ak.argsort(q_W_near_top_dr, axis=1)
        index_descend = ak.argsort(q_W_near_top_dr, ascending=False, axis=1)
        q_W_near_top_ascend = q_W_near_top[index_ascend]
        q_W_near_top_descend = q_W_near_top[index_descend]
        q_W_near_top_dr_ascend = q_W_near_top_dr[index_ascend]
        q_W_near_top_dr_descend = q_W_near_top_dr[index_descend]

        jet["dr_T"] = near_top_dr
        jet["dr_T_b"] = b_near_top_dr
        jet["dr_T_Wq_max"] = ak.fill_none(ak.firsts(q_W_near_top_dr_descend), 99)
        jet["dr_T_Wq_min"] = ak.fill_none(ak.firsts(q_W_near_top_dr_ascend), 99)
        jet["dr_T_Wq_max_pdgId"] = ak.fill_none(ak.firsts(q_W_near_top_descend).pdgId, 99)
        jet["dr_T_Wq_min_pdgId"] = ak.fill_none(ak.firsts(q_W_near_top_ascend).pdgId, 99)
        jet["dr_W_daus"] = ak.fill_none(ak.firsts(q_W_dr_descend), 99)

        top_matched = (
            (jet["dr_T_b"] < 0.8) 
            & (jet["dr_T_Wq_max"] < 0.8)
        )
        w_matched = (
            (
                (jet["dr_T_Wq_max_pdgId"] == 99) 
                & (jet["dr_W_daus"] < 0.8)
            ) | (
                (jet["dr_T_Wq_max_pdgId"] != 99) 
                & (jet["dr_T_b"] >= 0.8) 
                & (jet["dr_T_Wq_max"] < 0.8)
            )
        )
        non_matched = (
            (~top_matched) 
            & (~w_matched)
        )

        cat = np.zeros(len(jet.pt))
        cat = [1 if t else c for c, t in zip(cat, top_matched.to_numpy())]
        cat = [2 if w else c for c, w in zip(cat, w_matched.to_numpy())]
        cat = [3 if n else c for c, n in zip(cat, non_matched.to_numpy())]
        jet["cat"] = np.array(cat)
        
        return jet
