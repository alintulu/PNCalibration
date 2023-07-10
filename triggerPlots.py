import awkward as ak
import matplotlib.pyplot as plt
import os, sys
import subprocess
import json
import uproot
from coffea.nanoevents import NanoEventsFactory #, ScoutingNanoAODSchema
from coffea.lookup_tools.lookup_base import lookup_base
import numpy as np
from coffea import processor, util
from hist import Hist
import hist
from coffea.analysis_tools import Weights, PackedSelection
from collections import defaultdict
import mplhep
plt.style.use(mplhep.style.CMS)

out_o = util.load("outfiles/2022/data/trigger_Muon_offline.coffea")[0]
out_s = util.load("outfiles/2022/data/trigger_ScoutingPFMonitor_2022-CHS.coffea")[0]

doTriggers = False

pt = {
    "scouting" : {
            "ak8" : {
                "pt" : {
                    "before" : {
                        "range" : (0, 380),
                        "bin" : 1,
                    },
                    "after" : {
                        "range" : (380, 1000),
                        "bin" : 16,
                    }
                }
            },
            "ak4" : {
                "pt" : {
                    "before" : {
                        "range" : (0, 350),
                        "bin" : 1,
                    },
                    "after" : {
                        "range" : (350, 1000),
                        "bin" : 16,
                    }
                },
                "ht" : {
                    "before" : {
                        "range" : (0, 650),
                        "bin" : 1,
                    },
                    "after" : {
                        "range" : (650, 2000),
                        "bin" : 30,
                    }
                },
            }
        },
    "offline" : {
            "ak8" : {
                "pt" : {
                    "before" : {
                        "range" : (0, 800),
                        "bin" : 1,
                    },
                    "after" : {
                        "range" : (800, 1000),
                        "bin" : 16,
                    }
                }
            },
            "ak4" : {
                "pt" : {
                    "before" : {
                        "range" : (0, 750),
                        "bin" : 1,
                    },
                    "after" : {
                        "range" : (750, 1000),
                        "bin" : 16,
                    }
                },
                "ht" : {
                    "before" : {
                        "range" : (0, 1300),
                        "bin" : 1,
                    },
                    "after" : {
                        "range" : (1300, 2000),
                        "bin" : 30,
                    }
                },
            }
        }
    }

era = ""
#for era in ["C", "D", "E", "F", "G", ""]:

if doTriggers:

    #triggers = ['HLT_PFHT370', 'HLT_PFHT780', 'HLT_PFHT890', 'HLT_PFHT1050', 'HLT_PFJet450', 'HLT_PFJet500', 'HLT_PFJet550', 'HLT_PFJet320', 'HLT_PFJet400', 'HLT_PFHT430', 'HLT_PFJet260', 'HLT_PFHT590', 'HLT_PFHT680', 'HLT_PFHT510', 'HLT_PFJet110', 'HLT_PFJet140', 'HLT_PFHT180', 'HLT_PFHT250', 'HLT_PFJet200', 'HLT_PFJet40', 'HLT_PFJet60','any']
    #triggers = ['HLT_PFHT1050', 'HLT_PFHT890', 'HLT_PFHT780', 'HLT_PFHT680',  'HLT_PFHT590',  'HLT_PFHT510', 'HLT_PFHT430', 'HLT_PFHT370', 'HLT_PFHT250', 'HLT_PFHT180']
    triggers = ['HLT_PFJet550', 'HLT_PFJet500', 'HLT_PFJet450', 'HLT_PFJet400', 'HLT_PFJet320',  'HLT_PFJet260', 'HLT_PFJet200', 'HLT_PFJet140', 'HLT_PFJet110', 'HLT_PFJet40', 'HLT_PFJet60']

    fig, ax = plt.subplots(figsize=(10,10))
    ax.axhline(y=1, linestyle="--", color="gray")

    for trigger in triggers:

       dataset = hist.loc("2022" + era) if era != "" else sum

       ptproj = (
           out_o["ak4"]
           .project("ht", "trigger", "dataset")
       )
       denom = ptproj[:, hist.loc("none"), dataset]
       num = ptproj[:, hist.loc(trigger), dataset]

       hist_data_before, hist_bins = denom.to_numpy()
       hist_data_after, hist_bins = num.to_numpy()

       from scipy.stats import beta

       def binom_int(num, den, confint=0.68):
           quant = (1 - confint)/ 2.
           low = beta.ppf(quant, num, den - num + 1)
           high = beta.ppf(1 - quant, num + 1, den - num)
           return (np.nan_to_num(low), np.where(np.isnan(high), 1, high))

       # calculating efficiency
       efficiency = hist_data_after/hist_data_before

       # getting error band
       band_low, band_high = binom_int(hist_data_after, hist_data_before)
       error_low = efficiency - band_low
       error_high = band_high - efficiency

       # removing large errors in empty bins
       error_low[error_low == 1] = 0
       error_high[error_high == 1] = 0

       # stacking errors
       error = np.concatenate((error_low.reshape(error_low.shape[0], 1), error_high.reshape(error_high.shape[0], 1)), axis=1)

       data_err_opts = {
               'linestyle': 'none',
               'marker': '.',
               'markersize': 10.,
               'elinewidth': 1,
           }

       ax.errorbar(
                   num.axes[0].centers,
                   efficiency,
                   yerr=error.T,
                   #color="#FF66FF",
                   label=trigger if trigger != "any" else "Logical OR of all above",
                   **data_err_opts,
               )

       label = mplhep.cms.label(ax=ax, data=True, year="2022", com=13.6, label="Data")
       #xlabel = ax.set_xlabel(r"AK8 jet $p_T$ (GeV)")
       xlabel = ax.set_xlabel(r"$H_T$ (GeV)")

       ax.set_ylim(0, 1.1)
       lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
       #ax.legend(loc='best') #, fontsize=30)
       ax.set_ylabel("Efficiency")
       #ax.set_xlim(0, 1000)
       ax.set_xlim(0, 2000)

       fig.savefig(f"HT_{era}.png", bbox_extra_artists=(lgd, label[0], label[1], xlabel), bbox_inches='tight')

else:
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.axhline(y=1, linestyle="--", color="gray")

    for rec in ["scouting", "offline"]:
        
       eff = []
       err1 = []
       err2 = []
       c = []

       if rec == "scouting":
           out = out_s
       else:
           out = out_o

       dataset = hist.loc("2022" + era) if era != "" else sum
    
       for x_range, cone, var  in [("before", "ak8", "pt"), ("after", "ak8", "pt")]:

           ptproj = (
                out[cone]
                .project(var, "trigger", "dataset")
           )

           pt_range = pt[rec][cone][var][x_range]
            
           denom = ptproj[
                hist.rebin(pt_range["bin"]), hist.loc("none"), dataset][
                hist.loc(pt_range["range"][0]):hist.loc(pt_range["range"][1])
           ]
           num = ptproj[
                hist.rebin(pt_range["bin"]), hist.loc("any"), dataset][
                hist.loc(pt_range["range"][0]):hist.loc(pt_range["range"][1])
           ]
            
           centers = denom.axes[0].centers

           hist_data_before, hist_bins = denom.to_numpy()
           hist_data_after, hist_bins = num.to_numpy()

           from scipy.stats import beta

           def binom_int(num, den, confint=0.68):
               quant = (1 - confint)/ 2.
               low = beta.ppf(quant, num, den - num + 1)
               high = beta.ppf(1 - quant, num + 1, den - num)
               return (np.nan_to_num(low), np.where(np.isnan(high), 1, high))

           # calculating efficiency
           efficiency = hist_data_after/hist_data_before

           # getting error band
           band_low, band_high = binom_int(hist_data_after, hist_data_before)
           error_low = efficiency - band_low
           error_high = band_high - efficiency

           # removing large errors in empty bins
           error_low[error_low == 1] = 0
           error_high[error_high == 1] = 0

           # stacking errors
           error = np.concatenate((error_low.reshape(error_low.shape[0], 1), error_high.reshape(error_high.shape[0], 1)), axis=1)

           data_err_opts = {
                   'linestyle': 'none',
                   'marker': '.',
                   'markersize': 10.,
                   'elinewidth': 1,
           }
            
           eff =  np.concatenate((eff, efficiency))
           c = np.concatenate((c, num.axes[0].centers))
           err1 = np.concatenate((err1, error.T[0]))
           err2 = np.concatenate((err2, error.T[1]))
    
       ax.errorbar(
                   c,
                   eff,
                   xerr=[
                        [np.abs(c[i] - c[i-1]) / 2 if i > 0 else np.abs(c[i+1] - c[i]) / 2 for i, _ in enumerate(c)],
                        [np.abs(c[i+1] - c[i]) / 2 if i < len(c) - 1 else np.abs(c[i] - c[i-1]) / 2 for i, _ in enumerate(c)]
                    ],
                   yerr=[err1, err2],
                   color="#FF66FF" if rec == "scouting" else "#FF9900",
                   label="Scouting" if rec == "scouting" else "Offline",
                   **data_err_opts,
               )

    label = mplhep.cms.label(ax=ax, data=True, year="2022", com=13.6, label="Preliminary", lumi=34)
    if var == "pt":
       xlabel = ax.set_xlabel(f"{cone.upper()}" + r" jet $p_T$ (GeV)")
    else:
       xlabel = ax.set_xlabel(r"$H_T$ (GeV)")

    ax.set_ylim(0, 1.1)
    #lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    lgd = ax.legend(loc='lower right', fontsize=25)
    ax.set_ylabel("Efficiency")
    if var == "pt":
       ax.set_xlim(0, 1000)
    else:
       ax.set_xlim(0, 2000)

    fig.savefig(f"Temp.png", bbox_extra_artists=(lgd, label[0], label[1], xlabel), bbox_inches='tight')
