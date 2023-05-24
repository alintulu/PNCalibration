from processors import HistProcessor, CutflowProcessor, DebugProcessor, Nminus1Processor, TriggerProcessor, JERDijetProcessor, JERPhotonProcessor, JESProcessor, DiffProcessor, TriggerDijetProcessor
import os
import uproot
from coffea import processor, util
from coffea.nanoevents import NanoEventsFactory, ScoutingNanoAODSchema, ScoutingJMENanoAODSchema

fileset = {
    "Run2022E" : [
        "root://eoscms.cern.ch//eos/cms/store/group/phys_jetmet/adlintul/JME-Scouting-Nano/ScoutingPFMonitor/Run2022F/230403_185919/0000/jmenano_data_100.root"
    ],
#    "MC" : [
#        "root://eoscms.cern.ch//eos/cms/store/group/phys_jetmet/pinkaew/JME-Scouting-Nano/QCD_PT-15to7000_TuneCP5_13p6TeV_pythia8/Run3Summer22EE/230510_164629/0002/tree_jmenano_qcd_2000.root"
#    ],
}

output = processor.run_uproot_job(
            fileset,
            "Events",
            processor_instance=JESProcessor(triggers=["L1_HTT200er","L1_HTT255er","L1_HTT280er","L1_HTT320er","L1_HTT360er","L1_HTT400er","L1_HTT450er","L1_SingleJet180","L1_SingleJet200"],lowPt=True),
            #processor_instance=TriggerDijetProcessor(),
            executor=processor.futures_executor,
            executor_args={
                "schema": ScoutingJMENanoAODSchema,
                'skipbadfiles': True,
            },
            chunksize=10,
            maxchunks=10,
        )

outfile = "outfiles/Run3Summer22/local/debug.root"
util.save(output, outfile)
print("saved " + outfile)

