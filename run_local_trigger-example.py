from processors import TriggerProcessor
import os
import uproot
from coffea import processor, util
from coffea.nanoevents import NanoEventsFactory, ScoutingNanoAODSchema

fileset = {
    "Run2022E" : [
        "root://eoscms.cern.ch//eos/cms/store/group/ml/Tagging4ScoutingHackathon/Adelina/ScoutingPFRun3/Run2022E-HLT_Mu50/230315_124313/0002/scoutingnano_2535.root"
    ],
}

output = processor.run_uproot_job(
            fileset,
            "Events",
            processor_instance=TriggerProcessor(),
            executor=processor.futures_executor,
            executor_args={
                "schema": ScoutingNanoAODSchema,
                'skipbadfiles': True,
            },
            chunksize=10,
            maxchunks=10,
        )

if not os.path.exists(f'outfiles/Run3Summer22EE'):
    os.makedirs(f'outfiles/Run3Summer22EE')

outfile = f"outfiles/Run3Summer22EE/debug.root"
util.save(output, outfile)
print("saved " + outfile)

