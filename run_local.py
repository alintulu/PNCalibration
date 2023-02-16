from processors import HistProcessor, CutflowProcessor
import os
import uproot
from coffea import processor, util
from coffea.nanoevents import NanoEventsFactory, ScoutingNanoAODSchema

fileset = {
    "TTtoLNu2Q" : [
        "root://eosuser.cern.ch//eos/user/a/adlintul/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22/230131_071621/0000/nanoaod_1.root"
    ],
    "Run2022D" : [
        "root://eosuser.cern.ch//eos/user/a/adlintul/ScoutingPFRun3/Run2022D/230206_163934/0000/scoutingnano_1.root"
    ],
}

output = processor.run_uproot_job(
            fileset,
            "Events",
            processor_instance=CutflowProcessor(),
            executor=processor.futures_executor,
            executor_args={
                "schema": ScoutingNanoAODSchema,
                'skipbadfiles': True,
            },
            chunksize=10,
            maxchunks=10,
        )

outfile = f"outfiles/local.coffea"
util.save(output, outfile)
print("saved " + outfile)

