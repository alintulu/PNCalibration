from processors import HistProcessor 
import os
import uproot
from coffea import processor, util
from coffea.nanoevents import NanoEventsFactory, ScoutingNanoAODSchema

import json
fileset = {}
with open("inputfiles/run2022d.json") as fin:
    fileset = json.load(fin)


output = processor.run_uproot_job(
            fileset,
            "Events",
            processor_instance=HistProcessor(),
            executor=processor.futures_executor,
            executor_args={
                "schema": ScoutingNanoAODSchema,
                'skipbadfiles': True,
            },
            chunksize=10,
            maxchunks=10,
        )

outfile = f"outfiles/run2022d.coffea"
util.save(output, outfile)
print("saved " + outfile)

