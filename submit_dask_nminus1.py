from distributed import Client
from lpcjobqueue import LPCCondorCluster

from dask.distributed import performance_report
from dask_jobqueue import HTCondorCluster, SLURMCluster

from processors import HistProcessor, CutflowProcessor, ZbbProcessor, PVProcessor, JetIDProcessor, Nminus1Processor
import os, sys
import uproot
from coffea import processor, util
from coffea.nanoevents import NanoEventsFactory, ScoutingNanoAODSchema

import json

sample = sys.argv[1] #"Run2022D"
era = sys.argv[2] #"Run3Summer22"

fileset = {}
with open(f"inputfiles/{era}/{sample}.json") as fin:
    fileset = json.load(fin)

env_extra = [
    f"export PYTHONPATH=$PYTHONPATH:{os.getcwd()}",
]

cluster = LPCCondorCluster(
    transfer_input_files=["processors"],
    ship_env=True,
    memory="8GB",
)

cluster.adapt(minimum=1, maximum=100)
client = Client(cluster)

print("Waiting for at least one worker...")
client.wait_for_workers(1)

uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

output = processor.run_uproot_job(
            fileset,
            "Events",
            processor_instance=Nminus1Processor(do_isomuon=False, wp_btag=-1, do_jetid=True, is_postEE=True if "EE" in era else False),
            executor=processor.dask_executor,
            executor_args={
                "schema": ScoutingNanoAODSchema,
                "savemetrics": True,
                "retries": 3,
                "client": client,
                'skipbadfiles': True,
            },
            chunksize=10000,
            #maxchunks=args.max,
        )

outfile = f"outfiles/{era}/fulllumi/nminus1_{sample}_jetId_btag-1.coffea"
util.save(output, outfile)
print("saved " + outfile)
