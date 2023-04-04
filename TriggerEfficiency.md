# Compute trigger efficiency

## Reference method

Sources:

- [Efficiency measurement for hadronic triggers by Finn Labe](https://indico.cern.ch/event/1238936/contributions/5209612)

## How to run the code

### Start environment

1. Install [lpcjobqueue](https://github.com/CoffeaTeam/lpcjobqueue)

2. Create grid proxy

```
voms-proxy-init -voms cms -valid 192:00
```

3. Clone coffea version with ScoutingNanoAOD

```
git clone git@github.com:alintulu/coffea.git -b scouting
```

4. Clone this repository

```
git clone git@github.com:alintulu/PNCalibration.git
```

5. Start singularity image

```
./shell
```

6. Install coffea version with ScoutingNanoAOD

```
cd coffea
pip install .
cd ..
cd PNCalibration
```

### Run a test locally

1. Run processor

```
python run_local_trigger-example.py
```

2. Check output file

```
ls outfiles/Run3Summer22/debug.root
```

### Run full scale on DASK

1. Run processor

```
python submit_dask_trigger-example.py Run3Summer22EE_HLT_Mu50 Run3Summer22EE
```

2. Create trigger efficiency plot

```
jupyter nbclassic --no-browser --port=8765
```

Open `trigger-example.ipynb` and run all cells.
