# Reproducibility

## Dependencies

The Python dependencies are listed in the file `env.yaml` as conda environment. You might need to adapt the `cudatoolkit=11.3` version in the yaml file.

```bash
conda env create -f env.yaml -p ./conda-env
conda activate ./conda-env
```
We need to install `scikit-sparse` separately because it is problematic to install on Windows systems with conda. For updated instructions see the package's README [this thread](https://github.com/scikit-sparse/scikit-sparse).

On Windows:

```powershell
$env:SUITESPARSE_INCLUDE_DIR="$env:CONDA_PREFIX/Library/include/suitesparse"
$env:SUITESPARSE_LIBRARY_DIR="$env:CONDA_PREFIX/Library/lib"
pip install scikit-sparse
```

On Linux (not tested on Mac OS):
```bash
conda install -c conda-forge scikit-sparse
```

For downloading and pre-processing atmospheric and air pollution data we also need R. It wasn't possible to have a reproducible R conda environment since the packages required for the task conflicted with conda. We use Docker instead (or you can manually install the required packages by examining the `Dockerfile`).

```bash
docker build -t r-env .
```

## Simulation Study

### Generate data

```bash
python generate_simdata.py --verbose --output_dir=simulations/basic --ksize=13 --nsims=10
python generate_simdata.py --verbose --output_dir=simulations/nonlinear --ksize=13 --nsims=10 --nonlinear
```


### Results
```bash
for i in {0...9}; do python train_simstudy.py --sim="$i" --dir=simulations/basic --output=results/simstudy/basic --epochs 20000; done 
for i in {0...9}; do python train_simstudy.py --sim="$i" --dir=simulations/nonlinear --output=results/simstudy/nonlinear --epochs 20000; done
for i in {0...9}; do python train_simstudy.py --sparse --sim="$i" --dir=simulations/basic --output=results/simstudy/basic_sparse --epochs 10000; done 
for i in {0...9}; do python train_simstudy.py --sparse --sim="$i" --dir=simulations/nonlinear --output=results/simstudy/nonlinear_sparse --epochs 10000; done 
```

Then check out the notebooks
* `notebooks/explore_simdata.ipynb`
* `notebooks/potentials.ipynb`
* `notebooks/simstudy_results.ipynb`


## Applications: Data preprocessing and download (SO4 and NARR)


### First steps
It used to be possible to download the SO4 data automatically using Python and ftp. But as of May 2022, you will need to download it manual from [this link](https://wustl.box.com/s/wk3144jc6xfy6ujfvyv5m2yfk33nz2nn). For background see, or if the link stops working, see [https://sites.wustl.edu/acag/datasets/surface-pm2-5/](https://sites.wustl.edu/acag/datasets/surface-pm2-5/). The exact dataset we need is under `Monthly/ASCII/SO4` from the monthly `V4.NA.02` PM2.5 total mass and composition described in van Donkelaar et al. (2019).

In addition, you need to manually download the Power Plant Emission Data file `AMPD_Unit_with_Sulfur_Content_and_Regulations_with_Facility_Attributes.csv` from [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/M3D2NR) and place it in the  `data/` folder.

### Download and process the remaining data

From the root folder run:

```bash
docker run --rm -it -v $(pwd):/workspace/ r-env bash preprocessing/preprocess_raw.sh
```
***Note***: On Windows Powershell you can replace `$(pwd)` with `${pwd}`. The pipeline does seem to run slow on Windows (tested on Windows 11). If that affects you, you can run the preprocessing scripts in `prepare_training` using your native `R`. If you are running out of memory, try reducing the number of parallel processes at line 8 of `preprocessing/process_so4.R`.


It might take a few hours for the process to finish. After it's done, you can run (on the conda environment):

```
python preprocessing/prepare_training.py
```

The result should create a file `data/training_data.pkl` of approximately 266 MB.

## Application 1:

### Step 1. Self-supervised features from NARR

Run the commands
```bash
python train_app1_self_narr.py --radius=1 --odir=r1_w2vec
python train_app1_self_narr.py --radius=3 --odir=r3_w2vec
python train_app1_self_narr.py --radius=5 --odir=r5_w2vec
python train_app1_self_narr.py --radius=7 --odir=r7_w2vec
python train_app1_self_narr.py --radius=9 --odir=r9_w2vec
python train_app1_self_narr.py --radius=3 --odir=r3_nbrs --nbrs_av=3
python train_app1_self_narr.py --radius=9 --odir=r9_nbrs --nbrs_av=9
python train_app1_self_narr.py --radius=3 --odir=r3_local --local
python train_app1_self_narr.py --radius=9 --odir=r9_local --local
```
Then run the notebook `notebooks/visualize_selfsupervised.ipynb`.

### Step 2. Causal estimation and comparison with DAPSM.

First clone the DAPSm analysis repository from the previous analysis.

```bash
git clone https://github.com/gpapadog/DAPSm-Analysis dapsm
```

## Application 2: Meteorological De-trending with Prognostic Score

Run the following commands

```bash
python train_app2_detrend.py --odir=w2vec
python train_app2_detrend.py --odir=local --local
python train_app2_detrend.py --odir=n9 --av-nbrs=9
python train_app2_detrend.py --odir=unadjusted --unadjusted
```

Then to produce the plots and analysis run the Jupyter notebook `notebooks/visualize_supervised.ipynb`.