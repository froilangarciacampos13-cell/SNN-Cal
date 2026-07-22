# Neuromorphic Readout for Hadron Calorimeters — SNN Reconstruction Pipeline

Spiking Neural Networks (SNNs) for **transduction-less readout of hadronic
calorimeters**: reconstructing shower observables directly from the temporal
photon-scintillation signals of a homogeneous PbWO₄ calorimeter.

This repository is part of a research collaboration extending the paper
[*Neuromorphic Readout for Hadron Calorimeters*](https://arxiv.org/abs/2502.12693)
(Lupi et al., 2025). It contains my Bachelor's-thesis contribution to that
effort.

---

## My contribution — `tests_diego/`

A complete, reproducible SNN pipeline that reconstructs, **per event**, the
deposited energy and the 3D energy centroid of a shower in a PbWO₄ calorimeter
segmented into a 10×10×10 grid of cubelets (100 sensors), from raw
photon-scintillation data.

This pipeline was contributed to the collaboration's repository, reviewed, and
**integrated into the upstream project** (June 2026) — see
[PR #1](https://github.com/a-saborido/SNN-Cal/pull/1).

**→ Full documentation and usage: [`tests_diego/README.md`](tests_diego/README.md)**

Three stages, run in order:

```
generate_dataset.py   →   train_model.py   →   print_predictions.py
   (raw → .pt)              (.pt → .pth)         (.pt + .pth → metrics/plots)
```

Highlights:

- **Regression targets:** deposited energy (in `log10(E/MeV)`), spatial centroid
  `(x_c, y_c, z_c)` and dispersion — configurable per run.
- **Learned encoding:** the SNN encoder learns per-threshold exponents that map
  scintillation signals to spike trains.
- **Radial sectioning study:** the detector can be carved into regions (e.g. an
  inner cylinder vs. its complement) to compare a global model against
  region-specific ones.
- **Rigorous evaluation:** fixed-seed 70/15/15 train/val/test split and a
  figures-of-merit table per target (Pearson r, R², RMSE, MAE, bias, residual
  variance), plus deep diagnostics — binned bias/variance profiles, empirical
  vs. expected CDFs, per-neuron spike activity.

Built with **PyTorch** and **snnTorch** (NumPy, Matplotlib, SciPy,
scikit-learn).

---

## Setup

The pipeline runs on Python with `torch`, `snntorch`, `numpy`, `matplotlib`,
`scipy`, `pandas`, `scikit-learn` and `tqdm`. Using a package/environment manager
such as Anaconda is recommended.

### 1. Get the repository

[Fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)
this repository to get your own copy, then
[clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
it locally.

### 2. Install Miniconda

Follow the official [installation guide](https://docs.anaconda.com/miniconda/install/).
On a Linux machine:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```

### 3. Create the environment

For the exact environment used in this project (`numpy`, `PyTorch`, `snnTorch`,
Jupyter and the rest):

```bash
conda env create -f env/environmental_droplet.yml   # creates the snn_hgcal env
```

Then follow the step-by-step run instructions in
[`tests_diego/README.md`](tests_diego/README.md).

---

## Repository layout

| Path | Content |
|------|---------|
| `tests_diego/` | **My contribution** — the full reconstruction pipeline (see its README) |
| `SNN/` | Core neuromorphic-computing modules |
| `GenerateDataset/` | Dataset-creation scripts |
| `Data/` | Datasets |
| `env/` | Conda environment specification |

---

*Physics context: PbWO₄ (lead tungstate) is a homogeneous scintillating
calorimeter medium; "transduction-less" readout means inferring shower
observables directly from the light signal, without an intermediate segmented
active medium. See the [paper](https://arxiv.org/abs/2502.12693) for the full
detector setup.*
