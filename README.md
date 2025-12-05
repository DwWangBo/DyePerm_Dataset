
# DyePermDB — Membrane Permeability Dataset for Organic Dyes

DyePermDB is a curated dataset of **202 fluorescent and chromogenic dyes** with experimentally supported, manually validated **membrane-permeability annotations**.  
This repository provides the complete dataset, drug-information subset, reproducible train/test splits, and XGBoost-based baseline models used in the DyePermDB study.

The code allows users to reproduce all classification experiments and evaluate structure–permeability relationships using FP4 fingerprints.

---

##  Repository Structure

DyePerm_Dataset/
│
├── DyePerm_dataset.csv
├── DyePerm_dataset.xlsx
├── DyePerm_druginfo.csv
├── DyePerm_druginfo.xlsx
│
├── DyePerm_dataset_train.csv
├── DyePerm_dataset_test.csv
│
├── structure_picture.zip 
│
├── DyePerm_XGB_yes_no.py # Binary model: Yes vs No
├── DyePerm_XGB_yes_vs_conditional.py # Binary model: Yes vs Yes (conditional)
├── DyePerm_XGB_v3_multiclass.py # Three-class model: Yes / Cond / No
│
└── README.md



# Computational analysis and modelling

##  Create environment
conda create -n dyeperm python=3.9
conda activate dyeperm
pip install pandas numpy scikit-learn xgboost matplotlib rdkit-pypi



##  Machine-Learning Tasks (XGBoost + FP4 fingerprints)
    Three supervised classification tasks are implemented:
        1. Yes vs No (binary)                     Script: DyePerm_XGB_yes_no.py
            python DyePerm_XGB_yes_no.py \
                --train DyePerm_dataset_train.csv \
                --test DyePerm_dataset_test.csv
        
        2. Yes vs Yes (conditional) (binary)      Script: DyePerm_XGB_yes_vs_conditional.py
            python DyePerm_XGB_yes_vs_conditional.py \
                --train DyePerm_dataset_train.csv \
                --test DyePerm_dataset_test.csv

        3. Three-class prediction                 Script: DyePerm_XGB_v3_multiclass.py
            python DyePerm_XGB_v3_multiclass.py \
                --train DyePerm_dataset_train.csv \
                --test DyePerm_dataset_test.csv


# Web Interface
    The online version of DyePermDB is available at:http://47.83.134.12/DyePermDB/

