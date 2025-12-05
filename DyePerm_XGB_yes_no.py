#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

import xgboost as xgb
from openbabel import pybel
#python DyeMem_XGB_yes_no.py --data-dir "C:/Users/Administrator/Desktop/Model/data_split"


# =======================
#     FP4 指纹转换
# =======================
def smiles_to_fp4_bits(smiles_list: List[str]) -> np.ndarray:
    n_bits = 307
    mats = []
    for s in smiles_list:
        bitvec = np.zeros(n_bits, dtype=int)
        if not isinstance(s, str) or not s.strip() or s.strip() == "/":
            mats.append(bitvec)
            continue
        try:
            mol = pybel.readstring("smi", s.strip())
            fp = mol.calcfp("FP4")
            for b in getattr(fp, "bits", []):
                if 1 <= b <= n_bits:
                    bitvec[b - 1] = 1
        except Exception:
            mats.append(bitvec)
            continue
        mats.append(bitvec)
    return np.vstack(mats)


# =======================
#       计算指标
# =======================
def compute_metrics(y_true, y_prob, threshold=0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Acc": accuracy_score(y_true, y_pred),
        "Prec": precision_score(y_true, y_pred, zero_division=0),
        "Rec": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob),
    }


# =======================
#          主程序
# =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    data_dir = args.data_dir

    groups = sorted([int(re.findall(r"data_split_(\d+)_train\.csv", f)[0])
                     for f in os.listdir(data_dir) if "train" in f])

    print(f"将运行 split: {groups}")

    results = []

    for gid in groups:
        print(f"\n🔁 Split {gid} ...")

        tr = pd.read_csv(os.path.join(data_dir, f"data_split_{gid}_train.csv"))
        te = pd.read_csv(os.path.join(data_dir, f"data_split_{gid}_test.csv"))

        # ===== 标签：Yes + Yes(conditional) → 1 =====
        tr_label = tr["Membrane-Permeable"].astype(str).str.lower().str.strip()
        tr_label = tr_label.replace({"yes(conditional)": "yes", "yes (conditional)": "yes"})
        y_tr = tr_label.eq("yes").astype(int).values

        te_label = te["Membrane-Permeable"].astype(str).str.lower().str.strip()
        te_label = te_label.replace({"yes(conditional)": "yes", "yes (conditional)": "yes"})
        y_te = te_label.eq("yes").astype(int).values

        # FP4
        X_tr = smiles_to_fp4_bits(tr["SMILES"].tolist())
        X_te = smiles_to_fp4_bits(te["SMILES"].tolist())

        # ===== 特征选择（在 train 上 fit，然后应用到 test）=====
        var_thr = VarianceThreshold(threshold=0.0)
        X_tr_v = var_thr.fit_transform(X_tr)
        X_te_v = var_thr.transform(X_te)

        k = min(256, X_tr_v.shape[1])
        skb = SelectKBest(chi2, k=k)
        X_tr_s = skb.fit_transform(X_tr_v, y_tr)
        X_te_s = skb.transform(X_te_v)

        # ===== XGBoost 固定参数训练 =====
        dtrain = xgb.DMatrix(X_tr_s, label=y_tr)
        dtest = xgb.DMatrix(X_te_s, label=y_te)

        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": args.random_state,
        }

        booster = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=500)

        # ===== 评估 =====
        y_tr_prob = booster.predict(dtrain)
        y_te_prob = booster.predict(dtest)

        m_tr = compute_metrics(y_tr, y_tr_prob)
        m_te = compute_metrics(y_te, y_te_prob)

        diff = m_tr["AUC"] - m_te["AUC"]

        results.append({
            "Split": gid,
            "Train_AUC": m_tr["AUC"], "Test_AUC": m_te["AUC"], "Diff_AUC": diff,
            "Train_Acc": m_tr["Acc"], "Test_Acc": m_te["Acc"],
            "Train_Prec": m_tr["Prec"], "Test_Prec": m_te["Prec"],
            "Train_Rec": m_tr["Rec"], "Test_Rec": m_te["Rec"],
            "Train_F1": m_tr["F1"], "Test_F1": m_te["F1"],
        })

    # ===== 汇总输出 =====
    df = pd.DataFrame(results).sort_values("Test_AUC", ascending=False).reset_index(drop=True)

    col_order = [
        "Split",
        "Train_AUC", "Test_AUC", "Diff_AUC",
        "Train_Acc", "Test_Acc",
        "Train_Prec", "Test_Prec",
        "Train_Rec", "Test_Rec",
        "Train_F1", "Test_F1"
    ]
    df = df[col_order]

    out_path = os.path.join(data_dir, "xgb_train100_test100_results.csv")
    df.to_csv(out_path, index=False)

    print("\n📊 ======= XGBoost汇总输出——Yes/No =======")
    print(df.round(4))
    print(f"\n📁 已保存到: {out_path}")


if __name__ == "__main__":
    main()
