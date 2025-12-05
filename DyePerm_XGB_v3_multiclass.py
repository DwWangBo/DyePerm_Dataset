#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2

import xgboost as xgb
from openbabel import pybel

##python DyePerm_XGB_v3_multiclass.py --data-dir "C:/Users/Administrator/Desktop/Model/data_split"
# ==============================
#       FP4 指纹转换
# ==============================
def smiles_to_fp4_bits(smiles_list: List[str]) -> np.ndarray:
    n_bits = 307
    mats = []

    for s in smiles_list:
        bitvec = np.zeros(n_bits, dtype=int)
        if not isinstance(s, str) or not s.strip():
            mats.append(bitvec)
            continue
        try:
            mol = pybel.readstring("smi", s.strip())
            fp = mol.calcfp("FP4")
            for b in getattr(fp, "bits", []):
                if 1 <= b <= n_bits:
                    bitvec[b - 1] = 1
        except Exception:
            pass
        mats.append(bitvec)

    return np.vstack(mats)


# ==============================
#       三分类指标
# ==============================
def compute_multiclass_metrics(y_true, y_prob, num_classes=3) -> Dict[str, float]:
    y_pred = np.argmax(y_prob, axis=1)

    # Macro 指标
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # AUC（需 one-hot）
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")

    return {
        "Acc": acc,
        "Prec": prec,
        "Rec": rec,
        "F1": f1,
        "AUC": auc,
    }


# ==============================
#       主程序
# ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()

    data_dir = args.data_dir

    # ===== 识别 split 序号 =====
    groups = []
    for f in os.listdir(data_dir):
        m = re.findall(r"data_split_(\d+)_train\.csv", f)
        if m:
            groups.append(int(m[0]))
    groups = sorted(groups)

    print("📌 检测到 splits:", groups)

    results = []

    # ============================
    #      遍历每个 split
    # ============================
    for gid in groups:
        print(f"\n🔁 Split {gid} ...")

        tr = pd.read_csv(os.path.join(data_dir, f"data_split_{gid}_train.csv"))
        te = pd.read_csv(os.path.join(data_dir, f"data_split_{gid}_test.csv"))

        # ===== 标签映射 =====
        map_dict = {
            "no": 0,
            "yes(conditional)": 1,
            "yes (conditional)": 1,
            "yes": 2,
        }

        y_tr = tr["Membrane-Permeable"].astype(str).str.lower().str.strip().map(map_dict).values
        y_te = te["Membrane-Permeable"].astype(str).str.lower().str.strip().map(map_dict).values

        # ===== FP4 =====
        X_tr = smiles_to_fp4_bits(tr["SMILES"].tolist())
        X_te = smiles_to_fp4_bits(te["SMILES"].tolist())

        # ===== 特征选择 =====
        var_thr = VarianceThreshold(0.0)
        X_tr_v = var_thr.fit_transform(X_tr)
        X_te_v = var_thr.transform(X_te)

        k = min(256, X_tr_v.shape[1])
        skb = SelectKBest(chi2, k=k)
        X_tr_s = skb.fit_transform(X_tr_v, y_tr)
        X_te_s = skb.transform(X_te_v)

        # ===== 多分类 XGBoost =====
        dtrain = xgb.DMatrix(X_tr_s, label=y_tr)
        dtest = xgb.DMatrix(X_te_s, label=y_te)

        params = {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": 3,
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

        booster = xgb.train(params, dtrain, num_boost_round=500)

        # ===== 预测与评估 =====
        y_tr_prob = booster.predict(dtrain)
        y_te_prob = booster.predict(dtest)

        m_tr = compute_multiclass_metrics(y_tr, y_tr_prob)
        m_te = compute_multiclass_metrics(y_te, y_te_prob)

        diff_auc = m_tr["AUC"] - m_te["AUC"]

        results.append({
            "Split": gid,
            "Train_AUC": m_tr["AUC"], "Test_AUC": m_te["AUC"], "Diff_AUC": diff_auc,
            "Train_Acc": m_tr["Acc"], "Test_Acc": m_te["Acc"],
            "Train_Prec": m_tr["Prec"], "Test_Prec": m_te["Prec"],
            "Train_Rec": m_tr["Rec"], "Test_Rec": m_te["Rec"],
            "Train_F1": m_tr["F1"], "Test_F1": m_te["F1"],
        })

    # ============================
    #    汇总输出
    # ============================
    df = pd.DataFrame(results).sort_values("Test_AUC", ascending=False)

    col_order = [
        "Split",
        "Train_AUC", "Test_AUC", "Diff_AUC",
        "Train_Acc", "Test_Acc",
        "Train_Prec", "Test_Prec",
        "Train_Rec", "Test_Rec",
        "Train_F1", "Test_F1",
    ]
    df = df[col_order]

    out_path = os.path.join(data_dir, "xgb_multiclass_results.csv")
    df.to_csv(out_path, index=False)

    print("\n📊 ====== 三分类结果（按 Test_AUC 排序）======")
    print(df.round(4))
    print("\n📁 已保存到:", out_path)


if __name__ == "__main__":
    main()
