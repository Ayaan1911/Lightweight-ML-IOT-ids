"""Validation only: which categories are confused with Benign (uses saved xgb_multi, read-only)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.common import CLASSES
from src import data, predictors
X, y = data.load("val", "multi")
p = predictors.load("xgb_multi")(X.to_numpy(np.float32))
cm = pd.crosstab(pd.Series(y).map(dict(enumerate(CLASSES))).rename("true"), pd.Series(p).map(dict(enumerate(CLASSES))).rename("pred"))
cm.to_csv(ROOT / "diagnostics/step3b_val_cm_xgb_multi.csv")
ben = cm.loc["Benign"]; print("true Benign ->", (ben / ben.sum()).round(3).to_dict())
print("share of each true class predicted Benign:", (cm["Benign"] / cm.sum(1)).round(4).to_dict())
print("composition of rows predicted Benign:", (cm["Benign"] / cm["Benign"].sum()).round(3).to_dict())
pd.DataFrame({"share_of_true_class_predicted_Benign": cm["Benign"] / cm.sum(1), "n_true": cm.sum(1), "n_pred_Benign": cm["Benign"]}).to_csv(ROOT / "diagnostics/step3b_confusable_with_benign.csv")
