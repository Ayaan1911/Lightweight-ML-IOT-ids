"""Pick q8dyn vs full-integer q8 by VALIDATION macro F1 (then size). Never looks at test metrics."""
from .common import *
from . import predictors


def choose():
    out = {}
    for task in ("binary", "multi"):
        for base in ("q8", "distq8"):
            c = {}
            for suffix, label in (("", "full-integer"), ("dyn", "dynamic-range")):
                mid = f"mlp_{task}_{base}{suffix}"
                c[mid] = {"mode": label, "val_macro_f1": jload(RES / f"metrics_{mid}.json")["val_macro_f1"], "size_kb": predictors.size_kb(mid)}
            best = max(c, key=lambda m: (round(c[m]["val_macro_f1"], 2), -c[m]["size_kb"]))  # >=0.5pt F1 gap decides, else smaller
            out[f"mlp_{task}_{base}"] = {"chosen": best, "candidates": c}
    jdump(out, RES / "mlp_quant_choice.json")
    return out


if __name__ == "__main__":
    print(json.dumps(choose(), indent=1) if (json := __import__("json")) else "")
