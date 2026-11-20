"""Sampling, inspection, cleaning, splitting. Run: python -m src.data <sample|inspect|clean|split>"""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from .common import *

CHUNK = 500_000
BUDGET, SMALL, FLOOR = 1_000_000, 30_000, 10_000


def files():
    return sorted(CSV_DIR.glob("Merged*.csv"))


def label_col(cols):
    return next(c for c in cols if c.strip().lower() == "label")


def pass_a():
    out = RES / "label_counts_full.csv"
    if out.exists():
        return pd.read_csv(out, index_col=0)["count"]
    tot = pd.Series(dtype="int64")
    nrows = 0
    for f in files():
        lc = label_col(pd.read_csv(f, nrows=0).columns)
        s = pd.read_csv(f, usecols=[lc])[lc].map(norm_label)
        tot = tot.add(s.value_counts(), fill_value=0).astype("int64")
        nrows += len(s)
        print(f.name, nrows, flush=True)
    tot = tot.sort_values(ascending=False)
    tot.rename("count").to_frame().to_csv(out)
    return tot


def quota(counts):
    """Sampling rule: labels with < SMALL rows kept whole; the others share (BUDGET - kept)
    proportionally to their size, with a floor of FLOOR rows (or all rows if fewer)."""
    small = counts[counts < SMALL]
    big = counts[counts >= SMALL]
    remaining = BUDGET - small.sum()
    q = (remaining * big / big.sum()).round().astype(int)
    q = pd.concat([q, pd.Series(FLOOR, index=q.index)], axis=1).max(axis=1)
    q = pd.concat([q, big], axis=1).min(axis=1)
    return pd.concat([small, q]).astype(int)


def pass_b():
    out = PROC / "sample.parquet"
    if out.exists():
        return
    counts = pass_a()
    q = quota(counts)
    rng = np.random.default_rng(SEED)
    # exact stratified sample: pre-draw which occurrence indices of each label are kept
    keep = {l: (np.sort(rng.choice(counts[l], q[l], replace=False)) if q[l] < counts[l] else None) for l in counts.index}
    seen = {l: 0 for l in counts.index}
    parts = []
    for f in files():
        for ch in pd.read_csv(f, chunksize=CHUNK):
            lc = label_col(ch.columns)
            lab = ch[lc].map(norm_label).to_numpy()
            mask = np.zeros(len(ch), bool)
            for l in np.unique(lab):
                idx = np.flatnonzero(lab == l)
                k = keep[l]
                if k is None:
                    mask[idx] = True
                else:
                    mask[idx] = np.isin(np.arange(seen[l], seen[l] + len(idx)), k, assume_unique=True)
                seen[l] += len(idx)
            sub = ch[mask].copy()
            sub[lc] = lab[mask]
            sub = sub.rename(columns={lc: "Label"})
            feat = [c for c in sub.columns if c != "Label"]
            sub[feat] = sub[feat].astype("float32")
            parts.append(sub)
        print(f.name, sum(map(len, parts)), flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(out)
    pd.DataFrame({"before": counts, "after": df["Label"].value_counts()}).fillna(0).astype(int).to_csv(RES / "sampling_counts.csv")


def inspect():
    df = pd.read_parquet(PROC / "sample.parquet")
    feat = [c for c in df.columns if c != "Label"]
    pd.DataFrame({"dtype": df.dtypes.astype(str)}).to_csv(RES / "inspect_dtypes.csv")
    df["Label"].value_counts().rename("count").to_csv(RES / "inspect_label_counts.csv")
    X = df[feat]
    t = pd.DataFrame({"missing": X.isna().sum(), "infinite": np.isinf(X).sum(), "nunique": X.nunique()})
    t["constant"] = t["nunique"] <= 1
    t.to_csv(RES / "inspect_columns.csv")
    idnames = {"src ip", "dst ip", "src_ip", "dst_ip", "source ip", "destination ip", "source port", "destination port",
               "src port", "dst port", "timestamp", "flow id", "time", "date"}
    ident = [c for c in df.columns if c.strip().lower() in idnames]
    summ = {
        "rows": len(df), "columns": df.shape[1], "label_column": "Label", "n_labels": int(df["Label"].nunique()),
        "missing_total": int(t.missing.sum()), "infinite_total": int(t.infinite.sum()),
        "constant_columns": t.index[t.constant].tolist(),
        "duplicate_rows": int(df.duplicated().sum()),
        "identifier_columns_present": ident,
    }
    jdump(summ, RES / "inspect_summary.json")
    print(summ)


def clean():
    out = PROC / "clean.parquet"
    if out.exists():
        return
    df = pd.read_parquet(PROC / "sample.parquet")
    log = []

    def step(msg):
        log.append(f"{len(log)+1}. {msg}")
        print(log[-1], flush=True)

    step(f"Loaded sampled data (float32): {len(df)} rows x {df.shape[1]} cols. Identifier columns (IP/port/timestamp): none present (results/inspect_summary.json).")
    feat = [c for c in df.columns if c != "Label"]
    bad = df["Label"] == "NAN"
    df = df[~bad].reset_index(drop=True)
    step(f"Dropped rows with a missing label (read as 'NAN'): {int(bad.sum())}.")
    df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)
    step(f"Replaced +/-inf with NaN. Rows with any NaN: {int(df[feat].isna().any(axis=1).sum())}.")
    df = df.dropna().reset_index(drop=True)
    step(f"Missing-value policy: drop rows with any NaN (synopsis: drop rows with missing values). Remaining {len(df)}.")
    df["y_multi"] = df["Label"].map(LABEL2CAT)
    assert df["y_multi"].notna().all(), f"unmapped labels: {df.loc[df.y_multi.isna(), 'Label'].unique()}"
    n0 = len(df)
    df = df.drop_duplicates(subset=feat + ["Label"]).reset_index(drop=True)
    step(f"Removed exact duplicate rows (features+label) BEFORE split: {n0-len(df)} removed, {len(df)} remain.")
    n0 = len(df)
    nl = df.groupby(feat, sort=False)["y_multi"].transform("nunique")
    df = df[nl == 1].reset_index(drop=True)
    step(f"Dropped rows whose identical feature vector has conflicting task labels (8-class category; binary is implied): {n0-len(df)} removed, {len(df)} remain.")
    n0 = len(df)
    df = df.drop_duplicates(subset=feat).reset_index(drop=True)
    step(f"Same feature vector with different fine labels but the same category: kept first, so no identical feature vectors remain (prevents cross-split leakage): {n0-len(df)} removed, {len(df)} remain.")
    const = [c for c in feat if df[c].nunique() <= 1]
    df = df.drop(columns=const)
    step(f"Dropped constant columns: {const}.")
    df["y_fine"] = df["Label"]
    df["y_binary"] = (df["y_multi"] != "Benign").astype("int8")
    step("Built y_binary (0=Benign, 1=Attack), y_multi (8 classes, results/label_mapping.csv) and kept y_fine (original label).")
    pd.DataFrame({"label": list(LABEL2CAT), "category": list(LABEL2CAT.values())}).to_csv(RES / "label_mapping.csv", index=False)
    df = df.drop(columns=["Label"])
    df.to_parquet(out)
    df["y_multi"].value_counts().rename("count").to_csv(RES / "clean_class_counts.csv")
    step(f"Saved data_processed/clean.parquet ({df.shape[1]} cols); class counts in results/clean_class_counts.csv.")
    (RES / "cleaning_log.md").write_text("# Cleaning log (auto-generated)\n" + "\n".join(log) + "\n")


def split():
    out = PROC / "split.npz"
    if out.exists():
        return
    y = pd.read_parquet(PROC / "clean.parquet", columns=["y_multi"])["y_multi"].to_numpy()
    cnt = pd.Series(y).value_counts()
    small = cnt[cnt < 10]
    msg = f"classes with <10 rows (cannot stratify safely): {small.to_dict()}"
    print(msg)
    assert small.empty, msg
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=SEED)
    np.savez(out, train=np.sort(tr), val=np.sort(va), test=np.sort(te))
    (RES / "split_note.txt").write_text(msg + f"\nsizes train/val/test = {len(tr)}/{len(va)}/{len(te)}\n")


_CACHE = {}


def load(split_name, task="binary"):
    """X (float32 DataFrame), y (ndarray) for a split of the one shared split file."""
    if "df" not in _CACHE:
        _CACHE["df"] = pd.read_parquet(PROC / "clean.parquet")
        _CACHE["ix"] = np.load(PROC / "split.npz")
    df, ix = _CACHE["df"], _CACHE["ix"]
    feat = [c for c in df.columns if not c.startswith("y_")]
    s = ix[split_name]
    d = df.iloc[s]
    y = d["y_binary"].to_numpy() if task == "binary" else d["y_multi"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    return d[feat].reset_index(drop=True), y.astype("int64")


def feature_names():
    return [c for c in pd.read_parquet(PROC / "clean.parquet").columns if not c.startswith("y_")]


if __name__ == "__main__":
    {"sample": pass_b, "inspect": inspect, "clean": clean, "split": split}[sys.argv[1]]()
