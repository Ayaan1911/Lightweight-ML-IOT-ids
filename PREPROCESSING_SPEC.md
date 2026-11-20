# Preprocessing specification (CICIoT2023 data preparation)

Sources: `src/data.py`, `src/common.py`, `src/evaluation.py`, `src/training.py`, `results/cleaning_log.md`, `results/inspect_*.csv/json`, `results/label_counts_full.csv`, `results/sampling_counts.csv`, `results/clean_class_counts.csv`, `results/split_note.txt`, `data_processed/clean.parquet` and `split.npz` (read only, to count rows), `REPORT.md`, `DECISIONS.md`, `iot_ids.ipynb`, `diagnostics/step1_rows_per_label_per_step.csv`.
Nothing was re-run; counts from the parquet/npz files were only read.

## 1. Columns

**Received**: 63 files `MERGED_CSV/Merged01..63.csv`, 40 columns each: 39 numeric features plus `Label`. Header (from `Merged01.csv`):

`Header_Length, Protocol Type, Time_To_Live, Rate, fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number, ack_flag_number, ece_flag_number, cwr_flag_number, ack_count, syn_count, fin_count, rst_count, HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, TCP, UDP, DHCP, ARP, ICMP, IGMP, IPv, LLC, Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Variance, Label`

**Model inputs: the 39 feature columns above (all of them).** `Label` is the target only.

**Removed feature columns: none.**
* Identifier columns (IP, port, timestamp, flow id): none exist in the files, so none were dropped (`inspect_summary.json`: `identifier_columns_present: []`).
* Constant columns: none (`constant_columns: []`; step 8 of the cleaning log "Dropped constant columns: []"). Every feature has at least 10 distinct values in the sample (`inspect_columns.csv`).
* `Label` is renamed to `Label`, upper-cased and stripped, then turned into three targets stored in `clean.parquet` (42 columns = 39 features + `y_multi`, `y_binary`, `y_fine`); the three `y_*` columns are not model inputs.

## 2. Missing, infinite, constant values; duplicates; conflicts

Sampled data (1,055,885 rows) is held as float32. Cleaning (`data.clean`) is applied in this order:

1. Rows with a missing label (read as the string `NAN`) are dropped: 9 rows.
2. `+inf` and `-inf` in features are replaced by NaN (25 infinite values, all in `Rate`; `inspect_columns.csv`).
3. Rows with any NaN in any feature are dropped (`dropna`, no imputation): 25 rows after step 1 (193 missing feature cells and 25 infinite cells in the sample, per `inspect_columns.csv`). Remaining 1,055,851.
4. Exact duplicates removed, before the split: `drop_duplicates(subset = 39 features + Label)`, keep first; 238,729 removed, 817,122 remain. (Duplicate means identical features and identical fine label.)
5. Conflicting labels removed: rows whose **identical 39-feature vector** occurs with more than one **8-class category** (`groupby(features)["y_multi"].transform("nunique") > 1`) are dropped, **all copies of the vector** (not just one); 52,116 removed, 765,006 remain. Conflict is judged on the 8-class category, not on the fine label; the binary label follows from it, so a Benign-vs-Attack conflict is a special case.
6. Identical feature vectors that remain (same category, different fine labels): `drop_duplicates(subset = features)`, keep first; 6,092 removed, 758,914 remain. Purpose: no identical feature vector can fall into two splits.
7. Constant columns: none found.
8. Targets: `y_multi` from the mapping in section 3, `y_binary = 0` if Benign else `1`, `y_fine` = original upper-case label.
9. Feature scaling: not applied to the stored data; a `StandardScaler` is fitted on the training split only (`src/training.py: get_scaler`, `models/scaler.npz`) and used only for the MLP.

(The notebook-level count of duplicates, 238,732 rows, is computed before the NaN/label rows are dropped; `inspect_summary.json`.)

## 3. Label mapping (33 attack labels + Benign into 7 + Benign)

Source: `src/common.py` (`_CAT`, `LABEL2CAT`), saved as `results/label_mapping.csv`. `DECISIONS.md` #5: the dataset README (`README.pdf`) has no label-to-category table, so the standard CICIoT2023 33-attack grouping was used, with the 7 attack categories of the project synopsis. The mapping is the author's; it is not read from a file shipped with the data.

| Category | Original labels (as written in the CSV, upper case) |
|---|---|
| DDoS (12) | DDOS-RSTFINFLOOD, DDOS-PSHACK_FLOOD, DDOS-SYN_FLOOD, DDOS-UDP_FLOOD, DDOS-TCP_FLOOD, DDOS-ICMP_FLOOD, DDOS-SYNONYMOUSIP_FLOOD, DDOS-ACK_FRAGMENTATION, DDOS-UDP_FRAGMENTATION, DDOS-ICMP_FRAGMENTATION, DDOS-HTTP_FLOOD, DDOS-SLOWLORIS |
| DoS (4) | DOS-UDP_FLOOD, DOS-SYN_FLOOD, DOS-TCP_FLOOD, DOS-HTTP_FLOOD |
| Recon (5) | RECON-HOSTDISCOVERY, RECON-OSSCAN, RECON-PORTSCAN, RECON-PINGSWEEP, VULNERABILITYSCAN |
| Web-based (6) | SQLINJECTION, COMMANDINJECTION, BACKDOOR_MALWARE, UPLOADING_ATTACK, XSS, BROWSERHIJACKING |
| Brute force (1) | DICTIONARYBRUTEFORCE |
| Spoofing (2) | MITM-ARPSPOOFING, DNS_SPOOFING |
| Mirai (3) | MIRAI-GREETH_FLOOD, MIRAI-UDPPLAIN, MIRAI-GREIP_FLOOD |
| Benign | BENIGN (the files use `BENIGN`; `BENIGNTRAFFIC` is also mapped but does not occur) |

Binary task: Benign = 0, every other category = 1 (Attack). Class order for the 8-class task: Benign, DDoS, DoS, Recon, Web-based, Brute force, Spoofing, Mirai (`CLASSES`). The label values are 35 distinct strings in the data: 33 attacks, BENIGN and the string `NAN` (9 rows).

## 4. Row counts

| Stage | Rows |
|---|---|
| Original (63 files, before sampling) | 45,019,243 (`label_counts_full.csv`; includes 9 rows with a missing label) |
| After sampling | 1,055,885 (`sampling_counts.csv`) |
| After removing missing label / NaN / inf rows | 1,055,851 |
| After exact-duplicate removal | 817,122 |
| After conflict removal | 765,006 |
| After identical-vector removal = **cleaned data** | **758,914** |
| Train (70%) | 531,239 |
| Validation (15%) | 113,837 |
| Test (15%) | 113,838 |

Binary task:

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Benign (0) | 14,905 | 3,194 | 3,194 | 21,293 |
| Attack (1) | 516,334 | 110,643 | 110,644 | 737,621 |

8-class task (identical rows; counts from `split.npz` and `clean.parquet`):

| Class | Train | Val | Test | Total (= `clean_class_counts.csv`) |
|---|---|---|---|---|
| Benign | 14,905 | 3,194 | 3,194 | 21,293 |
| DDoS | 333,810 | 71,531 | 71,531 | 476,872 |
| DoS | 78,809 | 16,887 | 16,888 | 112,584 |
| Recon | 29,270 | 6,273 | 6,272 | 41,815 |
| Web-based | 16,542 | 3,545 | 3,545 | 23,632 |
| Brute force | 8,756 | 1,876 | 1,876 | 12,508 |
| Spoofing | 13,449 | 2,882 | 2,882 | 19,213 |
| Mirai | 35,698 | 7,649 | 7,650 | 50,997 |
| **Sum** | 531,239 | 113,837 | 113,838 | 758,914 |

Per fine label and step counts: `diagnostics/step1_rows_per_label_per_step.csv`.

## 5. Sampling, order of steps and split

**Order**: (1) sample from the 63 raw CSVs; (2) clean the sample (including all deduplication and conflict removal); (3) split the cleaned data into train/val/test. Deduplication therefore happens after sampling and before the split.

**Sampling** (`data.quota`, `data.pass_b`; constants `BUDGET = 1,000,000`, `SMALL = 30,000`, `FLOOR = 10,000`):
* Counts per label are taken over all 63 files (pass A, labels stripped and upper-cased).
* **Rare-label rule**: a label with fewer than 30,000 rows in the whole dataset is kept whole. There are 11 such labels (including `NAN`): DDOS-HTTP_FLOOD, DDOS-SLOWLORIS, DICTIONARYBRUTEFORCE, BROWSERHIJACKING, COMMANDINJECTION, SQLINJECTION, XSS, BACKDOOR_MALWARE, RECON-PINGSWEEP, UPLOADING_ATTACK and NAN; together 88,488 rows.
* **Proportional rule**: every other label gets `round((1,000,000 - 88,488) x label_count / sum of those labels' counts)` rows.
* **Floor**: that quota is raised to at least 10,000 rows, and never exceeds the label's own count. Because of the floor the sample (1,055,885) is larger than 1,000,000.
* **Seed and draw**: `np.random.default_rng(42)`; for each label the positions of the kept rows are drawn without replacement (`rng.choice(count, quota, replace=False)`) among the label's occurrences in file order, then the files are streamed in chunks of 500,000 rows and exactly those occurrences are kept (exact stratified sample).

**Split** (`data.split`): `train_test_split(test_size = 0.30, stratify = 8-class label, random_state = 42)`, then the remaining 30% is split `test_size = 0.50`, stratified, `random_state = 42`. One shared split file (`data_processed/split.npz`) for all models; the code asserts that no class has fewer than 10 rows. The scaler is fitted on train only.

## 6. Metric conventions (`src/evaluation.py: compute`)

* **Precision and recall are macro-averaged** over the classes (2 classes in the binary task, 8 in the 8-class task): `precision_recall_fscore_support(average=None)` then the mean over classes (`zero_division=0`). Macro F1 is the mean of per-class F1. Accuracy = trace / total.
* **FPR, binary task**: Attack is the positive class: FPR = FP / (FP + TN) = (benign rows predicted as attack) / (all benign rows). It is not macro-averaged (`fpr_c[1]`), so recall (macro) and FPR use different conventions (`DECISIONS.md` #11 documents the fix of an earlier bug that used Benign as positive).
* **FPR, 8-class task**: per class one-vs-rest FPR_c = FP_c / (FP_c + TN_c), then the unweighted mean over the 8 classes (macro FPR).
