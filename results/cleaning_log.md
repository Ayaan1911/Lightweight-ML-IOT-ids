# Cleaning log (auto-generated)
1. Loaded sampled data (float32): 1055885 rows x 40 cols. Identifier columns (IP/port/timestamp): none present (results/inspect_summary.json).
2. Dropped rows with a missing label (read as 'NAN'): 9.
3. Replaced +/-inf with NaN. Rows with any NaN: 25.
4. Missing-value policy: drop rows with any NaN (synopsis: drop rows with missing values). Remaining 1055851.
5. Removed exact duplicate rows (features+label) BEFORE split: 238729 removed, 817122 remain.
6. Dropped rows whose identical feature vector has conflicting task labels (8-class category; binary is implied): 52116 removed, 765006 remain.
7. Same feature vector with different fine labels but the same category: kept first, so no identical feature vectors remain (prevents cross-split leakage): 6092 removed, 758914 remain.
8. Dropped constant columns: [].
9. Built y_binary (0=Benign, 1=Attack), y_multi (8 classes, results/label_mapping.csv) and kept y_fine (original label).
10. Saved data_processed/clean.parquet (42 cols); class counts in results/clean_class_counts.csv.
