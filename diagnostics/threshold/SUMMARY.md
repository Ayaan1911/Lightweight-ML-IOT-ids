# Validation-chosen threshold analysis (binary baselines)

Post-hoc, predict-only analysis of the four saved binary baselines; nothing was retrained or tuned. Scores are P(Attack) (`01_scores.py`). Every threshold is chosen on the **validation split only** (`02_analysis.py`, written to `thresholds.csv` before test scoring) and applied unchanged to test. Rule: attack if score >= threshold; thresholds are actual validation score values (unique scores are the candidates, so recall is the value actually reached). The 0.5 rows use score > 0.5 and are a reference, not a choice.

Integrity: test macro F1 / FPR at 0.5 recomputed from the scores equal `results/metrics_*_binary.json` exactly and the stored predictions exactly (DT 0.7854/0.4211, RF 0.8021/0.4699, XGB 0.8194/0.4023, MLP 0.7800/0.5091).

## Operating points (val vs test at the same threshold; test 95% bootstrap CI, 1,000 resamples, seed 42)

| model | rule | threshold | val recall | test recall [CI] | val FPR | test FPR [CI] | val macro F1 | test macro F1 | test acc |
|---|---|---|---|---|---|---|---|---|---|
| dt | recall>=0.98 | 0.8182 | 0.9803 | 0.9803 [0.9794, 0.9811] | 0.3162 | 0.3012 [0.2864, 0.3168] | 0.7817 | 0.7863 | 0.9724 |
| dt | recall>=0.99 | 0.3216 | 0.9901 | 0.9902 [0.9896, 0.9908] | 0.5125 | 0.5085 [0.4928, 0.5260] | 0.7604 | 0.7625 | 0.9762 |
| dt | max_val_macro_f1 | 0.7667 | 0.9828 | 0.9828 [0.9820, 0.9836] | 0.3447 | 0.3328 [0.3176, 0.3494] | 0.7845 | 0.7881 | 0.9740 |
| rf | recall>=0.98 | 0.7700 | 0.9803 | 0.9810 [0.9802, 0.9818] | 0.2182 | 0.1997 [0.1864, 0.2139] | 0.8109 | 0.8191 | 0.9759 |
| rf | recall>=0.99 | 0.6400 | 0.9902 | 0.9901 [0.9895, 0.9907] | 0.3447 | 0.3381 [0.3224, 0.3548] | 0.8236 | 0.8253 | 0.9809 |
| rf | max_val_macro_f1 | 0.6700 | 0.9887 | 0.9886 [0.9879, 0.9892] | 0.3168 | 0.3018 [0.2872, 0.3178] | 0.8241 | 0.8283 | 0.9804 |
| xgb | recall>=0.98 | 0.8055 | 0.9800 | 0.9798 [0.9789, 0.9807] | 0.1897 | 0.1725 [0.1592, 0.1863] | 0.8173 | 0.8212 | 0.9755 |
| xgb | recall>=0.99 | 0.6171 | 0.9900 | 0.9897 [0.9891, 0.9903] | 0.3244 | 0.3153 [0.2994, 0.3323] | 0.8293 | 0.8306 | 0.9811 |
| xgb | max_val_macro_f1 | 0.6433 | 0.9893 | 0.9889 [0.9882, 0.9895] | 0.3071 | 0.2974 [0.2825, 0.3147] | 0.8307 | 0.8315 | 0.9808 |
| mlp | recall>=0.98 | 0.7655 | 0.9800 | 0.9810 [0.9801, 0.9818] | 0.2905 | 0.2736 [0.2590, 0.2895] | 0.7882 | 0.7978 | 0.9738 |
| mlp | recall>=0.99 | 0.6050 | 0.9900 | 0.9901 [0.9895, 0.9907] | 0.4217 | 0.4264 [0.4101, 0.4439] | 0.7949 | 0.7938 | 0.9784 |
| mlp | max_val_macro_f1 | 0.6382 | 0.9887 | 0.9889 [0.9883, 0.9895] | 0.3995 | 0.4014 [0.3847, 0.4188] | 0.7956 | 0.7963 | 0.9780 |
| dt | default 0.5 (reference) | 0.5000 | 0.9882 | 0.9882 [0.9876, 0.9889] | 0.4371 | 0.4211 [0.4052, 0.4377] | 0.7793 | 0.7854 | 0.9767 |
| rf | default 0.5 (reference) | 0.5000 | 0.9942 | 0.9943 [0.9938, 0.9947] | 0.4646 | 0.4699 [0.4534, 0.4887] | 0.8037 | 0.8021 | 0.9813 |
| xgb | default 0.5 (reference) | 0.5000 | 0.9930 | 0.9929 [0.9924, 0.9934] | 0.4101 | 0.4023 [0.3860, 0.4195] | 0.8172 | 0.8194 | 0.9818 |
| mlp | default 0.5 (reference) | 0.5000 | 0.9932 | 0.9934 [0.9929, 0.9938] | 0.5025 | 0.5091 [0.4923, 0.5274] | 0.7816 | 0.7800 | 0.9793 |

## Supplement only: balanced test subsets (threshold 0.5)

For comparability with other papers; NOT a replacement for the main results. 100 subsets, each = all 3,194 benign test rows + 3,194 attack test rows drawn at random, stratified in proportion to the 7 attack categories (seeds 1-100). SD with ddof=1.

| model | accuracy mean | accuracy SD | macro F1 mean | macro F1 SD |
|---|---|---|---|---|
| dt | 0.7835 | 0.0009 | 0.7741 | 0.0009 |
| rf | 0.7623 | 0.0006 | 0.7488 | 0.0005 |
| xgb | 0.7954 | 0.0007 | 0.7870 | 0.0006 |
| mlp | 0.7421 | 0.0007 | 0.7248 | 0.0006 |

## Reading

- Validation-chosen thresholds generalized well: test recall is within 0.0010 of validation recall in all 12 cases, and test FPR is lower than validation FPR in 10 of 12 (largest increase +0.0047, MLP recall>=0.99; all within the bootstrap CI width).
- Reaching recall 0.98 cuts FPR to 0.17-0.30 on test; recall 0.99 leaves FPR at 0.32-0.51. FPR stays high everywhere (0.17 at best), so the high-FPR finding is not a threshold artefact alone.

Files: `01_scores.py`, `02_analysis.py`, `scores_*.npz` (val/test scores), `thresholds.csv`, `operating_points.csv`, `balanced_supplement.csv`, `fpr_vs_recall_test.png`.
