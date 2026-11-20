# Algorithm specification (GA, PSO, ACOR, random search)

Sources: `optimization/core.py`, `optimization/optimizers.py`, `optimization/run.py`, `optimization/orchestrate.py`, `optimization/finalize.py`, `src/common.py`, `results/optimization/budgets.json`, `results/optimization/search_summary.csv`, `DECISIONS.md` (#18-#21, #23), `REPORT.md` section 9.
All values below were read from code or saved results; nothing was re-run.

## 0. Common setup (all four methods)

| Item | Value | Source |
|---|---|---|
| Search space | continuous box [0,1]^dim, dim = 39 + number of hyperparameter genes (dt 44, rf 44, xgb 45, mlp 46) | `core.dim`, `core.HP` |
| Population / swarm / archive / ants | 20 for all methods (`POP = 20`) | `run.py` |
| Search seed | `SEED = 1`; initial population `np.random.default_rng(1).random((20, dim))`; optimizer RNG `np.random.default_rng(seed + 1)` = seed 2 | `run.py`, `optimizers.init_pop`, `optimizers.run` |
| Number of search seeds | one per method and (task, model) pair | `DECISIONS.md` #18 |
| Objective | maximise fitness (section 7) | `core.fitness` |
| Termination | unique-evaluation budget exhausted (`BudgetDone`), or call guard (section 6) | `optimizers.Objective` |
| Initial population | identical for all four methods (same `X0`) | `optimizers.run` |

## 1. Genetic algorithm (GA)

| Item | Value (`optimizers.ga`) |
|---|---|
| Population size | 20 |
| Elites | 2 (best two copied unchanged into next generation; 18 children per generation) |
| Selection | tournament, size k = 3; indices drawn with replacement (`rng.integers(0, n, 3)`), the fittest of the three wins |
| Crossover type | blend crossover BLX-alpha, applied per gene |
| Crossover alpha | 0.5 (child gene ~ Uniform(min - 0.5 w, max + 0.5 w), w = \|a - b\|) |
| Crossover probability | 0.9 per pair of parents; otherwise the child is a copy of the first parent |
| Mutation type | additive Gaussian noise |
| Mutation probability | 0.1 **per gene** (independent Bernoulli draw for each of the dim genes) |
| Mutation sigma | 0.15 (absolute, in [0,1] units) |
| Bounds | after crossover and mutation the child is clipped: `np.clip(c, 0, 1)` |
| Replacement | generational with elitism: next population = 2 elites + 18 children |
| Parent identity | the two tournaments are independent, so both parents can be the same individual |

## 2. Particle swarm optimisation (PSO)

| Item | Value (`optimizers.pso`) |
|---|---|
| Swarm size | 20 |
| Inertia weight w | 0.72 (constant, no decay) |
| c1 (cognitive), c2 (social) | 1.49, 1.49 |
| Topology | global best (every particle is attracted to `g`, the best personal-best of the swarm; `g` is refreshed once per iteration, before the velocity update; no ring or local neighbourhood) |
| Initial positions | the shared initial population: Uniform[0,1] per dimension |
| Initial velocities | Uniform(-0.1, 0.1) per dimension |
| Velocity update | V = w V + c1 r1 (P - X) + c2 r2 (g - X), r1, r2 ~ U[0,1] drawn per particle and dimension |
| Velocity clamp | yes, each component clipped to [-0.25, 0.25] (vmax = 0.25) |
| Boundary handling | position clipped to [0,1] (`np.clip(X + V, 0, 1)`); velocity is not reset or reflected at the boundary |
| Update mode | synchronous per iteration for velocity and position; personal bests are updated particle by particle during evaluation |

## 3. Ant colony optimisation for continuous domains (ACOR)

Implementation follows Socha and Dorigo (2008), as stated in the docstring of `optimizers.acor`; it is an adaptation, not discrete ACO.

| Item | Value |
|---|---|
| Archive size k | 20 (initialised with the shared initial population and sorted by fitness, best first) |
| New ants per iteration | 20 (= k) |
| q | 0.2 |
| xi | 0.85 |
| Rank weights (rank l = 0 for the best, 0-indexed) | w_l = exp(-l^2 / (2 (q k)^2)) / sum_j exp(-j^2 / (2 (q k)^2)); with q k = 4 the denominator term is 2 x 16 = 32. The constant 1/(q k sqrt(2 pi)) of the original formula is omitted because it cancels in normalisation. |
| Guide choice | for each ant, an archive solution l is sampled with probability w_l (`rng.choice(k, size=k, p=wts)`) |
| Kernel standard deviation | for archive member l and dimension j: sigma_lj = xi x ( sum over e = 1..k of \|A_ej - A_lj\| ) / (k - 1); the sum runs over all archive members (the term e = l is zero) |
| Minimum sigma | 1e-3 (`np.maximum(sd, 1e-3)`) |
| Sampling | new_j ~ Normal(A_lj, sigma_lj), independently per dimension, around the chosen guide |
| Boundary handling | clip to [0,1] |
| Archive update | archive and the 20 new solutions are merged and the best 20 by fitness are kept (duplicates are not removed) |

## 4. Random search

| Item | Value (`optimizers.rand`) |
|---|---|
| Candidates | first the 20 shared initial candidates `X0`; afterwards each new candidate is `rng.random(dim)`, i.e. Uniform[0,1] in every dimension, drawn from the RNG with seed 2 |
| Shared first 20 | **Confirmed**: GA initial population, PSO initial positions, ACOR initial archive and random search's first 20 candidates are all the same array `init_pop(dim, 20, seed=1)`. In every run `n_evals >= 20` start from the same 20 vectors, so the first 20 unique evaluations coincide in all four methods (apart from duplicate decoded candidates, which would be counted once). |
| Selection or learning | none |

## 5. Shared encoding (vector in [0,1] to model)

Function `core.decode(model, x)`; decoding is deterministic and defines candidate identity.

* `x` is first clipped to [0,1].
* **Features**: the first 39 genes. Feature i is kept if `x_i > 0.5`. Minimum number of features = 5 (`MIN_FEAT`): if fewer than 5 genes exceed 0.5, the 5 genes with the highest values are forced on.
* **Remaining genes** map onto the hyperparameters in table order, per kind:
  * `int`: `round(lo + v (hi - lo))`
  * `float`: `round(lo + v (hi - lo), 4)`
  * `log`: `exp(log lo + v (log hi - log lo))`, formatted to 4 significant digits
  * `cat`: `choices[min(int(v * n_choices), n_choices - 1)]` (equal-width bins)
  * MLP: `u1, u2, u3` and `n_layers` are combined into `hidden = [u1, u2, u3][:n_layers]` (unused widths are ignored by the decoder but still evolve)

| Model | Hyperparameter | Kind | Range / choices |
|---|---|---|---|
| DT | max_depth | int | 3 to 40 |
| DT | min_samples_split | int | 2 to 50 |
| DT | min_samples_leaf | int | 1 to 30 |
| DT | criterion | cat | gini, entropy, log_loss |
| DT | class_weight | cat | None, balanced |
| RF | n_estimators | int | 10 to 150 |
| RF | max_depth | int | 3 to 30 |
| RF | min_samples_leaf | int | 1 to 20 |
| RF | max_features | cat | sqrt, log2, 0.5 |
| RF | class_weight | cat | None, balanced, balanced_subsample |
| XGB | n_estimators | int | 20 to 200 |
| XGB | max_depth | int | 2 to 10 |
| XGB | learning_rate | log | 0.02 to 0.4 |
| XGB | subsample | float | 0.5 to 1.0 |
| XGB | colsample_bytree | float | 0.5 to 1.0 |
| XGB | min_child_weight | int | 1 to 10 |
| MLP | n_layers | int | 1 to 3 |
| MLP | u1, u2, u3 (units per hidden layer) | int | 16 to 128 each |
| MLP | lr (Adam) | log | 1e-4 to 1e-2 |
| MLP | dropout | float | 0.0 to 0.4 (a Dropout layer after each hidden layer only if > 0) |
| MLP | batch size | cat | 256, 512, 1024, 2048 |

Fixed (not searched) settings: XGB `tree_method="hist"`; DT/RF/XGB `random_state = 42`; RF `n_jobs = 4`; XGB `n_jobs = 4`. MLP: ReLU hidden layers; output sigmoid (binary, 1 unit, loss binary cross-entropy) or softmax (8 units, sparse categorical cross-entropy); optimizer Adam; **10 epochs maximum; EarlyStopping(patience = 2, restore_best_weights = True)**, monitor left at the Keras default (`val_loss`). Inside the search fitness, MLP early stopping uses `validation_split = 0.1` of the 100k training subsample (no validation arrays are passed); inputs are standardised with the training scaler (`models/scaler.npz`) restricted to the selected columns. MLP prediction threshold 0.5 (binary) or argmax. Keras seed = 42.
Hyperparameter genes: DT 5, RF 5, XGB 6, MLP 7 (n_layers, u1, u2, u3, lr, dropout, batch).

## 6. Budget, caching and repeated candidates

| Task / model | Unique-evaluation budget (all four methods) |
|---|---|
| binary DT, RF, XGB | 800 each |
| binary MLP | 600 |
| 8-class DT, RF | 800 each |
| 8-class XGB | 650 |
| 8-class MLP | 580 |

(Source `budgets.json`; total 5830 unique evaluations per method if every run reaches its budget.)

* **Candidate identity**: `core.key` = (boolean feature mask bytes, sorted tuple of decoded hyperparameters as strings). Two vectors that decode to the same mask and hyperparameters are the same candidate.
* **Cache**: `Objective.cache` maps key to fitness. A cached candidate returns the stored fitness without retraining. It increments `calls` but not `n_evals`. Only an uncached candidate trains a model and is appended to the history, counted in `n_evals`.
* **Stop rules**: (a) a new (uncached) candidate arrives when `len(hist) >= budget` raises `BudgetDone`; (b) a cached candidate arrives when `calls > 100 x budget` raises `BudgetDone` ("converged onto cached points").
* **Repeated candidates**: the method is not penalised in fitness; it just wastes calls. Binary RF with PSO triggered rule (b): `n_calls = 80001`, `n_evals = 689` of 800 (swarm collapsed onto evaluated points). The run was kept as is (`DECISIONS.md` #21). Other runs with many repeats that still reached budget: GA (e.g. binary RF 1260 calls), ACOR (8-class RF 3155 calls), PSO 8-class DT 36007 calls (reached 800).
* Total PSO unique evaluations were 5719 instead of 5830 because of this single run.

## 7. Fitness function, subsamples and test isolation

* **Formula**: `fitness = macro_F1(validation subsample) - 0.001 x (n_selected_features / 39)` (`LAMBDA = 0.001`, penalty term = LAMBDA x `mask.mean()`), for a model trained on the training subsample using only the selected columns. Macro F1 is `sklearn.f1_score(average="macro", zero_division=0)` over 2 classes (binary) or 8 classes.
* **Training subsample**: 100,000 rows (`N_TR_SUB`) drawn from the training split; **validation subsample**: 50,000 rows (`N_VA_SUB`) drawn from the validation split. Fixed once per task (cached), identical for every candidate and every method.
* **Sampling rule**: stratified by the task label. For each class c, k = min(class size, max(round(n x class size / split size), floor)), floor = 200 for training and 100 for validation; k rows drawn without replacement with `np.random.default_rng(42)` (re-created for each of train and val); the concatenation is then permuted with the same RNG.
* **Seed**: 42 (`src.common.SEED`) for subsampling and model training inside the fitness. (The optimizer seed is 1 / 2, section 0.)
* **Test isolation**: `core.data` loads only `split.npz["train"]` and `split.npz["val"]`, and no code in `core.py`, `optimizers.py`, `run.py` or `orchestrate.py` reads the test split; the test split is only opened in `finalize.py` after the search, with each use written to `test_access.log`.
* Winners are afterwards retrained on the full training split (seeds 42, 43, 44) with early stopping on the full validation split for MLP (`finalize.py`).

## 8. Quote-ready paragraphs

**Genetic algorithm.** The genetic algorithm used a population of 20 real-coded individuals in [0,1]^d. In each generation the two best individuals were kept unchanged (elitism), and the remaining 18 children were produced from parents chosen by tournament selection with tournament size 3. With probability 0.9 two parents were recombined by blend crossover (BLX-alpha with alpha = 0.5); otherwise the child copied the first parent. Each gene was then mutated independently with probability 0.1 by adding Gaussian noise with standard deviation 0.15, and every gene was clipped back into [0,1].

**Particle swarm optimisation.** The particle swarm contained 20 particles with a global-best topology. Positions were initialised uniformly in [0,1]^d and velocities uniformly in [-0.1, 0.1]. The velocity update used an inertia weight of 0.72 and acceleration coefficients c1 = c2 = 1.49 with independent uniform random factors per particle and dimension, and each velocity component was clamped to [-0.25, 0.25]. Positions were clipped to [0,1] after each move.

**Ant colony optimisation (ACOR).** We used ACO for continuous domains (ACOR) with a solution archive of 20 candidates and 20 new ants per iteration. Archive members were ranked by fitness and given Gaussian rank weights w_l proportional to exp(-l^2 / (2 (qk)^2)) with q = 0.2 and k = 20. Each ant selected an archive member with probability w_l and sampled each dimension from a Gaussian centred on that member, with standard deviation xi times the mean absolute distance to the other archive members (xi = 0.85, lower bound 0.001); samples were clipped to [0,1]. The archive then kept the best 20 of the old and new solutions.

**Random search.** Random search evaluated the same 20 initial candidates as the other three methods and then drew every further candidate uniformly at random from [0,1]^d. All methods shared the same decoding of a candidate into a feature subset (genes above 0.5, at least five features) and model hyperparameters, the same unique-evaluation budget per model and task (580 to 800), and the same fitness: macro F1 on a 50,000-row validation subsample minus 0.001 times the fraction of selected features.
