"""GA, PSO, ACOR and random search on [0,1]^dim. Identical pop size / budget / seed / initial population.
Budget = number of UNIQUE candidate evaluations (cache hits are free; total calls are logged)."""
import time, numpy as np


class BudgetDone(Exception):
    pass


class Objective:
    def __init__(self, fn, keyf, budget):
        self.fn, self.keyf, self.budget = fn, keyf, budget
        self.cache, self.hist, self.calls = {}, [], 0
        self.best, self.best_x, self.t0, self.cpu0 = -np.inf, None, time.time(), time.process_time()

    def __call__(self, x):
        self.calls += 1
        k = self.keyf(x)
        if k in self.cache:
            if self.calls > 100 * self.budget: raise BudgetDone  # converged onto cached points; stop
            return self.cache[k]
        if len(self.hist) >= self.budget:
            raise BudgetDone
        f, info = self.fn(x)
        self.cache[k] = f
        if f > self.best: self.best, self.best_x = f, np.array(x, float)
        self.hist.append({"eval": len(self.hist) + 1, "fitness": f, "best_so_far": self.best, "calls": self.calls,
                          "wall_s": time.time() - self.t0, "cpu_s": time.process_time() - self.cpu0, **info})
        return f


def init_pop(dim, pop, seed):
    return np.random.default_rng(seed).random((pop, dim))


def ga(obj, X0, rng, elite=2, k=3, pc=0.9, alpha=0.5, pm=0.1, sigma=0.15):
    n, d = X0.shape
    X = X0.copy(); F = np.array([obj(x) for x in X])
    while True:
        o = np.argsort(-F); ex, ef = X[o[:elite]], F[o[:elite]]

        def tour():
            i = rng.integers(0, n, k); return X[i[np.argmax(F[i])]]
        ch = []
        while len(ch) < n - elite:
            a, b = tour(), tour()
            if rng.random() < pc:  # blend crossover (BLX-alpha)
                lo, hi = np.minimum(a, b), np.maximum(a, b); w = hi - lo
                c = rng.uniform(lo - alpha * w, hi + alpha * w)
            else:
                c = a.copy()
            m = rng.random(d) < pm
            c[m] += rng.normal(0, sigma, m.sum())  # Gaussian mutation
            ch.append(np.clip(c, 0, 1))
        ch = np.array(ch); fc = np.array([obj(c) for c in ch])
        X, F = np.vstack([ex, ch]), np.concatenate([ef, fc])


def pso(obj, X0, rng, w=0.72, c1=1.49, c2=1.49, vmax=0.25):
    n, d = X0.shape
    X = X0.copy(); V = rng.uniform(-0.1, 0.1, (n, d)); F = np.array([obj(x) for x in X])
    P, PF = X.copy(), F.copy()
    while True:
        g = P[np.argmax(PF)]
        r1, r2 = rng.random((n, d)), rng.random((n, d))
        V = np.clip(w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X), -vmax, vmax)
        X = np.clip(X + V, 0, 1)
        for i in range(n):
            F[i] = obj(X[i])
            if F[i] > PF[i]: P[i], PF[i] = X[i].copy(), F[i]


def acor(obj, X0, rng, q=0.2, xi=0.85):
    """ACO for continuous domains (Socha & Dorigo 2008): solution archive, rank weights, Gaussian kernel sampling."""
    k, d = X0.shape
    A = X0.copy(); AF = np.array([obj(x) for x in A])
    o = np.argsort(-AF); A, AF = A[o], AF[o]
    wts = np.exp(-(np.arange(k) ** 2) / (2 * (q * k) ** 2)); wts /= wts.sum()
    while True:
        sd = xi * np.abs(A[None, :, :] - A[:, None, :]).sum(1) / (k - 1)  # [k, d] spread around each archive solution
        new = np.array([np.clip(rng.normal(A[l], np.maximum(sd[l], 1e-3)), 0, 1) for l in rng.choice(k, size=k, p=wts)])  # k ants
        nf = np.array([obj(x) for x in new])
        A, AF = np.vstack([A, new]), np.concatenate([AF, nf])
        o = np.argsort(-AF)[:k]; A, AF = A[o], AF[o]


def rand(obj, X0, rng):
    for x in X0: obj(x)
    while True:
        obj(rng.random(X0.shape[1]))


METHODS = {"ga": ga, "pso": pso, "aco": acor, "random": rand}


def run(method, obj, dim, pop, seed):
    rng = np.random.default_rng(seed + 1)
    try:
        METHODS[method](obj, init_pop(dim, pop, seed), rng)
    except BudgetDone:
        pass
    return obj
