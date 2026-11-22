"""Learners for the walk-forward ladder. Each `make_*()` returns a callable
learner(X_fit, y_fit, X_val, y_val) -> model with .predict(X), .refit(X, y) and .chosen (selected hyperparameters).
Tuning uses only the validation slice inside the training window; refit uses the whole training window.
"""
import itertools

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

SEED = 42
N_JOBS = 6


def _mse(a, b):
    return float(np.mean((a - b) ** 2))


class ENet:
    GRID = {"alpha": [1e-4, 1e-3, 1e-2], "l1_ratio": [0.1, 0.5, 0.9]}

    def __init__(self, Xf, yf, Xv, yv):
        best = None
        for a, l1 in itertools.product(self.GRID["alpha"], self.GRID["l1_ratio"]):
            sc = StandardScaler().fit(Xf)
            m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=5000, random_state=SEED).fit(sc.transform(Xf), yf)
            e = _mse(m.predict(sc.transform(Xv)), yv)
            if best is None or e < best[0]:
                best = (e, a, l1)
        self.chosen = {"alpha": best[1], "l1_ratio": best[2]}

    def refit(self, X, y):
        self.scaler = StandardScaler().fit(X)                     # fitted on training rows only
        self.m = ElasticNet(**self.chosen, max_iter=5000, random_state=SEED).fit(self.scaler.transform(X), y)
        return self

    def predict(self, X):
        return self.m.predict(self.scaler.transform(X))


class LGBM:
    GRID = [{"num_leaves": 15}, {"num_leaves": 63}]
    BASE = dict(learning_rate=0.03, n_estimators=2000, min_child_samples=1000, subsample=0.7, subsample_freq=1,
                colsample_bytree=0.7, reg_lambda=10.0, max_bin=63, random_state=SEED, n_jobs=N_JOBS, verbose=-1)

    def __init__(self, Xf, yf, Xv, yv):
        import lightgbm as lgb
        best = None
        for g in self.GRID:
            m = lgb.LGBMRegressor(**self.BASE, **g).fit(Xf, yf, eval_set=[(Xv, yv)],
                                                        callbacks=[lgb.early_stopping(100, verbose=False)])
            e, it = m.best_score_["valid_0"]["l2"], max(m.best_iteration_, 20)
            if best is None or e < best[0]:
                best = (e, g, it)
        self.chosen = {**best[1], "n_estimators": best[2]}

    def refit(self, X, y):
        import lightgbm as lgb
        self.m = lgb.LGBMRegressor(**{**self.BASE, **self.chosen}).fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)


class XGB:
    GRID = [{"max_depth": 3}, {"max_depth": 6}]
    BASE = dict(learning_rate=0.03, n_estimators=2000, min_child_weight=200, subsample=0.7, colsample_bytree=0.7,
                reg_lambda=10.0, tree_method="hist", max_bin=63, random_state=SEED, n_jobs=N_JOBS)

    def __init__(self, Xf, yf, Xv, yv):
        import xgboost as xgb
        best = None
        for g in self.GRID:
            m = xgb.XGBRegressor(**self.BASE, **g, early_stopping_rounds=100, eval_metric="rmse")
            m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
            e, it = m.best_score, max(m.best_iteration + 1, 20)
            if best is None or e < best[0]:
                best = (e, g, it)
        self.chosen = {**best[1], "n_estimators": best[2]}

    def refit(self, X, y):
        import xgboost as xgb
        self.m = xgb.XGBRegressor(**{**self.BASE, **self.chosen}).fit(X, y, verbose=False)
        return self

    def predict(self, X):
        return self.m.predict(X)


class MLP:
    """2-hidden-layer net (32, 16), Adam; epochs and L2 picked on the validation slice; 2-seed average."""
    GRID = [1e-4, 1e-3]
    MAX_EPOCHS = 25
    SEEDS = (SEED, SEED + 1)

    def _net(self, alpha, seed):
        return MLPRegressor(hidden_layer_sizes=(32, 16), alpha=alpha, batch_size=1024, learning_rate_init=1e-3,
                            max_iter=1, warm_start=True, shuffle=True, random_state=seed)

    def __init__(self, Xf, yf, Xv, yv):
        import warnings
        best = None
        for a in self.GRID:
            net = self._net(a, SEED)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for ep in range(1, self.MAX_EPOCHS + 1):
                    net.fit(Xf, yf)
                    e = _mse(net.predict(Xv), yv)
                    if best is None or e < best[0]:
                        best = (e, a, ep)
        self.chosen = {"alpha": best[1], "epochs": best[2]}

    def refit(self, X, y):
        import warnings
        self.nets = []
        for s in self.SEEDS:
            net = self._net(self.chosen["alpha"], s)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(self.chosen["epochs"]):
                    net.fit(X, y)
            self.nets.append(net)
        return self

    def predict(self, X):
        return np.mean([n.predict(X) for n in self.nets], axis=0)


LEARNERS = {"enet": ENet, "lgbm": LGBM, "xgb": XGB, "mlp": MLP}
