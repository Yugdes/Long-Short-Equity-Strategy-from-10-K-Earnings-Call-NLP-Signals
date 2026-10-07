"""
Machine Learning models for the return prediction layer.
Includes LightGBM and ElasticNet implementations.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.utils.helpers import setup_logging

logger = setup_logging("models")


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame],
    y_val: Optional[pd.Series],
    X_test: pd.DataFrame,
    num_leaves: int = 31,
    min_data_in_leaf: int = 20,
    learning_rate: float = 0.03,
    feature_fraction: float = 0.7,
    n_estimators: int = 800,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
) -> Tuple[lgb.LGBMRegressor, np.ndarray]:
    """
    Train a LightGBM regressor with early stopping.
    """
    model = lgb.LGBMRegressor(
        num_leaves=num_leaves,
        min_child_samples=min_data_in_leaf,
        learning_rate=learning_rate,
        colsample_bytree=feature_fraction,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
    else:
        # If no validation set, just fit on everything
        model.fit(X_train, y_train)

    test_preds = model.predict(X_test)
    return model, test_preds


def train_elastic_net(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame],
    y_val: Optional[pd.Series],
    X_test: pd.DataFrame,
    l1_ratio: list = None,
    alphas: list = None,
    random_state: int = 42,
) -> Tuple[object, np.ndarray]:
    """
    Train an ElasticNet model with cross-validation for hyperparameter tuning.
    Requires scaling features first.
    """
    if l1_ratio is None:
        l1_ratio = [0.1, 0.5, 0.9]
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0]

    # Combine train and val for sklearn's internal CV
    if X_val is not None and y_val is not None:
        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])
    else:
        X_full = X_train
        y_full = y_train

    # Scale features (crucial for regularized linear models)
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)
    X_test_scaled = scaler.transform(X_test)

    # Note: Using TimeSeriesSplit inside ElasticNetCV would be better, 
    # but for simplicity we use default K-Fold since we already walk-forward at the macro level.
    model = ElasticNetCV(
        l1_ratio=l1_ratio,
        alphas=alphas,
        cv=3,
        random_state=random_state,
        n_jobs=-1,
        max_iter=2000,
    )
    
    model.fit(X_full_scaled, y_full)
    
    # Bundle the scaler with the model for later SHAP/interpretation
    class ScaledModel:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler
            
        def predict(self, X):
            return self.model.predict(self.scaler.transform(X))
            
    bundled_model = ScaledModel(model, scaler)
    test_preds = bundled_model.predict(X_test)
    
    return bundled_model, test_preds
