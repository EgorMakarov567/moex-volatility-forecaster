"""
model.py — XGBoost регрессор для прогноза волатильности + walk-forward
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

from volatility import VOL_FEATURE_COLS


def walk_forward_vol(
    df: pd.DataFrame,
    train_years: int = 2,
    test_months: int = 6,
) -> tuple:
    """Walk-forward валидация для прогноза волатильности."""
    df = df.sort_index()
    dates = df.index.unique().sort_values()

    results, fold_metrics = [], []
    current_date = dates[0] + pd.DateOffset(years=train_years)
    end_date = dates[-1] - pd.DateOffset(days=5)
    fold = 0

    while current_date < end_date:
        fold += 1
        test_end = current_date + pd.DateOffset(months=test_months)

        train = df[df.index < current_date]
        test  = df[(df.index >= current_date) & (df.index < test_end)]

        if len(train) < 300 or len(test) < 30:
            current_date = test_end
            continue

        X_train, y_train = train[VOL_FEATURE_COLS], train["target_rv"]
        X_test,  y_test  = test[VOL_FEATURE_COLS],  test["target_rv"]

        model = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=15, reg_alpha=0.1,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        corr = np.corrcoef(y_test, preds)[0, 1]

        fold_metrics.append({
            "fold": fold,
            "test_start": test.index.min().date(),
            "test_end":   test.index.max().date(),
            "rmse": round(rmse, 5),
            "mae":  round(mae,  5),
            "corr": round(corr, 4),
        })

        fold_res = test[["ticker", "target_rv"]].copy()
        fold_res["predicted_rv"] = preds
        fold_res["fold"] = fold
        results.append(fold_res)

        current_date = test_end
        print(f"  Fold {fold}: {test.index.min().date()} → {test.index.max().date()} | "
              f"RMSE={rmse:.4f} | Corr={corr:.3f}")

    predictions = pd.concat(results) if results else pd.DataFrame()
    metrics     = pd.DataFrame(fold_metrics)

    print(f"\n📊 Walk-forward summary ({fold} фолдов):")
    print(f"   Mean RMSE: {metrics['rmse'].mean():.5f} ± {metrics['rmse'].std():.5f}")
    print(f"   Mean Corr: {metrics['corr'].mean():.4f}")

    return predictions, metrics


def train_final_model(df: pd.DataFrame) -> XGBRegressor:
    """Финальная модель на всех данных — для SHAP и важности признаков."""
    X, y = df[VOL_FEATURE_COLS], df["target_rv"]
    model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=15, reg_alpha=0.1,
        random_state=42, n_jobs=-1,
    )
    model.fit(X, y, verbose=False)
    print(f"✅ Финальная модель обучена на {len(df)} строках")
    return model


if __name__ == "__main__":
    from data_loader import load_all_tickers
    from volatility import compute_realized_vol, build_vol_features

    raw  = load_all_tickers()
    rv   = compute_realized_vol(raw)
    feat = build_vol_features(rv)

    predictions, metrics = walk_forward_vol(feat)
    print(metrics)
