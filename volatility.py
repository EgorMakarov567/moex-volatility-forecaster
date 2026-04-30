"""
volatility.py — расчёт волатильности: realized vol, GARCH, признаки для ML
"""

import pandas as pd
import numpy as np
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. REALIZED VOLATILITY
# ─────────────────────────────────────────────

def compute_realized_vol(df: pd.DataFrame) -> pd.DataFrame:
    """Считает реализованную волатильность для каждого тикера."""
    results = []
    for ticker, g in df.groupby("ticker"):
        g = g.copy().sort_index()
        log_ret = np.log(g["close"] / g["close"].shift(1))

        for w in [5, 10, 20, 60]:
            g[f"rv_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

        # Целевая переменная: волатильность следующих 5 дней
        g["target_rv"] = log_ret.rolling(5).std().shift(-5) * np.sqrt(252)
        g["log_ret"] = log_ret
        g["ticker"] = ticker
        results.append(g)

    return pd.concat(results).sort_index()


# ─────────────────────────────────────────────
# 2. GARCH(1,1) ПРОГНОЗ
# ─────────────────────────────────────────────

def fit_garch_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Для каждого тикера обучает GARCH(1,1) на всей истории
    и возвращает in-sample условную волатильность.
    """
    all_garch = []
    tickers = df["ticker"].unique()

    print(f"📐 Обучаем GARCH(1,1) для {len(tickers)} тикеров...")
    for ticker in tickers:
        g = df[df["ticker"] == ticker].copy().sort_index()
        ret = g["log_ret"].dropna() * 100  # в процентах для GARCH

        try:
            model = arch_model(ret, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            res = model.fit(disp="off", show_warning=False)
            cond_vol = res.conditional_volatility / 100 * np.sqrt(252)
            garch_df = pd.DataFrame({
                "date": cond_vol.index,
                "ticker": ticker,
                "garch_vol": cond_vol.values,
            }).set_index("date")
            all_garch.append(garch_df)
            print(f"  ✓ {ticker}")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")

    garch_all = pd.concat(all_garch)
    return garch_all


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING ДЛЯ ML
# ─────────────────────────────────────────────

VOL_FEATURE_COLS = [
    "rv_5d", "rv_10d", "rv_20d", "rv_60d",
    "vol_ratio_5_20", "vol_ratio_10_20",
    "ret_abs_1d", "ret_abs_5d",
    "hl_range", "hl_range_ma10",
    "rv_change_5d", "rv_momentum",
    "month", "day_of_week", "is_month_end",
]

def build_vol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Строит признаки для предсказания волатильности."""
    results = []
    for ticker, g in df.groupby("ticker"):
        g = g.copy().sort_index()

        # Режимы волатильности
        g["vol_ratio_5_20"]  = g["rv_5d"]  / (g["rv_20d"]  + 1e-9)
        g["vol_ratio_10_20"] = g["rv_10d"] / (g["rv_20d"]  + 1e-9)

        # Абсолютные доходности (прокси текущей волатильности)
        g["ret_abs_1d"] = g["log_ret"].abs()
        g["ret_abs_5d"] = g["log_ret"].abs().rolling(5).mean()

        # Внутридневной диапазон
        g["hl_range"]     = (g["high"] - g["low"]) / (g["close"] + 1e-9)
        g["hl_range_ma10"] = g["hl_range"].rolling(10).mean()

        # Изменение волатильности
        g["rv_change_5d"]  = g["rv_5d"].pct_change(5)
        g["rv_momentum"]   = g["rv_5d"] / (g["rv_20d"] + 1e-9) - 1

        # Календарь
        g["month"]        = g.index.month
        g["day_of_week"]  = g.index.dayofweek
        g["is_month_end"] = (g.index.day >= 25).astype(int)

        results.append(g)

    result = pd.concat(results).sort_index()
    result = result.dropna(subset=VOL_FEATURE_COLS + ["target_rv"])
    print(f"✅ Vol features: {len(result)} строк, {len(VOL_FEATURE_COLS)} признаков")
    return result


if __name__ == "__main__":
    from data_loader import load_all_tickers
    raw = load_all_tickers()
    rv = compute_realized_vol(raw)
    feat = build_vol_features(rv)
    print(feat[["ticker", "rv_5d", "rv_20d", "target_rv"]].tail(10))
