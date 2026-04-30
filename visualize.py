"""
visualize.py — все графики для MOEX Volatility Forecaster
Упор на красивые и информативные визуализации
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
import shap
import os

os.makedirs("plots", exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f9fa",
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linestyle":   "--",
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

PALETTE  = ["#1565C0", "#E53935", "#2E7D32", "#6A1B9A", "#EF6C00",
            "#00838F", "#AD1457", "#4E342E", "#37474F", "#558B2F"]
BLUE     = "#1565C0"
RED      = "#E53935"
GREEN    = "#2E7D32"
ORANGE   = "#EF6C00"


# ─────────────────────────────────────────────
# 1. VOLATILITY REGIME MAP (тепловая карта)
# ─────────────────────────────────────────────

def plot_volatility_heatmap(df: pd.DataFrame, save=True):
    """
    Тепловая карта: по оси X — время, по оси Y — тикер.
    Цвет — реализованная волатильность (20-дневная).
    Показывает режимы волатильности по всему рынку.
    """
    pivot = df.pivot_table(index="ticker", columns=df.index, values="rv_20d")
    # Ресемплинг по неделям для читаемости
    pivot = pivot.T.resample("W").mean().T

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Реализованная волатильность (annualized)", "shrink": 0.6},
        linewidths=0,
        xticklabels=max(1, pivot.shape[1] // 20),
        yticklabels=True,
    )
    ax.set_title("🌡️ Карта волатильности MOEX — Топ-20 акций (2018–2025)",
                 fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Дата", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()
    if save:
        plt.savefig("plots/volatility_heatmap.png", dpi=150, bbox_inches="tight")
        print("  ✅ plots/volatility_heatmap.png")
    plt.show()


# ─────────────────────────────────────────────
# 2. GARCH vs REALIZED — сравнение по тикерам
# ─────────────────────────────────────────────

def plot_garch_vs_realized(df_feat: pd.DataFrame, garch_df: pd.DataFrame,
                           tickers=None, save=True):
    """
    Для каждого тикера: линия реализованной волатильности vs GARCH прогноз.
    """
    if tickers is None:
        tickers = ["SBER", "LKOH", "GAZP", "GMKN", "YNDX", "CHMF"]

    n = len(tickers)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    axes = axes.flatten()

    for i, ticker in enumerate(tickers):
        ax = axes[i]
        rv   = df_feat[df_feat["ticker"] == ticker]["rv_20d"].dropna()
        garch = garch_df[garch_df["ticker"] == ticker]["garch_vol"].dropna()

        common = rv.index.intersection(garch.index)
        ax.plot(common, rv.loc[common],    color=BLUE,   lw=1.5, label="Realized Vol (20d)", alpha=0.9)
        ax.plot(common, garch.loc[common], color=RED,    lw=1.5, label="GARCH(1,1)",         alpha=0.85, linestyle="--")

        # Заливка разницы
        ax.fill_between(common,
                        rv.loc[common], garch.loc[common],
                        where=(rv.loc[common] > garch.loc[common]),
                        alpha=0.15, color=RED,  label="_nolegend_")
        ax.fill_between(common,
                        rv.loc[common], garch.loc[common],
                        where=(rv.loc[common] <= garch.loc[common]),
                        alpha=0.15, color=BLUE, label="_nolegend_")

        ax.set_title(ticker, fontsize=12, fontweight="bold")
        ax.set_ylabel("Vol (annualized)", fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        if i == 0:
            ax.legend(fontsize=9)

    # Скрываем лишние оси
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("📐 GARCH(1,1) vs Реализованная волатильность — MOEX",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig("plots/garch_vs_realized.png", dpi=150, bbox_inches="tight")
        print("  ✅ plots/garch_vs_realized.png")
    plt.show()


# ─────────────────────────────────────────────
# 3. ML vs GARCH — scatter + residuals
# ─────────────────────────────────────────────

def plot_ml_vs_garch(predictions: pd.DataFrame, garch_df: pd.DataFrame, save=True):
    """
    Сравнение точности ML модели и GARCH:
    - Scatter: predicted vs actual
    - Residual distribution
    - RMSE по тикерам
    """
    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Panel 1: ML scatter ──
    ax1 = fig.add_subplot(gs[0])
    lim_max = predictions[["target_rv", "predicted_rv"]].quantile(0.99).max()
    ax1.scatter(predictions["target_rv"], predictions["predicted_rv"],
                alpha=0.15, s=8, color=BLUE, rasterized=True)
    ax1.plot([0, lim_max], [0, lim_max], color="black", lw=1, linestyle="--", label="Идеал")
    corr = np.corrcoef(predictions["target_rv"], predictions["predicted_rv"])[0, 1]
    ax1.set_title(f"XGBoost: predicted vs actual\nCorr = {corr:.3f}", fontweight="bold")
    ax1.set_xlabel("Actual Vol")
    ax1.set_ylabel("Predicted Vol")
    ax1.set_xlim(0, lim_max)
    ax1.set_ylim(0, lim_max)
    ax1.legend()

    # ── Panel 2: Residuals distribution ──
    ax2 = fig.add_subplot(gs[1])
    residuals = predictions["predicted_rv"] - predictions["target_rv"]
    ax2.hist(residuals, bins=60, color=BLUE, alpha=0.75, edgecolor="white")
    ax2.axvline(0,                color="black", lw=1.5, linestyle="--")
    ax2.axvline(residuals.mean(), color=RED,     lw=1.5, linestyle="-", label=f"Mean={residuals.mean():.4f}")
    ax2.set_title("Распределение ошибок (XGBoost)", fontweight="bold")
    ax2.set_xlabel("Residual (predicted − actual)")
    ax2.set_ylabel("Частота")
    ax2.legend()

    # ── Panel 3: RMSE по тикерам ──
    ax3 = fig.add_subplot(gs[2])
    rmse_per_ticker = (
        predictions.groupby("ticker")
        .apply(lambda x: np.sqrt(((x["predicted_rv"] - x["target_rv"]) ** 2).mean()))
        .sort_values(ascending=True)
    )
    colors = [GREEN if v < rmse_per_ticker.median() else ORANGE for v in rmse_per_ticker.values]
    ax3.barh(rmse_per_ticker.index, rmse_per_ticker.values, color=colors)
    ax3.axvline(rmse_per_ticker.median(), color="gray", lw=1.5, linestyle="--", label="Медиана")
    ax3.set_title("RMSE по тикерам (XGBoost)", fontweight="bold")
    ax3.set_xlabel("RMSE")
    ax3.legend()

    fig.suptitle("🤖 Качество прогноза волатильности — XGBoost",
                 fontsize=14, fontweight="bold", y=1.02)
    if save:
        plt.savefig("plots/ml_forecast_quality.png", dpi=150, bbox_inches="tight")
        print("  ✅ plots/ml_forecast_quality.png")
    plt.show()


# ─────────────────────────────────────────────
# 4. VOLATILITY CLUSTERING
# ─────────────────────────────────────────────

def plot_volatility_clustering(df: pd.DataFrame, save=True):
    """
    Демонстрация volatility clustering:
    автокорреляция квадратов доходностей по тикерам.
    """
    tickers_to_show = df["ticker"].unique()[:8]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    axes = axes.flatten()

    from pandas.plotting import autocorrelation_plot

    for i, ticker in enumerate(tickers_to_show):
        ax = axes[i]
        g   = df[df["ticker"] == ticker].copy()
        ret_sq = g["log_ret"].dropna() ** 2

        # Ручная автокорреляция
        lags = range(1, 31)
        acf_vals = [ret_sq.autocorr(lag=l) for l in lags]

        ax.bar(list(lags), acf_vals,
               color=[BLUE if v > 0 else RED for v in acf_vals],
               alpha=0.8, edgecolor="white")
        ax.axhline(0,    color="black", lw=0.8)
        ax.axhline(0.05, color="gray",  lw=0.8, linestyle="--", alpha=0.6)
        ax.set_title(ticker, fontweight="bold")
        ax.set_xlabel("Лаг (дни)")
        ax.set_ylabel("ACF(r²)")
        ax.set_ylim(-0.15, 0.4)

    fig.suptitle("📊 Volatility Clustering — автокорреляция квадратов доходностей\n"
                 "(положительные значения = волатильность кластеризуется)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig("plots/volatility_clustering.png", dpi=150, bbox_inches="tight")
        print("  ✅ plots/volatility_clustering.png")
    plt.show()


# ─────────────────────────────────────────────
# 5. SHAP IMPORTANCE
# ─────────────────────────────────────────────

def plot_shap(model, df, feature_cols, n_samples=2000, save=True):
    from volatility import VOL_FEATURE_COLS
    X = df[feature_cols].sample(min(n_samples, len(df)), random_state=42)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    plt.sca(axes[0])
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=15)
    axes[0].set_title("SHAP — средняя важность признаков", fontweight="bold")

    plt.sca(axes[1])
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    axes[1].set_title("SHAP — направление и сила эффекта", fontweight="bold")

    fig.suptitle("🔍 Что влияет на прогноз волатильности?",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig("plots/shap_importance.png", dpi=150, bbox_inches="tight")
        print("  ✅ plots/shap_importance.png")
    plt.show()


# ─────────────────────────────────────────────
# 6. VOLATILITY REGIME TIMELINE
# ─────────────────────────────────────────────

def plot_regime_timeline(df: pd.DataFrame, save=True):
    """
    Временной ряд средней волатильности по рынку с разметкой режимов:
    низкая / средняя / высокая волатильность.
    """
    market_rv = df.groupby(level=0)["rv_20d"].mean().dropna()
    market_rv = market_rv.resample("W").mean()

    low_thr  = market_rv.quantile(0.33)
    high_thr = market_rv.quantile(0.67)

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.fill_between(market_rv.index, 0, market_rv,
                    where=(market_rv <= low_thr),
                    color=GREEN, alpha=0.4, label="Низкая волатильность")
    ax.fill_between(market_rv.index, 0, market_rv,
                    where=((market_rv > low_thr) & (market_rv <= high_thr)),
                    color=ORANGE, alpha=0.4, label="Средняя волатильность")
    ax.fill_between(market_rv.index, 0, market_rv,
                    where=(market_rv > high_thr),
                    color=RED, alpha=0.4, label="Высокая волатильность")

    ax.plot(market_rv.index, market_rv, color="#333333", lw=1.2, alpha=0.8)
    ax.axhline(low_thr,  color=GREEN,  lw=1, linestyle=":", alpha=0.7)
    ax.axhline(high_thr, color=RED,    lw=1, linestyle=":", alpha=0.7)

    # Аннотации ключевых событий
    events = {
        "2020-03": "COVID\nобвал",
        "2022-02": "Февраль\n2022",
    }
    for date_str, label in events.items():
        dt = pd.Timestamp(date_str)
        if dt in market_rv.index or True:
            ax.axvline(dt, color="black", lw=1.2, linestyle="--", alpha=0.6)
            ax.text(dt, market_rv.max() * 0.92, label,
                    fontsize=9, ha="center", color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax.set_title("📈 Режимы волатильности российского рынка (2018–2025)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Средняя реализованная волатильность (annualized)")
    ax.set_xlabel("Дата")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(0)

    plt.tight_layout()
    if save:
        plt.savefig("plots/regime_timeline.png", dpi=150, bbox_inches="tight")
        print("  ✅ plots/regime_timeline.png")
    plt.show()


# ─────────────────────────────────────────────
# 7. FOLD METRICS
# ─────────────────────────────────────────────

def plot_fold_metrics(metrics: pd.DataFrame, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(metrics["fold"], metrics["rmse"], marker="o", color=BLUE, lw=2)
    axes[0].axhline(metrics["rmse"].mean(), color=GREEN, linestyle="--",
                    label=f"Mean RMSE = {metrics['rmse'].mean():.5f}")
    axes[0].set_title("RMSE по фолдам (Walk-Forward)", fontweight="bold")
    axes[0].set_xlabel("Фолд")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()

    axes[1].plot(metrics["fold"], metrics["corr"], marker="s", color=ORANGE, lw=2)
    axes[1].axhline(metrics["corr"].mean(), color=BLUE, linestyle="--",
                    label=f"Mean Corr = {metrics['corr'].mean():.3f}")
    axes[1].axhline(0, color="gray", linestyle=":", lw=1)
    axes[1].set_title("Корреляция прогноза с реальностью", fontweight="bold")
    axes[1].set_xlabel("Фолд")
    axes[1].set_ylabel("Pearson r")
    axes[1].legend()

    fig.suptitle("🔄 Walk-Forward Validation Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig("plots/fold_metrics.png", dpi=150, bbox_inches="tight")
        print("  ✅ plots/fold_metrics.png")
    plt.show()
