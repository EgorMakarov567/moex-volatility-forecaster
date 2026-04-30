"""
main.py — запускает весь пайплайн MOEX Volatility Forecaster
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  MOEX VOLATILITY FORECASTER — Top-20 IMOEX")
print("=" * 60)

# ── 1. Данные ───────────────────────────────────────────────
print("\n📡 STEP 1: Загрузка данных...")
from data_loader import load_all_tickers
raw = load_all_tickers()
print(f"   {len(raw)} строк, {raw['ticker'].nunique()} тикеров")
print(f"   Период: {raw.index.min().date()} → {raw.index.max().date()}")

# ── 2. Реализованная волатильность ─────────────────────────
print("\n📉 STEP 2: Расчёт реализованной волатильности...")
from volatility import compute_realized_vol, build_vol_features
rv_df = compute_realized_vol(raw)

# ── 3. GARCH(1,1) ──────────────────────────────────────────
print("\n📐 STEP 3: Обучение GARCH(1,1)...")
from volatility import fit_garch_forecasts
garch_df = fit_garch_forecasts(rv_df)

# ── 4. Feature Engineering ─────────────────────────────────
print("\n🔧 STEP 4: Построение признаков...")
feat_df = build_vol_features(rv_df)

# ── 5. Визуализация режимов ────────────────────────────────
print("\n🎨 STEP 5: Графики...")
from visualize import (plot_volatility_heatmap, plot_garch_vs_realized,
                       plot_volatility_clustering, plot_regime_timeline)

print("  Тепловая карта волатильности...")
plot_volatility_heatmap(feat_df)

print("  Временная ось режимов...")
plot_regime_timeline(feat_df)

print("  Volatility clustering...")
plot_volatility_clustering(feat_df)

print("  GARCH vs Realized...")
plot_garch_vs_realized(feat_df, garch_df)

# ── 6. Walk-Forward ML ─────────────────────────────────────
print("\n🔄 STEP 6: Walk-forward валидация XGBoost...")
from model import walk_forward_vol
predictions, metrics = walk_forward_vol(feat_df)

from visualize import plot_fold_metrics, plot_ml_vs_garch
plot_fold_metrics(metrics)
plot_ml_vs_garch(predictions, garch_df)

# ── 7. SHAP ────────────────────────────────────────────────
print("\n🧠 STEP 7: SHAP анализ...")
from model import train_final_model
from volatility import VOL_FEATURE_COLS
from visualize import plot_shap
final_model = train_final_model(feat_df)
plot_shap(final_model, feat_df, VOL_FEATURE_COLS)

# ── Итог ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅ ГОТОВО! Все графики сохранены в ./plots/")
print("=" * 60)

import os
print("\n📁 Сгенерированные файлы:")
for f in sorted(os.listdir("plots")):
    size = os.path.getsize(f"plots/{f}") // 1024
    print(f"   plots/{f} ({size} KB)")

print(f"\n📊 Итоговые метрики:")
print(f"   Mean RMSE: {metrics['rmse'].mean():.5f}")
print(f"   Mean Corr: {metrics['corr'].mean():.4f}")
