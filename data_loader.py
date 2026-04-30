"""
data_loader.py — загрузка данных с Московской биржи через apimoex
"""

import pandas as pd
import apimoex
import requests
from datetime import datetime
import time
import os

IMOEX_TOP20 = [
    "SBER", "LKOH", "GAZP", "GMKN", "NVTK",
    "ROSN", "YNDX", "TATN", "MGNT", "MTSS",
    "ALRS", "POLY", "PLZL", "SNGS", "CHMF",
    "NLMK", "MAGN", "PHOR", "IRAO", "FEES",
]

def load_ticker(session, ticker, start="2018-01-01", end=None):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    data = apimoex.get_board_history(
        session, ticker, start=start, end=end,
        columns=("TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"),
    )
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    df = df.rename(columns={"TRADEDATE": "date", "OPEN": "open",
                             "HIGH": "high", "LOW": "low",
                             "CLOSE": "close", "VOLUME": "volume"})
    df = df.set_index("date").sort_index()
    df["ticker"] = ticker
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df

def load_all_tickers(tickers=None, start="2018-01-01", cache_path="data/raw_prices.parquet"):
    if tickers is None:
        tickers = IMOEX_TOP20
    os.makedirs("data", exist_ok=True)
    if os.path.exists(cache_path):
        print(f"✅ Загружаем из кэша: {cache_path}")
        return pd.read_parquet(cache_path)
    print(f"📡 Загружаем данные с MOEX для {len(tickers)} тикеров...")
    all_data = []
    with requests.Session() as session:
        for i, ticker in enumerate(tickers):
            print(f"  [{i+1}/{len(tickers)}] {ticker}...", end=" ")
            try:
                df = load_ticker(session, ticker, start=start)
                if not df.empty:
                    all_data.append(df)
                    print(f"✓ {len(df)} строк")
                else:
                    print("пусто")
            except Exception as e:
                print(f"❌ {e}")
            time.sleep(0.3)
    result = pd.concat(all_data)
    result.to_parquet(cache_path)
    print(f"\n✅ Сохранено: {len(result)} строк, {result['ticker'].nunique()} тикеров")
    return result
