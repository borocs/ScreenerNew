# retrain_model.py - Version: 1.0.1
# Changelog:
# - Version 1.0.1: Updated model/scaler file naming to remove '/' and ':USDT' for consistency

import os
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import ccxt.async_support as ccxt
import asyncio
import logging
import joblib
from datetime import datetime
from ScreenerNew.config import PAIRS, API_KEY, SECRET, PASSWORD, get_logger

logger = get_logger("retrain_model")
logger.setLevel(logging.INFO)

async def fetch_new_data(exchange, symbol, timeframe='1m', limit=1440):  # 1 day of data
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        logger.info(f"Fetched {len(ohlcv)} new bars for {symbol}")
        return ohlcv
    except Exception as e:
        logger.error(f"Error fetching new data for {symbol}: {str(e)}")
        return []

def calculate_indicators(df):
    try:
        df['RSI'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        df['EMA9'] = ta.ema(df['close'], length=9)
        df['EMA21'] = ta.ema(df['close'], length=21)
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['Volume_MA14'] = df['volume'].rolling(window=14).mean()
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return df

async def retrain_model():
    exchange = ccxt.okx({
        'apiKey': API_KEY,
        'secret': SECRET,
        'password': PASSWORD,
        'enableRateLimit': True,
    })
    try:
        await exchange.load_markets()
        for symbol in PAIRS:
            logger.info(f"Retraining model for {symbol}...")
            symbol_clean = symbol.replace('/', '').replace(':USDT', '')
            model_path = f"./models/random_forest_model_{symbol_clean}.pkl"
            scaler_path = f"./models/scaler_{symbol_clean}.pkl"

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                logger.warning(f"Model or scaler not found for {symbol} at {model_path} or {scaler_path}, skipping")
                continue

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # Fetch new data
            ohlcv = await fetch_new_data(exchange, symbol)
            if not ohlcv:
                continue

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.name = symbol

            # Update historical CSV
            csv_path = f"./data/historical/{symbol_clean}.csv"
            if os.path.exists(csv_path):
                historical_df = pd.read_csv(csv_path)
                historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
                combined_df = pd.concat([historical_df, df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                combined_df.to_csv(csv_path, index=False)
                logger.info(f"Updated historical data for {symbol}")
            else:
                df.to_csv(csv_path, index=False)
                logger.info(f"Created new historical data for {symbol}")

            # Calculate indicators
            df = calculate_indicators(df)
            if df.empty or len(df) < 200:
                logger.warning(f"Not enough new data for {symbol}: {len(df)}")
                continue

            # Prepare features and target
            X = df[['RSI', 'MACD', 'MACD_signal']].dropna()
            if len(X) < 2:
                logger.warning(f"Not enough valid new data for {symbol}")
                continue

            y = [1 if df['close'].shift(-1)[i] > df['close'][i] else 0 for i in X.index]
            y = y[:-1]
            X = X.iloc[:-1]

            if len(set(y)) < 2:
                logger.warning(f"Insufficient class variation in new data for {symbol}")
                continue

            X_scaled = scaler.transform(X)
            model.fit(X_scaled, y)

            # === Оцінка якості ===
            from sklearn.metrics import accuracy_score
            import shutil

            y_pred = model.predict(X_scaled)
            acc = accuracy_score(y, y_pred)
            logger.info(f"[RETRAIN] Model accuracy for {symbol}: {acc:.4f}")

            if acc < 0.55:
                logger.warning(f"[RETRAIN] Accuracy below threshold. Skipping update for {symbol}")
                continue  # ❌ не зберігати модель

            # === Backup старих моделей ===
            if os.path.exists(model_path):
                shutil.copy2(model_path, model_path + ".bak")
                logger.info(f"[RETRAIN] Backup saved: {model_path}.bak")
            if os.path.exists(scaler_path):
                shutil.copy2(scaler_path, scaler_path + ".bak")

            # === Збереження нової моделі ===
            try:
                joblib.dump(model, model_path)
                joblib.dump(scaler, scaler_path)
                logger.info(f"Updated model for {symbol} at {model_path} and scaler at {scaler_path}")
            except Exception as e:
                logger.error(f"Error saving updated model/scaler for {symbol}: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error during retraining: {str(e)}")
    finally:
        await exchange.close()

async def schedule_retraining():
    while True:
        logger.info("Starting scheduled model retraining...")
        await retrain_model()
        logger.info("Retraining completed, sleeping for 24 hours...")
        await asyncio.sleep(24 * 3600)  # Retrain daily

if __name__ == "__main__":
    logger.info("Starting model retraining scheduler...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(schedule_retraining())
    except Exception as e:
        logger.error(f"Error in retraining: {str(e)}")
    finally:
        loop.close()
        logger.info("Retraining scheduler stopped")