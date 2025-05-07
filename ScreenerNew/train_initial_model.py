# train_initial_model.py - Version: 1.0.2
# Changelog:
# - Version 1.0.2: Updated model/scaler file naming to remove '/' and ':USDT' for consistency
# - Version 1.0.1: Added enhanced error handling and logging for insufficient data or API errors

import os
import sys
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import ccxt.async_support as ccxt
import asyncio
import logging
import joblib
from datetime import datetime
from config import PAIRS, API_KEY, SECRET, PASSWORD, get_logger

logger = get_logger("train_model")
logger.setLevel(logging.INFO)

async def fetch_historical_data(exchange, symbol, timeframe='1m', limit=172800):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        logger.info(f"Fetched {len(ohlcv)} bars for {symbol}")
        return ohlcv
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
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

async def train_model():
    exchange = ccxt.okx({
        'apiKey': API_KEY,
        'secret': SECRET,
        'password': PASSWORD,
        'enableRateLimit': True,
    })
    try:
        await exchange.load_markets()
        os.makedirs('./data/historical', exist_ok=True)
        os.makedirs('./models', exist_ok=True)

        for symbol in PAIRS:
            logger.info(f"Processing {symbol}...")
            ohlcv = await fetch_historical_data(exchange, symbol)
            if not ohlcv:
                logger.warning(f"No data fetched for {symbol}, skipping")
                continue

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.name = symbol

            # Save to CSV
            symbol_clean = symbol.replace('/', '').replace(':USDT', '')
            csv_path = f"./data/historical/{symbol_clean}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved data for {symbol} to {csv_path}")

            # Calculate indicators
            df = calculate_indicators(df)
            if df.empty or len(df) < 200:
                logger.warning(f"Not enough data for {symbol}: {len(df)} rows, need at least 200")
                continue

            # Prepare features and target
            X = df[['RSI', 'MACD', 'MACD_signal']].dropna()
            if len(X) < 2:
                logger.warning(f"Not enough valid data for {symbol} after dropping NaNs: {len(X)} rows")
                continue

            y = [1 if df['close'].shift(-1)[i] > df['close'][i] else 0 for i in X.index]
            y = y[:-1]
            X = X.iloc[:-1]

            if len(set(y)) < 2:
                logger.warning(f"Insufficient class variation for {symbol}: unique labels {set(y)}")
                continue

            # Split data
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            except Exception as e:
                logger.error(f"Error splitting data for {symbol}: {str(e)}")
                continue

            scaler = StandardScaler()
            try:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            except Exception as e:
                logger.error(f"Error scaling data for {symbol}: {str(e)}")
                continue

            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=7,
                min_samples_split=5,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42
            )
            try:
                model.fit(X_train_scaled, y_train)
            except Exception as e:
                logger.error(f"Error training model for {symbol}: {str(e)}")
                continue

            # Evaluate model
            try:
                accuracy = model.score(X_test_scaled, y_test)
                logger.info(f"Model accuracy for {symbol}: {accuracy:.4f}")
            except Exception as e:
                logger.error(f"Error evaluating model for {symbol}: {str(e)}")
                continue

            # Save model and scaler
            model_path = f"./models/random_forest_model_{symbol_clean}.pkl"
            scaler_path = f"./models/scaler_{symbol_clean}.pkl"
            try:
                joblib.dump(model, model_path)
                joblib.dump(scaler, scaler_path)
                logger.info(f"Saved model to {model_path} and scaler to {scaler_path}")
            except Exception as e:
                logger.error(f"Error saving model/scaler for {symbol}: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}")
    finally:
        await exchange.close()

if __name__ == "__main__":
    logger.info("Starting initial model training...")
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(train_model())
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
    finally:
        loop.close()
        logger.info("Training completed")