import sys
import os
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import logging
import joblib
import asyncio
from cachetools import TTLCache
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from ScreenerNew.config import BUY_THRESHOLD, SELL_THRESHOLD, EMA_PROXIMITY, VOLATILITY_THRESHOLD, MIN_PROB_THRESHOLD, get_logger, PAIRS

logger = get_logger("indicators")
logger.setLevel(logging.DEBUG)  # <-- важливо для відладки

class Indicators:
    def __init__(self, exchange=None):
        logger.info("Initializing Indicators...")
        self.support_threshold = 0.05
        self.min_data_points = 50
        self.exchange = exchange
        self.indicator_cache = {}  # Кеш для індикаторів
        self.cache_timeout = 5.0
        self.ohlcv_cache = TTLCache(maxsize=1000, ttl=60)
        self.models = {}
        self.scalers = {}
        self.load_models()

    def load_models(self):
        for symbol in PAIRS:
            symbol_clean = symbol.replace('/', '').replace(':USDT', '')
            model_path = f"./models/random_forest_model_{symbol_clean}.pkl"
            scaler_path = f"./models/scaler_{symbol_clean}.pkl"
            try:
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[symbol] = joblib.load(model_path)
                    self.scalers[symbol] = joblib.load(scaler_path)
                    logger.info(f"Loaded model and scaler for {symbol}")
                else:
                    logger.warning(f"Model or scaler not found for {symbol}, skipping ML inference")
                    self.models[symbol] = None
                    self.scalers[symbol] = None
            except Exception as e:
                logger.error(f"Error loading model for {symbol}: {str(e)}")
                self.models[symbol] = None
                self.scalers[symbol] = None

    async def update_ohlcv(self, symbol, timeframe='1m', limit=250):
        while True:
            try:
                ohlcv = await self.exchange.fetch_historical_data(symbol, timeframe, limit=limit)
                self.ohlcv_cache[symbol] = ohlcv
                logger.debug(f"Updated OHLCV cache for {symbol}")
            except Exception as e:
                logger.error(f"Error updating OHLCV for {symbol}: {str(e)}")
            await asyncio.sleep(60)

    async def start_ohlcv_updates(self):
        tasks = [self.update_ohlcv(symbol) for symbol in PAIRS]
        await asyncio.gather(*tasks, return_exceptions=True)

    def find_support_level(self, df):
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
        daily_lows = df.groupby('date')['low'].min()
        return daily_lows[-50:].min() if len(daily_lows) >= 50 else daily_lows.min()

    def generate_signals(self, df, positions):
        symbol = df.name
        if df.empty:
            logger.warning(f"[{symbol}] DataFrame is empty")
            return None, None, None

        if len(df) < self.min_data_points:
            logger.warning(f"[{symbol}] Not enough data: {len(df)} < {self.min_data_points}")
            return None, None, None

        # Check cache
        current_time = time.time()
        cache_key = f"{symbol}_{df['timestamp'].iloc[-1]}"
        if cache_key in self.indicator_cache and current_time - self.indicator_cache[cache_key]['timestamp'] < self.cache_timeout:
            logger.debug(f"[{symbol}] Using cached indicators")
            return self.indicator_cache[cache_key]['result']

        try:
            df = df.copy()  # Avoid modifying original DataFrame
            df['RSI'] = ta.rsi(df['close'], length=14)
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_signal'] = macd['MACDs_12_26_9']
            df['EMA9'] = ta.ema(df['close'], length=9)
            df['EMA21'] = ta.ema(df['close'], length=21)
            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['Volume_MA14'] = df['volume'].rolling(window=14).mean()
        except Exception as e:
            logger.error(f"[{symbol}] Error calculating indicators: {str(e)}")
            return None, None, None

        support_level = self.find_support_level(df)
        current_price = df['close'].iloc[-1]

        X = df[['RSI', 'MACD', 'MACD_signal']].dropna().tail(1)
        if X.empty:
            logger.warning(f"[{symbol}] Not enough valid data for prediction")
            return None, None, None

        model = self.models.get(symbol)
        scaler = self.scalers.get(symbol)
        if model is None or scaler is None:
            logger.warning(f"[{symbol}] No pre-trained model available, using default signal")
            signal_data = {
                'symbol': symbol,
                'rsi': df['RSI'].iloc[-1] if not df['RSI'].empty else None,
                'macd': df['MACD'].iloc[-1] if not df['MACD'].empty else None,
                'macd_signal': df['MACD_signal'].iloc[-1] if not df['MACD_signal'].empty else None,
                'ema9':   df['EMA9'].iloc[-1]   if not df['EMA9'].empty   else None,
                'ema21':  df['EMA21'].iloc[-1]  if not df['EMA21'].empty  else None,
                'atr': df['ATR'].iloc[-1] if not df['ATR'].empty else None,
                'volume': df['volume'].iloc[-1] if not df['volume'].empty else None,
                'volume_ma14': df['Volume_MA14'].iloc[-1] if not df['Volume_MA14'].empty else None,
                'probability': 0.5,
                'type': 'signal',
                'support_level': support_level,
                'current_price': current_price,
                'buy_prob': 0.5,
                'sell_prob': 0.5,
                'signal': 'hold',
                'reasons': ['No pre-trained model available']
            }
            self.indicator_cache[cache_key] = {'timestamp': current_time, 'result': (True, True, signal_data)}
            return True, True, signal_data

        X_scaled = scaler.transform(X)
        pred = model.predict_proba(X_scaled)[0]
        buy_prob, sell_prob = pred[1], pred[0]

        current_rsi = X['RSI'].iloc[0]
        current_macd = X['MACD'].iloc[0]
        current_macd_signal = X['MACD_signal'].iloc[0]
        current_ema9   = df['EMA9'].iloc[-1]
        current_ema21  = df['EMA21'].iloc[-1]
        current_atr = df['ATR'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_ma14 = df['Volume_MA14'].iloc[-1]

        missing_indicators = []
        if pd.isna(current_rsi):
            missing_indicators.append("RSI")
        if pd.isna(current_macd):
            missing_indicators.append("MACD")
        if pd.isna(current_macd_signal):
            missing_indicators.append("MACD_signal")
        if pd.isna(current_ema9):
            missing_indicators.append("EMA9")
        if pd.isna(current_ema21):
            missing_indicators.append("EMA21")
        if pd.isna(current_atr):
            missing_indicators.append("ATR")
        if pd.isna(current_volume) or pd.isna(volume_ma14):
            missing_indicators.append("Volume")

        signal_data = {
            'symbol': symbol,
            'rsi': current_rsi,
            'macd': current_macd,
            'macd_signal': current_macd_signal,
            'ema9':   current_ema9,
            'ema21':  current_ema21,
            'atr': current_atr,
            'volume': current_volume,
            'volume_ma14': volume_ma14,
            'probability': max(buy_prob, sell_prob),
            'type': 'signal' if symbol not in positions else 'position',
            'support_level': support_level,
            'current_price': current_price,
            'buy_prob': buy_prob,
            'sell_prob': sell_prob
        }

        signal = "hold"
        reasons = []

        if missing_indicators:
            logger.warning(f"[{symbol}] Missing indicators: {', '.join(missing_indicators)}")
            signal_data['signal'] = "hold"
            signal_data['reasons'] = [f"Missing indicators: {', '.join(missing_indicators)}"]
            self.indicator_cache[cache_key] = {'timestamp': current_time, 'result': (True, True, signal_data)}
            logger.debug(f"[{symbol}] Signal generated: {signal_data}")
            return True, True, signal_data

        if signal_data['type'] == 'signal':
            trend_up   = current_ema9  >= current_ema21 * 0.98  # трохи ширше

            trend_down = current_ema9  <= current_ema21 * 1.02

            volume_condition = current_volume > volume_ma14 * 0.1  # Relaxed to 0.3

            # Relaxed filters for high probabilities
            high_prob = buy_prob >= 0.7 or sell_prob >= 0.7
            relaxed_rsi_buy = current_rsi < 65 if high_prob else current_rsi < 60
            relaxed_rsi_sell = current_rsi > 35 if high_prob else current_rsi > 40
            relaxed_volume = current_volume > volume_ma14 * 0.2 if high_prob else current_volume > volume_ma14 * 0.3

            buy_conditions = (
                buy_prob >= MIN_PROB_THRESHOLD and
                relaxed_rsi_buy and
                (current_macd > current_macd_signal or abs(current_macd - current_macd_signal) / (abs(current_macd_signal) + 1e-10) < 0.1) and
                trend_up and
                current_atr / current_price > VOLATILITY_THRESHOLD and
                relaxed_volume
            )

            sell_conditions = (
                sell_prob >= MIN_PROB_THRESHOLD and
                relaxed_rsi_sell and
                (current_macd < current_macd_signal or abs(current_macd - current_macd_signal) / (abs(current_macd_signal) + 1e-10) < 0.1) and
                trend_down and
                current_atr / current_price > VOLATILITY_THRESHOLD and
                relaxed_volume
            )

            if buy_conditions:
                signal = "buy"
                reasons.append(f"Buy prob >= {MIN_PROB_THRESHOLD}, RSI < {'65' if high_prob else '60'}, MACD > MACD_signal or close (±10%), EMA9 >= EMA21 (±1%), price near EMA21 (±10%), ATR sufficient, volume > {'0.2' if high_prob else '0.3'} * Volume_MA14")
            elif sell_conditions:
                signal = "sell"
                reasons.append(f"Sell prob >= {MIN_PROB_THRESHOLD}, RSI > {'35' if high_prob else '40'}, MACD < MACD_signal or close (±10%), EMA9 <= EMA21 (±1%), price near EMA21 (±10%), ATR sufficient, volume > {'0.2' if high_prob else '0.3'} * Volume_MA14")
            else:
                signal = "hold"
                if signal != "hold":
                    logger.info(f"[SIGNAL] {symbol} - {signal.upper()} (buy_prob={buy_prob:.2f}, sell_prob={sell_prob:.2f})")
                    logger.info(
                        f"  RSI={df['RSI'].iloc[-1]:.2f}, MACD={df['MACD'].iloc[-1]:.4f}, "
                        f"MACD_hist={df['MACD'].iloc[-1] - df['MACD_signal'].iloc[-1]:.4f}, "
                        f"EMA9={df['EMA9'].iloc[-1]:.2f}, EMA21={df['EMA21'].iloc[-1]:.2f}, "
                        f"ATR={df['ATR'].iloc[-1]:.4f}, support_proximity={df['close'].iloc[-1] / df['EMA21'].iloc[-1]:.3f}, "
                        f"Volume_MA14={df['Volume_MA14'].iloc[-1]:.2f}"
                    )
                if buy_prob < MIN_PROB_THRESHOLD:
                    reasons.append(f"Buy prob {buy_prob:.2f} < {MIN_PROB_THRESHOLD}")
                if not relaxed_rsi_buy:
                    reasons.append(f"RSI {current_rsi:.2f} >= {'65' if high_prob else '60'}")
                if not (current_macd > current_macd_signal or abs(current_macd - current_macd_signal) / (abs(current_macd_signal) + 1e-10) < 0.1):
                    reasons.append("MACD not > MACD_signal or not close (±10%)")
                if not trend_up:
                    reasons.append("EMA9 < EMA21 or not within ±1%")
                if not (current_price >= current_ema21 * (1 - EMA_PROXIMITY) and current_price <= current_ema21 * (1 + EMA_PROXIMITY)):
                    reasons.append("Price not near EMA21 (±10%) for buy")
                if current_atr / current_price <= VOLATILITY_THRESHOLD:
                    reasons.append(f"ATR ratio {current_atr / current_price:.5f} <= {VOLATILITY_THRESHOLD}")
                if not relaxed_volume:
                    factor = 0.2 if high_prob else 0.3
                    reasons.append(f"Volume {current_volume:.2f} <= {factor:.1f} * Volume_MA14 {volume_ma14 * factor:.2f}")

                if sell_prob < MIN_PROB_THRESHOLD:
                    reasons.append(f"Sell prob {sell_prob:.2f} < {MIN_PROB_THRESHOLD}")
                if not relaxed_rsi_sell:
                    reasons.append(f"RSI {current_rsi:.2f} <= {'35' if high_prob else '40'}")
                if not (current_macd < current_macd_signal or abs(current_macd - current_macd_signal) / (abs(current_macd_signal) + 1e-10) < 0.1):
                    reasons.append("MACD not < MACD_signal or not close (±10%)")
                if not trend_down:
                    reasons.append("EMA9 > EMA21 or not within ±1%")
                if not (current_price >= current_ema21 * (1 - EMA_PROXIMITY) and current_price <= current_ema21 * (1 + EMA_PROXIMITY)):
                    reasons.append("Price not near EMA21 (±10%) for sell")
                if current_atr / current_price <= VOLATILITY_THRESHOLD:
                    reasons.append(f"ATR ratio {current_atr / current_price:.5f} <= {VOLATILITY_THRESHOLD}")
                if not relaxed_volume:
                    factor = 0.2 if high_prob else 0.3
                    reasons.append(f"Volume {current_volume:.2f} <= {factor:.1f} * Volume_MA14 {volume_ma14 * factor:.2f}")

        else:
            pos = positions[symbol]
            profit = (current_price - pos['entry_price']) * pos['amount'] if pos['side'] == 'buy' else (pos['entry_price'] - current_price) * pos['amount']
            signal_data['profit_usdt'] = profit
            signal_data['profit_percent'] = (profit / (pos['entry_price'] * pos['amount'])) * 100 if pos['entry_price'] and pos['amount'] else 0.0
            signal_data['max_profit_percent'] = pos.get('max_profit_percent', 0.0)
            signal = "hold"
            if signal_data['profit_percent'] >= 0.5:
                reasons.append("Profit >= 0.5%")
            elif signal_data['profit_percent'] <= -0.5:
                reasons.append("Loss <= -0.5%")
            else:
                reasons.append("No action required")

        signal_data['signal'] = signal
        signal_data['reasons'] = reasons if reasons else ["No significant conditions met"]
        logger.debug(f"[{symbol}] Buy prob: {buy_prob:.2f}, Sell prob: {sell_prob:.2f}, RSI: {current_rsi:.2f}, MACD: {current_macd:.6f}, EMA9: {current_ema9:.4f}, EMA21: {current_ema21:.4f}, Volume: {current_volume:.2f}, Volume_MA14: {volume_ma14:.2f}, Signal: {signal_data['signal']}")
        logger.debug(f"[{symbol}] Signal generated: {signal_data}")

        # Cache the result
        self.indicator_cache[cache_key] = {'timestamp': current_time, 'result': (True, True, signal_data)}
        return True, True, signal_data

    async def start_ohlcv_updates(self):
        """Start background OHLCV updates for all pairs."""
        tasks = [self.update_ohlcv(symbol) for symbol in PAIRS]
        await asyncio.gather(*tasks, return_exceptions=True)