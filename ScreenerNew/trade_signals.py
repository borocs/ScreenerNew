import logging
import pandas as pd
import asyncio
from datetime import datetime
import time

from ScreenerNew.config import MIN_VOLUME_24H, MAX_SPREAD_PCT, VOLATILITY_THRESHOLD

class TradeSignalProcessor:
    def __init__(self, exchange, indicators, ticker_history, filtered_pairs, probabilities, signal_stats):
        self.logger = logging.getLogger("trade_signals")
        self.exchange = exchange
        self.indicators = indicators
        self.ticker_history = ticker_history
        self.filtered_pairs = filtered_pairs
        self.probabilities = probabilities
        self.signal_stats = signal_stats

    async def process_pair(self, symbol, current_time):
        try:
            await self.exchange.safe_await(
                self.exchange.fetch_order_book(symbol),
                timeout=10,
                context=f"{symbol}.order_book"
            )

            ohlcv = self.indicators.ohlcv_cache.get(symbol)
            if not ohlcv:
                ohlcv = await self.exchange.safe_await(
                    self.exchange.fetch_historical_data(symbol, timeframe="1m", limit=250),
                    timeout=10,
                    context=f"{symbol}.ohlcv"
                )
                self.indicators.ohlcv_cache[symbol] = ohlcv
            if not ohlcv:
                self.filtered_pairs[symbol] = (current_time, "no historical data")
                self.logger.warning(f"[{symbol}] skipped: {self.filtered_pairs[symbol]}")
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.name = symbol

            ticker = await self.exchange.safe_await(
                self.exchange.fetch_ticker(symbol),
                timeout=10,
                context=f"{symbol}.fetch_ticker"
            )
            if ticker is None:
                self.filtered_pairs[symbol] = (current_time, "ticker fetch failed")
                self.logger.warning(f"[{symbol}] skipped: {self.filtered_pairs[symbol]}")
                return None

            current_price = ticker.get('last')
            if current_price is None:
                self.filtered_pairs[symbol] = (current_time, "current price is None")
                self.logger.warning(f"[{symbol}] skipped: {self.filtered_pairs[symbol]}")
                return None

            try:
                current_price = float(current_price)
                if current_price < 0.0001:
                    self.filtered_pairs[symbol] = (current_time, f"price too low ({current_price:.8f})")
                    self.logger.warning(f"[{symbol}] skipped: {self.filtered_pairs[symbol]}")
                    return None
            except Exception as e:
                self.filtered_pairs[symbol] = (current_time, f"invalid current_price")
                self.logger.warning(f"[{symbol}] skipped: {self.filtered_pairs[symbol]}")
                return None

            volume_24h = ticker.get('vol_24h')
            if volume_24h is None:
                self.filtered_pairs[symbol] = (current_time, "volume_24h is None")
                self.logger.warning(f"[{symbol}] skipped: {self.filtered_pairs[symbol]}")
                return None

            try:
                volume_24h = float(volume_24h)
                if volume_24h < MIN_VOLUME_24H:
                    self.filtered_pairs[symbol] = (current_time, f"low volume ({volume_24h:.2f})")
                    self.logger.warning(f"[{symbol}] skipped: {self.filtered_pairs[symbol]}")
                    return None
            except Exception as e:
                self.filtered_pairs[symbol] = (current_time, "invalid volume_24h")
                self.logger.warning(f"[{symbol}] skipped: {self.filtered_pairs[symbol]}")
                return None

            spread = self.exchange.spread(symbol)
            if spread > MAX_SPREAD_PCT:
                self.filtered_pairs[symbol] = (current_time, f"high spread ({spread:.2f}%)")
                self.logger.warning(f"[{symbol}] skipped: {self.filtered_pairs[symbol]}")
                return None

            prev_data = self.ticker_history.get(symbol, {'last_price': None, 'timestamp': 0})
            price_change = 0.0
            if prev_data['last_price'] is not None and current_price > 0:
                price_change = ((current_price - prev_data['last_price']) / prev_data['last_price']) * 100
            self.ticker_history[symbol] = {'last_price': current_price, 'timestamp': current_time}

            result = await asyncio.get_event_loop().run_in_executor(None, lambda: self.indicators.generate_signals(df, {}))
            if result is None or not isinstance(result[2], dict):
                self.filtered_pairs[symbol] = (current_time, "no valid signal data")
                self.logger.warning(f"[{symbol}] skipped: {self.filtered_pairs[symbol]}")
                return None

            _, _, signal_data = result
            signal_data['current_price'] = current_price
            signal_data['price_change'] = price_change
            signal_data['order_book_pressure'] = self.exchange.order_book_pressure(symbol)
            probability = signal_data.get("probability", 0)
            if probability is None:
                self.filtered_pairs[symbol] = (current_time, "probability is None")
                self.logger.warning(f"[{symbol}] skipped: {self.filtered_pairs[symbol]}")
                return None

            self.probabilities[symbol] = probability

            atr = signal_data.get('atr')
            if atr is None or current_price == 0:
                self.filtered_pairs[symbol] = (current_time, "ATR or current_price is None")
                self.logger.warning(f"[{symbol}] skipped: {self.filtered_pairs[symbol]}")
                return None

            try:
                atr = float(atr)
                atr_ratio = atr / current_price
            except Exception as e:
                self.filtered_pairs[symbol] = (current_time, "invalid atr or current_price")
                self.logger.warning(f"[{symbol}] skipped: {self.filtered_pairs[symbol]}")
                return None

            if atr_ratio < VOLATILITY_THRESHOLD:
                self.filtered_pairs[symbol] = (current_time, f"low ATR ratio ({atr_ratio:.5f})")
                self.logger.warning(f"[{symbol}] skipped: {self.filtered_pairs[symbol]}")
                return None

            self.signal_stats['valid'] += 1
            signal_data['atr'] = atr
            return (symbol, signal_data)

        except Exception as e:
            self.filtered_pairs[symbol] = (current_time, f"error: {str(e)}")
            self.logger.warning(f"[{symbol}] skipped: {self.filtered_pairs[symbol]}")
            return None
