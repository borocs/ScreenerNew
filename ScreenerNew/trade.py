import sys
import time
import os
import asyncio
import logging
from datetime import datetime
import pandas as pd

from ScreenerNew.trade_core import TradeCore
from ScreenerNew.trade_orders import TradeOrderManager
from ScreenerNew.trade_position import TradePositionManager
from ScreenerNew.trade_signals import TradeSignalProcessor
from ScreenerNew.exchange import AsyncExchange
from ScreenerNew.indicators import Indicators
from ScreenerNew.display import TradeDisplay
from ScreenerNew.config import (
    PAIRS, MAX_OPEN_POSITIONS, CLOSE_ON_SIGNAL_CHANGE,
    MIN_POSITION_HOLD_SEC, get_logger, MIN_MARGIN_PER_TRADE, LEVERAGE
)

if sys.platform == 'win32':
    try:
        import winloop
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except ImportError:
        print("Please install 'winloop' on Windows: pip install winloop")
        sys.exit(1)

logger = get_logger("trade")
logger.setLevel(logging.DEBUG)

class Trade:
    def __init__(self):
        logger.info("Initializing Trade...")

        self.exchange = AsyncExchange()
        self.indicators = Indicators(exchange=self.exchange)
        self.display = TradeDisplay()

        self.initial_balance = None
        self.virtual_balance = None
        self.max_open_positions = MAX_OPEN_POSITIONS
        self.position_metadata = {}
        self.trades = []
        self.rejected_signals = []
        self.signal_stats = {'total': 0, 'valid': 0, 'executed': 0, 'rejected': 0}
        self.ticker_history = {pair: {'last_price': None, 'timestamp': 0} for pair in PAIRS}
        self.filtered_pairs = {}
        self.probabilities = {pair: 0.0 for pair in PAIRS}

        self.order_manager = TradeOrderManager(
            exchange=self.exchange,
            position_metadata=self.position_metadata
        )

        self.core = TradeCore(
            exchange=self.exchange,
            virtual_balance=self.virtual_balance,
            initial_balance=self.initial_balance,
            trades=self.trades,
            log_file="C:/ScreenerNew/logs/trade_log.json",
            order_manager=self.order_manager
        )

        self.position_manager = TradePositionManager(
            exchange=self.exchange,
            max_open_positions=self.max_open_positions
        )

        self.signal_processor = TradeSignalProcessor(
            exchange=self.exchange,
            indicators=self.indicators,
            ticker_history=self.ticker_history,
            filtered_pairs=self.filtered_pairs,
            probabilities=self.probabilities,
            signal_stats=self.signal_stats
        )

        self.is_running = False
        self.update_interval = 0.2
        self.position_sync_cache_seconds = 1

    async def trade(self):
        logger.info("Starting trade method...")

        if self.is_running:
            logger.warning("Trading already running.")
            return

        self.is_running = True

        try:
            await self.exchange.initialize()

            self.initial_balance = await self.exchange.get_available_margin()
            if self.initial_balance is None:
                logger.warning("Initial balance unavailable, setting to 0")
                self.initial_balance = 0.0

            self.virtual_balance = self.initial_balance
            self.core.virtual_balance = self.virtual_balance
            self.core.initial_balance = self.initial_balance

            asyncio.create_task(self.indicators.start_ohlcv_updates())

            logger.info(f"Initial balance: {self.initial_balance:.2f} USDT")

            while True:
                await self.execute_cycle()
                await asyncio.sleep(self.update_interval)

        except Exception as e:
            logger.error(f"Trade loop crashed: {e}")
        finally:
            await self.exchange.close()

    async def execute_cycle(self):
        logger.info("[TRADE] New trading cycle started...")

        await self.position_manager.refresh_positions()
        await self.display.update_table(self.position_manager.open_positions)

        current_time = int(time.time())
        tasks = [self.signal_processor.process_pair(symbol, current_time) for symbol in self.exchange.valid_pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        tradable_pairs = [r for r in results if r is not None and not isinstance(r, Exception)]

        logger.info(f"[TRADE] Tradable pairs this cycle: {tradable_pairs}")

        if not tradable_pairs:
            await self.display.display_data(
                self.exchange,
                self.probabilities,
                self.trades,
                self.virtual_balance,
                self.initial_balance,
                self.signal_stats,
                self.position_metadata
            )
            return

        margin, _ = await self.exchange.get_total_margin()
        await self.display.update_balance(margin)
        logger.info(f"[TRADE] Available margin: {margin:.2f}")

        # 🧠 ВСТАВКА — моніторинг trigger price для трейлінгу
        current_prices = await self.exchange.get_symbol_prices()
        await self.position_manager.check_trailing_activation(
            position_metadata=self.position_metadata,
            current_prices=current_prices
        )

        for symbol, trade_signal in tradable_pairs:
            try:
                signal_type = trade_signal.get('signal')
                if signal_type not in ['buy', 'sell']:
                    logger.warning(f"[TRADE] No valid signal for {symbol}, skipping")
                    continue

                side = signal_type
                current_price = trade_signal['current_price']
                atr = trade_signal['atr']

                async with self.position_manager.position_lock:
                    total_positions = await self.position_manager.get_total_position_count()
                    if total_positions >= self.max_open_positions:
                        logger.warning(f"[TRADE] Max open positions reached ({total_positions}/{self.max_open_positions}), skipping {symbol}")
                        continue

                if await self.position_manager.has_open_position(symbol):
                    logger.info(f"[TRADE] Already have open position for {symbol}, skipping")
                    continue

                if not atr:
                    logger.warning(f"[TRADE] Invalid ATR for {symbol}, skipping")
                    continue

                margin, _ = await self.exchange.get_total_margin()
                if margin < MIN_MARGIN_PER_TRADE:
                    logger.warning(f"[TRADE] Not enough margin for {symbol}, skipping")
                    continue

                # ⬇️ ось тут: тепер ловимо meta з open_trade
                meta = await self.core.open_trade(symbol, side, current_price, atr)
                if meta:
                    self.position_metadata[symbol] = meta  # зберігаємо для моніторингу трейлінгу
                    logger.debug(f"[TRADE] Saved position metadata for {symbol}: {meta}")
                    
                    # Активація трейлінгу
                    if (
                        self.position_metadata[symbol].get("trailing_prepared") and
                        not self.position_metadata[symbol].get("trailing_started") and
                        current_price <= self.position_metadata[symbol].get("trigger_price", float('inf'))
                    ):
                        self.position_metadata[symbol]["trailing_started"] = True
                        logger.info(f"[TRAILING] Trigger met for {symbol}, starting monitor loop.")
                        await self.order_manager.monitor_trailing_stop(
                            symbol,
                            entry_price=self.position_metadata[symbol]["entry_price"],
                            quantity=self.position_metadata[symbol]["quantity"],
                            side=self.position_metadata[symbol]["side"],
                            trigger_price=self.position_metadata[symbol]["trigger_price"],
                            leverage=LEVERAGE
                        )
                        
                await self.display.update_table(self.position_manager.open_positions)

            except Exception as e:
                logger.error(f"[TRADE] Error processing {symbol}: {e}")

        await self.display.display_data(
            self.exchange,
            self.probabilities,
            self.trades,
            self.virtual_balance,
            self.initial_balance,
            self.signal_stats,
            self.position_metadata
        )

if __name__ == "__main__":
    logger.info("Starting trading system...")
    trade_bot = Trade()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(trade_bot.trade())
    except KeyboardInterrupt:
        logger.info("Trading stopped by user")
    except Exception as e:
        logger.error(f"Main loop error: {e}")
    finally:
        loop.run_until_complete(trade_bot.exchange.close())
        loop.close()
        logger.info("Event loop closed")