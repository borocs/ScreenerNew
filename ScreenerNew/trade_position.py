
import asyncio
import logging
import time
from ScreenerNew.config import TRAILING_OKX_CALLBACK_RATE

class TradePositionManager:
    def __init__(self, exchange, max_open_positions):
        self.logger = logging.getLogger("position_manager")
        self.exchange = exchange
        self.max_open_positions = max_open_positions
        self.cached_active_positions = {}
        self.open_positions = {}
        self.position_lock = asyncio.Lock()

    async def refresh_positions(self):
        try:
            if hasattr(self.exchange, "valid_pairs") and self.exchange.valid_pairs:
                positions = await self.exchange.sync_positions(self.exchange.valid_pairs)
                self.logger.debug(f"[POSITION] Synced raw positions: {positions}")
                if positions is not None:
                    self.cached_active_positions = positions
                    self.open_positions = {
                        symbol: pos
                        for symbol, pos in positions.items()
                        if pos.get("contracts") and pos.get("entryPrice")
                    }
                    self.logger.info(f"[POSITION] Positions refreshed. {len(self.open_positions)} open positions.")
                else:
                    self.cached_active_positions = {}
                    self.open_positions = {}
                    self.logger.warning("[POSITION] No positions fetched, cache cleared")
        except Exception as e:
            self.logger.error(f"[POSITION] Error refreshing positions: {e}")

    async def check_trailing_activation(self, position_metadata, current_prices):
        for symbol, meta in position_metadata.items():
            self.logger.debug(
                f"[{symbol}] Checking trailing: prepared={meta.get('trailing_prepared')}, "
                f"started={meta.get('trailing_started')}, trigger_price={meta.get('trigger_price')}, "
                f"current_price={current_prices.get(symbol)}"
            )

            try:
                # Якщо трейлінг не підготовлений або вже активований — пропустити
                if not meta.get("trailing_prepared") or meta.get("trailing_started"):
                    continue

                current_price = current_prices.get(symbol)
                if current_price is None:
                    self.logger.warning(f"[{symbol}] No current price for trailing check")
                    continue

                trigger_price = meta.get("trigger_price")
                if trigger_price is None:
                    self.logger.warning(f"[{symbol}] No trigger price for trailing check")
                    continue

                side = meta["side"]
                amount = meta["amount"]

                # Перевірка на досягнення trigger
                if (side == "buy" and current_price > trigger_price) or (side == "sell" and current_price < trigger_price):
                    self.logger.info(f"[{symbol}] Trigger price {trigger_price:.4f} reached at {current_price:.4f}. Activating trailing.")

                    # Скасувати початковий SL
                    sl_algo_id = meta.get("trailing_sl_algo_id")
                    if sl_algo_id:
                        self.logger.info(f"[{symbol}] Cancelling initial SL algoId={sl_algo_id}")
                        await self.exchange.cancel_algo_order(sl_algo_id)

                    # Виставити трейлінг-ордер
                    trailing_algo_id = await self.exchange.place_trailing_order(
                        symbol=symbol,
                        side=side,
                        size=amount,
                        trigger_price=current_price,
                        callback_rate=TRAILING_OKX_CALLBACK_RATE
                    )

                    if trailing_algo_id:
                        self.logger.info(f"[{symbol}] Trailing order placed successfully: algoId={trailing_algo_id}")
                        meta["trailing_started"] = True
                        meta["algo_id"] = trailing_algo_id
                        meta["trail_stop_price"] = current_price
                        meta["last_trailing_update"] = int(time.time())
                    else:
                        self.logger.warning(f"[{symbol}] Failed to place trailing stop for {symbol}")

            except Exception as e:
                self.logger.error(f"[{symbol}] Error during trailing activation: {str(e)}")

    async def has_open_position(self, symbol):
        try:
            pos = self.open_positions.get(symbol)
            self.logger.debug(f"[POSITION] Checking open position for {symbol}: {pos}")
            if not pos:
                return False
            contracts = pos.get("contracts", 0)
            entry_price = pos.get("entryPrice", 0)
            return contracts > 0 and entry_price > 0
        except Exception as e:
            self.logger.error(f"[POSITION] Error checking open position for {symbol}: {e}")
            return False

    async def get_total_position_count(self):
        try:
            return len(self.open_positions)
        except Exception as e:
            self.logger.error(f"[POSITION] Error getting total position count: {e}")
            return 0
