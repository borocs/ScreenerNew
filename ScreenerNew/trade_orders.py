#Trade_orders 0.1

import logging
import asyncio
import time
import math

class TradeOrderManager:
    def __init__(self, exchange, position_metadata):
        self.logger = logging.getLogger("trade_orders")
        self.exchange = exchange
        self.position_metadata = position_metadata

    async def set_exit_orders(self, symbol, side, amount, entry_price, stop_price, take_price, trigger_price):
        self.logger.info(
            f"[TRADE_ORDERS] Setting exit orders: {symbol}, side={side}, amount={amount:.4f}, "
            f"entry={entry_price:.4f}, stop={stop_price:.4f}, take={take_price:.4f}"
        )

        if stop_price is None or take_price is None or amount <= 0:
            self.logger.error(f"[EXIT_ORDERS] Invalid SL/TP params for {symbol}, skipping")
            return

        try:
            close_side = 'sell' if side == 'buy' else 'buy'

            algo_id = await self.exchange.place_stop_loss_take_profit(
                symbol=symbol,
                side=close_side,
                amount=amount,
                stop_price=stop_price,
                take_price=take_price
            )

            if algo_id:
                self.logger.info(f"[{symbol}] OCO order set successfully, algoId={algo_id}")
                return

            self.logger.warning(f"[{symbol}] OCO creation failed, applying fallback strategy")

            if USE_EXCHANGE_TRAILING:
                await self.place_trailing_order(symbol, close_side, amount, entry_price)
                self.logger.info(f"[{symbol}] Fallback: Trailing stop placed")
            else:
                await self.place_stop_loss_order(symbol, close_side, amount, stop_price)
                self.logger.info(f"[{symbol}] Fallback: Simple stop loss placed")

        except Exception as e:
            self.logger.error(f"[{symbol}] Exception during set_exit_orders: {e}")

    async def monitor_trailing_stop(self, symbol, entry_price, quantity, side, trigger_price, leverage):
        from ScreenerNew.config import TRAILING_OKX_CALLBACK_RATE, TRAILING_SL_PERCENT
        max_profit = trigger_price
        trail_step = TRAILING_SL_PERCENT

        self.logger.info(f"[TRAILING] Monitoring {symbol}, starting at {trigger_price}")

        while True:
            try:
                orderbook = await self.exchange.fetch_order_book(symbol)
                price = orderbook["bids"][0][0] if side == "sell" else orderbook["asks"][0][0]

                if price > max_profit:
                    max_profit = price

                stop_price = max_profit * (1 - trail_step)

                if price <= stop_price:
                    self.logger.info(f"[TRAILING] Triggered for {symbol}: price {price:.4f}, SL {stop_price:.4f}")
                    await self.close_position_market(symbol, quantity, side)
                    # === Cancel TP/SL orders after market exit
                    try:
                        await self.exchange.cancel_all_orders(symbol)
                        self.logger.info(f"[TRAILING] Cancelled all TP/SL orders for {symbol}")
                    except Exception as e:
                        self.logger.warning(f"[TRAILING] Failed to cancel orders for {symbol}: {str(e)}")

                    break  # exit monitoring loop

                await asyncio.sleep(TRAILING_OKX_CALLBACK_RATE)

            except Exception as e:
                self.logger.warning(f"[TRAILING] Error for {symbol}: {str(e)}")
                await asyncio.sleep(1)

    async def close_position_market(self, symbol, quantity, side):
        try:
            # Реверс напрямку: якщо buy → sell
            close_side = "sell" if side == "buy" else "buy"

            # Виконуємо маркет-ордер
            await self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=close_side,
                amount=quantity,
                params={"reduceOnly": True}
            )
            self.logger.info(f"[TRAILING] Closed position {symbol} via market order, qty={quantity}")
        except Exception as e:
            self.logger.error(f"[TRAILING] Failed to close position {symbol}: {str(e)}")

    async def update_trailing_stop(self, symbol, side, amount, current_price):
        from ScreenerNew.config import TRAILING_CALLBACK, TRAILING_REACTIVATION_GAP, TRAILING_UPDATE_INTERVAL

        if symbol not in self.position_metadata:
            self.logger.debug(f"[{symbol}] [TRAILING] No position metadata found, skipping update")
            return

        meta = self.position_metadata[symbol]
        entry_price = meta.get("entry_price", 0)
        old_trigger = meta.get("trail_stop_price", 0)
        algo_id = meta.get("algo_id")
        last_trail_update = meta.get("last_trailing_update", 0)
        time_since_update = time.time() - last_trail_update

        if not algo_id:
            self.logger.debug(f"[{symbol}] [TRAILING] No trailing stop algo_id found, skipping update")
            return

        new_trigger = None
        if side == "buy" and current_price > meta.get("max_price", 0):
            meta["max_price"] = current_price
            new_trigger = current_price * (1 - TRAILING_CALLBACK / 100)
        elif side == "sell" and current_price < meta.get("min_price", float('inf')):
            meta["min_price"] = current_price
            new_trigger = current_price * (1 + TRAILING_CALLBACK / 100)

        if new_trigger is None:
            self.logger.debug(f"[{symbol}] [TRAILING] No new trigger price calculated, skipping update")
            return

        diff_pct = abs(new_trigger - old_trigger) / old_trigger * 100 if old_trigger > 0 else 100
        if diff_pct < TRAILING_REACTIVATION_GAP or time_since_update < TRAILING_UPDATE_INTERVAL:
            self.logger.debug(f"[{symbol}] [TRAILING] Update skipped: diff_pct={diff_pct:.2f}% < {TRAILING_REACTIVATION_GAP}% or time_since_update={time_since_update:.2f}s < {TRAILING_UPDATE_INTERVAL}s")
            return

        self.logger.info(f"[{symbol}] [TRAILING] Ціна={current_price:.4f}, старий SL={old_trigger:.4f}, новий SL={new_trigger:.4f}")

        try:
            self.logger.info(f"[{symbol}] [TRAILING] Скасування попереднього трейлінг SL (algoId={algo_id}) перед оновленням")
            await self.exchange.safe_await(
                self.exchange.cancel_algo_order(algo_id),
                timeout=10,
                context=f"{symbol}.cancel_trailing"
            )
        except AttributeError as e:
            self.logger.error(f"[{symbol}] [TRAILING] Метод cancel_algo_order відсутній: {str(e)}")
            self.logger.warning(f"[{symbol}] [TRAILING] Пропущено оновлення трейлінгу, щоб уникнути дублювання SL")
            return
        except Exception as e:
            self.logger.error(f"[{symbol}] [TRAILING] Помилка при скасуванні трейлінг SL: {str(e)}")
            return

        try:
            new_algo_id = await self.exchange.safe_await(
                self.exchange.place_trailing_order(
                    symbol=symbol,
                    side=side,
                    size=amount,
                    trigger_price=new_trigger,
                    callback_rate=TRAILING_CALLBACK
                ),
                timeout=10,
                context=f"{symbol}.place_trailing"
            )
            if new_algo_id:
                meta["algo_id"] = new_algo_id
                meta["trail_stop_price"] = new_trigger
                meta["last_trailing_update"] = time.time()
                self.position_metadata[symbol] = meta
                self.logger.info(f"[{symbol}] [TRAILING] Новий трейлінг SL встановлено: {new_trigger:.4f}, algo_id={new_algo_id}")
            else:
                self.logger.warning(f"[{symbol}] [TRAILING] Failed to set new trailing stop")
        except Exception as e:
            self.logger.error(f"[{symbol}] [TRAILING] Помилка при встановленні нового трейлінг SL: {str(e)}")