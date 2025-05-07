#Trade_core 0.5

import logging
import json
from datetime import datetime
import math
import asyncio
from ScreenerNew.config import TRADE_MARGIN_RATIO, MIN_MARGIN_PER_TRADE, LEVERAGE, FIXED_SL_PCT, FIXED_TP_PCT, TRAILING_TRIGGER_RATIO, TRAILING_OKX_CALLBACK_RATE, USE_EXCHANGE_TRAILING, USE_BOTH_SL_TRAILING
class TradeCore:
    def __init__(self, exchange, virtual_balance, initial_balance, trades, log_file, order_manager):
        self.logger = logging.getLogger("trade_core")
        self.order_manager = order_manager
        self.exchange = exchange
        self.virtual_balance = virtual_balance
        self.initial_balance = initial_balance
        self.trades = trades
        self.log_file = log_file

    def calculate_margin(self, available_margin):
        return max(available_margin * TRADE_MARGIN_RATIO, MIN_MARGIN_PER_TRADE)

    def calculate_amount_from_margin(self, margin, price, lot_size):
        try:
            raw_amount = (margin * LEVERAGE) / price
            amount = math.floor(raw_amount / lot_size) * lot_size
            return amount
        except Exception as e:
            self.logger.error(f"Error calculating amount from margin: {e}")
            return 0.0

    def validate_amount(self, amount, min_amount):
        return amount >= min_amount

    def log_action(self, symbol, action, side, price, amount, profit=None, extra=None):
        try:
            if amount is None or price is None:
                self.logger.warning(f"[{symbol}] log_action: amount or price is None — margin set to 0")
                margin = 0.0
            else:
                margin = float(amount) * float(price) / LEVERAGE
        except Exception as e:
            self.logger.error(f"[{symbol}] log_action: Failed to calculate margin: {str(e)}")
            margin = 0.0

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "side": side,
            "price": price,
            "amount": amount,
            "margin": margin,
            "profit": profit if action == "close" else None,
            "balance": self.virtual_balance,
            "winrate": self.calculate_winrate() if action == "close" else None
        }
        if extra:
            log_entry.update(extra)
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write log for {symbol}: {str(e)}")
        self.logger.info(f"Logged action for {symbol}: {log_entry}")

    def calculate_winrate(self):
        if not self.trades:
            return 0.0
        wins = sum(1 for trade in self.trades if trade.get('profit', 0) > 0)
        return (wins / len(self.trades)) * 100 if self.trades else 0.0

    async def calculate_locked_balance(self):
        balance = await self.exchange.safe_await(
            self.exchange.exchange.fetch_balance(),
            timeout=10,
            context="fetch_balance"
        )
        if balance is None:
            self.logger.error("Failed to fetch balance, using default values")
            balance = {'USDT': {'used': 0.0}}
        used = float(balance.get('USDT', {}).get('used', 0))

        active_positions = await self.exchange.safe_await(
            self.exchange.sync_positions(self.exchange.valid_pairs),
            timeout=15,
            context="sync_positions"
        )
        if active_positions is None:
            self.logger.error("Failed to sync positions, using empty dict")
            active_positions = {}

        local_locked = 0.0
        for pos in active_positions.values():
            entry_price = pos.get('entryPrice')
            amount = pos.get('contracts')
            if entry_price is None or amount is None:
                self.logger.warning(f"Zero entry_price or amount detected for {pos.get('symbol', 'unknown')} in calculate_locked_balance")
                continue
            try:
                entry_price = float(entry_price)
                amount = float(amount)
            except Exception as e:
                self.logger.error(f"Failed to convert entry_price or amount to float: {str(e)} | pos={pos}")
                continue
            if entry_price and amount:
                position_value = amount * entry_price
                margin = position_value / LEVERAGE
                local_locked += margin
            else:
                self.logger.warning(f"Zero entry_price or amount detected for {pos.get('symbol', 'unknown')} in calculate_locked_balance")

        if abs(local_locked - used) > 0.01:
            self.logger.warning(f"Local locked balance {local_locked:.2f} differs from exchange used {used:.2f}")

        self.logger.info(f"Locked balance from exchange: {used:.2f} USDT")
        return used

    async def calculate_roi(self):
        if self.initial_balance is None or self.initial_balance == 0:
            self.logger.warning("Initial balance is None or 0, returning ROI as 0.0%")
            return 0.0
        total = self.virtual_balance + await self.calculate_locked_balance()
        if total == 0 or self.initial_balance == 0:
            self.logger.warning("Total or initial balance is 0, returning ROI as 0.0%")
            return 0.0
        return ((total - self.initial_balance) / self.initial_balance) * 100

    async def open_trade(self, symbol, side, current_price, atr):
        try:
            self.logger.info(f"[TRADE] Opening trade for {symbol} with side {side}")

            market_info = await self.exchange.get_market_info(symbol)
            lot_size = market_info["lot_size"]

            margin = await self.exchange.get_available_margin()
            calculated_margin = self.calculate_margin(margin)

            amount = self.calculate_amount_from_margin(
                margin=calculated_margin,
                price=current_price,
                lot_size=lot_size
            )

            if amount is None or amount <= 0:
                self.logger.warning(f"[TRADE] Invalid amount for {symbol}, skipping trade")
                return None

            if side == "buy":
                stop_price = current_price * (1 - FIXED_SL_PCT / 100)
                take_price = current_price * (1 + FIXED_TP_PCT / 100)
            else:
                stop_price = current_price * (1 + FIXED_SL_PCT / 100)
                take_price = current_price * (1 - FIXED_TP_PCT / 100)

            trigger_price = None
            if USE_EXCHANGE_TRAILING and USE_BOTH_SL_TRAILING:
                if side == "buy":
                    trigger_price = current_price * (1 + TRAILING_TRIGGER_RATIO / 100)
                else:
                    trigger_price = current_price * (1 - TRAILING_TRIGGER_RATIO / 100)

                self.logger.info(
                    f"[TRAILING] Trigger price for {symbol}: {trigger_price:.4f} "
                    f"(TRIGGER {TRAILING_TRIGGER_RATIO}%, CALLBACK {TRAILING_OKX_CALLBACK_RATE}%)"
                )

            self.logger.info(
                f"[TRADE] {symbol} SL={stop_price:.4f}, TP={take_price:.4f}, Qty={amount:.4f}"
            )

            entry_order, stop_price, take_price, amount, algo_ids = await self.exchange.place_order(
                symbol=symbol,
                side=side,
                price=current_price,
                margin=calculated_margin,
                leverage=LEVERAGE,
                stop_price=stop_price,
                take_price=take_price,
                extra_params={}
            )

            if entry_order is None:
                self.logger.error(f"[TRADE] Failed to place entry order for {symbol}")
                return None

            self.logger.info(f"[TRADE] Entry order placed for {symbol}, setting exit orders")

            await self.order_manager.set_exit_orders(
                symbol=symbol,
                side=side,
                amount=amount,
                entry_price=current_price,
                stop_price=stop_price,
                take_price=take_price,
                trigger_price=trigger_price
            )

            self.logger.debug(f"[{symbol}] Returning metadata for trailing: " + f"trigger={trigger_price}, amount={amount}, stop_loss_id={algo_ids.get('stop_loss')}")


            # 📌 Ось що тепер повертаємо:
            return {
                "symbol": symbol,
                "trailing_prepared": True,
                "trigger_price": trigger_price,
                "trailing_started": False,
                "trailing_sl_algo_id": algo_ids.get('stop_loss'),
                "side": side,
                "amount": amount
            }

        except Exception as e:
            self.logger.error(f"[TRADE] Exception during open_trade for {symbol}: {e}")
            return None
