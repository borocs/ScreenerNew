# display.py - Version: 0.2


import os
from prettytable import PrettyTable
from colorama import init, Fore, Style
import pandas as pd
import logging
import asyncio
from datetime import datetime
import textwrap
from ScreenerNew.config import PAIRS, MIN_PROB_THRESHOLD, get_logger, LEVERAGE

logger = get_logger("display")
logger.setLevel(logging.INFO)

class TradeDisplay:
    def __init__(self):
        init()
        logger.info("Initializing TradeDisplay...")
        self.active_positions_cache = None
        self.last_position_sync_time = 0
        self.position_cache_timeout = 1.0  # Cache timeout in seconds
        self.positions = []
        self.balance = 0.0

    async def update_table(self, positions):
            self.positions = positions
            logger.info(f"[DISPLAY] Updated table with {len(positions)} open positions.")

    async def update_balance(self, balance):
            self.balance = balance
            logger.info(f"[DISPLAY] Updated balance: {balance:.2f} USDT.")

    async def display_data(self, exchange, probabilities, trades, virtual_balance, initial_balance, signal_stats, position_metadata):
        try:
            # Fetch available margin
            available = await exchange.get_available_margin()

            # Cache active positions
            current_time = datetime.now().timestamp()
            if self.active_positions_cache is None or (current_time - self.last_position_sync_time) > self.position_cache_timeout:
                logger.debug("Fetching active positions from exchange")
                self.active_positions_cache = await exchange.sync_positions(exchange.valid_pairs)
                self.last_position_sync_time = current_time
            else:
                logger.debug("Using cached active positions")
            active_positions = self.active_positions_cache

            # Safe calculation of locked balance
            locked = 0.0
            for symbol, pos in active_positions.items():
                contracts = pos.get('contracts')
                entry_price = pos.get('entryPrice')
                if contracts is None or entry_price is None:
                    logger.warning(f"[{symbol}] Invalid position data (NoneType): contracts={contracts}, entryPrice={entry_price}")
                    continue
                try:
                    contracts = float(contracts)
                    entry_price = float(entry_price)
                    if contracts > 0 and entry_price > 0:
                        locked += (contracts * entry_price) / LEVERAGE
                    else:
                        logger.debug(f"[{symbol}] Skipping position: contracts={contracts}, entryPrice={entry_price}")
                except Exception as e:
                    logger.error(f"[{symbol}] Error calculating locked balance: {str(e)} | Pos: {pos}")
                    continue

            total = virtual_balance + locked

            # Balance table
            balance_table = PrettyTable()
            balance_table.field_names = ["Metric", "Value (USDT)"]
            balance_table.align["Metric"] = "l"
            balance_table.align["Value (USDT)"] = "r"
            balance_table.add_row(["Available Balance", f"{available:.2f}"])
            balance_table.add_row(["In Trades (Locked)", f"{locked:.2f}"])
            balance_table.add_row(["Total Equity", f"{total:.2f}"])
            balance_table.add_row(["ROI", f"{((total - initial_balance) / initial_balance * 100 if initial_balance != 0 else 0.0):+.1f}%"])

            # Positions table
            table = PrettyTable()
            table.field_names = [
                "Symbol", "Status", "Side", "Entry", "Price",
                "Pos(USDT)", "P%", "Progress", "Action",
                "Support", "TP/SL", "TS Trigger", "Win%", "Prob%", "Signal", "Reason"
            ]

            tickers = await asyncio.gather(*[exchange.fetch_ticker(symbol) for symbol in exchange.valid_pairs], return_exceptions=True)
            for symbol, ticker in zip(exchange.valid_pairs, tickers):
                if isinstance(ticker, Exception):
                    logger.warning(f"Failed to fetch ticker for {symbol}: {str(ticker)}")
                    current_price = 0
                    current_price_str = "-"
                else:
                    current_price = ticker.get('last', 0) if ticker else 0
                    current_price_str = f"{current_price:.4f}" if current_price >= 0.0001 else "-"

                status = "Open" if symbol in active_positions and active_positions[symbol].get("contracts", 0) > 0 else "Closed"
                probability = probabilities.get(symbol, 0.0)
                prob_str = f"{probability*100:.1f}" if probability else "-"

                if status == "Closed":
                    ohlcv = await exchange.fetch_historical_data(symbol, timeframe="1m", limit=300)
                    if not ohlcv:
                        reason = "No data"
                    else:
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.name = symbol
                        from ScreenerNew.indicators import Indicators
                        indicators = Indicators(exchange=exchange)
                        result = indicators.generate_signals(df, {})
                        if result is None or not isinstance(result[2], dict):
                            reason = "No signal"
                        else:
                            _, _, signal_data = result
                            signal_data['current_price'] = current_price
                            signal_data['price_change'] = 0.0
                            probability = signal_data.get("probability", 0.0)
                            buy_prob = signal_data.get('buy_prob', 0.0)
                            sell_prob = signal_data.get('sell_prob', 0.0)
                            rsi = signal_data.get('rsi', 50.0)
                            reasons = signal_data.get('reasons', [])
                            signal = signal_data.get('signal', 'hold')

                            adjusted_buy_prob = buy_prob
                            adjusted_sell_prob = sell_prob
                            for r in reasons:
                                if 'Buy prob' in r:
                                    adjusted_buy_prob = max(0.0, adjusted_buy_prob - 0.05)
                                if 'Sell prob' in r:
                                    adjusted_sell_prob = max(0.0, adjusted_sell_prob - 0.05)
                                if 'MACD' in r or 'RSI' in r or 'Volume' in r:
                                    if signal.lower() == 'buy':
                                        adjusted_buy_prob = max(0.0, adjusted_buy_prob - 0.05)
                                    else:
                                        adjusted_sell_prob = max(0.0, adjusted_sell_prob - 0.05)

                            side = 'hold'
                            if adjusted_buy_prob >= MIN_PROB_THRESHOLD:
                                if rsi < 50:
                                    side = 'buy'
                                else:
                                    reasons.append("RSI too high for buy (>=50)")
                            elif adjusted_sell_prob >= MIN_PROB_THRESHOLD:
                                if rsi > 50:
                                    side = 'sell'
                                else:
                                    reasons.append("RSI too low for sell (<=50)")

                            reason = f"Buy: p={adjusted_buy_prob:.2f}" if side == 'buy' else f"Sell: p={adjusted_sell_prob:.2f}" if side == 'sell' else (reasons[0] if reasons else "No signal")
                            if probability >= MIN_PROB_THRESHOLD and side == 'hold':
                                reason = f"High prob ({probability:.2f}) but {reason}"

                    reason = textwrap.shorten(reason, width=25, placeholder="...")
                    table.add_row([
                        symbol[:10], status, "-", "-", current_price_str,
                        "-", "-", "-", "search", "-", "-", "-",
                        "-", prob_str, signal, reason
                    ])

                    logger.debug("[DISPLAY] data updated")

                    continue

                # Open position handling
                pos = active_positions[symbol]
                meta = position_metadata.get(symbol, {})
                if not meta:
                    logger.warning(f"[{symbol}] No position metadata found")

                contracts = pos.get('contracts')
                entry_price = pos.get('entryPrice')
                if contracts is None or entry_price is None:
                    logger.warning(f"[{symbol}] Invalid position data: contracts={contracts}, entryPrice={entry_price}")
                    continue
                try:
                    amount = float(contracts)
                    entry_price = float(entry_price)
                except Exception as e:
                    logger.error(f"[{symbol}] Error parsing position: {str(e)} | Pos: {pos}")
                    continue

                take_price = meta.get('take_price', 0)
                stop_price = meta.get('stop_price', 0)
                support_level = meta.get('support_level', 0)
                atr = meta.get('atr', 0)
                side = "buy" if pos.get('side') == "long" else "sell"

                position_value = amount * entry_price if entry_price and amount else 0.0
                if current_price == 0 or entry_price == 0 or amount == 0:
                    logger.warning(f"[{symbol}] Invalid values for profit calculation: current_price={current_price}, entry_price={entry_price}, amount={amount}")
                    profit = 0.0
                    profit_percent = 0.0
                else:
                    profit = (current_price - entry_price) * amount if side == 'buy' else (entry_price - current_price) * amount
                    profit *= LEVERAGE
                    profit_percent = (profit / (entry_price * amount)) * 100 if entry_price and amount else 0.0

                total_bars = 10
                progress = 0
                if take_price and stop_price and entry_price and current_price:
                    try:
                        if side == 'buy':
                            progress = min((current_price - entry_price) / (take_price - entry_price), 1.0) if current_price > entry_price else min((entry_price - current_price) / (entry_price - stop_price), 1.0)
                        elif side == 'sell':
                            progress = min((entry_price - current_price) / (entry_price - take_price), 1.0) if current_price < entry_price else min((current_price - entry_price) / (stop_price - entry_price), 1.0)
                    except ZeroDivisionError:
                        logger.warning(f"[{symbol}] ZeroDivisionError in progress calculation: take_price={take_price}, stop_price={stop_price}, entry_price={entry_price}")
                        progress = 0

                progress_str = f"{Fore.GREEN}{'■' * int(progress * total_bars)}{'□' * (total_bars - int(progress * total_bars))}{Style.RESET_ALL}" if (side == 'buy' and current_price > entry_price) or (side == 'sell' and current_price < entry_price) else f"{Fore.RED}{'■' * int(progress * total_bars)}{'□' * (total_bars - int(progress * total_bars))}{Style.RESET_ALL}"

                symbol_trades = [t for t in trades if t['symbol'] == symbol]
                coin_winrate = "-" if not symbol_trades else f"{(sum(1 for t in symbol_trades if t['profit'] > 0) / len(symbol_trades)) * 100:.0f}"
                prob_str = f"{probability*100:.1f}" if probability else "-"
                tp_sl_str = f"{take_price:.2f}/{stop_price:.2f}" if take_price and stop_price else "-"
                ts_trigger_str = f"{meta.get('trail_stop_price', '-'):.4f}" if meta.get('trail_stop_price') else "-"

                entry_price_str = f"{entry_price:.4f}" if entry_price >= 0.0001 else f"{entry_price:.2e}"
                support_level_str = f"{support_level:.2f}" if support_level >= 0.0001 else "-"
                reason = f"Open: {profit_percent:.1f}%"
                reason = textwrap.shorten(reason, width=25, placeholder="...")

                table.add_row([
                    symbol[:10], status, side, entry_price_str, current_price_str,
                    f"{position_value:.1f}", f"{profit_percent:.1f}", progress_str,
                    "hold", support_level_str, tp_sl_str, ts_trigger_str,
                    coin_winrate, prob_str, "hold", reason
                ])

            total_trades = len(trades)
            successful_trades = sum(1 for trade in trades if trade['profit'] > 0)

            output = []
            output.append(f"{Fore.CYAN}=== Balance Summary ==={Style.RESET_ALL}")
            output.append(str(balance_table))
            output.append(f"{Fore.MAGENTA}Winrate: {(successful_trades / total_trades * 100 if total_trades > 0 else 0.0):.1f}%{Style.RESET_ALL}")
            output.append(f"{Fore.BLUE}Trades: {successful_trades}/{total_trades}{Style.RESET_ALL}")
            output.append(f"{Fore.BLUE}Signals (Total/Valid/Executed): {signal_stats['total']}/{signal_stats['valid']}/{signal_stats['executed']}{Style.RESET_ALL}")
            output.append("")
            output.append(f"{Fore.CYAN}=== Open Positions ==={Style.RESET_ALL}")
            output.append(str(table))
            output.append("")

            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n".join(output))

        except Exception as e:
            logger.error(f"Error displaying data: {str(e)}")