# exchange.py - Version: 1.10.6
# Changelog:
# - Version 1.10.6: Added place_stop_loss_take_profit method to support OCO orders for stop-loss and take-profit.
# - Version 1.10.5: Added __all__ to explicitly export AsyncExchange and safe_await for use in trade.py.
# - Version 1.10.4: Added timeout handling using asyncio.wait_for for all exchange API calls.
# - Version 1.10.3: Added robust error handling for NoneType, invalid types, and missing data.
# - Version 1.10.2: Added cancel_algo_order method to support cancelling algo orders for trailing stops.

import sys
import os
from typing import Optional
import ccxt.async_support as ccxt
import asyncio
import logging
import json
import time
import math
from datetime import datetime
from ScreenerNew.config import API_KEY, SECRET, PASSWORD, PAIRS, get_logger, LEVERAGE, MIN_MARGIN_PER_TRADE, USE_EXCHANGE_STOPLOSS, USE_EXCHANGE_TRAILING

# Додаємо кореневий каталог до sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logger = get_logger("exchange")

class AsyncExchange:
    def __init__(self):
        from ScreenerNew.config import get_logger
        self.logger = get_logger("AsyncExchange")

        self.exchange = ccxt.okx({
            'apiKey': API_KEY,
            'secret': SECRET,
            'password': PASSWORD,
            'enableRateLimit': True,
        })
        self.ticker_data = {}
        self.order_book_data = {}
        self.candlestick_data = {pair: [] for pair in PAIRS}
        self.candlestick_timestamp = {pair: 0 for pair in PAIRS}
        self.lock = asyncio.Lock()
        self.lot_sizes = {}
        self.markets = None
        self.last_trade_time = {}
        self.valid_pairs = []
        self.position_mode = 'net_mode'  # За замовчуванням One-Way
        self.client = self.exchange  # Для сумісності з новим place_order

    async def safe_await(self, coro, timeout=5, default=None, context=""):
        """
        Виконує асинхронний виклик із таймаутом.
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"[TIMEOUT] {context} не повернувся за {timeout} секунд")
            return default
        except Exception as e:
            logger.error(f"[ERROR] {context} викликав помилку: {str(e)}")
            return default

    async def fetch_position_mode(self):
        context = "fetch_position_mode"
        try:
            resp = await self.safe_await(
                self.exchange.private_get_account_config(),
                timeout=5,
                default={'data': [{'posMode': 'net_mode'}]},
                context=context
            )
            mode = resp['data'][0].get('posMode', 'net_mode')
            return mode
        except Exception as e:
            logger.error(f"Error in {context}: {str(e)}")
            return 'net_mode'

    async def initialize(self):
        context = "initialize"
        try:
            self.markets = await self.safe_await(
                self.exchange.load_markets(),
                timeout=10,
                default=None,
                context=f"{context}: load_markets"
            )
            if not self.markets:
                raise Exception("Failed to load markets")
            logger.info("Markets loaded successfully")
            
            self.position_mode = await self.fetch_position_mode()
            logger.info(f"Position mode detected: {self.position_mode}")
            self.valid_pairs = []

            async def set_leverage_for_pair(symbol):
                try:
                    await self.set_leverage(symbol, LEVERAGE)
                    return symbol
                except Exception as e:
                    logger.warning(f"Skipping {symbol}: failed to set leverage {LEVERAGE}: {str(e)}")
                    return None

            tasks = [set_leverage_for_pair(symbol) for symbol in PAIRS]
            results = await self.safe_await(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=15,
                default=[],
                context=f"{context}: set_leverage_for_pairs"
            )
            self.valid_pairs = [r for r in results if r is not None]

            if not self.valid_pairs:
                logger.warning("No pairs support required leverage, proceeding with all pairs")
                self.valid_pairs = PAIRS.copy()
            logger.info(f"Valid pairs: {self.valid_pairs}")
        except Exception as e:
            logger.error(f"Failed to {context}: {str(e)}")
            raise

    async def set_leverage(self, symbol, leverage):
        context = f"set_leverage for {symbol}"
        try:
            await self.safe_await(
                self.exchange.set_leverage(leverage, symbol, params={'mgnMode': 'isolated'}),
                timeout=5,
                default=None,
                context=context
            )
            logger.info(f"Leverage set to {leverage} for {symbol}")
        except Exception as e:
            logger.error(f"Failed to {context}: {str(e)}")
            raise

    def get_inst_id(self, symbol: str) -> Optional[str]:
        if not self.markets:
            logger.error("[exchange] - Markets not loaded, cannot get instId")
            return None

        market = self.markets.get(symbol)
        if market and 'id' in market:
            return market['id']
        logger.error(f"[exchange] - Unable to find instId for {symbol}")
        return None

    async def get_market_info(self, symbol):
        if not self.markets or symbol not in self.markets:
            logger.error(f"[{symbol}] Market not found in self.markets")
            return {
                'min_notional': 0.01,
                'lot_size': 0.01,
                'max_amount': 1e9,
                'min_amount': 0.01,
                'price_precision': 8
            }

        try:
            market = self.markets[symbol]
            return {
                'min_notional': market['limits']['cost'].get('min', 0.01) or 0.01,
                'lot_size': market['precision']['amount'] or 0.01,
                'max_amount': market['limits']['amount'].get('max', 1e9) or 1e9,
                'min_amount': market['limits']['amount'].get('min', 0.01) or 0.01,
                'price_precision': market['precision']['price'] or 8
            }
        except Exception as e:
            logger.error(f"Failed to fetch market info for {symbol}: {str(e)}")
            return {
                'min_notional': 0.01,
                'lot_size': 0.01,
                'max_amount': 1e9,
                'min_amount': 0.01,
                'price_precision': 8
            }

    async def fetch_historical_data(self, symbol, timeframe="1m", limit=300):
        context = f"fetch_historical_data for {symbol}"
        async with self.lock:
            current_time = datetime.now().timestamp()
            if (current_time - self.candlestick_timestamp.get(symbol, 0)) < 30:
                candles = self.candlestick_data.get(symbol, [])
                if candles:
                    logger.debug(f"Returning cached historical data for {symbol}")
                    return candles[-limit:]

            try:
                ohlcv = await self.safe_await(
                    self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit),
                    timeout=5,
                    default=[],
                    context=context
                )
                self.candlestick_data[symbol] = ohlcv
                self.candlestick_timestamp[symbol] = current_time
                logger.debug(f"Fetched {len(ohlcv)} candles for {symbol}")
                return ohlcv
            except Exception as e:
                logger.error(f"Error in {context}: {str(e)}")
                return []

    async def fetch_ticker(self, symbol):
        context = f"fetch_ticker for {symbol}"
        async with self.lock:
            ticker = self.ticker_data.get(symbol)
            current_time = datetime.now().timestamp()
            if ticker and (current_time - ticker.get('timestamp', 0)) < 5:
                return ticker

        retries = 3
        for attempt in range(retries):
            try:
                ticker = await self.safe_await(
                    self.exchange.fetch_ticker(symbol),
                    timeout=5,
                    default={},
                    context=f"{context} (attempt {attempt + 1}/{retries})"
                )
                if not ticker or not isinstance(ticker, dict):
                    logger.error(f"[{symbol}] Invalid ticker format or None")
                    ticker = {}
                logger.debug(f"Ticker keys for {symbol}: {ticker.keys()}")
                last_price = ticker.get("last")
                vol_24h = ticker.get("baseVolume") or ticker.get("volume") or float(ticker.get("info", {}).get("vol24h", 0))
                bid = ticker.get("bid")
                ask = ticker.get("ask")

                last_price = float(last_price) if last_price is not None else 0.0
                vol_24h = float(vol_24h) if vol_24h is not None else 0.0
                bid = float(bid) if bid is not None else 0.0
                ask = float(ask) if bid is not None else 0.0

                async with self.lock:
                    self.ticker_data[symbol] = {
                        "last": last_price,
                        "vol_24h": vol_24h,
                        "bid": bid,
                        "ask": ask,
                        "timestamp": current_time
                    }
                logger.info(f"Fetched ticker for {symbol}: last={last_price:.4f}, vol_24h={vol_24h:.2f}, bid={bid:.4f}, ask={ask:.4f}")
                return self.ticker_data[symbol]
            except Exception as e:
                logger.error(f"Error in {context} (attempt {attempt + 1}/{retries}): {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(1)
                else:
                    async with self.lock:
                        self.ticker_data[symbol] = {
                            "last": 0.0,
                            "vol_24h": 0.0,
                            "bid": 0.0,
                            "ask": 0.0,
                            "timestamp": current_time
                        }
                    return self.ticker_data[symbol]

    async def get_total_margin(self):
        """Отримання вільної маржі та загального балансу."""
        try:
            balance = await self.exchange.fetch_balance()
            available_margin = balance.get('free', {}).get('USDT', 0.0)
            total_balance = balance.get('total', {}).get('USDT', 0.0)
            self.logger.info(f"Fetched available margin: {available_margin:.2f}, total: {total_balance:.2f}, used={balance.get('used', {}).get('USDT', 0.0):.2f}")
            return available_margin, total_balance
        except Exception as e:
            self.logger.error(f"[exchange] Error fetching margin: {str(e)}")
            return 0.0, 0.0

    async def get_total_margin(self):
        """
        Отримати доступну і зайняту маржу для акаунту.
        """
        try:
            balance = await self.client.fetch_balance()
            total = balance['total']['USDT']
            used = balance['used']['USDT']
            free = balance['free']['USDT']
            return free, used
        except Exception as e:
            self.logger.error(f"[EXCHANGE] Error fetching total margin: {e}")
            return 0, 0

    async def fetch_order_book(self, symbol):
        context = f"fetch_order_book for {symbol}"
        try:
            order_book = await self.safe_await(
                self.exchange.fetch_order_book(symbol, limit=20),
                timeout=5,
                default={'bids': [], 'asks': []},
                context=context
            )
            self.order_book_data[symbol] = {
                "bids": order_book['bids'],
                "asks": order_book['asks'],
                "timestamp": datetime.now().timestamp()
            }
            best_bid = order_book['bids'][0][0] if order_book['bids'] else 0
            best_ask = order_book['asks'][0][0] if order_book['asks'] else 0
            logger.info(f"Fetched order book for {symbol}: bestBid={best_bid:.4f}, bestAsk={best_ask:.4f}")
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in {context}: {str(e)}")
            self.order_book_data[symbol] = {"bids": [], "asks": [], "timestamp": datetime.now().timestamp()}

    def order_book_pressure(self, symbol):
        order_book = self.order_book_data.get(symbol, {"bids": [], "asks": []})
        bid_volume = sum([bid[1] for bid in order_book["bids"]]) if order_book["bids"] else 0
        ask_volume = sum([ask[1] for ask in order_book["asks"]]) if order_book["asks"] else 0
        pressure = bid_volume / ask_volume if ask_volume > 0 else 1.0
        return pressure

    def spread(self, symbol):
        ticker = self.ticker_data.get(symbol, {"bid": 0, "ask": 0})
        bid = ticker.get("bid", 0)
        ask = ticker.get("ask", 0)
        if bid == 0 or ask == 0:
            return float('inf')
        return ((ask - bid) / bid) * 100 if bid > 0 else float('inf')

    async def get_available_margin(self):
        context = "get_available_margin"
        retries = 3
        for attempt in range(retries):
            try:
                balance = await self.safe_await(
                    self.exchange.fetch_balance(),
                    timeout=5,
                    default={'USDT': {'free': 0.0, 'total': 0.0, 'used': 0.0}},
                    context=f"{context} (attempt {attempt + 1}/{retries})"
                )
                available = float(balance['USDT']['free'])
                total = float(balance['USDT']['total'])
                used = float(balance['USDT']['used'])
                logger.info(f"Fetched available margin: {available:.2f}, total: {total:.2f}, used={used:.2f}")
                return available
            except Exception as e:
                logger.error(f"Error in {context} (attempt {attempt + 1}/{retries}): {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(2)
                else:
                    logger.error(f"All retries failed to {context}")
                    return 0.0

    async def sync_positions(self, symbols):
        """
        Отримує відкриті позиції для заданих символів та фільтрує всі порожні (contracts=0 або entryPrice=None).
        Повертає словник лише активних позицій.
        """
        context = f"sync_positions for {len(symbols)} symbols"
        logger.info(f"[exchange] Syncing positions for {len(symbols)} symbols...")
        retries = 3
        positions = {}

        for attempt in range(retries):
            try:
                chunk_size = 10
                for i in range(0, len(symbols), chunk_size):
                    chunk = symbols[i:i + chunk_size]
                    try:
                        raw_positions = await self.safe_await(
                            self.exchange.fetch_positions(chunk),
                            timeout=10,
                            default=[],
                            context=f"{context}: fetch_positions for {chunk}"
                        )
                        if not raw_positions or not isinstance(raw_positions, list):
                            logger.warning(f"[exchange] Empty or invalid positions list for {chunk}")
                            continue
                    except Exception as e:
                        logger.error(f"[exchange] Exception in {context}: {str(e)}")
                        continue

                    for pos in raw_positions:
                        symbol = pos['symbol']
                        contracts = pos.get('contracts')
                        entry_price = pos.get('entryPrice')

                        if contracts is None or entry_price is None:
                            logger.debug(f"[{symbol}] Skipping position: contracts or entryPrice is None")
                            continue

                        try:
                            contracts = float(contracts)
                            entry_price = float(entry_price)
                        except (TypeError, ValueError) as e:
                            logger.debug(f"[{symbol}] Skipping position: invalid contracts={contracts} or entryPrice={entry_price} ({str(e)})")
                            continue

                        if contracts == 0.0 or entry_price == 0.0:
                            logger.debug(f"[{symbol}] Skipping empty position: contracts={contracts}, entryPrice={entry_price}")
                            continue

                        positions[symbol] = {
                            'symbol': symbol,
                            'contracts': contracts,
                            'entryPrice': entry_price,
                            'side': pos.get('side', 'long'),  # OKX повертає 'long' або 'short'
                            'openTime': pos.get('timestamp', int(datetime.now().timestamp() * 1000))  # OKX повертає час у мс
                        }
                        logger.debug(f"[{symbol}] Active position synced: contracts={contracts:.2f}, entryPrice={entry_price:.4f}")

                logger.info(f"[exchange] Synced {len(positions)} active positions")
                cleaned_positions = {}
                for symbol, pos in positions.items():
                    contracts = float(pos.get("contracts", 0))
                    if contracts > 0:
                        cleaned_positions[symbol] = pos
                    else:
                        logger.info(f"[{symbol}] Position is empty on exchange — skipping from active list")
                return cleaned_positions

            except Exception as e:
                logger.error(f"Error in {context} (attempt {attempt + 1}/{retries}): {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(2)
                else:
                    logger.error(f"All retries failed to {context}")
                    return {}

    async def place_order(self, symbol, side, price, margin, leverage, stop_price, take_price, extra_params):
        context = f"place_order for {symbol}"
        retries = 3
        algo_ids = {'stop_loss': None, 'take_profit': None}
        for attempt in range(retries):
            try:
                market = self.markets.get(symbol)
                if not market:
                    logger.error(f"[{symbol}] Market info not found for symbol")
                    return None, stop_price, take_price, 0, algo_ids

                precision = market['precision']['amount']
                min_notional = market['limits']['cost'].get('min', 0.01) or 0.01
                min_amount = market['limits']['amount'].get('min', 0.01) or 0.01
                max_amount = market['limits']['amount'].get('max', 1e9) or 1e9

                if price == 0:
                    logger.error(f"[{symbol}] Price is zero, cannot calculate amount")
                    return None, stop_price, take_price, 0, algo_ids
                amount = (margin / price) * leverage

                if amount < min_amount:
                    logger.warning(f"[{symbol}] Calculated amount {amount:.6f} is below minimum {min_amount:.6f}, skipping")
                    return None, stop_price, take_price, 0, algo_ids

                amount = max(amount, min_amount)
                amount = min(amount, max_amount)
                amount = math.floor(amount / precision) * precision

                position_value = amount * price
                calculated_margin = position_value / leverage
                if position_value < min_notional or calculated_margin < MIN_MARGIN_PER_TRADE:
                    logger.warning(f"[{symbol}] Order rejected: notional={position_value:.2f} (min={min_notional:.2f}), margin={calculated_margin:.2f} (min={MIN_MARGIN_PER_TRADE:.2f}), precision={precision}, price={price:.4f}")
                    return None, stop_price, take_price, 0, algo_ids

                inst_id = self.get_inst_id(symbol)
                if inst_id is None:
                    logger.error(f"[{symbol}] Order setup aborted: invalid instId")
                    return None, stop_price, take_price, 0, algo_ids

                params = {
                    'tdMode': 'isolated',
                    'ordType': 'market',
                    'reduceOnly': False,
                    'leverage': leverage
                }
                if 'clOrdId' in params:
                    del params['clOrdId']
                for forbidden in ['posSide', 'clOrdId']:
                    extra_params.pop(forbidden, None)
                params.update(extra_params)

                logger.debug(f"[{symbol}] Market info: min_notional={min_notional:.2f}, min_amount={min_amount:.6f}, calculated: notional={position_value:.2f}, amount={amount:.6f}")

                logger.debug(f"[{symbol}] Attempt {attempt + 1}/{retries} - Placing market order: side={side}, amount={amount:.6f}, price={price:.4f}, margin={margin:.2f}, params={json.dumps(params, indent=2)}")
                logger.info(f"[{symbol}] [ORDER] Calling create_order: side={side}, amount={amount:.4f}, margin={margin:.4f}, price={price}")

                order = await self.safe_await(
                    self.exchange.create_order(
                        symbol=symbol,
                        type='market',
                        side=side,
                        amount=amount,
                        price=None,
                        params=params
                    ),
                    timeout=10,
                    default=None,
                    context=f"{context}: create_order"
                )
                if not order:
                    logger.error(f"[{symbol}] Order creation returned None")
                    return None, stop_price, take_price, 0, algo_ids
                logger.info(f"[{symbol}] Placed market order: {side} {amount:.6f} at {price:.4f}")

                if USE_EXCHANGE_STOPLOSS and stop_price and take_price:
                    close_side = "sell" if side == "buy" else "buy"
                    oco_payload = {
                        "instId": inst_id,
                        "tdMode": "isolated",
                        "side": close_side,
                        "ordType": "oco",
                        "sz": str(amount),
                        "slTriggerPx": str(stop_price),
                        "slOrdPx": str(stop_price),
                        "tpTriggerPx": str(take_price),
                        "tpOrdPx": str(take_price)
                    }
                    if self.position_mode == "long_short_mode":
                        oco_payload["posSide"] = "long" if side == "buy" else "short"
                    else:
                        oco_payload.pop("posSide", None)

                    logger.debug(f"[{symbol}] Placing OCO order: {json.dumps(oco_payload, indent=2)}")
                    oco_response = await self.safe_await(
                        self.exchange.private_post_trade_order_algo(params=oco_payload),
                        timeout=5,
                        default=None,
                        context=f"{context}: place_oco_order"
                    )
                    if not oco_response or 'data' not in oco_response:
                        logger.error(f"[{symbol}] Invalid OCO response format: {json.dumps(oco_response, indent=2)}")
                        return None, stop_price, take_price, 0, algo_ids
                    if 'code' in oco_response and oco_response['code'] != '0':
                        logger.error(f"[{symbol}] OCO API error: {oco_response.get('msg', 'No message')} (code: {oco_response['code']})")
                        if oco_response['code'] in ['50014', '50015']:
                            logger.warning(f"[{symbol}] Invalid ordPx/triggerPx, retrying with ordPx=-1")
                            oco_payload["slOrdPx"] = "-1"
                            oco_payload["tpOrdPx"] = "-1"
                            oco_response = await self.safe_await(
                                self.exchange.private_post_trade_order_algo(params=oco_payload),
                                timeout=5,
                                default=None,
                                context=f"{context}: place_oco_order (retry)"
                            )
                    if oco_response and 'data' in oco_response and len(oco_response['data']) > 0:
                        algo_ids['stop_loss'] = oco_response['data'][0].get('algoId')
                        algo_ids['take_profit'] = oco_response['data'][0].get('algoId')
                        logger.info(f"[{symbol}] Placed OCO order: TP={take_price:.4f}, SL={stop_price:.4f}, algoId={algo_ids['stop_loss']}")
                    else:
                        logger.error(f"[{symbol}] Failed to place OCO order: Invalid response {json.dumps(oco_response, indent=2)}")
                        return None, stop_price, take_price, 0, algo_ids

                if not order or 'id' not in order or order['id'] is None:
                    logger.error(f"[{symbol}] Order placement failed: order_id is missing or None")
                    return None, stop_price, take_price, 0, algo_ids

                return order['id'], stop_price, take_price, amount, algo_ids
            except Exception as e:
                logger.error(f"Error in {context} (attempt {attempt + 1}/{retries}): {str(e)}")
                if "51000" in str(e):
                    logger.critical(f"[{symbol}] posSide error 51000 – check params: {json.dumps(params, indent=2)}")
                if "50014" in str(e) or "50015" in str(e):
                    logger.warning(f"[{symbol}] Invalid ordPx/triggerPx error, retrying")
                if attempt < retries - 1:
                    await asyncio.sleep(2)
                else:
                    logger.error(f"[{symbol}] All retries failed to {context}")
                    return None, stop_price, take_price, 0, algo_ids

    async def place_stop_loss_take_profit(self, symbol, side, amount, stop_price, take_price):
        try:
            params = {
                "instId": symbol,
                "tdMode": "isolated",
                "side": side.lower(),
                "ordType": "oco",
                "sz": str(round(amount, 8)),
                "tpTriggerPx": str(round(take_price, 8)),
                "tpOrdPx": "-1",  # execute at market
                "slTriggerPx": str(round(stop_price, 8)),
                "slOrdPx": "-1",  # execute at market
                "reduceOnly": True,
            }

            self.logger.info(f"[EXCHANGE] Placing OCO order: {params}")

            response = await self.safe_await(
                self.client.private_post_trade_order_algo(params),
                timeout=20,  # збільшений таймаут
                context=f"place_oco-{symbol}"
            )

            if response is None or 'data' not in response:
                self.logger.error(f"[EXCHANGE] Failed to place OCO for {symbol}, response: {response}")
                return None

            algo_id = response['data'][0].get('algoId')
            if not algo_id:
                self.logger.error(f"[EXCHANGE] No algoId returned for OCO {symbol}, full response: {response}")
                return None

            self.logger.info(f"[EXCHANGE] Successfully placed OCO for {symbol}, algoId: {algo_id}")
            return algo_id

        except Exception as e:
            self.logger.error(f"[EXCHANGE] Exception placing OCO for {symbol}: {e}")
            return None


    async def place_trailing_order(self, symbol, side, size, trigger_price, callback_rate, dry_run=False):
        context = f"place_trailing_order for {symbol}"
        retries = 3
        callback_rates = [float(callback_rate), 0.8]
        for attempt in range(retries):
            for attempt_rate in callback_rates[:2 if attempt < retries - 1 else 1]:
                try:
                    inst_id = self.get_inst_id(symbol)
                    if inst_id is None:
                        logger.error(f"[{symbol}] Trailing stop setup aborted: invalid instId")
                        return None

                    adjusted_rate = max(0.3, min(attempt_rate, 5.0))
                    if adjusted_rate != attempt_rate:
                        logger.warning(f"[{symbol}] Adjusted callBackRatio from {attempt_rate:.2f} to {adjusted_rate:.2f}")

                    payload = {
                        "instId": inst_id,
                        "tdMode": "isolated",
                        "side": "sell" if side == "buy" else "buy",
                        "ordType": "move_order_stop",
                        "sz": str(size),
                        "triggerPx": str(trigger_price),
                        "callBackRatio": str(adjusted_rate),
                        "moveTriggerPx": str(trigger_price),
                        "moveOrderPx": "-1"
                    }
                    if self.position_mode == "long_short_mode":
                        payload["posSide"] = "long" if side == "buy" else "short"
                    else:
                        payload.pop("posSide", None)

                    logger.debug(f"[{symbol}] Attempt {attempt + 1}/{retries} - Placing trailing stop order: {json.dumps(payload, indent=2)}")

                    if dry_run:
                        logger.info(f"[{symbol}] Dry run: Simulating trailing stop order placement")
                        return "simulated_algo_id"

                    response = await self.safe_await(
                        self.exchange.private_post_trade_order_algo(params=payload),
                        timeout=5,
                        default=None,
                        context=f"{context}: place_trailing_order"
                    )
                    logger.debug(f"[{symbol}] API response: {json.dumps(response, indent=2)}")

                    if not response or 'data' not in response:
                        logger.error(f"[{symbol}] Invalid response format: {json.dumps(response, indent=2)}")
                        continue

                    if 'code' in response and response['code'] != '0':
                        logger.error(f"[{symbol}] API error: {response.get('msg', 'No message')} (code: {response['code']})")
                        if response['code'] in ['51000', '51100']:
                            logger.warning(f"[{symbol}] Parameter error (code {response['code']}): Likely issue with amount={size} or triggerPx={trigger_price}")
                        if response['code'] in ['50014', '50015']:
                            logger.warning(f"[{symbol}] Invalid ordPx/triggerPx error, retrying")
                        continue

                    if not response['data'] or len(response['data']) == 0:
                        logger.error(f"[{symbol}] No data in response: {json.dumps(response, indent=2)}")
                        continue

                    algo_id = response['data'][0].get('algoId')
                    if not algo_id:
                        logger.error(f"[{symbol}] No algoId in response: {json.dumps(response, indent=2)}")
                        continue

                    logger.info(f"[{symbol}] Trailing stop order placed: algoId={algo_id}, triggerPx={trigger_price}, callBackRatio={adjusted_rate}")
                    return algo_id

                except Exception as e:
                    logger.error(f"Error in {context} (attempt {attempt + 1}/{retries}, callBackRatio={adjusted_rate:.2f}): {str(e)}")
                    if "51000" in str(e):
                        logger.critical(f"[{symbol}] posSide error 51000 – check payload: {json.dumps(payload, indent=2)}")
                    if "50014" in str(e) or "50015" in str(e):
                        logger.warning(f"[{symbol}] Invalid ordPx/triggerPx error, retrying")
                    if attempt < retries - 1:
                        await asyncio.sleep(2)
                    continue

            logger.error(f"[{symbol}] All retries failed to {context}")
            return None

    async def place_stop_loss_only(self, symbol, side, size, stop_price):
        context = f"place_stop_loss_only for {symbol}"
        retries = 3
        for attempt in range(retries):
            try:
                inst_id = self.get_inst_id(symbol)
                if inst_id is None:
                    logger.error(f"[{symbol}] SL-only setup aborted: invalid instId")
                    return None

                close_side = "sell" if side == "buy" else "buy"
                payload = {
                    "instId": inst_id,
                    "tdMode": "isolated",
                    "side": close_side,
                    "ordType": "trigger",
                    "sz": str(size),
                    "triggerPx": str(stop_price),
                    "ordPx": str(stop_price),
                    "triggerPxType": "last"
                }
                if self.position_mode == "long_short_mode":
                    payload["posSide"] = "long" if side == "buy" else "short"
                else:
                    payload.pop("posSide", None)
                if self.position_mode == "net_mode":
                    payload.pop("reduceOnly", None)

                logger.debug(f"[{symbol}] Placing SL-only order: {json.dumps(payload, indent=2)}")
                response = await self.safe_await(
                    self.exchange.private_post_trade_order_algo(params=payload),
                    timeout=5,
                    default=None,
                    context=f"{context}: place_stop_loss_only"
                )
                if not response or 'data' not in response:
                    logger.error(f"[{symbol}] Invalid SL-only response format: {json.dumps(response, indent=2)}")
                    continue
                if 'code' in response and response['code'] != '0':
                    logger.error(f"[{symbol}] SL-only API error: {response.get('msg', 'No message')} (code: {response['code']})")
                    if response['code'] in ['50014', '50015']:
                        logger.warning(f"[{symbol}] Invalid ordPx/triggerPx, retrying with ordPx=market")
                        payload["ordPx"] = "market"
                        response = await self.safe_await(
                            self.exchange.private_post_trade_order_algo(params=payload),
                            timeout=5,
                            default=None,
                            context=f"{context}: place_stop_loss_only (retry)"
                        )
                    continue
                if response and 'data' in response and len(response['data']) > 0:
                    algo_id = response['data'][0].get('algoId')
                    logger.info(f"[{symbol}] SL-only order placed: algoId={algo_id}, stop_price={stop_price}")
                    return algo_id
                else:
                    logger.error(f"[{symbol}] Failed to place SL-only order: Invalid response {json.dumps(response, indent=2)}")
                    return None
            except Exception as e:
                logger.error(f"Error in {context} (attempt {attempt + 1}/{retries}): {str(e)}")
                if "51000" in str(e):
                    logger.critical(f"[{symbol}] posSide error 51000 – check payload: {json.dumps(payload, indent=2)}")
                if "50014" in str(e) or "50015" in str(e):
                    logger.warning(f"[{symbol}] Invalid ordPx/triggerPx error, retrying")
                if attempt < retries - 1:
                    await asyncio.sleep(2)
                else:
                    logger.error(f"[{symbol}] All retries failed to {context}")
                    return None

    async def close(self):
        context = "close_exchange_connection"
        try:
            await self.safe_await(
                self.exchange.close(),
                timeout=5,
                default=None,
                context=context
            )
            logger.info("Exchange connection closed")
        except Exception as e:
            logger.error(f"Error in {context}: {str(e)}")

    async def cancel_algo_order(self, algo_id):
        """
        Скасування алгоритмічного ордера (SL/TP/TS) за ID.
        """
        context = f"cancel_algo_order {algo_id}"
        try:
            payload = {
                "algoId": [algo_id]  # OKX очікує список
            }
            logger.info(f"[exchange] Cancelling algo order: {algo_id}")
            response = await self.safe_await(
                self.exchange.private_post_trade_cancel_algo_order(params=payload),
                timeout=5,
                default=None,
                context=context
            )

            if isinstance(response, dict) and response.get("code") != "0":
                logger.error(f"[exchange] Failed to cancel algo order {algo_id}: {response.get('msg')}")
            else:
                logger.info(f"[exchange] Algo order cancelled successfully: {algo_id}")

            return response

        except Exception as e:
            logger.error(f"[exchange] Exception in {context}: {str(e)}")
            return None

    async def get_symbol_prices(self):
        prices = {}
        try:
            for symbol in self.valid_pairs:
                ticker = await self.safe_await(
                    self.exchange.fetch_ticker(symbol),
                    timeout=5,
                    default=None,
                    context=f"fetch_ticker {symbol}"
                )
                if ticker is not None:
                    last_price = ticker.get('last')
                    if last_price is not None:
                        prices[symbol] = last_price
                    else:
                        self.logger.warning(f"[{symbol}] No last price in ticker")
        except Exception as e:
            self.logger.error(f"[exchange] Failed to fetch symbol prices: {str(e)}")
        return prices

__all__ = ["AsyncExchange", "safe_await"]