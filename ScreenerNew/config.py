import logging
import os

# API ключи для OKX (реальний API)
API_KEY = "daab8f9b-5470-4419-a0b1-1e0a845108b3"
SECRET = "5586EAE4FF88B4308D191B4DA6CFD31F"
PASSWORD = "234ghhhuZ!"

# Список пар для торгівлі
PAIRS = [
"OP/USDT:USDT",
"SOL/USDT:USDT",
"INJ/USDT:USDT",
"ADA/USDT:USDT",
"AVAX/USDT:USDT",
"LINK/USDT:USDT",
"APT/USDT:USDT",
"TIA/USDT:USDT",
"SUI/USDT:USDT",
"STRK/USDT:USDT"

]

# Параметри бота
LEVERAGE = 50
VOLATILITY_THRESHOLD = 0.0001  # Було 0.0003
MIN_MARGIN_PER_TRADE = 0.01
MIN_POSITION_HOLD_SEC = 10
USE_EXCHANGE_STOPLOSS = True
USE_EXCHANGE_TRAILING = True
USE_BOTH_SL_TRAILING = True
MAX_OPEN_POSITIONS = 3
TRADE_MARGIN_RATIO = 0.2

# Множники для SL/TP
FIXED_SL_PCT = 0.7  
FIXED_TP_PCT = 0.2  

# Параметри трейлінг-стопа
TRAILING_OKX_CALLBACK_RATE = 0.7
TRAILING_TRIGGER_RATIO = 10
TRAILING_SL_PERCENT = 0.99  # 0.8% — оптимально для altcoins

# Пороги для фільтрації
MIN_VOLUME_24H = 500_000  # Було 1_000_000
MAX_SPREAD_PCT = 1.0  # Було 0.1

# Пороги для сигналів
BUY_THRESHOLD = 0.35
SELL_THRESHOLD = 0.55
EMA_PROXIMITY = 0.10
MIN_PROB_THRESHOLD = 0.78
CLOSE_ON_SIGNAL_CHANGE = True

# Логер

def get_logger(name, file_path=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Важливо: не передавати логи вище

    # Очистити всі попередні хендлери, щоб не дублювалося
    if logger.hasHandlers():
        logger.handlers.clear()

    # Створити директорію, якщо потрібно
    log_dir = "C:/ScreenerNew/logs"
    os.makedirs(log_dir, exist_ok=True)
    if not file_path:
        file_path = os.path.join(log_dir, f"{name}.log")

    # Лог у файл
    file_handler = logging.FileHandler(file_path, mode='a')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
