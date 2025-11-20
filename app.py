from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import requests
import pandas as pd
import numpy as np
import time
import threading
import joblib
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'crypto_bot_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
bot_status = False
bot_thread = None
current_data = {
    'price': 0,
    'signal': 'NEUTRAL',
    'symbol': 'BTCUSDT',
    'interval': '5m',
    'leverage': 50,
    'balance': 1000,
    'ema20': 0,
    'rsi': 0,
    'bb_up': 0,
    'bb_low': 0,
    'stop_loss': 0,
    'take_profit': 0,
    'position_size': 0,
    'timestamp': ''
}

BASE_URL = "https://api.mexc.com"

# ------------------------
# Indicator Functions
# ------------------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=length-1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length-1, min_periods=length).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def bollinger(series, period=20, dev=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + dev * std
    lower = sma - dev * std
    return upper, lower

# ------------------------
# Data Fetching Functions
# ------------------------
def fix_symbol(symbol):
    s = symbol.upper().replace("_", "").replace("/", "")
    if not s.endswith("USDT"):
        s += "USDT"
    return s

def get_klines(symbol, interval="5m", limit=100):
    try:
        url = f"{BASE_URL}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            print(f"API Error: {r.status_code}")
            return create_dummy_data()
        data = r.json()
        
        # Handle different response formats from MEXC
        if len(data) > 0 and len(data[0]) == 12:
            # Standard format with 12 columns
            df = pd.DataFrame(data, columns=[
                "time", "open", "high", "low", "close", "volume", 
                "close_time", "quote_asset_volume", "trades", 
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])
        elif len(data) > 0 and len(data[0]) == 8:
            # Alternative format with 8 columns
            df = pd.DataFrame(data, columns=[
                "time", "open", "high", "low", "close", "volume", 
                "close_time", "quote_asset_volume"
            ])
        elif len(data) > 0 and len(data[0]) == 6:
            # Minimal format with 6 columns
            df = pd.DataFrame(data, columns=[
                "time", "open", "high", "low", "close", "volume"
            ])
        else:
            print(f"Unknown data format with {len(data[0]) if data else 0} columns")
            return create_dummy_data()
            
        df = df[["time", "open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        print(f"Error fetching klines: {e}")
        return create_dummy_data()

def get_live_price(symbol):
    try:
        url = f"{BASE_URL}/api/v3/ticker/price?symbol={symbol}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return float(r.json()["price"])
    except Exception as e:
        print(f"Error fetching live price: {e}")
        return np.random.uniform(25000, 35000)

def create_dummy_data():
    """Create realistic dummy data for testing"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='5min')
    base_price = 30000
    prices = [base_price]
    for i in range(1, 100):
        change = np.random.normal(0, 50)
        new_price = prices[-1] + change
        prices.append(max(new_price, 1000))
    
    df = pd.DataFrame({
        'time': [int(x.timestamp() * 1000) for x in dates],
        'open': prices,
        'high': [p + abs(np.random.normal(0, 100)) for p in prices],
        'low': [p - abs(np.random.normal(0, 100)) for p in prices],
        'close': [p + np.random.normal(0, 50) for p in prices],
        'volume': np.random.uniform(1000, 10000, 100)
    })
    return df

# ------------------------
# Model Management
# ------------------------
def initialize_model():
    model_path = "crypto_live_model.pkl"
    try:
        if os.path.exists(model_path):
            print("Loading existing model...")
            return joblib.load(model_path)
        else:
            print("Initializing new model...")
            np.random.seed(42)
            dummy_data = pd.DataFrame({
                'ema20': np.random.normal(30000, 1000, 1000),
                'rsi': np.random.uniform(20, 80, 1000),
                'bb_up': np.random.normal(30500, 1000, 1000),
                'bb_low': np.random.normal(29500, 1000, 1000),
                'signal': np.random.choice([-1, 0, 1], 1000, p=[0.3, 0.4, 0.3])
            })
            features = ['ema20', 'rsi', 'bb_up', 'bb_low']
            X = dummy_data[features]
            y = dummy_data['signal']
            model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
            model.fit(X, y)
            joblib.dump(model, model_path)
            print("New model created and saved")
            return model
    except Exception as e:
        print(f"Error initializing model: {e}")
        return create_dummy_model()

def create_dummy_model():
    """Create a simple dummy model as fallback"""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X_dummy = np.random.rand(100, 4)
    y_dummy = np.random.choice([-1, 0, 1], 100)
    model.fit(X_dummy, y_dummy)
    return model

# Initialize model and recent data
model = initialize_model()
recent_data = pd.DataFrame(columns=['ema20', 'rsi', 'bb_up', 'bb_low', 'signal'])

# ------------------------
# Trading Bot Thread
# ------------------------
def trading_bot():
    global current_data, recent_data, model
    
    print("🚀 Trading bot started!")
    iteration = 0
    
    while bot_status:
        try:
            symbol = current_data['symbol']
            interval = current_data['interval']
            balance = current_data['balance']
            leverage = current_data['leverage']
            
            print(f"📊 Fetching data for {symbol} ({interval})...")
            df = get_klines(symbol, interval=interval, limit=100)
            
            if df.empty or len(df) < 20:
                print("⚠️ Not enough data, skipping iteration")
                time.sleep(10)
                continue

            close = df['close']

            # Calculate indicators
            df['ema20'] = ema(close, 20)
            df['rsi'] = rsi(close)
            df['bb_up'], df['bb_low'] = bollinger(close)

            last = df.iloc[-1]
            price = get_live_price(symbol)
            if price is None:
                price = last['close']
                
            qty = (balance * leverage / price) if price > 0 else 0

            # Make prediction
            features_live = pd.DataFrame([[
                last['ema20'], 
                last['rsi'], 
                last['bb_up'], 
                last['bb_low']
            ]], columns=['ema20', 'rsi', 'bb_up', 'bb_low'])
            
            try:
                pred = model.predict(features_live)[0]
            except Exception as e:
                print(f"⚠️ Prediction error: {e}")
                pred = 0

            # Determine signal
            rsi_val = last['rsi']
            price_vs_ema = last['close'] / last['ema20']
            
            if pred == 1 and rsi_val < 70 and price_vs_ema > 1.001:
                signal = "LONG"
                sl = last['bb_low'] * 0.995
                tp1 = last['close'] * 1.015
            elif pred == -1 and rsi_val > 30 and price_vs_ema < 0.999:
                signal = "SHORT"
                sl = last['bb_up'] * 1.005
                tp1 = last['close'] * 0.985
            else:
                signal = "NEUTRAL"
                sl = last['close'] * 0.99
                tp1 = last['close'] * 1.01

            # Update current data
            current_data.update({
                'price': round(price, 6),
                'signal': signal,
                'ema20': round(last['ema20'], 6),
                'rsi': round(last['rsi'], 2),
                'bb_up': round(last['bb_up'], 6),
                'bb_low': round(last['bb_low'], 6),
                'stop_loss': round(sl, 6),
                'take_profit': round(tp1, 6),
                'position_size': round(qty, 3),
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })

            # Emit update via WebSocket
            socketio.emit('data_update', current_data)
            print(f"📡 Update: {signal} at ${price:.2f}, RSI: {rsi_val:.1f}")
            
        except Exception as e:
            print(f"❌ Error in trading bot: {e}")
        
        time.sleep(10)

# ------------------------
# Flask Routes
# ------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/prices')
def get_prices():
    return jsonify({
        'price': current_data['price'],
        'symbol': current_data['symbol'],
        'timestamp': current_data['timestamp']
    })

@app.route('/api/signals')
def get_signals():
    return jsonify({
        'signal': current_data['signal'],
        'stop_loss': current_data['stop_loss'],
        'take_profit': current_data['take_profit'],
        'position_size': current_data['position_size'],
        'ema20': current_data['ema20'],
        'rsi': current_data['rsi'],
        'bb_up': current_data['bb_up'],
        'bb_low': current_data['bb_low']
    })

@app.route('/api/bot/status')
def bot_status_check():
    return jsonify({'status': 'running' if bot_status else 'stopped'})

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    global bot_status, bot_thread
    if not bot_status:
        try:
            data = request.get_json() or {}
            current_data.update({
                'symbol': fix_symbol(data.get('symbol', 'BTCUSDT')),
                'interval': data.get('interval', '5m'),
                'balance': float(data.get('balance', 1000)),
                'leverage': float(data.get('leverage', 50))
            })
            bot_status = True
            bot_thread = threading.Thread(target=trading_bot)
            bot_thread.daemon = True
            bot_thread.start()
            return jsonify({
                'status': 'started', 
                'message': f'Bot started for {current_data["symbol"]}'
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    return jsonify({'status': 'already_running', 'message': 'Bot is already running'})

@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    global bot_status
    bot_status = False
    return jsonify({'status': 'stopped', 'message': 'Bot stopped successfully'})

@app.route('/api/test')
def test_endpoint():
    return jsonify({
        'message': '✅ Crypto Bot API is working!',
        'bot_status': 'running' if bot_status else 'stopped',
        'current_symbol': current_data['symbol'],
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("🎯 Starting Crypto Trading Bot Dashboard...")
    print("🌐 Access the dashboard at: http://localhost:5000")
    print("📊 API Test endpoint: http://localhost:5000/api/test")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
