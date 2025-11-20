# TradeVision

## MEXC Futures Signal Bot

A Python bot for **MEXC Futures** that fetches live crypto prices, calculates EMA, RSI, Bollinger Bands,  
and provides real-time trading signals with suggested entry, stop-loss, and take-profit levels.

## Features

- Fetch live prices from MEXC Futures API  
- Technical indicators: EMA, RSI, Bollinger Bands  
- Generates real-time trading signals: **LONG**, **SHORT**, or **NO SIGNAL**  
- Calculates estimated position size based on balance and leverage  
- Auto-refresh option to continuously scan the market  
- Color-coded console output using `colorama`  

## Supported Coins

Supports a wide range of coins such as: BTC_USDT, ETH_USDT, SOL_USDT, XRP_USDT, ADA_USDT, DOGE_USDT, and many more.  
(*Full list in the code comments*)


## Installation

1. Clone the repository:  
git clone https://github.com/Sahab00/TradeVision.git
cd TradeVision

2. Install required packages
pip install -r requirements.txt

## Run the app in the browser
1.Run the command 
python app.py
2.Then open your browser and go to: http://localhost:5000

## Demoo for Using the TradeVision Bot
<video src="assets/demo.mp4" width="600" controls></video>



