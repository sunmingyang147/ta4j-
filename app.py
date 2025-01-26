import json
import requests
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Binance API endpoint for Klines
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

def fetch_klines(symbol, interval, start_time, end_time):
    """
    Fetch Kline data from Binance API in batches to handle large datasets.
    :param symbol: Trading pair (e.g., BTCUSDT)
    :param interval: Kline interval (e.g., '3m', '5m')
    :param start_time: Start time in milliseconds
    :param end_time: End time in milliseconds
    :return: List of Kline data
    """
    klines = []
    limit = 1000  # Max limit per request as per Binance API
    current_start_time = start_time

    while current_start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start_time,
            "endTime": end_time,
            "limit": limit,
        }
        response = requests.get(BINANCE_KLINES_URL, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching data from Binance: {response.text}")
        data = response.json()
        if not data:
            break
        klines.extend(data)
        # Update the start time for the next batch
        current_start_time = data[-1][6] + 1  # Use the last Kline's close time + 1ms

        # Print progress
        print(f"Fetched {len(data)} klines, total so far: {len(klines)}")

    return klines

@app.route('/get_klines', methods=['GET'])
def get_klines():
    """
    Flask endpoint to fetch Kline data.
    Parameters:
    - symbol: Trading pair (e.g., BTCUSDT)
    - interval: Kline interval (e.g., '3m', '5m')
    - start_time: Start time in 'YYYY-MM-DD HH:MM:SS' format
    - end_time: End time in 'YYYY-MM-DD HH:MM:SS' format
    """
    try:
        # Parse query parameters
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '3m')  # Default to 3m interval
        start_time_str = request.args.get('start_time', '2021-01-01 00:00:00')  # Default to 2021-01-01
        end_time_str = request.args.get('end_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))  # Default to now

        # Convert to timestamps
        start_time = int(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
        end_time = int(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)

        if start_time >= end_time:
            return jsonify({"error": "start_time must be earlier than end_time"}), 400

        # Fetch Kline data
        klines = fetch_klines(symbol, interval, start_time, end_time)

        # Save data to JSON file
        output_file = f"{symbol}_{interval}_{start_time_str.replace(':', '-')}_{end_time_str.replace(':', '-')}.json"
        with open(output_file, 'w') as f:
            json.dump(klines, f, indent=4)

        return jsonify({"message": "Data fetched successfully", "file": output_file, "data_length": len(klines)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)