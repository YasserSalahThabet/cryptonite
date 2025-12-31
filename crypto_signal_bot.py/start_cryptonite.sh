#!/bin/zsh
set -e
cd "$(dirname "$0")"

UI_PORT=8502

echo "=== Cryptonite Starter ==="
echo "Project: $(pwd)"

mkdir -p logs

# Kill anything using the UI port
if lsof -ti tcp:$UI_PORT >/dev/null 2>&1; then
  echo "Port $UI_PORT is in use. Killing..."
  lsof -ti tcp:$UI_PORT | xargs kill -9
fi

echo "Starting signal bot (background)..."
nohup python3 signal_bot.py > logs/logs_signal_bot.txt 2>&1 &

echo "Starting dashboard: http://localhost:$UI_PORT"
python3 -m streamlit run app.py --server.port $UI_PORT
