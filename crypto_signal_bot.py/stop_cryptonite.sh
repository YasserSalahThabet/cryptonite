#!/bin/zsh
set +e
cd "$(dirname "$0")"

echo "Stopping Cryptonite..."

pkill -f "python3 signal_bot.py" 2>/dev/null
pkill -f "streamlit run app.py" 2>/dev/null
pkill -f "python3 -m streamlit run app.py" 2>/dev/null

echo "Done."
