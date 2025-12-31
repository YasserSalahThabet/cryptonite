#!/bin/zsh
set -e

cd "$HOME/Desktop/crypto_signal_bot.py"

STREAMLIT="/Library/Frameworks/Python.framework/Versions/3.12/bin/streamlit"

"$STREAMLIT" run app.py --server.port 8502

