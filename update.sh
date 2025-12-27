#!/bin/bash
# Nuke the old file
rm app.py
# Download the fresh fixed one from your own github (replace with your raw link)
curl -L https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/app_fixed.py -o app.py
# Kill the old process
pkill -9 python
# Start the new one
python3 app.py &
