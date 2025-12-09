#!/bin/bash
# Setup-Skript für RunPod Server

echo "🚀 Setting up Super Mario RL Trainer on RunPod..."

# Prüfe CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA verfügbar:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "⚠️  CUDA nicht gefunden - verwende CPU"
fi

# Python Version prüfen
python3 --version

# Virtual Environment erstellen
if [ ! -d "venv" ]; then
    echo "📦 Erstelle Virtual Environment..."
    python3 -m venv venv
fi

# Aktivieren
source venv/bin/activate

# Dependencies installieren
echo "📥 Installiere Dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup abgeschlossen!"
echo ""
echo "Starte die App mit:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py --server.port 8501 --server.address 0.0.0.0"

