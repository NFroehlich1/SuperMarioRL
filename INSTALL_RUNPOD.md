# RunPod Installation - Python 3.12 Fix

## Problem
`gym==0.21.0` ist nicht mit Python 3.12 kompatibel.

## Lösung 1: Verwende Python 3.11 (Empfohlen)

```bash
# Auf RunPod: Python 3.11 installieren und verwenden
apt-get update
apt-get install -y python3.11 python3.11-venv python3.11-dev

# Virtual Environment mit Python 3.11 erstellen
python3.11 -m venv venv
source venv/bin/activate

# Dann installieren
pip install --upgrade pip
pip install -r requirements.txt
```

## Lösung 2: Verwende neuere gym-Version

```bash
cd /workspace/SuperMarioRL
source venv/bin/activate

# Installiere zuerst Build-Tools
pip install wheel "setuptools<70.0.0"

# Installiere gym aus Wheel (wenn verfügbar)
pip install --only-binary gym gym==0.21.0 || pip install gym==0.26.2

# Dann restliche Dependencies
pip install gym-super-mario-bros==7.4.0 nes-py==8.2.1
pip install stable-baselines3[extra]==1.8.0
pip install torch opencv-python matplotlib streamlit tensorboard
```

## Lösung 3: Manuelle Installation (Schritt für Schritt)

```bash
cd /workspace/SuperMarioRL
source venv/bin/activate

# 1. Build-Tools
pip install --upgrade pip wheel
pip install "setuptools<70.0.0"

# 2. Versuche gym aus Wheel
pip install --only-binary :all: gym==0.21.0 || \
pip install gym==0.26.2 || \
pip install gymnasium

# 3. Restliche Pakete
pip install gym-super-mario-bros==7.4.0
pip install nes-py==8.2.1
pip install "stable-baselines3[extra]==1.8.0"
pip install torch opencv-python matplotlib streamlit tensorboard
```

## Schnellste Lösung (Kopieren & Einfügen)

```bash
cd /workspace/SuperMarioRL && \
python3.11 -m venv venv || python3 -m venv venv && \
source venv/bin/activate && \
pip install --upgrade pip wheel "setuptools<70.0.0" && \
pip install gym==0.26.2 gym-super-mario-bros==7.4.0 nes-py==8.2.1 && \
pip install "stable-baselines3[extra]==1.8.0" torch opencv-python matplotlib streamlit tensorboard
```

## Nach erfolgreicher Installation

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

