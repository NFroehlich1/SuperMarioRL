# 🍄 Super Mario Bros RL Trainer

Ein Reinforcement Learning Projekt zum Trainieren eines KI-Agenten, der Super Mario Bros spielt. Das Projekt bietet eine benutzerfreundliche Web-Oberfläche für Training, Visualisierung und Benchmarking verschiedener RL-Algorithmen.

## ✨ Features

- **Web-Interface**: Interaktive Streamlit-App für Training und Visualisierung
- **Mehrere RL-Algorithmen**: PPO, DQN, A2C mit optimierten Hyperparametern
- **Progress-Based Reward Shaping**: Intelligente Belohnungsfunktion fokussiert auf Level-Completion
- **Hardware-Optimierung**: Parallele Environments für schnelleres Training
- **Modell-Management**: Automatisches Speichern des besten Modells, Fortsetzen von Trainings
- **Live-Visualisierung**: Controller-Input-Anzeige während des Spielens
- **TensorBoard Integration**: Detaillierte Trainings-Metriken

## 📋 Voraussetzungen

- Python 3.8 oder höher
- macOS, Linux oder Windows
- Für macOS: `ffmpeg` und `sdl2` (via Homebrew: `brew install ffmpeg sdl2`)

## 🚀 Installation

1. **Repository klonen oder herunterladen**

2. **Virtuelle Umgebung erstellen (empfohlen):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Abhängigkeiten installieren:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Nutzung

### Web-Interface (Empfohlen)

Starte die interaktive Web-App:

```bash
streamlit run app.py
```

Die App öffnet sich automatisch im Browser unter `http://localhost:8501`.

#### Features im Web-Interface:

**Training Tab:**
- Algorithmus auswählen (PPO, DQN, A2C)
- Trainingsschritte konfigurieren
- Hardware-Optimierung (parallele Environments)
- Bestehendes Modell weiter trainieren
- Live-Fortschrittsanzeige

**Play / Watch Tab:**
- Agent beim Spielen zusehen
- Controller-Input in Echtzeit anzeigen
- Bestes Modell oder Standard-Modell wählen

### Terminal-Skripte

Alternativ können die Skripte direkt ausgeführt werden:

**Training:**
```bash
python train.py
```

**Testen:**
```bash
python test.py
```

## 🎯 Reward-Shaping Strategie

Das Projekt verwendet eine **Progress-Based Reward Function**, die darauf ausgelegt ist, Level-Completion zu erreichen:

- **Progress Reward**: Belohnung nur für neuen Fortschritt (neue maximale X-Position)
- **Stagnation Penalty**: Strafe für langes Verharren ohne Fortschritt
- **Level Completion**: Massive Belohnung (+1000) für das Erreichen des Ziels
- **Death Penalty**: Strafe für Tod, aber ausgewogen, um Exploration zu ermöglichen

Diese Strategie verhindert "Reward Hacking" (z.B. Hin-und-Herlaufen) und zwingt den Agenten, echten Fortschritt zu machen.

## ⚙️ Konfiguration

### Trainingsparameter

- **Trainingsschritte**: Empfohlen 500.000+ für Level-Completion
- **Parallele Environments**: 1-8 (mehr = schneller, aber mehr RAM)
- **Subprocess-Modus**: Schneller, aber mehr RAM-Verbrauch

### Algorithmus-Empfehlungen

- **PPO**: Meist am besten für Super Mario Bros (empfohlen)
- **DQN**: Gut für Sample-Efficiency
- **A2C**: Schneller, aber oft weniger stabil

## 📊 TensorBoard

Um detaillierte Trainings-Metriken zu sehen:

```bash
tensorboard --logdir logs_web/
```

Öffne dann `http://localhost:6006` im Browser.

## 📁 Projektstruktur

```
SuperMario/
├── app.py              # Streamlit Web-Interface
├── train.py            # Terminal-Training-Skript
├── test.py             # Terminal-Test-Skript
├── requirements.txt    # Python-Abhängigkeiten
├── README.md           # Diese Datei
├── train_web/          # Gespeicherte Modelle
└── logs_web/           # TensorBoard Logs
```

## 🔧 Troubleshooting

### OverflowError: Python integer out of bounds for uint8

Dieser Fehler wurde bereits in den installierten Bibliotheken behoben. Falls er auftritt, starte den Streamlit-Server neu.

### Training dauert sehr lange

Das NES-Environment ist CPU-intensiv. Optimierungen:
- Mehr parallele Environments verwenden
- Subprocess-Modus aktivieren
- Frame-Skipping ist bereits auf 4 gesetzt (optimal)

### Agent lernt nicht / läuft nur nach rechts

- Starte ein **neues Training** mit den aktuellen Reward-Funktionen
- Alte Modelle haben möglicherweise "falsches" Verhalten gelernt
- Verwende mindestens 500.000 Trainingsschritte

## 🎓 Technische Details

### Environment Wrappers

- **SkipFrame**: Verarbeitet jeden 4. Frame (beschleunigt Training)
- **GrayScaleObservation**: Reduziert Dimensionalität
- **ResizeObservation**: 84x84 Pixel (Standard für Deep RL)
- **VecFrameStack**: Stackt 4 Frames für Bewegungsinformation
- **RewardShaping**: Custom Reward-Funktion für Level-Completion

### Hyperparameter

Die Hyperparameter sind für Super Mario Bros optimiert:
- PPO: `learning_rate=2.5e-4`, `n_steps=2048`, `n_epochs=10`
- DQN: `buffer_size=100000`, `learning_starts=5000`
- A2C: `learning_rate=7e-4`, `n_steps=5`

## 📝 Lizenz

Dieses Projekt verwendet:
- `gym-super-mario-bros` (MIT License)
- `stable-baselines3` (MIT License)
- `nes-py` (MIT License)

## 🤝 Beitragen

Verbesserungen und Pull Requests sind willkommen!

## 📧 Support

Bei Problemen oder Fragen, öffne ein Issue im Repository.

---

**Viel Erfolg beim Trainieren deines Super Mario Agenten! 🍄**
