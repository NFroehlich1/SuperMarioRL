#!/bin/bash
# Sync-Skript: Synchronisiert lokale Dateien mit RunPod Server

RUNPOD_HOST="198.13.252.15"
RUNPOD_PORT="15267"
RUNPOD_USER="root"
RUNPOD_KEY="~/.ssh/id_ed25519"
RUNPOD_PATH="/workspace/SuperMarioRL"

echo "🔄 Synchronisiere Projekt mit RunPod..."

# Dateien die synchronisiert werden sollen
FILES_TO_SYNC=(
    "app.py"
    "train.py"
    "test.py"
    "requirements.txt"
    "setup_runpod.sh"
    "README.md"
    ".gitignore"
)

# Erstelle Verzeichnis auf RunPod falls nicht vorhanden
ssh -p $RUNPOD_PORT -i $RUNPOD_KEY $RUNPOD_USER@$RUNPOD_HOST "mkdir -p $RUNPOD_PATH"

# Synchronisiere Dateien
for file in "${FILES_TO_SYNC[@]}"; do
    if [ -f "$file" ]; then
        echo "📤 Übertrage $file..."
        scp -P $RUNPOD_PORT -i $RUNPOD_KEY "$file" $RUNPOD_USER@$RUNPOD_HOST:$RUNPOD_PATH/
    else
        echo "⚠️  $file nicht gefunden, überspringe..."
    fi
done

echo "✅ Synchronisation abgeschlossen!"
echo ""
echo "Verbinde dich jetzt mit:"
echo "  ssh -p $RUNPOD_PORT -i $RUNPOD_KEY $RUNPOD_USER@$RUNPOD_HOST"
echo ""
echo "Dann auf dem Server:"
echo "  cd $RUNPOD_PATH"
echo "  ./setup_runpod.sh"

