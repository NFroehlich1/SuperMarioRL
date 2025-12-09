#!/bin/bash
# Port-Forwarding und SSH-Verbindung zu RunPod

RUNPOD_HOST="198.13.252.15"
RUNPOD_PORT="15267"
RUNPOD_USER="root"
RUNPOD_KEY="~/.ssh/id_ed25519"
LOCAL_PORT="8501"
REMOTE_PORT="8501"

echo "🔗 Verbinde mit RunPod..."
echo ""
echo "Optionen:"
echo "1. SSH-Verbindung (interaktiv)"
echo "2. Port-Forwarding (Streamlit lokal nutzen)"
echo "3. Beides"
echo ""
read -p "Wähle Option (1/2/3): " option

case $option in
    1)
        echo "🔌 Öffne SSH-Verbindung..."
        ssh -p $RUNPOD_PORT -i $RUNPOD_KEY $RUNPOD_USER@$RUNPOD_HOST
        ;;
    2)
        echo "🔌 Port-Forwarding aktiviert..."
        echo "Streamlit läuft auf RunPod, erreichbar unter: http://localhost:$LOCAL_PORT"
        echo "Drücke Ctrl+C zum Beenden"
        ssh -p $RUNPOD_PORT -i $RUNPOD_KEY -L $LOCAL_PORT:localhost:$REMOTE_PORT -N $RUNPOD_USER@$RUNPOD_HOST
        ;;
    3)
        echo "🔌 Öffne SSH mit Port-Forwarding..."
        echo "Streamlit erreichbar unter: http://localhost:$LOCAL_PORT"
        ssh -p $RUNPOD_PORT -i $RUNPOD_KEY -L $LOCAL_PORT:localhost:$REMOTE_PORT $RUNPOD_USER@$RUNPOD_HOST
        ;;
    *)
        echo "❌ Ungültige Option"
        exit 1
        ;;
esac

