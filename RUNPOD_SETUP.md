# RunPod Setup Anleitung

## 🔑 SSH-Key auf RunPod hinterlegen

**WICHTIG:** Bevor du dich verbinden kannst, musst du deinen öffentlichen SSH-Key auf RunPod hinterlegen:

1. **Öffne deinen öffentlichen Key:**
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

2. **Kopiere den gesamten Key** (beginnt mit `ssh-ed25519...`)

3. **Auf RunPod:**
   - Gehe zu deinem Pod im RunPod Dashboard
   - Öffne "Settings" oder "SSH Keys"
   - Füge den öffentlichen Key hinzu
   - Oder per Web-Terminal:
     ```bash
     mkdir -p ~/.ssh
     echo "DEIN_PUBLIC_KEY_HIER" >> ~/.ssh/authorized_keys
     chmod 600 ~/.ssh/authorized_keys
     ```

## 📤 Projekt auf RunPod übertragen

### Option 1: Sync-Skript (Empfohlen)
```bash
chmod +x sync_to_runpod.sh
./sync_to_runpod.sh
```

### Option 2: Manuell per SCP
```bash
scp -P 15267 -i ~/.ssh/id_ed25519 -r . root@198.13.252.15:/workspace/SuperMarioRL
```

### Option 3: Git Clone (auf RunPod)
```bash
ssh -p 15267 -i ~/.ssh/id_ed25519 root@198.13.252.15
cd /workspace
git clone https://github.com/NFroehlich1/SuperMarioRL.git
```

## 🚀 Setup auf RunPod

1. **SSH-Verbindung:**
   ```bash
   ssh -p 15267 -i ~/.ssh/id_ed25519 root@198.13.252.15
   ```

2. **Zum Projekt:**
   ```bash
   cd /workspace/SuperMarioRL
   ```

3. **Setup ausführen:**
   ```bash
   chmod +x setup_runpod.sh
   ./setup_runpod.sh
   ```

4. **Virtual Environment aktivieren:**
   ```bash
   source venv/bin/activate
   ```

5. **Streamlit starten:**
   ```bash
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

## 🔗 Lokale Verbindung (Port-Forwarding)

Um die RunPod-App lokal zu nutzen:

```bash
chmod +x runpod_connection.sh
./runpod_connection.sh
# Wähle Option 2 oder 3
```

Dann öffne: `http://localhost:8501`

## 📝 Workflow: Lokal entwickeln, auf RunPod trainieren

1. **Lokal entwickeln** (auf MacBook)
2. **Synchronisieren:**
   ```bash
   ./sync_to_runpod.sh
   ```
3. **Auf RunPod trainieren:**
   ```bash
   ssh -p 15267 -i ~/.ssh/id_ed25519 root@198.13.252.15
   cd /workspace/SuperMarioRL
   source venv/bin/activate
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```
4. **Lokal zugreifen:**
   ```bash
   ./runpod_connection.sh  # Option 2
   ```

## ✅ CUDA prüfen

```bash
ssh -p 15267 -i ~/.ssh/id_ed25519 root@198.13.252.15 "nvidia-smi"
```

## 🐛 Troubleshooting

### Permission denied
- Prüfe, ob SSH-Key auf RunPod hinterlegt ist
- Prüfe Key-Berechtigungen: `chmod 600 ~/.ssh/id_ed25519`

### Port bereits belegt
```bash
ssh -p 15267 -i ~/.ssh/id_ed25519 root@198.13.252.15 "lsof -i :8501"
```

### CUDA nicht erkannt
```bash
ssh -p 15267 -i ~/.ssh/id_ed25519 root@198.13.252.15 "python3 -c 'import torch; print(torch.cuda.is_available())'"
```
