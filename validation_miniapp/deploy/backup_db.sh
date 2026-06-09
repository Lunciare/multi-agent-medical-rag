#!/usr/bin/env bash
# Consistent, timestamped backup of the SQLite DB using sqlite3's online .backup
# (safe even while the app is writing). Simple count-based retention.
#
# Optional automation (systemd timer) instead of cron — see the comment block at
# the bottom. Manual invocation is the baseline; run it at least once before close.
set -euo pipefail

DB_PATH="${DB_PATH:-/home/melodiz/miniapp-data/app.sqlite3}"
BACKUP_DIR="${BACKUP_DIR:-/home/melodiz/miniapp-data/backups}"
RETENTION="${RETENTION:-14}"   # keep this many most-recent backups

if [ ! -f "${DB_PATH}" ]; then
    echo "[backup_db] ERROR: DB not found at ${DB_PATH}" >&2
    exit 1
fi

mkdir -p "${BACKUP_DIR}"
STAMP="$(date +%Y%m%d-%H%M%S)"
DEST="${BACKUP_DIR}/app-${STAMP}.sqlite3"

sqlite3 "${DB_PATH}" ".backup '${DEST}'"
echo "[backup_db] wrote ${DEST}"

count="$(find "${BACKUP_DIR}" -maxdepth 1 -type f -name 'app-*.sqlite3' | wc -l | tr -d ' ')"
if [ "${count}" -gt "${RETENTION}" ]; then
    remove=$((count - RETENTION))
    echo "[backup_db] pruning ${remove} old backup(s) (keeping ${RETENTION})"
    find "${BACKUP_DIR}" -maxdepth 1 -type f -name 'app-*.sqlite3' | sort | head -n "${remove}" \
        | while IFS= read -r old; do
            rm -f -- "${old}"
            echo "[backup_db] removed ${old}"
        done
fi

# --- Optional: run hourly via a systemd timer instead of by hand ---------------
# /etc/systemd/system/miniapp-backup.service
#   [Unit]
#   Description=Backup validation Mini App DB
#   [Service]
#   Type=oneshot
#   User=melodiz
#   Environment=DB_PATH=/home/melodiz/miniapp-data/app.sqlite3
#   ExecStart=/home/melodiz/miniapp/deploy/backup_db.sh
# /etc/systemd/system/miniapp-backup.timer
#   [Unit]
#   Description=Hourly Mini App DB backup
#   [Timer]
#   OnCalendar=hourly
#   Persistent=true
#   [Install]
#   WantedBy=timers.target
# Then: sudo systemctl enable --now miniapp-backup.timer
