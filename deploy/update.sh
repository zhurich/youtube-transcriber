#!/usr/bin/env bash
# Обновление кода и зависимостей: sudo bash deploy/update.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_USER="${SERVICE_USER:-transcriber}"
SERVICE_NAME="transcriber-bot"

if [[ $EUID -ne 0 ]]; then
  echo "Запустите через sudo: sudo bash deploy/update.sh" >&2
  exit 1
fi

if [[ -d "$APP_DIR/.git" ]]; then
  git -C "$APP_DIR" pull --ff-only
fi

# yt-dlp обновляем всегда: YouTube регулярно меняет отдачу плейлистов и подписи URL
"$APP_DIR/.venv/bin/pip" install --upgrade --quiet -r "$APP_DIR/requirements.txt" yt-dlp

chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR"
systemctl restart "$SERVICE_NAME"
systemctl --no-pager status "$SERVICE_NAME" | head -n 12
