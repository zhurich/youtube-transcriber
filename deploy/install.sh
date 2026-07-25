#!/usr/bin/env bash
# Установка бота как systemd-сервиса на Ubuntu.
# Запускать из корня проекта: sudo bash deploy/install.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_USER="${SERVICE_USER:-transcriber}"
SERVICE_NAME="transcriber-bot"

if [[ $EUID -ne 0 ]]; then
  echo "Запустите через sudo: sudo bash deploy/install.sh" >&2
  exit 1
fi

echo "==> Пакеты системы"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip ffmpeg ca-certificates curl unzip

# Современный yt-dlp исполняет JS со страниц YouTube; без рантайма часть форматов недоступна
if ! command -v deno &>/dev/null; then
  echo "==> JavaScript-рантайм deno"
  case "$(uname -m)" in
    x86_64) DENO_TARGET="x86_64-unknown-linux-gnu" ;;
    aarch64 | arm64) DENO_TARGET="aarch64-unknown-linux-gnu" ;;
    *) DENO_TARGET="" ;;
  esac

  if [[ -z "$DENO_TARGET" ]]; then
    echo "!! Архитектура $(uname -m) не поддерживается, deno пропущен"
  else
    TMP_DIR="$(mktemp -d)"
    if curl -fsSL "https://github.com/denoland/deno/releases/latest/download/deno-${DENO_TARGET}.zip" -o "$TMP_DIR/deno.zip"; then
      unzip -qo "$TMP_DIR/deno.zip" -d /usr/local/bin
      chmod 755 /usr/local/bin/deno
    else
      echo "!! Не удалось скачать deno — YouTube может отдавать не все форматы"
    fi
    rm -rf "$TMP_DIR"
  fi
fi

if ! id "$SERVICE_USER" &>/dev/null; then
  echo "==> Пользователь $SERVICE_USER"
  useradd --system --user-group --shell /usr/sbin/nologin --home-dir "$APP_DIR" "$SERVICE_USER"
fi

echo "==> Виртуальное окружение"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip --quiet
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt" --quiet

mkdir -p "$APP_DIR/data"

if [[ ! -f "$APP_DIR/.env" ]]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
  chmod 600 "$APP_DIR/.env"
  echo
  echo "Создан $APP_DIR/.env — впишите BOT_TOKEN и OPENAI_API_KEY, затем запустите:"
  echo "  sudo systemctl start $SERVICE_NAME"
  NEEDS_ENV=1
fi

chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR"

echo "==> systemd"
sed -e "s|__APP_DIR__|$APP_DIR|g" -e "s|__USER__|$SERVICE_USER|g" \
  "$APP_DIR/deploy/$SERVICE_NAME.service" > "/etc/systemd/system/$SERVICE_NAME.service"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME" --quiet

if [[ -z "${NEEDS_ENV:-}" ]]; then
  systemctl restart "$SERVICE_NAME"
  echo "==> Готово. Логи: journalctl -u $SERVICE_NAME -f"
else
  echo "==> Сервис включён в автозапуск, но не стартовал: сначала заполните .env"
fi
