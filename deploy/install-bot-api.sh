#!/usr/bin/env bash
# Сборка и установка локального Telegram Bot API сервера.
# Он снимает лимит отправки с 50 МБ до 2000 МБ — без него бот не может прислать
# длинное видео в 1080p.
#
# Запускать из корня проекта: sudo bash deploy/install-bot-api.sh
#
# Нужны api_id и api_hash с https://my.telegram.org/apps (обычный аккаунт Telegram,
# не бот). Скрипт спросит их и положит в /etc/telegram-bot-api.env под root.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_USER="${SERVICE_USER:-transcriber}"
SERVICE_NAME="telegram-bot-api"
BOT_SERVICE="transcriber-bot"
API_PORT="${API_PORT:-8081}"
API_DATA_DIR="${API_DATA_DIR:-/var/lib/telegram-bot-api}"
ENV_FILE="/etc/telegram-bot-api.env"
BUILD_DIR="${BUILD_DIR:-/usr/local/src/telegram-bot-api}"

if [[ $EUID -ne 0 ]]; then
  echo "Запустите через sudo: sudo bash deploy/install-bot-api.sh" >&2
  exit 1
fi

# --- api_id / api_hash ---------------------------------------------------------

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

if [[ -z "${TELEGRAM_API_ID:-}" || -z "${TELEGRAM_API_HASH:-}" ]]; then
  echo "Нужны api_id и api_hash с https://my.telegram.org/apps"
  echo "(войдите своим номером телефона -> API development tools)"
  echo
  read -rp "api_id: " TELEGRAM_API_ID
  read -rp "api_hash: " TELEGRAM_API_HASH
fi

if [[ ! "$TELEGRAM_API_ID" =~ ^[0-9]+$ ]]; then
  echo "api_id должен быть числом, получено: '$TELEGRAM_API_ID'" >&2
  exit 1
fi
if [[ ! "$TELEGRAM_API_HASH" =~ ^[0-9a-fA-F]{32}$ ]]; then
  echo "api_hash должен быть 32 символами hex" >&2
  exit 1
fi

# --- сборка --------------------------------------------------------------------

echo "==> Пакеты для сборки"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git cmake g++ make zlib1g-dev libssl-dev gperf ca-certificates curl

# TDLib собирается тяжело: один g++ съедает больше гигабайта. На маленькой VPS
# без подкачки сборка падает с OOM, поэтому считаем доступную память заранее.
MEM_TOTAL_MB=$(($(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024))
SWAP_TOTAL_MB=$(($(awk '/SwapTotal/ {print $2}' /proc/meminfo) / 1024))
USABLE_MB=$((MEM_TOTAL_MB + SWAP_TOTAL_MB))
echo "==> Память: ${MEM_TOTAL_MB} МБ RAM + ${SWAP_TOTAL_MB} МБ swap"

if [[ $USABLE_MB -lt 2600 && ! -f /swapfile-botapi ]]; then
  echo "!! Для сборки желательно от 2.5 ГБ (RAM + swap), сейчас ${USABLE_MB} МБ."
  read -rp "   Создать файл подкачки на 2 ГБ? [Y/n] " ANSWER
  if [[ ! "${ANSWER:-Y}" =~ ^[Nn] ]]; then
    fallocate -l 2G /swapfile-botapi || dd if=/dev/zero of=/swapfile-botapi bs=1M count=2048
    chmod 600 /swapfile-botapi
    mkswap /swapfile-botapi >/dev/null
    swapon /swapfile-botapi
    echo "/swapfile-botapi none swap sw 0 0" >> /etc/fstab
    USABLE_MB=$((USABLE_MB + 2048))
    echo "   Подкачка включена. Удалить потом: swapoff /swapfile-botapi && rm /swapfile-botapi"
  fi
fi

# Примерно 1.2 ГБ на поток компиляции, минимум один
JOBS=$((USABLE_MB / 1200))
[[ $JOBS -lt 1 ]] && JOBS=1
[[ $JOBS -gt $(nproc) ]] && JOBS=$(nproc)
echo "==> Сборка в $JOBS поток(а). Это надолго: от 10 минут до часа."

if [[ -d "$BUILD_DIR/.git" ]]; then
  git -C "$BUILD_DIR" fetch --depth 1 origin master
  git -C "$BUILD_DIR" reset --hard origin/master
  git -C "$BUILD_DIR" submodule update --init --recursive --depth 1
else
  rm -rf "$BUILD_DIR"
  git clone --recursive --depth 1 https://github.com/tdlib/telegram-bot-api.git "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR/build"
cd "$BUILD_DIR/build"
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..
cmake --build . --target install -j "$JOBS"

command -v telegram-bot-api >/dev/null || { echo "Сборка не дала бинарник" >&2; exit 1; }
echo "==> Установлено: $(telegram-bot-api --version 2>&1 | head -n1)"

# --- пользователь и каталоги ---------------------------------------------------

# Сервер запускаем под тем же пользователем, что и бота. В режиме --local getFile
# отдаёт не ссылку, а путь на диске, и бот читает файл напрямую — с общим
# пользователем не приходится городить права доступа между двумя сервисами.
if ! id "$SERVICE_USER" &>/dev/null; then
  useradd --system --user-group --shell /usr/sbin/nologin --home-dir "$APP_DIR" "$SERVICE_USER"
fi

mkdir -p "$API_DATA_DIR/temp"
chown -R "$SERVICE_USER:$SERVICE_USER" "$API_DATA_DIR"
chmod 750 "$API_DATA_DIR"

umask 077
cat > "$ENV_FILE" <<EOF
TELEGRAM_API_ID=$TELEGRAM_API_ID
TELEGRAM_API_HASH=$TELEGRAM_API_HASH
EOF
chmod 600 "$ENV_FILE"
umask 022

# --- systemd -------------------------------------------------------------------

echo "==> systemd"
sed -e "s|__USER__|$SERVICE_USER|g" \
    -e "s|__PORT__|$API_PORT|g" \
    -e "s|__DATA_DIR__|$API_DATA_DIR|g" \
  "$APP_DIR/deploy/$SERVICE_NAME.service" > "/etc/systemd/system/$SERVICE_NAME.service"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME" --quiet
systemctl restart "$SERVICE_NAME"

sleep 3
if ! systemctl is-active --quiet "$SERVICE_NAME"; then
  echo "!! Сервер не запустился. Логи: journalctl -u $SERVICE_NAME -n 40" >&2
  exit 1
fi
echo "==> Сервер слушает 127.0.0.1:$API_PORT"

# --- переключение бота ---------------------------------------------------------

BOT_TOKEN="$(sed -n 's/^BOT_TOKEN=//p' "$APP_DIR/.env" 2>/dev/null | head -n1)"
BOT_TOKEN="${BOT_TOKEN%$'\r'}"   # .env мог быть сохранён в Windows с CRLF
BOT_TOKEN="${BOT_TOKEN//\"/}"
BOT_TOKEN="${BOT_TOKEN//\'/}"
BOT_TOKEN="${BOT_TOKEN// /}"
if [[ -z "$BOT_TOKEN" ]]; then
  echo "!! В $APP_DIR/.env нет BOT_TOKEN — допишите TELEGRAM_API_URL вручную." >&2
  exit 1
fi

echo
echo "Осталось разлогинить бота из облачного Bot API — без этого локальный"
echo "сервер его не примет. После этого бот работает ТОЛЬКО через локальный"
echo "сервер (вернуться назад: вызвать logOut уже у локального сервера)."
read -rp "Выполнить logOut сейчас? [Y/n] " ANSWER
if [[ "${ANSWER:-Y}" =~ ^[Nn] ]]; then
  echo "Пропущено. Потом: curl \"https://api.telegram.org/bot<TOKEN>/logOut\""
else
  # Бот мог быть разлогинен ранее — тогда Telegram ответит ошибкой, это не беда
  LOGOUT_RESULT="$(curl -fsS "https://api.telegram.org/bot${BOT_TOKEN}/logOut" || true)"
  echo "   Ответ Telegram: ${LOGOUT_RESULT:-(нет ответа)}"
  # Токен освобождается не мгновенно
  sleep 5
fi

echo "==> Прописываю TELEGRAM_API_URL в .env"
API_URL="http://127.0.0.1:$API_PORT"
if grep -q '^TELEGRAM_API_URL=' "$APP_DIR/.env"; then
  sed -i "s|^TELEGRAM_API_URL=.*|TELEGRAM_API_URL=$API_URL|" "$APP_DIR/.env"
else
  echo "TELEGRAM_API_URL=$API_URL" >> "$APP_DIR/.env"
fi

systemctl restart "$BOT_SERVICE" 2>/dev/null || true

echo
echo "==> Готово."
echo "    Лимит отправки вырос с 50 МБ до 2000 МБ, скачивание 1080p заработает."
echo "    Логи сервера: journalctl -u $SERVICE_NAME -f"
echo "    Логи бота:    journalctl -u $BOT_SERVICE -f"
echo
echo "    Учтите: в режиме --local сервер складывает скачанные файлы в"
echo "    $API_DATA_DIR и сам их не удаляет. Чистка раз в сутки:"
echo "      echo '0 4 * * * root find $API_DATA_DIR -type f -mtime +1 -delete' > /etc/cron.d/telegram-bot-api-cleanup"
