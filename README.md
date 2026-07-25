# Транскрибатор YouTube для Telegram

Телеграм-бот: присылаете ссылку на YouTube — бот скачивает звуковую дорожку,
расшифровывает её через OpenAI Audio API и возвращает текст. Под расшифровкой
есть кнопка «Сделать конспект» (краткие тезисы через chat-модель).

Что внутри:

- `yt-dlp` скачивает только аудио, `ffmpeg` жмёт его в mp3 16 кГц моно (64 кбит/с);
- длинное аудио автоматически режется на части по 15 минут, чтобы влезать в лимит OpenAI 25 МБ,
  части расшифровываются параллельно;
- готовые расшифровки кэшируются в `data/transcripts`, повторный запрос того же видео бесплатен;
- очередь задач: одновременная обработка ограничена, у пользователя не больше 3 задач;
- доступ по списку Telegram ID (иначе за ваш ключ OpenAI будут платить незнакомцы);
- расшифровка до 3500 символов приходит сообщением, длиннее — файлом `.txt`.

## Требования

- Python 3.10+
- ffmpeg и ffprobe в `PATH`
- [deno](https://deno.com/) в `PATH` — свежий yt-dlp исполняет JavaScript со страниц
  YouTube, без рантайма часть форматов недоступна (на Ubuntu ставится скриптом установки)
- токен бота от [@BotFather](https://t.me/BotFather)
- ключ [OpenAI API](https://platform.openai.com/api-keys) с положительным балансом

## Быстрый старт локально

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # впишите BOT_TOKEN и OPENAI_API_KEY
python -m app
```

Дальше в Telegram: `/start`, затем `/id` — полученный ID впишите в `ALLOWED_USER_IDS`
и перезапустите бота.

## Развёртывание на VPS с Ubuntu

Всё делается за пять команд от пользователя с sudo:

```bash
sudo apt update && sudo apt install -y git
sudo git clone <URL-вашего-репозитория> /opt/transcriber
cd /opt/transcriber
sudo bash deploy/install.sh        # пакеты, venv, systemd-юнит, автозапуск
sudo nano .env                     # BOT_TOKEN, OPENAI_API_KEY, ALLOWED_USER_IDS
sudo systemctl start transcriber-bot
```

Если репозитория нет, скопируйте папку проекта с локальной машины:

```bash
# из корня проекта на своей машине
scp -r . root@IP:/opt/transcriber
```

Скрипт `deploy/install.sh` ставит `python3-venv`, `ffmpeg` и `deno`, создаёт системного
пользователя `transcriber`, собирает виртуальное окружение и включает сервис
`transcriber-bot` в автозапуск.

Полезные команды:

```bash
sudo systemctl status transcriber-bot     # состояние
journalctl -u transcriber-bot -f          # живые логи
sudo systemctl restart transcriber-bot    # после правки .env
sudo bash deploy/update.sh                # git pull + обновление зависимостей + рестарт
```

Бот работает на long polling, поэтому открытые порты, домен и сертификаты не нужны —
достаточно исходящего доступа в интернет. Минимальной VPS (1 ядро, 1 ГБ RAM, 10 ГБ диска)
хватает: тяжёлые вычисления происходят на стороне OpenAI, локально нужен только ffmpeg.
Для 1 ГБ RAM полезно добавить swap:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Настройки `.env`

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `BOT_TOKEN` | — | токен от @BotFather |
| `OPENAI_API_KEY` | — | ключ OpenAI |
| `ALLOWED_USER_IDS` | пусто (все) | список ID через запятую |
| `TRANSCRIBE_MODEL` | `whisper-1` | `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe` |
| `TRANSCRIBE_LANGUAGE` | `ru` | подсказка языка; пусто — автоопределение |
| `TIMESTAMPS` | `false` | таймкоды в тексте (только `whisper-1`) |
| `SUMMARY_MODEL` | `gpt-4o-mini` | модель для конспекта |
| `MAX_VIDEO_MINUTES` | `180` | лимит длительности видео |
| `WORKERS` | `1` | сколько видео обрабатывать параллельно |
| `TRANSCRIBE_CONCURRENCY` | `3` | сколько частей одного видео отправлять в OpenAI параллельно |
| `YT_COOKIES_FILE` | пусто | cookies для YouTube |
| `YT_PROXY` | пусто | прокси для yt-dlp |

## YouTube требует «подтвердить, что вы не робот»

Самая частая проблема на VPS: с IP дата-центров YouTube иногда отдаёт ошибку
`Sign in to confirm you're not a bot`. Варианты решения:

1. **Cookies.** В браузере, где вы залогинены на YouTube, экспортируйте cookies
   в формате Netscape (расширение вроде «Get cookies.txt»), положите файл на сервер
   и укажите путь в `YT_COOKIES_FILE=/opt/transcriber/cookies.txt`. Файл держите
   в режиме `chmod 600` — это доступ к вашему аккаунту. Желательно использовать
   отдельный аккаунт: за автоматизацию его могут заблокировать.
2. **Прокси.** Жилой или мобильный прокси в `YT_PROXY=http://user:pass@host:port`.
3. **Свежий yt-dlp.** `sudo bash deploy/update.sh` — обходы блокировок появляются
   почти каждую неделю.

## Стоимость

Платите только за OpenAI: транскрибация тарифицируется за минуты аудио
(у `whisper-1` порядка $0.006 за минуту, то есть примерно $0.36 за час видео),
конспект — за токены и на фоне транскрибации почти незаметен. Актуальные цены
смотрите на [странице тарифов OpenAI](https://openai.com/api/pricing/).
Поэтому список `ALLOWED_USER_IDS` — не формальность.

## Структура

```
app/
  __main__.py    точка входа, логирование
  bot.py         хендлеры aiogram, очередь задач, отправка результата
  audio.py       yt-dlp, перекодирование и нарезка через ffmpeg
  transcriber.py вызовы OpenAI: расшифровка и конспект
  config.py      настройки из .env
  utils.py       мелкие помощники
deploy/
  install.sh                 установка на Ubuntu
  update.sh                  обновление и рестарт
  transcriber-bot.service    шаблон systemd-юнита
```

## Возможные проблемы

- **`Не найден ffmpeg`** — `sudo apt install -y ffmpeg`.
- **В логах `Не найден JavaScript-рантайм`** — поставьте deno вручную:

```bash
curl -fsSL https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip -o /tmp/deno.zip
sudo unzip -o /tmp/deno.zip -d /usr/local/bin && sudo chmod 755 /usr/local/bin/deno
```

- **`OpenAI отклонил ключ API`** — ключ неверный или отозван.
- **`OpenAI вернул 429`** — закончились деньги на балансе или сработал лимит запросов.
- **Бот молчит** — смотрите `journalctl -u transcriber-bot -n 50`; при конфликте
  `TelegramConflictError` где-то запущена вторая копия бота с тем же токеном.
- **Диск кончился** — кэш расшифровок лежит в `data/transcripts`, чистится вручную:
  `find data/transcripts -mtime +30 -delete`.
