# AutoCon

Локальный дорожный видеоассистент на YOLO11. Приложение открывает видео или live-камеру, распознаёт дорожные знаки, транспорт, номера, собирает последовательности событий и передаёт структурированную сцену в локальную Ollama-модель для коротких комментариев.

AutoCon не является сертифицированной ADAS-системой и не должен использоваться как единственный источник решений на дороге. Интерфейс показывает confidence и работает как ассистент/аналитический инструмент.

## Быстрый старт

```powershell
python build.py install
python build.py run
```

Сборка:

```powershell
python build.py build
```

Полный форсированный цикл:

```powershell
python build.py --force
```

`--force` сначала выполняет `uv pip install --python <current python> -r requirements.txt`, затем собирает `dist\AutoCon.exe` и запускает его.

## Первый запуск

Скомпилированный `.exe` остаётся лёгким. Тяжёлые ML-зависимости ставятся в `%APPDATA%\AutoCon\runtime`:

- `ultralytics`, `opencv-python`, `numpy`, `pillow` для YOLO11;
- `lap` для трекинга;
- `fast-alpr`, `fast-plate-ocr`, `onnxruntime` для номеров;
- `huggingface-hub` для curated model packs;
- `torch`, `torchvision`, `onnxruntime-gpu` при выборе CUDA;
- `pyttsx3` только если нужна локальная озвучка.

Все runtime-команды запускаются скрыто через Windows `CREATE_NO_WINDOW`; прогресс и ошибки идут в UI и логи `%APPDATA%\AutoCon\logs`.

## Модели

Стартовый набор ориентирован на Россия/СНГ:

- `yolo11s.pt` как базовый официальный детектор транспорта/людей;
- community traffic sign pack на 100 классов;
- YOLO11 license plate detector;
- опциональный VehicleDINO INT8 ONNX.

Если community-веса недоступны или нужна своя точность, в настройках можно указать собственные `.pt` или `.onnx`.

## Структура

```text
app/
  main.py                 pywebview entry point
  api.py                  JS <-> Python bridge
  config.py               paths and settings
  core/
    runtime.py            managed uv runtime
    hidden.py             no-window subprocess helpers
    vision.py             YOLO11 processing and aggregation
    model_registry.py     curated model packs
    llm.py                Ollama client and prompts
    project.py            project JSON store
    server.py             local HTTP server with Range support
web/
  index.html
  css/styles.css
  js/app.js
  assets/logo.svg
build.py
tests/
```

## Проверки

```powershell
python -m unittest
python build.py doctor
python build.py build
```
