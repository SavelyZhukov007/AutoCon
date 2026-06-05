# -*- coding: utf-8 -*-
"""AutoCon desktop entry point."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import config

    config.bootstrap_runtime_packages()
    from app.api import Api
    from app.core.single_instance import ensure_single_instance
else:
    from . import config

    config.bootstrap_runtime_packages()
    from .api import Api
    from .core.single_instance import ensure_single_instance

import webview


def main() -> None:
    guard = ensure_single_instance("AutoCon")
    api = Api()
    try:
        api._server.start(web_root=config.web_dir())
        url = api._server.index_url()
    except Exception:
        url = str(config.web_dir() / "index.html")

    webview.create_window(
        title="AutoCon — дорожный видеоассистент",
        url=url,
        js_api=api,
        width=1420,
        height=900,
        min_size=(1080, 700),
        background_color="#0f1417",
        text_select=True,
    )
    try:
        webview.start(debug=False)
    finally:
        api.stop_camera()
        guard.release()


if __name__ == "__main__":
    main()
