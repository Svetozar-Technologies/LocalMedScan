"""Internationalization system for LocalMedScan.

Supports 8 languages with RTL detection and format string interpolation.
Usage: from i18n import t; t("key", name=value)
"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSettings

LANGUAGES = OrderedDict([
    ("en", {"name": "English", "native_name": "English"}),
    ("hi", {"name": "Hindi", "native_name": "\u0939\u093f\u0928\u094d\u0926\u0940"}),
    ("ru", {"name": "Russian", "native_name": "\u0420\u0443\u0441\u0441\u043a\u0438\u0439"}),
    ("zh", {"name": "Chinese", "native_name": "\u4e2d\u6587"}),
    ("ja", {"name": "Japanese", "native_name": "\u65e5\u672c\u8a9e"}),
    ("es", {"name": "Spanish", "native_name": "Espa\u00f1ol"}),
    ("fr", {"name": "French", "native_name": "Fran\u00e7ais"}),
    ("ar", {"name": "Arabic", "native_name": "\u0627\u0644\u0639\u0631\u0628\u064a\u0629"}),
])

_translations: dict = {}
_fallback: dict = {}
_current_lang: str = "en"


def _get_i18n_dir() -> Path:
    """Get the directory containing translation JSON files."""
    import sys
    if getattr(sys, "frozen", False):
        import sys as _sys
        return Path(_sys._MEIPASS) / "i18n"
    return Path(__file__).parent


def _load_json(lang_code: str) -> dict:
    """Load a translation JSON file."""
    path = _get_i18n_dir() / f"{lang_code}.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def init():
    """Initialize the translation system. Call once at app startup."""
    global _translations, _fallback, _current_lang
    settings = QSettings("Svetozar Technologies", "LocalMedScan")
    _current_lang = settings.value("language", "en")
    if _current_lang not in LANGUAGES:
        _current_lang = "en"

    _fallback = _load_json("en")
    if _current_lang != "en":
        _translations = _load_json(_current_lang)
    else:
        _translations = _fallback


def t(key: str, **kwargs) -> str:
    """Translate a key with optional format arguments.

    Falls back: current language -> English -> raw key.
    """
    text = _translations.get(key) or _fallback.get(key) or key
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return text
    return text


def get_current_language() -> str:
    """Get the current language code."""
    return _current_lang


def set_language(code: str):
    """Save language preference. Takes effect on restart."""
    settings = QSettings("Svetozar Technologies", "LocalMedScan")
    settings.setValue("language", code)


def is_rtl() -> bool:
    """Check if the current language uses right-to-left layout."""
    meta = _translations.get("_meta", {})
    if isinstance(meta, dict):
        return meta.get("direction", "ltr") == "rtl"
    return False
