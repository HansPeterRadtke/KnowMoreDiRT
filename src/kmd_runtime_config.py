"""Validated KMD runtime configuration with XML defaults and overrides.

Precedence is environment > optional user XML > packaged defaults.  Merely
loading defaults never mutates ``os.environ``; callers can therefore distinguish
an explicit override from the default and preserve historical cache identities.
"""
from __future__ import annotations

import math
import os
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).with_name("knowmoredirt") / "default_config.xml"
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class SettingSpec:
    name: str
    value: str
    value_type: str
    unit: str = ""
    minimum: float | None = None
    maximum: float | None = None
    enforce_range: bool = False
    choices: tuple[str, ...] = ()
    group: str = ""
    risk: str = ""
    change_frequency: str = ""
    description: str = ""


_DEFAULT_LOCK = threading.Lock()
_DEFAULT_SPECS: dict[str, SettingSpec] | None = None
_USER_CACHE: tuple[str, int, int, dict[str, str]] | None = None
_USER_LOCK = threading.Lock()


def _bool_attr(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    raise ValueError(f"invalid boolean attribute {value!r}")


def _optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"configuration bound must be finite, got {value!r}")
    return parsed


def _parse_default_config(path: Path) -> dict[str, SettingSpec]:
    root = ET.parse(path).getroot()
    if root.tag != "knowmoredirt-config":
        raise ValueError(f"unexpected KMD config root {root.tag!r} in {path}")
    specs: dict[str, SettingSpec] = {}
    for element in root.findall("./settings/setting"):
        name = str(element.get("name") or "").strip()
        if not name or not name.startswith("KMD_"):
            raise ValueError(f"invalid KMD setting name {name!r} in {path}")
        if name in specs:
            raise ValueError(f"duplicate KMD setting {name!r} in {path}")
        choices = tuple(
            item.strip() for item in str(element.get("choices") or "").split(",") if item.strip()
        )
        spec = SettingSpec(
            name=name,
            value=str(element.get("value") or ""),
            value_type=str(element.get("type") or "str").strip().lower(),
            unit=str(element.get("unit") or "").strip(),
            minimum=_optional_float(element.get("minimum")),
            maximum=_optional_float(element.get("maximum")),
            enforce_range=_bool_attr(element.get("enforce_range"), False),
            choices=choices,
            group=str(element.get("group") or "").strip(),
            risk=str(element.get("risk") or "").strip(),
            change_frequency=str(element.get("change_frequency") or "").strip(),
            description=str(element.get("description") or "").strip(),
        )
        _validate_value(spec, spec.value, source=str(path))
        specs[name] = spec
    if not specs:
        raise ValueError(f"KMD default config has no settings: {path}")
    return specs


def default_specs() -> dict[str, SettingSpec]:
    global _DEFAULT_SPECS
    if _DEFAULT_SPECS is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_SPECS is None:
                _DEFAULT_SPECS = _parse_default_config(DEFAULT_CONFIG_PATH)
    return _DEFAULT_SPECS


def _user_config_path() -> Path | None:
    text = os.environ.get("KMD_CONFIG_FILE", "").strip()
    return Path(text).expanduser() if text else None


def _parse_user_config(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    if root.tag != "knowmoredirt-config":
        raise ValueError(f"unexpected KMD config root {root.tag!r} in {path}")
    known = default_specs()
    values: dict[str, str] = {}
    for element in root.findall("./settings/setting"):
        name = str(element.get("name") or "").strip()
        if name not in known:
            raise ValueError(f"unknown KMD setting {name!r} in {path}")
        if name in values:
            raise ValueError(f"duplicate KMD setting {name!r} in {path}")
        value = str(element.get("value") or "")
        _validate_value(known[name], value, source=str(path))
        values[name] = value
    return values


def user_values() -> dict[str, str]:
    global _USER_CACHE
    path = _user_config_path()
    if path is None:
        return {}
    stat = path.stat()
    key = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    with _USER_LOCK:
        if _USER_CACHE is not None and _USER_CACHE[:3] == key:
            return _USER_CACHE[3]
        values = _parse_user_config(path)
        _USER_CACHE = (*key, values)
        return values


def _numeric_value(spec: SettingSpec, raw: str) -> float | None:
    if spec.value_type in {"int", "float"}:
        value = float(raw)
    elif spec.value_type in {"int_or_empty", "float_or_empty"}:
        if not raw.strip():
            return None
        value = float(raw)
    else:
        return None
    if not math.isfinite(value):
        raise ValueError(f"{spec.name} must be finite")
    return value


def _validate_value(spec: SettingSpec, raw: str, *, source: str) -> None:
    normalized = raw.strip()
    kind = spec.value_type
    try:
        if kind == "bool":
            if normalized.lower() not in _TRUE | _FALSE:
                raise ValueError("expected boolean")
        elif kind in {"int", "int_or_empty"}:
            if kind == "int_or_empty" and not normalized:
                pass
            else:
                int(normalized)
        elif kind in {"float", "float_or_empty"}:
            if kind == "float_or_empty" and not normalized:
                pass
            else:
                value = float(normalized)
                if not math.isfinite(value):
                    raise ValueError("expected finite number")
        elif kind == "csv_int":
            for item in normalized.split(","):
                if item.strip():
                    int(item.strip())
        elif kind == "csv_float":
            for item in normalized.split(","):
                if item.strip():
                    value = float(item.strip())
                    if not math.isfinite(value):
                        raise ValueError("expected finite comma-separated numbers")
        elif kind in {"enum", "enum_or_empty"}:
            if kind == "enum_or_empty" and not normalized:
                pass
            elif normalized not in spec.choices:
                raise ValueError(f"expected one of {spec.choices!r}")
        elif kind in {"str", "path"}:
            pass
        else:
            raise ValueError(f"unsupported setting type {kind!r}")
        numeric = _numeric_value(spec, normalized)
        if spec.enforce_range and numeric is not None:
            if spec.minimum is not None and numeric < spec.minimum:
                raise ValueError(f"must be >= {spec.minimum:g}{(' ' + spec.unit) if spec.unit else ''}")
            if spec.maximum is not None and numeric > spec.maximum:
                raise ValueError(f"must be <= {spec.maximum:g}{(' ' + spec.unit) if spec.unit else ''}")
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {spec.name}={raw!r} from {source}: {error}") from error


def explicit_raw(name: str) -> str | None:
    specs = default_specs()
    if name not in specs:
        raise KeyError(f"unknown KMD setting {name!r}")
    if name in os.environ:
        value = os.environ[name]
        _validate_value(specs[name], value, source="environment")
        return value
    values = user_values()
    return values.get(name)


def raw(name: str) -> str:
    override = explicit_raw(name)
    if override is not None:
        return override
    return default_specs()[name].value


def text(name: str) -> str:
    return raw(name)


def integer(name: str) -> int:
    return int(raw(name).strip())


def floating(name: str) -> float:
    return float(raw(name).strip())


def boolean(name: str) -> bool:
    value = raw(name).strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"invalid boolean setting {name}={value!r}")


def optional_float(name: str) -> float | None:
    value = raw(name).strip()
    return float(value) if value else None


def optional_int(name: str) -> int | None:
    value = raw(name).strip()
    return int(value) if value else None


def csv_integers(name: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw(name).split(",") if item.strip())


def csv_floats(name: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw(name).split(",") if item.strip())


def source(name: str) -> str:
    if name in os.environ:
        return "environment"
    if name in user_values():
        return str(_user_config_path())
    return str(DEFAULT_CONFIG_PATH)



DEFAULT_MODEL_CACHE_ROOT = Path("/data/var/knowmoredirt/model_cache")
MODEL_CACHE_NAMESPACES: dict[str, str] = {
    "KMD_FRAME_CACHE_DIR": "frame",
    "KMD_CHUNK_FRAME_CACHE_DIR": "chunk_frame",
    "KMD_CHUNK_DRS_CACHE_DIR": "chunk_drs",
    "KMD_QUERY_PLAN_CACHE_DIR": "query_plan",
    "KMD_QUERY_DRS_CACHE_DIR": "query_drs",
    "KMD_QUERY_EVIDENCE_REPAIR_CACHE_DIR": "query_evidence_repair",
    "KMD_QUERY_EVIDENCE_CACHE_DIR": "query_evidence",
    "KMD_EVIDENCE_ANSWER_CACHE_DIR": "evidence_answer",
    "KMD_VERIFIER_CACHE_DIR": "verifier",
    "KMD_QUERY_VERIFIER_CACHE_DIR": "verifier",
    "KMD_ANSWER_CANONICALIZATION_CACHE_DIR": "answer_canonicalization",
    "KMD_QUERY_CANONICAL_CACHE_DIR": "answer_canonicalization",
    "KMD_IDENTITY_CACHE_DIR": "identity",
    "KMD_IDENTITY_CANONICAL_CACHE_DIR": "identity",
    "KMD_SOURCE_RESOLUTION_CACHE_DIR": "source_resolution",
    "KMD_DOCUMENT_CONTEXT_CACHE_DIR": "document_context",
    "KMD_EVALUATION_JUDGE_CACHE_DIR": "evaluation_judge",
}

def model_cache_root() -> Path:
    configured = raw("KMD_SHARED_MODEL_CACHE_ROOT").strip()
    return Path(configured).expanduser() if configured else DEFAULT_MODEL_CACHE_ROOT

def model_cache_dir(setting_name: str) -> Path:
    if setting_name not in MODEL_CACHE_NAMESPACES:
        raise KeyError(f"unknown KMD model-cache setting {setting_name!r}")
    override = explicit_raw(setting_name)
    if override is not None and override.strip():
        return Path(override.strip()).expanduser()
    return model_cache_root() / MODEL_CACHE_NAMESPACES[setting_name]

def configure_model_cache_environment() -> dict[str, str]:
    root = model_cache_root()
    os.environ.setdefault("KMD_SHARED_MODEL_CACHE_ROOT", str(root))
    resolved: dict[str, str] = {"KMD_SHARED_MODEL_CACHE_ROOT": str(root)}
    for setting_name in MODEL_CACHE_NAMESPACES:
        path = model_cache_dir(setting_name)
        os.environ.setdefault(setting_name, str(path))
        path.mkdir(parents=True, exist_ok=True)
        resolved[setting_name] = str(path)
    root.mkdir(parents=True, exist_ok=True)
    return resolved

def validate_all() -> None:
    for name, spec in default_specs().items():
        _validate_value(spec, raw(name), source=source(name))
