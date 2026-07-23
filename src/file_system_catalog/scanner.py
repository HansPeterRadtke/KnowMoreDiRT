from __future__ import annotations

import array
import base64
import ctypes
import ctypes.util
import datetime as dt
import errno
import fcntl
import grp
import hashlib
import json
import mimetypes
import os
import pwd
import shutil
import sqlite3
import stat
import struct
import subprocess
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from xml.etree import ElementTree

from . import __version__
from .metadata_extractors import ExifToolClient, collect_raw_metadata
from .content_schema import (
    CHUNK_TABLE_NAME,
    CONTENT_CREATE_SQL,
    CONTENT_INDEX_SQL,
    REPRESENTATION_TABLE_NAME,
)
from .schema import COLUMN_NAMES, CREATE_TABLE_SQL, INDEX_SQL, SCHEMA_VERSION, TABLE_NAME

PENDING_CONTENT = "PENDING_CONTENT_EXTRACTION"
AT_FDCWD = -100
AT_SYMLINK_NOFOLLOW = 0x100
STATX_BASIC_STATS = 0x07FF
STATX_BTIME = 0x0800
FS_IOC_GETFLAGS = 0x80086601

FS_FLAG_NAMES = {
    0x00000001: "secure_deletion",
    0x00000002: "undelete",
    0x00000004: "compressed",
    0x00000008: "synchronous_updates",
    0x00000010: "immutable",
    0x00000020: "append_only",
    0x00000040: "nodump",
    0x00000080: "noatime",
    0x00000100: "dirty_compressed",
    0x00000200: "compressed_blocks",
    0x00000400: "no_compression",
    0x00000800: "encrypted",
    0x00001000: "indexed_directory_or_btree",
    0x00002000: "imagic",
    0x00004000: "journal_data",
    0x00008000: "no_tail_merge",
    0x00010000: "directory_synchronous",
    0x00020000: "top_directory",
    0x00040000: "huge_file",
    0x00080000: "extents",
    0x00100000: "verity",
    0x00200000: "ea_inode",
    0x00400000: "eof_blocks",
    0x10000000: "inline_data",
    0x20000000: "project_inherit",
    0x40000000: "casefold",
}


class StatxTimestamp(ctypes.Structure):
    _fields_ = [
        ("tv_sec", ctypes.c_int64),
        ("tv_nsec", ctypes.c_uint32),
        ("reserved", ctypes.c_int32),
    ]


class Statx(ctypes.Structure):
    _fields_ = [
        ("mask", ctypes.c_uint32),
        ("blksize", ctypes.c_uint32),
        ("attributes", ctypes.c_uint64),
        ("nlink", ctypes.c_uint32),
        ("uid", ctypes.c_uint32),
        ("gid", ctypes.c_uint32),
        ("mode", ctypes.c_uint16),
        ("spare0", ctypes.c_uint16),
        ("ino", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("blocks", ctypes.c_uint64),
        ("attributes_mask", ctypes.c_uint64),
        ("atime", StatxTimestamp),
        ("btime", StatxTimestamp),
        ("ctime", StatxTimestamp),
        ("mtime", StatxTimestamp),
        ("rdev_major", ctypes.c_uint32),
        ("rdev_minor", ctypes.c_uint32),
        ("dev_major", ctypes.c_uint32),
        ("dev_minor", ctypes.c_uint32),
        ("mnt_id", ctypes.c_uint64),
        ("dio_mem_align", ctypes.c_uint32),
        ("dio_offset_align", ctypes.c_uint32),
        ("spare3", ctypes.c_uint64 * 12),
    ]


@dataclass(frozen=True)
class MountInfo:
    filesystem_type: str | None
    source: str | None
    mount_point: bytes
    options: str | None


class LinuxStatx:
    def __init__(self) -> None:
        self._call = None
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            call = libc.statx
            call.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(Statx)]
            call.restype = ctypes.c_int
            self._call = call
        except (AttributeError, OSError):
            self._call = None

    def birth_time_ns(self, path: bytes) -> tuple[int | None, str | None]:
        if self._call is None:
            return None, "statx unavailable"
        result = Statx()
        return_code = self._call(
            AT_FDCWD,
            ctypes.c_char_p(path),
            AT_SYMLINK_NOFOLLOW,
            STATX_BASIC_STATS | STATX_BTIME,
            ctypes.byref(result),
        )
        if return_code != 0:
            error = ctypes.get_errno()
            return None, f"statx: {os.strerror(error)}"
        if not result.mask & STATX_BTIME:
            return None, None
        return result.btime.tv_sec * 1_000_000_000 + result.btime.tv_nsec, None


class LibMagic:
    MAGIC_NONE = 0x000000
    MAGIC_MIME_TYPE = 0x000010
    MAGIC_MIME_ENCODING = 0x000400
    MAGIC_ERROR = 0x000200

    def __init__(self) -> None:
        self._lib = None
        self._cookies: dict[str, int] = {}
        library = ctypes.util.find_library("magic")
        if not library:
            return
        try:
            lib = ctypes.CDLL(library)
            lib.magic_open.argtypes = [ctypes.c_int]
            lib.magic_open.restype = ctypes.c_void_p
            lib.magic_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            lib.magic_load.restype = ctypes.c_int
            lib.magic_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            lib.magic_file.restype = ctypes.c_char_p
            lib.magic_error.argtypes = [ctypes.c_void_p]
            lib.magic_error.restype = ctypes.c_char_p
            lib.magic_close.argtypes = [ctypes.c_void_p]
            for name, flags in (
                ("description", self.MAGIC_NONE | self.MAGIC_ERROR),
                ("mime_type", self.MAGIC_MIME_TYPE | self.MAGIC_ERROR),
                ("mime_encoding", self.MAGIC_MIME_ENCODING | self.MAGIC_ERROR),
            ):
                cookie = lib.magic_open(flags)
                if not cookie or lib.magic_load(cookie, None) != 0:
                    if cookie:
                        lib.magic_close(cookie)
                    raise RuntimeError("libmagic database failed to load")
                self._cookies[name] = cookie
            self._lib = lib
        except Exception:
            self.close()

    def inspect(self, path: bytes) -> tuple[dict[str, str | None], str | None]:
        result: dict[str, str | None] = {"description": None, "mime_type": None, "mime_encoding": None}
        if self._lib is None:
            return result, "libmagic unavailable"
        errors: list[str] = []
        for name, cookie in self._cookies.items():
            raw = self._lib.magic_file(cookie, ctypes.c_char_p(path))
            if raw:
                result[name] = raw.decode("utf-8", "backslashreplace")
            else:
                error = self._lib.magic_error(cookie)
                errors.append(error.decode("utf-8", "backslashreplace") if error else f"libmagic {name} failed")
        return result, "; ".join(errors) if errors else None

    def close(self) -> None:
        if self._lib is not None:
            for cookie in self._cookies.values():
                self._lib.magic_close(cookie)
        self._cookies = {}
        self._lib = None

    def __enter__(self) -> "LibMagic":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def display_bytes(value: bytes) -> str:
    return os.fsdecode(value).encode("utf-8", "backslashreplace").decode("utf-8")


def safe_error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}".encode("utf-8", "backslashreplace").decode("utf-8")


def timestamp_iso(value_ns: int | None) -> str | None:
    if value_ns is None:
        return None
    seconds, nanoseconds = divmod(value_ns, 1_000_000_000)
    try:
        moment = dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return f"{moment.strftime('%Y-%m-%dT%H:%M:%S')}.{nanoseconds:09d}Z"


def timestamp_storage(value_ns: int | None) -> str | None:
    return None if value_ns is None else str(value_ns)


def entry_type(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISBLK(mode):
        return "block_device"
    if stat.S_ISCHR(mode):
        return "character_device"
    return "other"


def decode_mount_escape(value: bytes) -> bytes:
    for encoded, decoded in ((b"\\040", b" "), (b"\\011", b"\t"), (b"\\012", b"\n"), (b"\\134", b"\\")):
        value = value.replace(encoded, decoded)
    return value


def find_mount_info(path: bytes) -> MountInfo:
    absolute = os.path.abspath(path)
    best: tuple[int, MountInfo] | None = None
    with open("/proc/self/mountinfo", "rb") as handle:
        for line in handle:
            try:
                left, right = line.rstrip(b"\n").split(b" - ", 1)
                left_fields = left.split()
                right_fields = right.split()
                mount_point = decode_mount_escape(left_fields[4])
                if absolute != mount_point and not absolute.startswith(mount_point.rstrip(b"/") + b"/"):
                    continue
                info = MountInfo(
                    filesystem_type=display_bytes(right_fields[0]),
                    source=display_bytes(decode_mount_escape(right_fields[1])),
                    mount_point=mount_point,
                    options=display_bytes(left_fields[5] + b"," + right_fields[2]),
                )
                score = len(mount_point)
                if best is None or score > best[0]:
                    best = (score, info)
            except (IndexError, ValueError):
                continue
    return best[1] if best else MountInfo(None, None, b"/", None)


def username(uid: int) -> str | None:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return None


def groupname(gid: int) -> str | None:
    try:
        return grp.getgrgid(gid).gr_name
    except KeyError:
        return None


def extensions(name: bytes) -> list[bytes]:
    result: list[bytes] = []
    remaining = name
    while True:
        stem, suffix = os.path.splitext(remaining)
        if not suffix or suffix == b".":
            break
        result.insert(0, suffix)
        remaining = stem
    return result


def filesystem_flags(path: bytes, mode: int) -> tuple[int | None, list[str], str | None]:
    if stat.S_ISLNK(mode) or stat.S_ISSOCK(mode):
        return None, [], None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    if stat.S_ISDIR(mode):
        flags |= getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        return None, [], safe_error(error)
    try:
        buffer = array.array("I", [0])
        fcntl.ioctl(descriptor, FS_IOC_GETFLAGS, buffer, True)
        value = int(buffer[0])
        names = [name for bit, name in FS_FLAG_NAMES.items() if value & bit]
        return value, names, None
    except OSError as error:
        if error.errno in (errno.ENOTTY, errno.EOPNOTSUPP, errno.EINVAL):
            return None, [], None
        return None, [], safe_error(error)
    finally:
        os.close(descriptor)


def read_xattrs(path: bytes) -> tuple[dict[str, str], list[str], str | None]:
    values: dict[str, str] = {}
    errors: list[str] = []
    try:
        names = os.listxattr(path, follow_symlinks=False)
    except OSError as error:
        return values, [], safe_error(error)
    for name in names:
        key = name if isinstance(name, str) else display_bytes(name)
        try:
            values[key] = b64(os.getxattr(path, name, follow_symlinks=False))
        except OSError as error:
            errors.append(f"{key}: {safe_error(error)}")
    return values, sorted(values), "; ".join(errors) if errors else None


def read_acl(path: bytes, xattr_names: list[str]) -> tuple[int, int, str | None, str | None]:
    access = int("system.posix_acl_access" in xattr_names)
    default = int("system.posix_acl_default" in xattr_names)
    if not access and not default:
        return access, default, None, None
    executable = shutil.which("getfacl")
    if not executable:
        return access, default, None, "getfacl unavailable"
    try:
        process = subprocess.run(
            [os.fsencode(executable), b"-cp", b"--absolute-names", b"--", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        text = process.stdout.decode("utf-8", "backslashreplace")
        error = process.stderr.decode("utf-8", "backslashreplace").strip()
        if process.returncode != 0:
            return access, default, text or None, error or f"getfacl returned {process.returncode}"
        return access, default, text or None, error or None
    except Exception as error:
        return access, default, None, safe_error(error)

def open_for_read(path: bytes) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOATIME"):
        try:
            return os.open(path, flags | os.O_NOATIME)
        except OSError as error:
            if error.errno not in (errno.EPERM, errno.EINVAL, errno.EOPNOTSUPP):
                raise
    return os.open(path, flags)


def sample_hashes(path: bytes, size: int, sample_bytes: int = 65536) -> tuple[str | None, str | None, str | None]:
    try:
        descriptor = open_for_read(path)
        try:
            head = os.read(descriptor, min(sample_bytes, size))
            if size <= sample_bytes:
                tail = head
            else:
                os.lseek(descriptor, max(0, size - sample_bytes), os.SEEK_SET)
                tail = os.read(descriptor, min(sample_bytes, size))
            return hashlib.sha256(head).hexdigest(), hashlib.sha256(tail).hexdigest(), None
        finally:
            os.close(descriptor)
    except Exception as error:
        return None, None, safe_error(error)


def sparse_metadata(path: bytes, size: int) -> tuple[list[dict[str, Any]], str | None, int, str | None]:
    if size == 0 or not hasattr(os, "SEEK_DATA") or not hasattr(os, "SEEK_HOLE"):
        return [], None, 0, None
    extents: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    physical_bytes = 0
    try:
        descriptor = open_for_read(path)
        try:
            position = 0
            while position < size:
                try:
                    start = os.lseek(descriptor, position, os.SEEK_DATA)
                except OSError as error:
                    if error.errno == errno.ENXIO:
                        break
                    raise
                try:
                    end = os.lseek(descriptor, start, os.SEEK_HOLE)
                except OSError:
                    end = size
                end = min(end, size)
                os.lseek(descriptor, start, os.SEEK_SET)
                extent_hash = hashlib.sha256()
                remaining = end - start
                combined.update(struct.pack(">QQ", start, remaining))
                while remaining:
                    chunk = os.read(descriptor, min(1 << 20, remaining))
                    if not chunk:
                        raise OSError("unexpected end of sparse extent")
                    extent_hash.update(chunk)
                    combined.update(chunk)
                    physical_bytes += len(chunk)
                    remaining -= len(chunk)
                extents.append({"offset": start, "length": end - start, "sha256": extent_hash.hexdigest()})
                position = end
        finally:
            os.close(descriptor)
        return extents, combined.hexdigest(), physical_bytes, None
    except Exception as error:
        return extents, combined.hexdigest() if extents else None, physical_bytes, safe_error(error)


def hash_regular_file(path: bytes, size: int, is_sparse: bool, max_hash_bytes: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "hash_algorithm": "sha256",
        "content_sha256": None,
        "hash_status": None,
        "hash_bytes_read": 0,
        "head_sha256": None,
        "tail_sha256": None,
        "hash_error": None,
    }
    if size > max_hash_bytes:
        result["hash_status"] = "skipped_sparse_large" if is_sparse else "skipped_large"
        head, tail, error = sample_hashes(path, size)
        result["head_sha256"] = head
        result["tail_sha256"] = tail
        result["hash_error"] = error
        return result
    try:
        digest = hashlib.sha256()
        descriptor = open_for_read(path)
        try:
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                result["hash_bytes_read"] += len(chunk)
        finally:
            os.close(descriptor)
        result["content_sha256"] = digest.hexdigest()
        result["hash_status"] = "complete"
        result["head_sha256"] = result["content_sha256"] if size <= 65536 else None
        result["tail_sha256"] = result["content_sha256"] if size <= 65536 else None
    except Exception as error:
        result["hash_status"] = "error"
        result["hash_error"] = safe_error(error)
    return result


def parse_key_value_output(raw: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in raw.decode("utf-8", "backslashreplace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def pdf_metadata(path: bytes) -> tuple[dict[str, Any], str | None]:
    executable = shutil.which("pdfinfo")
    if not executable:
        return {}, "pdfinfo unavailable"
    try:
        process = subprocess.run(
            [os.fsencode(executable), path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
        fields = parse_key_value_output(process.stdout)
        error = process.stderr.decode("utf-8", "backslashreplace").strip()
        if process.returncode != 0:
            return {"raw": fields}, error or f"pdfinfo returned {process.returncode}"
        return {
            "raw": fields,
            "title": fields.get("Title"),
            "author": fields.get("Author"),
            "subject": fields.get("Subject"),
            "keywords": fields.get("Keywords"),
            "creator": fields.get("Creator"),
            "producer": fields.get("Producer"),
            "creation_date": fields.get("CreationDate"),
            "modification_date": fields.get("ModDate"),
            "pages": int(fields["Pages"]) if fields.get("Pages", "").isdigit() else None,
            "encrypted": fields.get("Encrypted"),
            "page_size": fields.get("Page size"),
            "version": fields.get("PDF version"),
        }, error or None
    except Exception as error:
        return {}, safe_error(error)


def ffprobe_metadata(path: bytes) -> tuple[dict[str, Any], str | None]:
    executable = shutil.which("ffprobe")
    if not executable:
        return {}, "ffprobe unavailable"
    try:
        process = subprocess.run(
            [os.fsencode(executable), b"-v", b"error", b"-show_format", b"-show_streams", b"-of", b"json", b"--", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
            check=False,
        )
        error = process.stderr.decode("utf-8", "backslashreplace").strip()
        if process.returncode != 0:
            return {}, error or f"ffprobe returned {process.returncode}"
        return json.loads(process.stdout.decode("utf-8", "strict")), error or None
    except Exception as error:
        return {}, safe_error(error)


def image_metadata(path: bytes) -> tuple[dict[str, Any], str | None]:
    try:
        from PIL import Image, ExifTags
    except ImportError:
        return {}, "Pillow unavailable"
    try:
        with Image.open(path) as image:
            exif: dict[str, Any] = {}
            try:
                for key, value in image.getexif().items():
                    name = ExifTags.TAGS.get(key, str(key))
                    if isinstance(value, bytes):
                        exif[name] = {"base64": b64(value)}
                    elif isinstance(value, (str, int, float, bool)) or value is None:
                        exif[name] = value
                    else:
                        exif[name] = repr(value)
            except Exception as error:
                exif["_error"] = safe_error(error)
            return {
                "format": image.format,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "frame_count": int(getattr(image, "n_frames", 1)),
                "info": {
                    str(key): (value if isinstance(value, (str, int, float, bool)) or value is None else repr(value))
                    for key, value in image.info.items()
                },
                "exif": exif,
            }, None
    except Exception as error:
        return {}, safe_error(error)


def office_metadata(path: bytes) -> tuple[dict[str, Any], str | None]:
    try:
        descriptor = open_for_read(path)
        file_object = os.fdopen(descriptor, "rb", closefd=True)
        with file_object, zipfile.ZipFile(file_object) as archive:
            result: dict[str, Any] = {"members": len(archive.infolist())}
            for member, section in (("docProps/core.xml", "core"), ("docProps/app.xml", "app"), ("docProps/custom.xml", "custom")):
                if member not in archive.namelist():
                    continue
                root = ElementTree.fromstring(archive.read(member))
                values: dict[str, Any] = {}
                for element in root.iter():
                    if element is root:
                        continue
                    tag = element.tag.rsplit("}", 1)[-1]
                    text = element.text.strip() if element.text else None
                    if text is not None:
                        values[tag] = text
                result[section] = values
            return result, None
    except Exception as error:
        return {}, safe_error(error)


def collect_embedded_metadata(path: bytes, extension: str | None, mime_type: str | None) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    combined: dict[str, Any] = {}
    flat: dict[str, Any] = {}
    errors: list[str] = []
    extension_lower = (extension or "").lower()
    mime_value = mime_type or ""
    if mime_value == "application/pdf":
        value, error = pdf_metadata(path)
        combined["pdf"] = value
        if error:
            errors.append(error)
        flat.update({
            "pdf_title": value.get("title"),
            "pdf_author": value.get("author"),
            "pdf_subject": value.get("subject"),
            "pdf_keywords": value.get("keywords"),
            "pdf_creator": value.get("creator"),
            "pdf_producer": value.get("producer"),
            "pdf_creation_date": value.get("creation_date"),
            "pdf_modification_date": value.get("modification_date"),
            "pdf_pages": value.get("pages"),
            "pdf_encrypted": value.get("encrypted"),
            "pdf_page_size": value.get("page_size"),
            "pdf_version": value.get("version"),
        })
    if mime_value.startswith(("audio/", "video/")):
        value, error = ffprobe_metadata(path)
        combined["media"] = value
        if error:
            errors.append(error)
        format_value = value.get("format", {}) if isinstance(value, dict) else {}
        streams = value.get("streams", []) if isinstance(value, dict) else []
        try:
            duration = float(format_value["duration"]) if format_value.get("duration") is not None else None
        except (TypeError, ValueError):
            duration = None
        try:
            bit_rate = int(format_value["bit_rate"]) if format_value.get("bit_rate") is not None else None
        except (TypeError, ValueError):
            bit_rate = None
        flat.update({
            "media_format_name": format_value.get("format_name"),
            "media_duration_seconds": duration,
            "media_bit_rate": bit_rate,
            "media_stream_count": len(streams),
            "media_tags_json": json_text(format_value.get("tags", {})),
        })
    if mime_value in {"image/jpeg", "image/png", "image/gif", "image/tiff", "image/bmp", "image/webp", "image/x-icon"}:
        value, error = image_metadata(path)
        combined["image"] = value
        if error:
            errors.append(error)
        flat.update({
            "image_format": value.get("format"),
            "image_width": value.get("width"),
            "image_height": value.get("height"),
            "image_mode": value.get("mode"),
            "image_frame_count": value.get("frame_count"),
            "image_exif_json": json_text(value.get("exif", {})) if value else None,
        })
    office_extensions = {".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp", ".epub"}
    office_mime_types = {
        "application/epub+zip",
        "application/vnd.oasis.opendocument.presentation",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.text",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
    }
    if extension_lower in office_extensions and mime_value in office_mime_types:
        value, error = office_metadata(path)
        combined["office"] = value
        if error:
            errors.append(error)
        core = value.get("core", {}) if isinstance(value, dict) else {}
        flat.update({
            "office_title": core.get("title"),
            "office_subject": core.get("subject"),
            "office_creator": core.get("creator"),
            "office_keywords": core.get("keywords"),
            "office_description": core.get("description"),
            "office_last_modified_by": core.get("lastModifiedBy"),
            "office_created": core.get("created"),
            "office_modified": core.get("modified"),
            "office_revision": core.get("revision"),
            "office_category": core.get("category"),
            "office_content_status": core.get("contentStatus"),
            "office_language": core.get("language"),
            "office_version": core.get("version"),
        })
    return combined, flat, errors

class FilesystemScanner:
    def __init__(
        self,
        root: os.PathLike[str] | str | bytes,
        *,
        max_hash_bytes: int = 256 * 1024 * 1024,
        content_placeholder: str = PENDING_CONTENT,
        progress_every: int = 1000,
    ) -> None:
        root_bytes = os.fsencode(root)
        self.root = os.path.abspath(root_bytes).rstrip(b"/") or b"/"
        if not os.path.isdir(self.root):
            raise NotADirectoryError(display_bytes(self.root))
        self.max_hash_bytes = max_hash_bytes
        self.content_placeholder = content_placeholder
        self.progress_every = progress_every
        self.scan_id = str(uuid.uuid4())
        self.scanned_at_ns = time.time_ns()
        self.mount = find_mount_info(self.root)
        self.statvfs = os.statvfs(self.root)
        self.statx = LinuxStatx()
        self.now_ns = time.time_ns()

    def iter_entries(self) -> Iterator[tuple[bytes, str | None]]:
        try:
            children = sorted(os.listdir(self.root), reverse=True)
        except OSError as error:
            raise PermissionError(f"cannot enumerate scan root: {safe_error(error)}") from error
        stack: list[bytes] = list(children)
        while stack:
            relative = stack.pop()
            full_path = os.path.join(self.root, relative)
            traversal_error: str | None = None
            try:
                metadata = os.lstat(full_path)
                if stat.S_ISDIR(metadata.st_mode):
                    try:
                        names = sorted(os.listdir(full_path), reverse=True)
                        stack.extend(os.path.join(relative, name) for name in names)
                    except OSError as error:
                        traversal_error = safe_error(error)
            except OSError:
                pass
            yield relative, traversal_error

    def base_row(self, row_id: int, relative: bytes, traversal_error: str | None) -> dict[str, Any]:
        row = {name: None for name in COLUMN_NAMES}
        name = os.path.basename(relative)
        parent = os.path.dirname(relative)
        suffixes = extensions(name)
        extension = display_bytes(suffixes[-1]) if suffixes else None
        stem = name
        for suffix in suffixes:
            stem = stem[: -len(suffix)]
        row.update({
            "id": row_id,
            "scan_id": self.scan_id,
            "scanner_version": __version__,
            "scanned_at_ns": self.scanned_at_ns,
            "scan_root_display": display_bytes(self.root),
            "scan_root_b64": b64(self.root),
            "relative_path_display": display_bytes(relative),
            "relative_path_b64": b64(relative),
            "relative_path_hex": relative.hex(),
            "parent_path_display": display_bytes(parent),
            "parent_path_b64": b64(parent),
            "name_display": display_bytes(name),
            "name_b64": b64(name),
            "name_hex": name.hex(),
            "path_depth": relative.count(b"/") + 1,
            "name_length_bytes": len(name),
            "extension": extension,
            "extensions_json": json_text([display_bytes(value) for value in suffixes]),
            "stem_display": display_bytes(stem),
            "is_hidden": int(name.startswith(b".")),
            "entry_type": "unstatable",
            "is_regular": 0,
            "is_directory": 0,
            "is_symlink": 0,
            "is_fifo": 0,
            "is_socket": 0,
            "is_block_device": 0,
            "is_character_device": 0,
            "is_other": 1,
            "filesystem_type": self.mount.filesystem_type,
            "mount_source": self.mount.source,
            "mount_point_display": display_bytes(self.mount.mount_point),
            "mount_options": self.mount.options,
            "statvfs_block_size": self.statvfs.f_bsize,
            "statvfs_fragment_size": self.statvfs.f_frsize,
            "statvfs_blocks": self.statvfs.f_blocks,
            "statvfs_blocks_free": self.statvfs.f_bfree,
            "statvfs_blocks_available": self.statvfs.f_bavail,
            "statvfs_files": self.statvfs.f_files,
            "statvfs_files_free": self.statvfs.f_ffree,
            "statvfs_files_available": self.statvfs.f_favail,
            "statvfs_name_max": self.statvfs.f_namemax,
            "xattr_count": 0,
            "xattr_names_json": "[]",
            "xattrs_b64_json": "{}",
            "acl_access_present": 0,
            "acl_default_present": 0,
            "extension_mime_type": mimetypes.guess_type(display_bytes(name), strict=False)[0],
            "embedded_metadata_status": "not_applicable",
            "metadata_extraction_status": "not_applicable",
            "metadata_parser_attempts_json": "[]",
            "raw_metadata_json": "{}",
            "raw_metadata_source_count": 0,
            "exiftool_status": "not_applicable",
            "media_tags_json": "{}",
            "scan_error": traversal_error,
            "content_summary_short": self.content_placeholder,
            "content_summary_long": self.content_placeholder,
            "content_description": self.content_placeholder,
            "content_keywords": self.content_placeholder,
            "content_entities": self.content_placeholder,
        })
        return row

    def build_row(self, row_id: int, relative: bytes, traversal_error: str | None, magic: LibMagic, exiftool: ExifToolClient) -> dict[str, Any]:
        row = self.base_row(row_id, relative, traversal_error)
        path = os.path.join(self.root, relative)
        metadata_errors: list[str] = []
        try:
            metadata = os.lstat(path)
        except OSError as error:
            row["stat_error"] = safe_error(error)
            return row

        kind = entry_type(metadata.st_mode)
        permissions = stat.S_IMODE(metadata.st_mode)
        allocated = metadata.st_blocks * 512
        sparse = int(kind == "file" and metadata.st_size > 0 and allocated < metadata.st_size)
        birth_ns, birth_error = self.statx.birth_time_ns(path)
        if birth_error:
            metadata_errors.append(birth_error)
        times = [metadata.st_atime_ns, metadata.st_mtime_ns, metadata.st_ctime_ns]
        if birth_ns is not None:
            times.append(birth_ns)
        row.update({
            "entry_type": kind,
            "is_regular": int(kind == "file"),
            "is_directory": int(kind == "directory"),
            "is_symlink": int(kind == "symlink"),
            "is_fifo": int(kind == "fifo"),
            "is_socket": int(kind == "socket"),
            "is_block_device": int(kind == "block_device"),
            "is_character_device": int(kind == "character_device"),
            "is_other": int(kind == "other"),
            "device_id": str(metadata.st_dev),
            "device_major": os.major(metadata.st_dev),
            "device_minor": os.minor(metadata.st_dev),
            "inode": str(metadata.st_ino),
            "hardlink_key": f"{metadata.st_dev}:{metadata.st_ino}" if kind == "file" and metadata.st_nlink > 1 else None,
            "link_count": metadata.st_nlink,
            "mode_int": metadata.st_mode,
            "mode_octal": f"{metadata.st_mode:o}",
            "mode_symbolic": stat.filemode(metadata.st_mode),
            "permissions_octal": f"{permissions:04o}",
            "uid": metadata.st_uid,
            "username": username(metadata.st_uid),
            "gid": metadata.st_gid,
            "groupname": groupname(metadata.st_gid),
            "size_bytes": metadata.st_size,
            "allocated_bytes": allocated,
            "blocks_512": metadata.st_blocks,
            "io_block_size": metadata.st_blksize,
            "is_sparse": sparse,
            "sparse_ratio": (allocated / metadata.st_size) if metadata.st_size else None,
            "rdev": str(metadata.st_rdev),
            "rdev_major": os.major(metadata.st_rdev) if metadata.st_rdev else 0,
            "rdev_minor": os.minor(metadata.st_rdev) if metadata.st_rdev else 0,
            "atime_ns": timestamp_storage(metadata.st_atime_ns),
            "mtime_ns": timestamp_storage(metadata.st_mtime_ns),
            "ctime_ns": timestamp_storage(metadata.st_ctime_ns),
            "birth_time_ns": timestamp_storage(birth_ns),
            "atime_iso": timestamp_iso(metadata.st_atime_ns),
            "mtime_iso": timestamp_iso(metadata.st_mtime_ns),
            "ctime_iso": timestamp_iso(metadata.st_ctime_ns),
            "birth_time_iso": timestamp_iso(birth_ns),
            "mtime_before_birth": int(birth_ns is not None and metadata.st_mtime_ns < birth_ns),
            "atime_before_mtime": int(metadata.st_atime_ns < metadata.st_mtime_ns),
            "has_pre_epoch_time": int(any(value < 0 for value in times)),
            "has_future_time": int(any(value > self.now_ns + 86_400_000_000_000 for value in times)),
        })

        flags_value, flags_names, flags_error = filesystem_flags(path, metadata.st_mode)
        row["filesystem_flags_int"] = flags_value
        row["filesystem_flags_json"] = json_text(flags_names) if flags_value is not None else None
        if flags_error:
            metadata_errors.append(flags_error)

        xattrs, xattr_names, xattr_error = read_xattrs(path)
        row["xattr_count"] = len(xattrs)
        row["xattr_names_json"] = json_text(xattr_names)
        row["xattrs_b64_json"] = json_text(xattrs)
        if xattr_error:
            metadata_errors.append(xattr_error)
        access_acl, default_acl, acl_text, acl_error = read_acl(path, xattr_names)
        row["acl_access_present"] = access_acl
        row["acl_default_present"] = default_acl
        row["acl_text"] = acl_text
        if acl_error:
            metadata_errors.append(acl_error)

        if kind == "symlink":
            try:
                target = os.readlink(path)
                row["symlink_target_display"] = display_bytes(target)
                row["symlink_target_b64"] = b64(target)
                row["symlink_target_hex"] = target.hex()
                row["symlink_target_is_absolute"] = int(os.path.isabs(target))
                candidate = target if os.path.isabs(target) else os.path.join(os.path.dirname(path), target)
                row["symlink_target_exists"] = int(os.path.exists(candidate))
                try:
                    row["symlink_target_type"] = entry_type(os.stat(candidate).st_mode)
                except OSError:
                    row["symlink_target_type"] = None
                resolved = os.path.realpath(candidate)
                root_prefix = self.root.rstrip(b"/") + b"/"
                row["symlink_resolves_outside_root"] = int(resolved != self.root and not resolved.startswith(root_prefix))
            except OSError as error:
                metadata_errors.append(safe_error(error))

        if kind == "file":
            magic_values, magic_error = magic.inspect(path)
            row["magic_description"] = magic_values["description"]
            row["magic_mime_type"] = magic_values["mime_type"]
            row["magic_mime_encoding"] = magic_values["mime_encoding"]
            if magic_error:
                metadata_errors.append(magic_error)
            if sparse:
                extents, physical_hash, physical_bytes, sparse_error = sparse_metadata(path, metadata.st_size)
                row["sparse_extents_json"] = json_text(extents)
                row["physical_data_sha256"] = physical_hash
                row["physical_data_bytes"] = physical_bytes
                if sparse_error:
                    metadata_errors.append(sparse_error)
            row.update(hash_regular_file(path, metadata.st_size, bool(sparse), self.max_hash_bytes))
            raw_metadata, parser_attempts, parser_errors, extraction_status = collect_raw_metadata(
                path,
                row["extension"],
                row["magic_mime_type"],
                row["magic_description"],
                exiftool,
            )
            row["metadata_extraction_status"] = extraction_status
            row["metadata_parser_attempts_json"] = json_text(parser_attempts)
            row["raw_metadata_json"] = json_text(raw_metadata)
            row["raw_metadata_source_count"] = len(raw_metadata)
            exif_attempt = next((item for item in parser_attempts if item["parser"] == "exiftool"), None)
            row["exiftool_status"] = exif_attempt["status"] if exif_attempt else "error"
            row["exiftool_metadata_json"] = json_text(raw_metadata["exiftool"]) if "exiftool" in raw_metadata else None
            row["exiftool_error"] = parser_errors.get("exiftool")
            for parser_name, metadata_column, error_column in (
                ("archive", "archive_metadata_json", "archive_error"),
                ("sqlite", "database_metadata_json", "database_error"),
                ("message", "message_metadata_json", "message_error"),
                ("certificate", "certificate_metadata_json", "certificate_error"),
                ("font", "font_metadata_json", "font_error"),
                ("executable", "executable_metadata_json", "executable_error"),
                ("scientific", "scientific_metadata_json", "scientific_error"),
                ("version_control", "version_control_metadata_json", "version_control_error"),
                ("compressed_stream", "compressed_stream_metadata_json", "compressed_stream_error"),
                ("byte_structure", "byte_structure_metadata_json", "byte_structure_error"),
                ("intentional_unsupported", "intentional_unsupported_metadata_json", None),
            ):
                row[metadata_column] = json_text(raw_metadata[parser_name]) if parser_name in raw_metadata else None
                if error_column is not None:
                    row[error_column] = parser_errors.get(parser_name)
            specialized_complete = any(
                item["parser"] not in {"exiftool", "intentional_unsupported"} and item["status"] == "complete"
                for item in parser_attempts
            )
            parser_failure_names = {
                item["parser"]
                for item in parser_attempts
                if item["status"] in {"error", "partial", "unavailable"}
                and not (item["parser"] == "exiftool" and specialized_complete)
            }
            metadata_errors.extend(
                f"{name}: {error}" for name, error in parser_errors.items() if error and name in parser_failure_names
            )
            embedded, embedded_flat, embedded_errors = collect_embedded_metadata(
                path,
                row["extension"],
                row["magic_mime_type"],
            )
            row.update(embedded_flat)
            row["embedded_metadata_json"] = json_text(embedded) if embedded else None
            if embedded:
                row["embedded_metadata_status"] = "partial" if embedded_errors else "complete"
            elif embedded_errors:
                row["embedded_metadata_status"] = "error"
            if embedded_errors:
                metadata_errors.extend(embedded_errors)

        row["metadata_error"] = "; ".join(metadata_errors) if metadata_errors else None
        return row

    def scan_to_database(self, database_path: os.PathLike[str] | str, *, replace: bool = False) -> dict[str, Any]:
        destination = Path(database_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not replace:
            raise FileExistsError(f"database already exists: {destination}")
        partial = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
        partial.unlink(missing_ok=True)
        started = time.monotonic()
        counts: dict[str, int] = {}
        errors = 0
        metadata_errors = 0
        inserted = 0
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(partial)
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA locking_mode=EXCLUSIVE")
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute("PRAGMA cache_size=-65536")
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            connection.execute(CREATE_TABLE_SQL)
            for statement in CONTENT_CREATE_SQL:
                connection.execute(statement)
            placeholders = ",".join("?" for _ in COLUMN_NAMES)
            columns = ",".join(f'"{name}"' for name in COLUMN_NAMES)
            insert_sql = f"INSERT INTO {TABLE_NAME} ({columns}) VALUES ({placeholders})"
            connection.execute("BEGIN")
            with LibMagic() as magic, ExifToolClient() as exiftool:
                for row_id, (relative, traversal_error) in enumerate(self.iter_entries(), start=1):
                    row = self.build_row(row_id, relative, traversal_error, magic, exiftool)
                    connection.execute(insert_sql, [row[name] for name in COLUMN_NAMES])
                    inserted += 1
                    counts[row["entry_type"]] = counts.get(row["entry_type"], 0) + 1
                    if row["scan_error"] or row["stat_error"] or row["hash_error"]:
                        errors += 1
                    if row["metadata_error"]:
                        metadata_errors += 1
                    if self.progress_every and inserted % self.progress_every == 0:
                        print(json_text({"progress": inserted, "path": row["relative_path_display"]}), flush=True)
            connection.commit()
            for statement in INDEX_SQL + CONTENT_INDEX_SQL:
                connection.execute(statement)
            connection.commit()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            table_count = connection.execute(
                "SELECT count(*) FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchone()[0]
            row_count = connection.execute(f"SELECT count(*) FROM {TABLE_NAME}").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"SQLite integrity check failed: {integrity}")
            if table_count != 3:
                raise RuntimeError(f"expected three user tables, found {table_count}")
            chunk_row_count = connection.execute(f"SELECT count(*) FROM {CHUNK_TABLE_NAME}").fetchone()[0]
            representation_row_count = connection.execute(f"SELECT count(*) FROM {REPRESENTATION_TABLE_NAME}").fetchone()[0]
            if chunk_row_count != 0 or representation_row_count != 0:
                raise RuntimeError(
                    f"new content tables are not empty: chunks={chunk_row_count}, representations={representation_row_count}"
                )
            if row_count != inserted:
                raise RuntimeError(f"row count mismatch: inserted={inserted}, stored={row_count}")
            connection.close()
            connection = None
            os.replace(partial, destination)
            duration = time.monotonic() - started
            return {
                "status": "ok",
                "scan_id": self.scan_id,
                "database": str(destination),
                "root": display_bytes(self.root),
                "rows": inserted,
                "counts": counts,
                "rows_with_scan_or_hash_errors": errors,
                "rows_with_metadata_parser_errors": metadata_errors,
                "columns": len(COLUMN_NAMES),
                "user_tables": table_count,
                "database_bytes": destination.stat().st_size,
                "duration_seconds": round(duration, 6),
                "content_placeholder": self.content_placeholder,
            }
        except Exception:
            if connection is not None:
                connection.close()
            partial.unlink(missing_ok=True)
            raise
