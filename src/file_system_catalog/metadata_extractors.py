from __future__ import annotations

import base64
import email.policy
import email.parser
import hashlib
import json
import math
import os
import re
import select
import shutil
import sqlite3
import struct
import subprocess
import time
import urllib.parse
import zlib
from typing import Any, Callable


def safe_error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}".encode("utf-8", "backslashreplace").decode("utf-8")


def decode_output(value: bytes) -> str:
    return value.decode("utf-8", "backslashreplace")


def run_command(arguments: list[bytes], *, timeout: int = 30) -> dict[str, Any]:
    try:
        process = subprocess.run(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
        return {
            "status": "complete" if process.returncode == 0 else "error",
            "returncode": process.returncode,
            "stdout": decode_output(process.stdout),
            "stderr": decode_output(process.stderr),
        }
    except Exception as error:
        return {"status": "error", "returncode": None, "stdout": "", "stderr": safe_error(error)}


class ExifToolClient:
    def __init__(self, timeout: int = 45) -> None:
        self.timeout = timeout
        self.executable = shutil.which("exiftool")
        self.process: subprocess.Popen[bytes] | None = None
        self.sequence = 0
        if self.executable:
            self._start()

    def _start(self) -> None:
        if not self.executable:
            return
        self.process = subprocess.Popen(
            [os.fsencode(self.executable), b"-stay_open", b"True", b"-@", b"-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def _stop(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.write(b"-stay_open\nFalse\n")
                process.stdin.flush()
            process.wait(timeout=5)
        except Exception:
            process.kill()
            try:
                process.wait(timeout=2)
            except Exception:
                pass
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()

    def _restart(self) -> None:
        self._stop()
        self._start()

    def _direct(self, path: bytes) -> tuple[dict[str, Any] | None, str | None]:
        if not self.executable:
            return None, "exiftool unavailable"
        arguments = [
            os.fsencode(self.executable), b"-json", b"-G1:4", b"-a", b"-u", b"-n", b"-struct", b"-ee3",
            b"-api", b"LargeFileSupport=1", b"--", path,
        ]
        try:
            process = subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.timeout, check=False)
            if not process.stdout:
                error = decode_output(process.stderr).strip() or f"exiftool returned {process.returncode}"
                return None, error
            values = json.loads(process.stdout.decode("utf-8", "surrogateescape"))
            metadata = values[0] if values else None
            errors = []
            if process.returncode != 0:
                errors.append(f"exiftool returned {process.returncode}")
            if process.stderr:
                errors.append(decode_output(process.stderr).strip())
            if isinstance(metadata, dict) and metadata.get("ExifTool:Error"):
                errors.append(str(metadata["ExifTool:Error"]))
            return metadata, "; ".join(value for value in errors if value) or None
        except Exception as error:
            return None, safe_error(error)

    def inspect(self, path: bytes) -> tuple[dict[str, Any] | None, str | None]:
        if not self.executable:
            return None, "exiftool unavailable"
        if b"\n" in path or b"\r" in path:
            return self._direct(path)
        if self.process is None or self.process.poll() is not None:
            self._restart()
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            return None, "exiftool process unavailable"
        self.sequence += 1
        identifier = str(self.sequence).encode("ascii")
        command = (
            b"-json\n-G1:4\n-a\n-u\n-n\n-struct\n-ee3\n-api\nLargeFileSupport=1\n"
            + path + b"\n-execute" + identifier + b"\n"
        )
        try:
            process.stdin.write(command)
            process.stdin.flush()
            marker = b"{ready" + identifier + b"}"
            output: list[bytes] = []
            deadline = time.monotonic() + self.timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"exiftool timed out after {self.timeout}s")
                readable, _, _ = select.select([process.stdout], [], [], remaining)
                if not readable:
                    raise TimeoutError(f"exiftool timed out after {self.timeout}s")
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError("exiftool terminated unexpectedly")
                if line.rstrip(b"\r\n") == marker:
                    break
                output.append(line)
            raw = b"".join(output)
            if not raw.strip():
                return None, "exiftool returned no metadata"
            values = json.loads(raw.decode("utf-8", "surrogateescape"))
            metadata = values[0] if values else None
            error = None
            if isinstance(metadata, dict) and metadata.get("ExifTool:Error"):
                error = str(metadata["ExifTool:Error"])
            return metadata, error
        except Exception as error:
            self._restart()
            return None, safe_error(error)

    def close(self) -> None:
        self._stop()

    def __enter__(self) -> "ExifToolClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()



def archive_metadata(path: bytes, extension: str, mime_type: str, description: str) -> tuple[dict[str, Any] | None, str | None, bool]:
    lowered = description.lower()
    applicable = (
        any(term in lowered for term in (
            "zip archive", "rar archive", "7-zip archive", "tar archive", "gzip compressed data",
            "bzip2 compressed data", "xz compressed data", "iso 9660",
        ))
        or mime_type in {"application/zip", "application/x-tar", "application/gzip", "application/x-7z-compressed", "application/x-xz", "application/x-bzip2"}
    )
    if not applicable:
        return None, None, False
    executable = shutil.which("7z")
    if not executable:
        return None, "7z unavailable", True
    result = run_command([os.fsencode(executable), b"l", b"-slt", b"-ba", b"--", path], timeout=45)
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in result["stdout"].splitlines():
        if not line.strip():
            if current:
                records.append(current)
                current = {}
            continue
        if " = " in line:
            key, value = line.split(" = ", 1)
            if key in current:
                existing = current[key]
                current[key] = existing + "\n" + value
            else:
                current[key] = value
        else:
            current.setdefault("_unparsed", "")
            current["_unparsed"] += line + "\n"
    if current:
        records.append(current)
    value = {"tool": "7z", "records": records, "raw_stdout": result["stdout"], "raw_stderr": result["stderr"], "returncode": result["returncode"]}
    error = result["stderr"].strip() if result["status"] != "complete" else None
    if result["status"] != "complete" and not error:
        error = f"7z returned {result['returncode']}"
    return value, error, True


def _pragma(connection: sqlite3.Connection, name: str) -> list[list[Any]]:
    return [list(row) for row in connection.execute(f"PRAGMA {name}").fetchall()]


def sqlite_metadata(path: bytes, extension: str, mime_type: str, description: str) -> tuple[dict[str, Any] | None, str | None, bool]:
    ext = extension.lower()
    lowered = description.lower()
    if ext == ".sqlite-wal" or "write-ahead log" in lowered:
        try:
            size = os.lstat(path).st_size
            with open(path, "rb") as handle:
                header = handle.read(32)
            if len(header) < 32:
                raise ValueError("SQLite WAL header is truncated")
            magic, version, page_size, checkpoint, salt1, salt2, checksum1, checksum2 = struct.unpack(">8I", header)
            effective_page_size = 65536 if page_size == 1 else page_size
            frame_size = effective_page_size + 24 if effective_page_size else None
            frame_count = (size - 32) // frame_size if frame_size and size >= 32 else None
            return {
                "kind": "sqlite_wal",
                "magic": f"0x{magic:08x}",
                "format_version": version,
                "page_size": effective_page_size,
                "checkpoint_sequence": checkpoint,
                "salt": [salt1, salt2],
                "header_checksum": [checksum1, checksum2],
                "checksum_byte_order": "big" if magic == 0x377F0683 else "little" if magic == 0x377F0682 else "unknown",
                "file_size": size,
                "frame_size": frame_size,
                "frame_count": frame_count,
                "trailing_bytes": (size - 32) % frame_size if frame_size and size >= 32 else None,
            }, None, True
        except Exception as error:
            return None, safe_error(error), True
    if ext == ".sqlite-shm" or "wal-index" in lowered:
        try:
            size = os.lstat(path).st_size
            with open(path, "rb") as handle:
                header = handle.read(48)
            if len(header) < 48:
                raise ValueError("SQLite SHM header is truncated")
            values = struct.unpack("<IIIBBHIIIIIIII", header)
            return {
                "kind": "sqlite_shm",
                "version": values[0],
                "unused": values[1],
                "change_counter": values[2],
                "is_initialized": values[3],
                "big_endian_checksum": values[4],
                "page_size": values[5],
                "max_frame": values[6],
                "database_pages": values[7],
                "frame_checksum": [values[8], values[9]],
                "salt": [values[10], values[11]],
                "header_checksum": [values[12], values[13]],
                "file_size": size,
            }, None, True
        except Exception as error:
            return None, safe_error(error), True
    sqlite_signature = "sqlite 3.x database" in lowered or mime_type == "application/vnd.sqlite3"
    extension_candidate = ext in {".sqlite", ".sqlite3", ".db"}
    recognized_other_content = (
        mime_type.startswith(("text/", "image/", "audio/", "video/"))
        or mime_type in {"application/pdf", "application/zip", "message/rfc822"}
    )
    if extension_candidate and recognized_other_content and not sqlite_signature:
        return None, None, False
    applicable = sqlite_signature or extension_candidate
    if not applicable:
        return None, None, False
    uri = "file:" + urllib.parse.quote_from_bytes(path, safe="/") + "?mode=ro"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=2)
        connection.row_factory = sqlite3.Row
        pragmas: dict[str, Any] = {}
        for name in (
            "application_id", "auto_vacuum", "cache_size", "encoding", "freelist_count", "journal_mode", "legacy_file_format",
            "locking_mode", "max_page_count", "page_count", "page_size", "schema_version", "secure_delete", "synchronous", "user_version",
        ):
            try:
                pragmas[name] = _pragma(connection, name)
            except sqlite3.Error as error:
                pragmas[name] = {"error": safe_error(error)}
        schema = [dict(row) for row in connection.execute("SELECT type,name,tbl_name,rootpage,sql FROM sqlite_master ORDER BY type,name")]
        objects: dict[str, Any] = {}
        for item in schema:
            if item["type"] != "table" or item["name"].startswith("sqlite_"):
                continue
            quoted = item["name"].replace('"', '""')
            details: dict[str, Any] = {}
            for label, statement in (
                ("table_xinfo", f'PRAGMA table_xinfo("{quoted}")'),
                ("index_list", f'PRAGMA index_list("{quoted}")'),
                ("foreign_key_list", f'PRAGMA foreign_key_list("{quoted}")'),
            ):
                try:
                    details[label] = [dict(row) for row in connection.execute(statement)]
                except sqlite3.Error as error:
                    details[label] = {"error": safe_error(error)}
            objects[item["name"]] = details
        integrity = [list(row) for row in connection.execute("PRAGMA integrity_check").fetchall()]
        value = {"sqlite_version": sqlite3.sqlite_version, "pragmas": pragmas, "schema": schema, "objects": objects, "integrity_check": integrity}
        return value, None, True
    except Exception as error:
        return None, safe_error(error), True
    finally:
        if connection is not None:
            connection.close()


def message_metadata(path: bytes, extension: str, mime_type: str) -> tuple[dict[str, Any] | None, str | None, bool]:
    applicable = extension.lower() in {".eml", ".mime"} or mime_type == "message/rfc822"
    if not applicable:
        return None, None, False
    try:
        with open(path, "rb") as handle:
            message = email.parser.BytesParser(policy=email.policy.default).parse(handle)
        parts: list[dict[str, Any]] = []
        for index, part in enumerate(message.walk()):
            payload = part.get_payload(decode=True)
            parts.append({
                "index": index,
                "content_type": part.get_content_type(),
                "content_maintype": part.get_content_maintype(),
                "content_subtype": part.get_content_subtype(),
                "content_disposition": part.get_content_disposition(),
                "filename": part.get_filename(),
                "charset": part.get_content_charset(),
                "content_transfer_encoding": part.get("Content-Transfer-Encoding"),
                "content_id": part.get("Content-ID"),
                "payload_bytes": len(payload) if payload is not None else None,
                "is_multipart": part.is_multipart(),
                "defects": [str(defect) for defect in part.defects],
            })
        value = {
            "headers": [[name, str(value)] for name, value in message.raw_items()],
            "decoded_headers": [[name, str(value)] for name, value in message.items()],
            "defects": [str(defect) for defect in message.defects],
            "is_multipart": message.is_multipart(),
            "parts": parts,
        }
        return value, None, True
    except Exception as error:
        return None, safe_error(error), True


CERTIFICATE_EXTENSIONS = {".cer", ".crt", ".der", ".p7b", ".p7c", ".pem"}


def certificate_metadata(path: bytes, extension: str, description: str) -> tuple[dict[str, Any] | None, str | None, bool]:
    applicable = extension.lower() in CERTIFICATE_EXTENSIONS or any(term in description.lower() for term in ("certificate", "pem", "private key", "public key"))
    if not applicable:
        return None, None, False
    executable = shutil.which("openssl")
    if not executable:
        return None, "openssl unavailable", True
    base = os.fsencode(executable)
    attempts: list[dict[str, Any]] = []
    commands = [
        [base, b"x509", b"-in", path, b"-noout", b"-subject", b"-issuer", b"-serial", b"-dates", b"-fingerprint", b"-sha256", b"-ocsp_uri", b"-email", b"-nameopt", b"RFC2253"],
        [base, b"x509", b"-inform", b"DER", b"-in", path, b"-noout", b"-subject", b"-issuer", b"-serial", b"-dates", b"-fingerprint", b"-sha256", b"-ocsp_uri", b"-email", b"-nameopt", b"RFC2253"],
        [base, b"pkey", b"-in", path, b"-check", b"-noout"],
        [base, b"pkcs7", b"-in", path, b"-print_certs", b"-noout"],
    ]
    for command in commands:
        result = run_command(command, timeout=20)
        attempts.append({"command": [decode_output(item) for item in command[1:]], **result})
    successes = [attempt for attempt in attempts if attempt["status"] == "complete"]
    error = None if successes else "; ".join(attempt["stderr"].strip() for attempt in attempts if attempt["stderr"].strip())
    return {"tool": "openssl", "attempts": attempts}, error or None, True


def font_metadata(path: bytes, extension: str, mime_type: str, description: str) -> tuple[dict[str, Any] | None, str | None, bool]:
    applicable = extension.lower() in {".otf", ".ttc", ".ttf", ".woff", ".woff2"} or mime_type.startswith("font/") or "font" in description.lower()
    if not applicable:
        return None, None, False
    executable = shutil.which("fc-scan")
    if not executable:
        return None, "fc-scan unavailable", True
    format_value = "%{family}|%{style}|%{fullname}|%{fontversion}|%{fontformat}|%{index}|%{lang}|%{charset}\n"
    result = run_command([os.fsencode(executable), b"--format", format_value.encode(), b"--", path], timeout=20)
    value = {"tool": "fc-scan", "format": format_value, "raw_stdout": result["stdout"], "raw_stderr": result["stderr"], "returncode": result["returncode"]}
    error = result["stderr"].strip() if result["status"] != "complete" else None
    return value, error or None, True


def mz_pe_metadata(path: bytes) -> dict[str, Any]:
    file_size = os.lstat(path).st_size
    with open(path, "rb") as handle:
        header = handle.read(64)
        if len(header) < 64 or header[:2] != b"MZ":
            raise ValueError("DOS MZ header is missing or truncated")
        fields = struct.unpack_from("<14H", header, 0)
        e_lfanew = struct.unpack_from("<I", header, 0x3C)[0]
        value: dict[str, Any] = {
            "kind": "dos_mz",
            "file_size": file_size,
            "magic": header[:2].decode("ascii"),
            "bytes_on_last_page": fields[1],
            "pages_in_file": fields[2],
            "relocations": fields[3],
            "header_paragraphs": fields[4],
            "minimum_extra_paragraphs": fields[5],
            "maximum_extra_paragraphs": fields[6],
            "initial_ss": fields[7],
            "initial_sp": fields[8],
            "checksum": fields[9],
            "initial_ip": fields[10],
            "initial_cs": fields[11],
            "relocation_table_offset": fields[12],
            "overlay_number": fields[13],
            "new_header_offset": e_lfanew,
        }
        if e_lfanew <= 0 or e_lfanew + 24 > file_size:
            value["new_header_status"] = "absent_or_out_of_range"
            return value
        handle.seek(e_lfanew)
        signature = handle.read(4)
        value["new_header_signature_hex"] = signature.hex()
        if signature != b"PE\0\0":
            value["new_header_status"] = "not_pe"
            return value
        coff = handle.read(20)
        if len(coff) != 20:
            raise ValueError("PE COFF header is truncated")
        machine, section_count, timestamp, symbol_table_pointer, symbol_count, optional_size, characteristics = struct.unpack("<HHIIIHH", coff)
        optional = handle.read(optional_size)
        if len(optional) != optional_size:
            raise ValueError("PE optional header is truncated")
        optional_magic = struct.unpack_from("<H", optional, 0)[0] if len(optional) >= 2 else None
        pe: dict[str, Any] = {
            "kind": "pe",
            "machine": machine,
            "section_count": section_count,
            "timestamp": timestamp,
            "symbol_table_pointer": symbol_table_pointer,
            "symbol_count": symbol_count,
            "optional_header_size": optional_size,
            "characteristics": characteristics,
            "optional_magic": optional_magic,
        }
        if optional_magic in (0x10B, 0x20B) and len(optional) >= (32 if optional_magic == 0x10B else 32):
            pe["format"] = "PE32" if optional_magic == 0x10B else "PE32+"
            pe["linker_version"] = [optional[2], optional[3]]
            pe["size_of_code"] = struct.unpack_from("<I", optional, 4)[0]
            pe["size_of_initialized_data"] = struct.unpack_from("<I", optional, 8)[0]
            pe["size_of_uninitialized_data"] = struct.unpack_from("<I", optional, 12)[0]
            pe["address_of_entry_point"] = struct.unpack_from("<I", optional, 16)[0]
            pe["base_of_code"] = struct.unpack_from("<I", optional, 20)[0]
            if optional_magic == 0x10B and len(optional) >= 32:
                pe["base_of_data"] = struct.unpack_from("<I", optional, 24)[0]
                pe["image_base"] = struct.unpack_from("<I", optional, 28)[0]
            elif optional_magic == 0x20B and len(optional) >= 32:
                pe["image_base"] = struct.unpack_from("<Q", optional, 24)[0]
            if len(optional) >= 64:
                pe["section_alignment"] = struct.unpack_from("<I", optional, 32)[0]
                pe["file_alignment"] = struct.unpack_from("<I", optional, 36)[0]
                pe["size_of_image"] = struct.unpack_from("<I", optional, 56)[0]
                pe["size_of_headers"] = struct.unpack_from("<I", optional, 60)[0]
        sections: list[dict[str, Any]] = []
        for index in range(section_count):
            raw = handle.read(40)
            if len(raw) != 40:
                pe["section_table_error"] = f"section {index} header truncated"
                break
            name = raw[:8].rstrip(b"\0").decode("ascii", "backslashreplace")
            virtual_size, virtual_address, raw_size, raw_pointer, relocation_pointer, line_pointer, relocation_count, line_count, section_characteristics = struct.unpack_from("<IIIIIIHHI", raw, 8)
            sections.append({
                "index": index,
                "name": name,
                "virtual_size": virtual_size,
                "virtual_address": virtual_address,
                "raw_size": raw_size,
                "raw_pointer": raw_pointer,
                "relocation_pointer": relocation_pointer,
                "line_number_pointer": line_pointer,
                "relocation_count": relocation_count,
                "line_number_count": line_count,
                "characteristics": section_characteristics,
            })
        pe["sections"] = sections
        value["new_header_status"] = "pe"
        value["pe"] = pe
        return value


def executable_metadata(path: bytes, extension: str, description: str) -> tuple[dict[str, Any] | None, str | None, bool]:
    lowered = description.lower()
    applicable = any(term in lowered for term in ("elf ", "pe32", "ms-dos executable", "mach-o", "shared object", "linux kernel module"))
    if not applicable:
        return None, None, False
    attempts: list[dict[str, Any]] = []
    if "ms-dos executable" in lowered or "pe32" in lowered:
        try:
            attempts.append({"tool": "builtin_mz_pe", "status": "complete", "metadata": mz_pe_metadata(path)})
        except Exception as error:
            attempts.append({"tool": "builtin_mz_pe", "status": "error", "stderr": safe_error(error)})
    if "elf" in lowered or extension.lower() in {".elf", ".ko", ".so"}:
        executable = shutil.which("readelf")
        if executable:
            result = run_command([os.fsencode(executable), b"-a", b"-W", b"--", path], timeout=30)
            attempts.append({"tool": "readelf", **result})
    executable = shutil.which("objdump")
    if executable:
        result = run_command([os.fsencode(executable), b"-x", b"--", path], timeout=30)
        attempts.append({"tool": "objdump", **result})
    if extension.lower() == ".ko":
        executable = shutil.which("modinfo")
        if executable:
            result = run_command([os.fsencode(executable), path], timeout=20)
            attempts.append({"tool": "modinfo", **result})
    if not attempts:
        return None, "no executable metadata reader available", True
    successes = [attempt for attempt in attempts if attempt["status"] == "complete"]
    error = None if successes else "; ".join(attempt["stderr"].strip() for attempt in attempts if attempt["stderr"].strip())
    return {"attempts": attempts}, error or None, True


def scientific_metadata(path: bytes, extension: str, description: str) -> tuple[dict[str, Any] | None, str | None, bool]:
    ext = extension.lower()
    applicable = ext in {".h5", ".hdf5", ".nc", ".netcdf", ".npy", ".parquet", ".fits", ".fit", ".fts"} or any(term in description.lower() for term in ("hierarchical data format", "netcdf", "numpy array", "parquet", "fits image"))
    if not applicable:
        return None, None, False
    attempts: list[dict[str, Any]] = []
    if ext in {".h5", ".hdf5"} or "hierarchical data format" in description.lower():
        executable = shutil.which("h5dump")
        if executable:
            result = run_command([os.fsencode(executable), b"-H", b"-A", path], timeout=30)
            attempts.append({"tool": "h5dump", **result})
        else:
            attempts.append({"tool": "h5dump", "status": "unavailable", "stderr": "h5dump unavailable"})
    if ext in {".nc", ".netcdf"} or "netcdf" in description.lower():
        executable = shutil.which("ncdump")
        if executable:
            result = run_command([os.fsencode(executable), b"-h", path], timeout=30)
            attempts.append({"tool": "ncdump", **result})
        else:
            attempts.append({"tool": "ncdump", "status": "unavailable", "stderr": "ncdump unavailable"})
    if ext in {".fits", ".fit", ".fts"} or "fits image" in description.lower():
        try:
            file_size = os.lstat(path).st_size
            hdus: list[dict[str, Any]] = []
            with open(path, "rb") as handle:
                offset = 0
                hdu_index = 0
                while offset < file_size:
                    handle.seek(offset)
                    cards: list[str] = []
                    values: dict[str, list[str]] = {}
                    header_bytes = 0
                    end_found = False
                    while not end_found:
                        block = handle.read(2880)
                        if not block:
                            break
                        if len(block) != 2880:
                            raise ValueError("FITS header block is truncated")
                        header_bytes += 2880
                        for card_offset in range(0, 2880, 80):
                            card_bytes = block[card_offset:card_offset + 80]
                            card = card_bytes.decode("ascii", "replace")
                            cards.append(card)
                            keyword = card[:8].strip()
                            if keyword:
                                values.setdefault(keyword, []).append(card[10:].rstrip() if card[8:10] == "= " else card[8:].rstrip())
                            if keyword == "END":
                                end_found = True
                                break
                    if not cards:
                        break
                    if not end_found:
                        raise ValueError("FITS END card is missing")
                    def first_number(name: str, default: int = 0) -> int:
                        raw = values.get(name, [str(default)])[0].split("/", 1)[0].strip()
                        return int(raw)
                    bitpix = first_number("BITPIX")
                    naxis = first_number("NAXIS")
                    dimensions = [first_number(f"NAXIS{index}") for index in range(1, naxis + 1)]
                    pcount = first_number("PCOUNT")
                    gcount = first_number("GCOUNT", 1)
                    elements = 1
                    for dimension in dimensions:
                        elements *= dimension
                    data_bytes = ((abs(bitpix) // 8) * elements + pcount) * gcount if naxis > 0 else pcount * gcount
                    padded_data_bytes = ((data_bytes + 2879) // 2880) * 2880 if data_bytes else 0
                    hdus.append({
                        "index": hdu_index,
                        "offset": offset,
                        "header_bytes": header_bytes,
                        "data_bytes": data_bytes,
                        "padded_data_bytes": padded_data_bytes,
                        "bitpix": bitpix,
                        "naxis": naxis,
                        "dimensions": dimensions,
                        "pcount": pcount,
                        "gcount": gcount,
                        "keywords": values,
                        "cards": cards,
                    })
                    next_offset = offset + header_bytes + padded_data_bytes
                    if next_offset <= offset or next_offset >= file_size:
                        break
                    offset = next_offset
                    hdu_index += 1
            attempts.append({"tool": "builtin_fits_header", "status": "complete", "file_size": file_size, "hdus": hdus})
        except Exception as error:
            attempts.append({"tool": "builtin_fits_header", "status": "error", "stderr": safe_error(error)})
    if ext == ".npy" or "numpy array" in description.lower():
        try:
            import numpy
            array_value = numpy.load(path, mmap_mode="r", allow_pickle=False)
            attempts.append({
                "tool": "numpy",
                "status": "complete",
                "numpy_version": numpy.__version__,
                "shape": list(array_value.shape),
                "dtype": str(array_value.dtype),
                "dtype_descriptor": array_value.dtype.descr if array_value.dtype.fields else None,
                "fortran_order": bool(array_value.flags.f_contiguous and not array_value.flags.c_contiguous),
                "dimensions": int(array_value.ndim),
                "elements": int(array_value.size),
            })
        except Exception as error:
            attempts.append({"tool": "numpy", "status": "error", "stderr": safe_error(error)})
    if ext == ".parquet" or "parquet" in description.lower():
        try:
            import pyarrow.parquet as parquet
            value = parquet.ParquetFile(os.fsdecode(path))
            metadata = value.metadata
            attempts.append({
                "tool": "pyarrow",
                "status": "complete",
                "schema": str(value.schema),
                "arrow_schema": str(value.schema_arrow),
                "created_by": metadata.created_by,
                "format_version": metadata.format_version,
                "num_columns": metadata.num_columns,
                "num_rows": metadata.num_rows,
                "num_row_groups": metadata.num_row_groups,
                "serialized_size": metadata.serialized_size,
                "metadata": {decode_output(key): decode_output(item) for key, item in (metadata.metadata or {}).items()},
            })
        except ImportError:
            attempts.append({"tool": "pyarrow", "status": "unavailable", "stderr": "pyarrow unavailable"})
        except Exception as error:
            attempts.append({"tool": "pyarrow", "status": "error", "stderr": safe_error(error)})
    successes = [attempt for attempt in attempts if attempt.get("status") == "complete"]
    errors = [attempt.get("stderr", "") for attempt in attempts if attempt.get("status") != "complete" and attempt.get("stderr")]
    return {"attempts": attempts}, None if successes and not errors else "; ".join(errors) or None, True




BYTE_ANALYSIS_FULL_LIMIT = 64 * 1024 * 1024
BYTE_ANALYSIS_CHUNK = 1024 * 1024
GIT_INDEX_PATTERN = re.compile(br"(?:^|/)\.git/index$")


def byte_structure_metadata(path: bytes, mime_type: str, description: str) -> tuple[dict[str, Any] | None, str | None, bool]:
    try:
        metadata = os.stat(path, follow_symlinks=False)
        size = metadata.st_size
        allocated = getattr(metadata, "st_blocks", 0) * 512
        sample_records: list[dict[str, Any]] = []
        analyzed = bytearray()
        complete = size <= BYTE_ANALYSIS_FULL_LIMIT
        if complete:
            with open(path, "rb") as handle:
                value = handle.read()
            analyzed.extend(value)
            sample_records.append({
                "offset": 0,
                "length": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            })
        elif size > 0:
            offsets = sorted({
                0,
                max(0, size // 2 - BYTE_ANALYSIS_CHUNK // 2),
                max(0, size - BYTE_ANALYSIS_CHUNK),
            })
            with open(path, "rb") as handle:
                for offset in offsets:
                    handle.seek(offset)
                    value = handle.read(min(BYTE_ANALYSIS_CHUNK, size - offset))
                    analyzed.extend(value)
                    sample_records.append({
                        "offset": offset,
                        "length": len(value),
                        "sha256": hashlib.sha256(value).hexdigest(),
                    })
        counts = [0] * 256
        for byte in analyzed:
            counts[byte] += 1
        analyzed_size = len(analyzed)
        entropy = 0.0
        if analyzed_size:
            for count in counts:
                if count:
                    probability = count / analyzed_size
                    entropy -= probability * math.log2(probability)
        printable = sum(counts[byte] for byte in range(32, 127)) + counts[9] + counts[10] + counts[13]
        nul_count = counts[0]
        printable_ratio = printable / analyzed_size if analyzed_size else 0.0
        unique_byte_count = sum(1 for count in counts if count)
        analyzed_bytes = bytes(analyzed)
        utf8_valid = True
        try:
            analyzed_bytes.decode("utf-8")
        except UnicodeDecodeError:
            utf8_valid = False
        bom = None
        for prefix, name in (
            (b"\xef\xbb\xbf", "utf-8"),
            (b"\xff\xfe\x00\x00", "utf-32-le"),
            (b"\x00\x00\xfe\xff", "utf-32-be"),
            (b"\xff\xfe", "utf-16-le"),
            (b"\xfe\xff", "utf-16-be"),
        ):
            if analyzed_bytes.startswith(prefix):
                bom = name
                break
        if size == 0:
            classification = "empty_byte_stream"
        elif size == 1:
            classification = "single_byte_stream"
        elif size > 0 and allocated < size and allocated * 8 < size:
            classification = "sparse_byte_stream"
        elif complete and unique_byte_count == 1 and counts[0] == size:
            classification = "all_zero_byte_stream"
        elif mime_type.startswith("text/") or (printable_ratio >= 0.85 and nul_count == 0):
            classification = "text_byte_stream"
        elif printable_ratio >= 0.65 and nul_count > 0:
            classification = "mixed_text_binary_stream"
        elif entropy >= 7.5:
            classification = "high_entropy_byte_stream"
        else:
            classification = "generic_binary_stream"
        if bom:
            probable_encoding = bom
        elif utf8_valid and nul_count == 0:
            probable_encoding = "ascii" if all(byte < 128 for byte in analyzed) else "utf-8"
        elif printable_ratio >= 0.65:
            probable_encoding = "single-byte-8-bit-or-mixed"
        else:
            probable_encoding = "binary"
        histogram = {f"{index:02x}": count for index, count in enumerate(counts) if count}
        return {
            "kind": "byte_structure",
            "classification": classification,
            "logical_size": size,
            "allocated_bytes": allocated,
            "allocation_ratio": allocated / size if size else None,
            "analysis_complete": complete,
            "analyzed_bytes": analyzed_size,
            "analysis_coverage_ratio": analyzed_size / size if size else 1.0,
            "samples": sample_records,
            "analyzed_sha256": hashlib.sha256(analyzed_bytes).hexdigest(),
            "entropy_bits_per_byte": entropy,
            "unique_byte_count": unique_byte_count,
            "byte_histogram_hex": histogram,
            "all_zero": counts[0] == analyzed_size if analyzed_size else size == 0,
            "all_zero_is_exact": complete,
            "ascii_only": all(byte < 128 for byte in analyzed),
            "utf8_valid": utf8_valid,
            "utf8_valid_is_exact": complete,
            "byte_order_mark": bom,
            "probable_encoding_class": probable_encoding,
            "printable_or_whitespace_bytes": printable,
            "printable_or_whitespace_ratio": printable_ratio,
            "nul_bytes": nul_count,
            "line_feed_bytes": counts[10],
            "carriage_return_bytes": counts[13],
            "tab_bytes": counts[9],
            "mime_type": mime_type,
            "file_description": description,
        }, None, True
    except Exception as error:
        return None, safe_error(error), True


def _git_command(arguments: list[bytes], *, timeout: int = 30) -> dict[str, Any]:
    try:
        process = subprocess.run(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env={**os.environ, "LC_ALL": "C", "GIT_OPTIONAL_LOCKS": "0"},
        )
        return {
            "status": "complete" if process.returncode == 0 else "error",
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
    except Exception as error:
        return {
            "status": "error",
            "returncode": None,
            "stdout": b"",
            "stderr": safe_error(error).encode("utf-8"),
        }


def git_index_metadata(path: bytes, description: str) -> tuple[dict[str, Any] | None, str | None, bool]:
    if not GIT_INDEX_PATTERN.search(path) and not description.lower().startswith("git index"):
        return None, None, False
    try:
        with open(path, "rb") as handle:
            data = handle.read()
        if len(data) < 32:
            raise ValueError("Git index is too short")
        if data[:4] != b"DIRC":
            raise ValueError("Git index DIRC signature is missing")
        version, declared_entries = struct.unpack_from(">II", data, 4)
        git_dir = os.path.dirname(path)
        work_tree = os.path.dirname(git_dir)
        prefix = [b"git", b"--git-dir=" + git_dir, b"--work-tree=" + work_tree]
        object_format_result = _git_command(prefix + [b"rev-parse", b"--show-object-format"])
        object_format = object_format_result["stdout"].strip().decode("ascii", "replace") if object_format_result["status"] == "complete" else "sha1"
        digest_size = 32 if object_format == "sha256" else 20
        if len(data) < 12 + digest_size:
            raise ValueError("Git index is shorter than its header and checksum")
        checksum_offset = len(data) - digest_size
        checksum_stored = data[checksum_offset:]
        checksum_computed = hashlib.new(object_format, data[:checksum_offset]).digest()
        entries: list[dict[str, Any]] = []
        extensions: list[dict[str, Any]] = []
        offset = 12
        native_parse_status = "complete"
        if version in (2, 3):
            for entry_index in range(declared_entries):
                entry_start = offset
                fixed_size = 40 + digest_size + 2
                if offset + fixed_size > checksum_offset:
                    raise ValueError(f"Git index entry {entry_index} is truncated")
                stat_values = struct.unpack_from(">10I", data, offset)
                offset += 40
                object_id = data[offset:offset + digest_size].hex()
                offset += digest_size
                flags = struct.unpack_from(">H", data, offset)[0]
                offset += 2
                extended_flags = None
                if version >= 3 and flags & 0x4000:
                    if offset + 2 > checksum_offset:
                        raise ValueError(f"Git index entry {entry_index} extended flags are truncated")
                    extended_flags = struct.unpack_from(">H", data, offset)[0]
                    offset += 2
                path_end = data.find(b"\0", offset, checksum_offset)
                if path_end < 0:
                    raise ValueError(f"Git index entry {entry_index} pathname terminator is missing")
                path_bytes = data[offset:path_end]
                encoded_length = flags & 0x0FFF
                entry_length = path_end + 1 - entry_start
                offset = entry_start + ((entry_length + 7) // 8) * 8
                entries.append({
                    "index": entry_index,
                    "ctime_seconds": stat_values[0],
                    "ctime_nanoseconds": stat_values[1],
                    "mtime_seconds": stat_values[2],
                    "mtime_nanoseconds": stat_values[3],
                    "device": stat_values[4],
                    "inode": stat_values[5],
                    "mode": stat_values[6],
                    "mode_octal": f"{stat_values[6]:06o}",
                    "uid": stat_values[7],
                    "gid": stat_values[8],
                    "size": stat_values[9],
                    "object_id": object_id,
                    "flags_raw": flags,
                    "assume_valid": bool(flags & 0x8000),
                    "extended": bool(flags & 0x4000),
                    "stage": (flags >> 12) & 0x3,
                    "pathname_length_field": encoded_length,
                    "extended_flags_raw": extended_flags,
                    "intent_to_add": bool(extended_flags is not None and extended_flags & 0x2000),
                    "skip_worktree": bool(extended_flags is not None and extended_flags & 0x4000),
                    "path_display": path_bytes.decode("utf-8", "backslashreplace"),
                    "path_b64": base64.b64encode(path_bytes).decode("ascii"),
                    "path_hex": path_bytes.hex(),
                    "entry_offset": entry_start,
                    "entry_storage_bytes": offset - entry_start,
                })
            while offset < checksum_offset:
                if offset + 8 > checksum_offset:
                    raise ValueError("Git index extension header is truncated")
                signature = data[offset:offset + 4]
                length = struct.unpack_from(">I", data, offset + 4)[0]
                payload_start = offset + 8
                payload_end = payload_start + length
                if payload_end > checksum_offset:
                    raise ValueError(f"Git index extension {signature!r} is truncated")
                payload = data[payload_start:payload_end]
                extensions.append({
                    "signature": signature.decode("ascii", "backslashreplace"),
                    "signature_hex": signature.hex(),
                    "optional": bool(signature and 65 <= signature[0] <= 90),
                    "length": length,
                    "offset": offset,
                    "data_b64": base64.b64encode(payload).decode("ascii"),
                    "data_sha256": hashlib.sha256(payload).hexdigest(),
                })
                offset = payload_end
        else:
            native_parse_status = f"version_{version}_delegated_to_git"
        stage_result = _git_command(prefix + [b"ls-files", b"-z", b"--stage"])
        debug_result = _git_command(prefix + [b"ls-files", b"--stage", b"--debug"])
        git_version_result = _git_command([b"git", b"--version"])
        git_entries: list[dict[str, Any]] = []
        if stage_result["status"] == "complete":
            for record in stage_result["stdout"].split(b"\0"):
                if not record:
                    continue
                header, separator, path_bytes = record.partition(b"\t")
                parts = header.split()
                if not separator or len(parts) != 3:
                    raise ValueError(f"Unexpected git ls-files stage record: {record!r}")
                git_entries.append({
                    "mode": parts[0].decode("ascii", "replace"),
                    "object_id": parts[1].decode("ascii", "replace"),
                    "stage": int(parts[2]),
                    "path_display": path_bytes.decode("utf-8", "backslashreplace"),
                    "path_b64": base64.b64encode(path_bytes).decode("ascii"),
                    "path_hex": path_bytes.hex(),
                })
        errors = []
        for name, result in (("object-format", object_format_result), ("ls-files-stage", stage_result), ("ls-files-debug", debug_result), ("git-version", git_version_result)):
            if result["status"] != "complete":
                errors.append(f"{name}: {decode_output(result['stderr']).strip() or result['returncode']}")
        if checksum_stored != checksum_computed:
            errors.append("Git index trailing checksum does not match")
        if version in (2, 3) and len(entries) != declared_entries:
            errors.append(f"native entry count {len(entries)} does not match declared {declared_entries}")
        if stage_result["status"] == "complete" and len(git_entries) != declared_entries:
            errors.append(f"git entry count {len(git_entries)} does not match declared {declared_entries}")
        value = {
            "kind": "git_index",
            "signature": "DIRC",
            "version": version,
            "declared_entry_count": declared_entries,
            "file_size": len(data),
            "object_format": object_format,
            "checksum_algorithm": object_format,
            "checksum_stored": checksum_stored.hex(),
            "checksum_computed": checksum_computed.hex(),
            "checksum_matches": checksum_stored == checksum_computed,
            "native_parse_status": native_parse_status,
            "native_entries": entries,
            "native_entry_count": len(entries),
            "extensions": extensions,
            "extension_count": len(extensions),
            "git_entries": git_entries,
            "git_entry_count": len(git_entries),
            "git_version": decode_output(git_version_result["stdout"]).strip(),
            "git_ls_files_debug_stdout": decode_output(debug_result["stdout"]),
            "git_ls_files_debug_stderr": decode_output(debug_result["stderr"]),
            "git_ls_files_debug_returncode": debug_result["returncode"],
        }
        return value, "; ".join(errors) or None, True
    except Exception as error:
        return None, safe_error(error), True


GIT_OBJECT_PATTERN = re.compile(br"(?:^|/)\.git/objects/([0-9a-fA-F]{2})/([0-9a-fA-F]{38})$")
INTENTIONALLY_UNSUPPORTED_EXTENSIONS = {".ifc", ".dxf", ".stl"}


def git_object_metadata(path: bytes, description: str) -> tuple[dict[str, Any] | None, str | None, bool]:
    match = GIT_OBJECT_PATTERN.search(path)
    if not match:
        return None, None, False
    try:
        with open(path, "rb") as handle:
            compressed = handle.read()
        uncompressed = zlib.decompress(compressed)
        header, separator, payload = uncompressed.partition(b"\0")
        if not separator:
            raise ValueError("Git object header terminator is missing")
        object_type, declared_size_raw = header.split(b" ", 1)
        declared_size = int(declared_size_raw)
        computed = hashlib.sha1(uncompressed).hexdigest()
        object_id = (match.group(1) + match.group(2)).decode("ascii").lower()
        return {
            "kind": "git_loose_object",
            "object_id": object_id,
            "computed_object_id": computed,
            "object_type": object_type.decode("ascii", "replace"),
            "declared_size": declared_size,
            "actual_payload_size": len(payload),
            "header": header.decode("ascii", "replace"),
            "compressed_size": len(compressed),
            "compression_ratio": len(compressed) / len(uncompressed) if uncompressed else None,
            "object_id_matches": computed == object_id,
            "declared_size_matches": declared_size == len(payload),
        }, None, True
    except Exception as error:
        return None, safe_error(error), True


def version_control_metadata(path: bytes, description: str) -> tuple[dict[str, Any] | None, str | None, bool]:
    value, error, applicable = git_index_metadata(path, description)
    if applicable:
        return value, error, True
    return git_object_metadata(path, description)



def compressed_stream_metadata(path: bytes, description: str) -> tuple[dict[str, Any] | None, str | None, bool]:
    if "zlib compressed data" not in description.lower() or GIT_OBJECT_PATTERN.search(path):
        return None, None, False
    try:
        with open(path, "rb") as handle:
            compressed = handle.read()
        uncompressed = zlib.decompress(compressed)
        return {
            "kind": "zlib_stream",
            "compressed_size": len(compressed),
            "uncompressed_size": len(uncompressed),
            "compression_ratio": len(compressed) / len(uncompressed) if uncompressed else None,
            "uncompressed_sha256": hashlib.sha256(uncompressed).hexdigest(),
            "zlib_header_hex": compressed[:2].hex(),
        }, None, True
    except Exception as error:
        return None, safe_error(error), True


def intentional_unsupported_metadata(extension: str) -> tuple[dict[str, Any] | None, str | None, bool]:
    ext = extension.lower()
    if ext not in INTENTIONALLY_UNSUPPORTED_EXTENSIONS:
        return None, None, False
    return {
        "status": "intentionally_unsupported",
        "extension": ext,
        "reason": "Domain-specific parser deliberately omitted to test unsupported-format storage",
    }, None, True

def collect_raw_metadata(
    path: bytes,
    extension: str | None,
    mime_type: str | None,
    description: str | None,
    exiftool: ExifToolClient,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str | None], str]:
    extension_value = extension or ""
    mime_value = mime_type or ""
    description_value = description or ""
    raw: dict[str, Any] = {}
    attempts: list[dict[str, Any]] = []
    errors: dict[str, str | None] = {}

    exif_value, exif_error = exiftool.inspect(path)
    if exif_value is not None:
        raw["exiftool"] = exif_value
    errors["exiftool"] = exif_error
    if exif_value is not None and exif_error in {"Unknown file type", "File is empty", "Entire file is binary zeros"}:
        exif_status = "unsupported"
    elif exif_value is not None and exif_error:
        exif_status = "partial"
    elif exif_value is not None:
        exif_status = "complete"
    elif exif_error and "unavailable" in exif_error.lower():
        exif_status = "unavailable"
    else:
        exif_status = "error"
    attempts.append({"parser": "exiftool", "applicable": True, "status": exif_status, "error": exif_error, "metadata_items": len(exif_value) if isinstance(exif_value, dict) else 0})

    parsers: list[tuple[str, Callable[..., tuple[dict[str, Any] | None, str | None, bool]], tuple[Any, ...]]] = [
        ("archive", archive_metadata, (path, extension_value, mime_value, description_value)),
        ("sqlite", sqlite_metadata, (path, extension_value, mime_value, description_value)),
        ("message", message_metadata, (path, extension_value, mime_value)),
        ("certificate", certificate_metadata, (path, extension_value, description_value)),
        ("font", font_metadata, (path, extension_value, mime_value, description_value)),
        ("executable", executable_metadata, (path, extension_value, description_value)),
        ("scientific", scientific_metadata, (path, extension_value, description_value)),
        ("version_control", version_control_metadata, (path, description_value)),
        ("compressed_stream", compressed_stream_metadata, (path, description_value)),
        ("byte_structure", byte_structure_metadata, (path, mime_value, description_value)),
        ("intentional_unsupported", intentional_unsupported_metadata, (extension_value,)),
    ]
    for name, parser, arguments in parsers:
        value, error, applicable = parser(*arguments)
        if not applicable:
            continue
        if value is not None:
            raw[name] = value
        errors[name] = error
        if name == "intentional_unsupported" and value is not None:
            parser_status = "intentionally_unsupported"
        elif value is not None and error:
            parser_status = "partial"
        elif value is not None:
            parser_status = "complete"
        elif error and "unavailable" in error.lower():
            parser_status = "unavailable"
        else:
            parser_status = "error"
        attempts.append({"parser": name, "applicable": True, "status": parser_status, "error": error})

    authoritative_success = any(
        attempt["parser"] != "exiftool" and attempt["status"] == "complete"
        for attempt in attempts
    )
    exif_attempt = next(attempt for attempt in attempts if attempt["parser"] == "exiftool")
    if authoritative_success and exif_attempt["status"] in {"unsupported", "partial"}:
        exif_attempt["diagnostic"] = exif_attempt["error"]
        exif_attempt["error"] = None
        exif_attempt["status"] = "not_applicable" if exif_status == "unsupported" else "superseded"
        errors["exiftool"] = None
        exif_status = exif_attempt["status"]

    specialized_attempts = [attempt for attempt in attempts if attempt["parser"] not in {"exiftool", "intentional_unsupported"}]
    specialized_statuses = [attempt["status"] for attempt in specialized_attempts]
    if any(attempt["status"] == "intentionally_unsupported" for attempt in attempts):
        status = "intentionally_unsupported"
    elif specialized_attempts and all(value == "complete" for value in specialized_statuses):
        status = "complete"
    elif any(value in {"error", "partial", "unavailable"} for value in specialized_statuses):
        status = "partial" if raw else "error"
    elif exif_status == "complete":
        status = "complete"
    elif raw:
        status = "unsupported"
    else:
        status = "error"
    return raw, attempts, errors, status
