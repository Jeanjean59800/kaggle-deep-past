from __future__ import annotations

import base64
import os
import re
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def decode_b85(path: Path) -> str:
    data = path.read_bytes().strip()
    try:
        return zlib.decompress(base64.b85decode(data)).decode("utf-8")
    except Exception:
        # The first connector upload accidentally appended its source SHA.
        # Strip only a trailing 64-character hexadecimal digest, then verify
        # the compressed payload by actually decoding it.
        if len(data) >= 64 and re.fullmatch(rb"[0-9a-f]{64}", data[-64:]):
            data = data[:-64]
            text = zlib.decompress(base64.b85decode(data)).decode("utf-8")
            path.write_bytes(data)
            return text
        raise


# Validate and normalize the controller payload before the benchmark imports it.
decode_b85(ROOT / "overlay.b85")
source = decode_b85(ROOT / "benchmark.b85")
code_path = ROOT / "benchmark_v2_runtime.py"
code_path.write_text(source, encoding="utf-8")

# Execute from a real Python file. ProcessPoolExecutor workers must be able to
# import functions from __main__; an exec()-only namespace is not picklable.
os.execv(sys.executable, [sys.executable, str(code_path)])
