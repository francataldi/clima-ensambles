from __future__ import annotations

import argparse
import ctypes
import sys
import time
from datetime import datetime
from pathlib import Path

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def finished(done: Path, start: float) -> bool:
    return done.exists() and done.stat().st_mtime > start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hasta", default="runs/listo.txt")
    parser.add_argument("--latido", default="runs/despierta.txt")
    parser.add_argument("--horas", type=float, default=30.0)
    parser.add_argument("--cada", type=float, default=60.0)
    args = parser.parse_args()
    if sys.platform != "win32":
        sys.exit("Solo hace falta en Windows.")
    kernel = ctypes.windll.kernel32
    done = Path(args.hasta)
    beat = Path(args.latido)
    beat.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    limit = start + args.horas * 3600
    print(f"La PC no se va a suspender hasta que aparezca {done} o pasen {args.horas:g} horas.")
    print("Cerrar esta ventana devuelve a Windows su configuración de energía de siempre.")
    try:
        while time.time() < limit and not finished(done, start):
            kernel.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            beat.write_text(f"despierta {now}\n", encoding="utf-8")
            print(f"\r{now}", end="", flush=True)
            time.sleep(args.cada)
    finally:
        kernel.SetThreadExecutionState(ES_CONTINUOUS)
    beat.write_text(f"terminado {datetime.now():%Y-%m-%d %H:%M:%S}\n", encoding="utf-8")
    print("\nListo: Windows vuelve a su configuración de energía de siempre.")


if __name__ == "__main__":
    main()
