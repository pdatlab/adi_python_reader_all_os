#!/usr/bin/env python
r"""Regenerate the CFFI .pyd for every installed CPython on Windows.

Each CPython ABI (3.9, 3.10, ... 3.14) needs its own compiled extension, so
this discovers the interpreters registered with the Windows ``py`` launcher and
runs the in-place CFFI build (``adic/_build_ffi.py``) under each one. Only
the resulting ``adic/_adic_cffi.<abi>.pyd`` files remain afterwards; every
intermediate build artifact is discarded.

Usage:
    py build_all_pyds.py                 # all installed versions >= 3.9
    py build_all_pyds.py 3.11 3.13       # only the listed versions
    py build_all_pyds.py --clean         # delete existing .pyd files first
    py build_all_pyds.py --no-pip        # do not auto-install cffi if missing
    py build_all_pyds.py --list          # list discovered interpreters, then exit

Building needs ``cffi`` and a C compiler (MSVC Build Tools, "Desktop
development with C++") in each target interpreter. A missing ``cffi`` is
installed automatically unless ``--no-pip`` is given.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(HERE, "adic")
BUILDER = os.path.join(PKG_DIR, "_build_ffi.py")

MIN_VERSION = (3, 9)

# Intermediate artifacts that must never linger in the package directory.
_STRAY_GLOBS = ["_adic_cffi.c", "*.obj", "*.o", "*.exp", "*.lib"]
_STRAY_DIRS = ["Release", "build"]


def discover_pythons():
    """Return {(major, minor): exe_path} for interpreters known to ``py``."""
    try:
        out = subprocess.run(["py", "-0p"], capture_output=True, text=True)
    except FileNotFoundError:
        return {}
    found = {}
    for line in out.stdout.splitlines():
        # e.g.  " -V:3.14 *        C:\...\python.exe"
        m = re.match(r"\s*-V:(\d+)\.(\d+)\S*\s+\*?\s*(.+\.exe)\s*$", line)
        if m:
            found.setdefault((int(m.group(1)), int(m.group(2))), m.group(3).strip())
    return found


def ensure_cffi(exe, allow_pip):
    """Ensure ``cffi`` and ``setuptools`` (needed to compile on Python 3.12+)
    are importable under ``exe``; optionally pip-install them."""
    if subprocess.run([exe, "-c", "import cffi, setuptools"], capture_output=True).returncode == 0:
        return True
    if not allow_pip:
        return False
    print(f"  cffi/setuptools missing -> {exe} -m pip install cffi setuptools")
    return subprocess.run([exe, "-m", "pip", "install", "--upgrade", "cffi",
                           "setuptools"]).returncode == 0


def build_for(exe):
    """Run the in-place CFFI build under ``exe``. Returns (ok, completed_proc)."""
    proc = subprocess.run([exe, BUILDER], capture_output=True, text=True)
    return proc.returncode == 0, proc


def sweep_artifacts():
    """Remove any stray build intermediates from the package directory."""
    for pat in _STRAY_GLOBS:
        for p in glob.glob(os.path.join(PKG_DIR, pat)):
            try:
                os.remove(p)
            except OSError:
                pass
    for d in _STRAY_DIRS:
        shutil.rmtree(os.path.join(PKG_DIR, d), ignore_errors=True)


def _abi_tag(ver):
    return "cp" + ver.replace(".", "")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build the _adic_cffi .pyd for all installed Pythons.")
    ap.add_argument("versions", nargs="*",
                    help="specific X.Y versions to build (default: all >= 3.9)")
    ap.add_argument("--clean", action="store_true",
                    help="delete existing _adic_cffi*.pyd before building")
    ap.add_argument("--no-pip", dest="pip", action="store_false",
                    help="do not auto-install cffi when it is missing")
    ap.add_argument("--list", action="store_true",
                    help="list discovered interpreters and exit")
    args = ap.parse_args(argv)

    if not os.path.isfile(BUILDER):
        print(f"Builder not found: {BUILDER}", file=sys.stderr)
        return 1

    pythons = discover_pythons()
    if not pythons:
        print("No interpreters found via the `py` launcher "
              "(try: py --list-paths).", file=sys.stderr)
        return 1

    if args.list:
        for (maj, minr), exe in sorted(pythons.items()):
            print(f"{maj}.{minr}\t{exe}")
        return 0

    # Decide which versions to build.
    if args.versions:
        want = set()
        for v in args.versions:
            m = re.fullmatch(r"(\d+)\.(\d+)", v.strip())
            if m:
                want.add((int(m.group(1)), int(m.group(2))))
            else:
                print(f"Ignoring bad version '{v}' (expected e.g. 3.12)",
                      file=sys.stderr)
        targets = {k: pythons[k] for k in sorted(want) if k in pythons}
        for k in sorted(want - set(pythons)):
            print(f"Requested {k[0]}.{k[1]} not found via py launcher; skipping.",
                  file=sys.stderr)
    else:
        targets = {k: v for k, v in sorted(pythons.items()) if k >= MIN_VERSION}

    if not targets:
        print("Nothing to build.", file=sys.stderr)
        return 1

    if args.clean:
        for p in (glob.glob(os.path.join(PKG_DIR, "_adic_cffi*.pyd"))
                  + glob.glob(os.path.join(PKG_DIR, "_adic_cffi*.so"))):
            print("removing", os.path.basename(p))
            os.remove(p)
    sweep_artifacts()

    results = {}
    for (maj, minr), exe in targets.items():
        ver = f"{maj}.{minr}"
        print(f"\n=== Python {ver}  ({exe}) ===")
        if not ensure_cffi(exe, args.pip):
            print(f"  SKIP: cffi unavailable for {ver}"
                  + ("" if args.pip else " (omit --no-pip to auto-install)"))
            results[ver] = "no cffi"
            continue
        print("  compiling (this builds the C library, ~10-30s)...")
        ok, proc = build_for(exe)
        results[ver] = "ok" if ok else "FAILED"
        if not ok:
            sys.stdout.write(proc.stdout[-2000:])
            sys.stderr.write(proc.stderr[-2000:])
        sweep_artifacts()  # tidy up even if a build bailed mid-way

    # Report.
    print("\n----- summary -----")
    width = max((len(v) for v in results), default=4)
    for ver in sorted(results, key=lambda s: tuple(int(x) for x in s.split("."))):
        status = results[ver]
        tag = ""
        if status == "ok":
            hits = glob.glob(os.path.join(PKG_DIR, f"_adic_cffi.{_abi_tag(ver)}-*.pyd"))
            tag = "  -> " + os.path.basename(hits[0]) if hits else ""
        print(f"  {ver:<{width}}  {status}{tag}")

    return 0 if results and all(s == "ok" for s in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
