r"""
Compare adic with the ADInstruments SDK Python package (adi), Windows only.

    python tests/compare_to_sdk.py FILE [FILE ...] [--sdk PATH_TO_SDK_REPO]

Checks records, channels, units, sample counts, rates, start times, comments
and every sample (float32 values within 1e-6 relative: both are single
precision, rounded slightly differently), the time output, sample/time
ranges, and that float64/int16/thread/SIMD options agree with each other.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import adic  # noqa: E402

n_bad = 0


def check(ok, label, what):
    global n_bad
    if not ok:
        n_bad += 1
        print(f"MISMATCH {label}: {what}")


def rel_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    scale = max(np.max(np.abs(b)) * 1e-3, 1e-300)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), scale)))


def compare(fp, adi, rng):
    label = os.path.basename(fp)
    bad0 = n_bad
    a = adic.read_file(fp)
    s = adi.read_file(fp)

    check(a.n_records == s.n_records, label, "n_records")
    check(a.n_channels == s.n_channels, label, "n_channels")
    check(a.channel_names == s.channel_names, label, "channel names")

    for ra, rs in zip(a.records, s.records):
        r = ra.id
        check(ra.n_ticks == rs.n_ticks, label, f"rec {r} n_ticks")
        check(abs(ra.tick_dt - rs.tick_dt) <= 1e-12 * rs.tick_dt, label, f"rec {r} tick_dt")
        d = abs((ra.record_time.rec_datetime - rs.record_time.rec_datetime).total_seconds())
        check(d < 1e-3, label, f"rec {r} start time ({d} s)")
        check(len(ra.comments) == len(rs.comments), label, f"rec {r} # comments")
        for ca, cs in zip(ra.comments, rs.comments):
            # the SDK package gives the SDK's 0 based channel
            ch = cs.channel + 1 if cs.channel >= 0 else cs.channel
            ok = (ca.text == cs.text and ca.id == cs.id and ca.tick_position == cs.tick_position
                  and ca.channel == ch and abs(ca.time - cs.time) < 1e-9)
            check(ok, label, f"rec {r} comment {cs.id}")

    for ca, cs in zip(a.channels, s.channels):
        check(ca.n_samples == cs.n_samples, label, f"chan {ca.id} n_samples")
        for r in range(1, a.n_records + 1):
            n = ca.n_samples[r - 1]
            if n == 0:
                continue
            where = f"chan {ca.id} rec {r}"
            check(abs(ca.dt[r-1] - cs.dt[r-1]) <= 1e-12 * cs.dt[r-1], label, where + " dt")
            if cs.units[r-1].isascii():     # the SDK mangles some non-ASCII units
                check(ca.units[r-1] == cs.units[r-1], label, where + " units")

            ta, ya = ca.get_data(r, return_time=True)
            ts, ys = cs.get_data(r, return_time=True)
            e = rel_err(ya, ys)
            check(ya.dtype == ys.dtype and ya.shape == ys.shape and e < 1e-6, label,
                  f"{where} data (max rel err {e:g})")
            check(np.array_equal(ta, ts), label, where + " time")

            # other options must agree exactly with each other
            y64 = ca.get_data(r, dtype=np.float64)
            check(np.array_equal(ya, y64.astype(np.float32)), label, where + " float64 vs float32")
            y1 = ca.get_data(r, dtype=np.float64, n_threads=1, simd="none")
            check(np.array_equal(y1, y64), label, where + " 1 thread, no SIMD")
            if ca.data_type[r-1] == "int16":
                raw = ca.get_data(r, dtype=np.float64, scaling="raw")
                y16 = ca.get_data(r, dtype=np.int16, scaling="raw")
                check(np.array_equal(y16, raw.astype(np.int16)), label, where + " int16")

            # ranges, by sample and by time (same rules as the SDK)
            if n > 10:
                k1 = int(rng.integers(1, n // 2))
                k2 = int(rng.integers(k1 + 1, n + 1))
                yk = ca.get_data(r, sample_range=[k1, k2])
                check(np.array_equal(yk, ya[k1-1:k2]), label, where + " sample_range")
                t1 = (k1 - 1) * ca.dt[r-1] + 0.3 * ca.dt[r-1]
                t2 = (k2 - 1) * ca.dt[r-1]
                tta, yta = ca.get_data(r, time_range=[t1, t2], return_time=True)
                tts, yts = cs.get_data(r, time_range=[t1, t2], return_time=True)
                check(np.array_equal(tta, tts) and yta.shape == yts.shape, label, where + " time_range")
    if n_bad == bad0:
        print(f"PASS {label}")
    a.close()
    del s


def timing(fp, adi):
    a = adic.read_file(fp)
    s = adi.read_file(fp)
    out = {}
    for name, f in (("adic", a), ("SDK", s)):
        t0 = time.perf_counter()
        for c in f.channels:
            for r in range(1, f.n_records + 1):
                if c.n_samples[r-1] > 0:
                    c.get_data(r)
        out[name] = time.perf_counter() - t0
    print(f"{os.path.basename(fp)}, every channel as float32: adic {out['adic']:.2f} s, "
          f"SDK {out['SDK']:.2f} s")
    a.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--sdk", default=r"C:\repos\python\adinstruments_sdk_python",
                    help="folder containing the SDK package 'adi'")
    ap.add_argument("--time", action="store_true", help="also time the last file")
    args = ap.parse_args()
    sys.path.insert(0, args.sdk)
    import adi

    rng = np.random.default_rng(0)
    for fp in args.files:
        compare(fp, adi, rng)
    if args.time:
        timing(args.files[-1], adi)
    print("PASS: adic matches the SDK" if n_bad == 0 else f"FAIL: {n_bad} problems")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
