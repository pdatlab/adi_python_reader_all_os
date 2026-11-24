"""
Check Channel.get_pyramid against pyramids built with NumPy from get_data,
and (if the SDK package ``adi`` is installed) the record trigger offsets
against the SDK:

    python tests/test_pyramid.py FILE [FILE ...]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import adic  # noqa: E402

n_bad = 0


def check(ok, what):
    global n_bad
    if not ok:
        n_bad += 1
        print("MISMATCH", what)


def reference(y, first, factor=4, min_level_bins=512, max_levels=24):
    def bins(a, r, fn):
        nb = a.size // r
        out = fn(a[:nb * r].reshape(nb, r), axis=1)
        if nb * r < a.size:
            out = np.append(out, fn(a[nb * r:]))
        return out
    lo, hi = bins(y, first, np.min), bins(y, first, np.max)
    levels, f = [(first, lo, hi)], first
    while lo.size // factor >= min_level_bins and len(levels) < max_levels:
        lo, hi = bins(lo, factor, np.min), bins(hi, factor, np.max)
        f *= factor
        levels.append((f, lo, hi))
    return levels


def same(pyr, levels):
    if len(pyr.levels) != len(levels):
        return False
    for got, (f, lo, hi) in zip(pyr.levels, levels):
        if got.factor != f or got.bmin.dtype != lo.dtype:
            return False
        nan = lo.dtype.kind == "f"
        if not (np.array_equal(got.bmin, lo, equal_nan=nan) and
                np.array_equal(got.bmax, hi, equal_nan=nan)):
            return False
    return True


def main(files):
    try:
        import adi
    except ImportError:
        adi = None
    rng = np.random.default_rng(0)
    for fp in files:
        label = os.path.basename(fp)
        bad0 = n_bad
        f = adic.read_file(fp)
        for c in f.channels:
            for r in range(1, f.n_records + 1):
                n = c.n_samples[r - 1]
                if n == 0:
                    continue
                where = f"{label} chan {c.id} rec {r}"
                kinds = [(np.float32, "display"), (np.float64, "base")]
                if c.data_type[r - 1] == "int16":
                    kinds.append((np.int16, "raw"))
                for dtype, scaling in kinds:
                    y = c.get_data(r, dtype=dtype, scaling=scaling)
                    for first in (None, 7, 1000):
                        pyr = c.get_pyramid(r, first=first, dtype=dtype, scaling=scaling,
                                            min_level_bins=3)
                        fst = pyr.first
                        check(fst == (first or max(256, -(-n // 131072))), where + " first")
                        check(pyr.n_samples == n and same(pyr, reference(y, fst, min_level_bins=3)),
                              f"{where} {np.dtype(dtype)} {scaling} first {fst}")
                if n > 20:
                    k1 = int(rng.integers(1, n // 2))
                    k2 = int(rng.integers(k1 + 1, n + 1))
                    y = c.get_data(r, sample_range=[k1, k2])
                    pyr = c.get_pyramid(r, first=11, sample_range=[k1, k2], min_level_bins=2)
                    check(same(pyr, reference(y, 11, min_level_bins=2)), where + " sample_range")
        if adi is not None:
            s = adi.read_file(fp)
            for ra, rs in zip(f.records, s.records):
                check(ra.record_time.trig_start_delta == rs.record_time.trig_start_delta,
                      f"{label} rec {ra.id} trigger")
                d = abs((ra.record_time.trig_datetime - rs.record_time.trig_datetime).total_seconds())
                check(d < 1e-3, f"{label} rec {ra.id} trigger time ({d} s)")
        if n_bad == bad0:
            print("PASS", label)
        f.close()
    print("PASS: pyramids match NumPy" if n_bad == 0 else f"FAIL: {n_bad} problems")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
