"""CFFI build script (API mode) for the adi_reader C library.

Build the extension for the current interpreter with:

    python adic/_build_ffi.py        # from the repository root

(Run it as a file, not with -m: "python -m adic._build_ffi" imports the adic
package first, which loads the existing extension and locks it on Windows.)

It compiles the C library (see ``_C_SRC`` below, or set the environment
variable ADI_C_READER_DIR) into a single ``_adic_cffi.<abi>.pyd`` (``.so`` off
Windows) next to this file. The C sources are compiled from a copy in a
temporary directory, so no intermediate files are left anywhere. On Windows
the MSVC Build Tools are found automatically by setuptools.

The AVX2 kernels (decode_avx2.c) are compiled separately with AVX2 code
generation and linked in; they only run on CPUs that have AVX2.

To build for every installed Python version, run ``build_all_pyds.py`` at the
repository root.
"""
import glob
import os
import platform
import shutil
import tempfile

from cffi import FFI

_HERE = os.path.dirname(os.path.abspath(__file__))

# The C library lives in its own (private) repository.
_C_SRC = os.environ.get("ADI_C_READER_DIR", r"C:\repos\c\adi_c_reader_all_os")

# Declarations visible to Python, from include/adi_reader.h (keep in sync).
_CDEF = r"""
    enum adi_status {
        ADI_OK = 0, ADI_ERR_ARG = -1, ADI_ERR_OPEN = -2, ADI_ERR_IO = -3,
        ADI_ERR_FORMAT = -4, ADI_ERR_NOMEM = -5, ADI_ERR_RANGE = -6,
        ADI_ERR_BLOCK = -7, ADI_ERR_TYPE = -8, ADI_ERR_BUFFER = -9,
        ADI_ERR_UNSUPPORTED = -10
    };
    enum adi_out_type { ADI_OUT_DOUBLE = 0, ADI_OUT_SINGLE = 1, ADI_OUT_INT16 = 2 };
    enum adi_scaling { ADI_SCALE_DISPLAY = 0, ADI_SCALE_BASE = 1, ADI_SCALE_RAW = 2 };
    enum adi_simd { ADI_SIMD_AUTO = 0, ADI_SIMD_NONE = 1, ADI_SIMD_SSE2 = 2, ADI_SIMD_AVX2 = 3 };
    enum adi_data_type { ADI_DATA_NONE = 0, ADI_DATA_INT16 = 1, ADI_DATA_FLOAT32 = 2 };
    enum adi_calc_kind { ADI_CALC_OTHER = 0, ADI_CALC_ARITHMETIC = 1,
                         ADI_CALC_CYCLIC_MEASUREMENTS = 2, ADI_CALC_DIGITAL_FILTER = 3,
                         ADI_CALC_SMOOTHING = 4 };
    enum adi_calc_text { ADI_CALC_TEXT_NAME = 0, ADI_CALC_TEXT_CLASS = 1,
                         ADI_CALC_TEXT_EXPRESSION = 2, ADI_CALC_TEXT_UNITS = 3 };

    typedef struct adi_file adi_file;

    int adi_version(void);
    int adi_last_error(char *buf, int buf_len, int *len_needed);
    const char *adi_status_name(int status);
    int adi_simd_available(int simd);
    int adi_simd_best(void);
    int adi_cpu_count(void);

    int adi_open(const char *path_utf8, adi_file **file);
    void adi_close(adi_file *file);

    int adi_n_records(const adi_file *file, int *n_records);
    int adi_n_channels(const adi_file *file, int *n_channels);
    int adi_file_version(const adi_file *file, int *major, int *minor);

    int adi_record_info(const adi_file *file, int record, int64_t *n_ticks,
                        double *tick_dt, double *start, double *tz_offset);
    int adi_record_trigger(const adi_file *file, int record,
                           int64_t *trig_minus_start);

    int adi_n_slots(const adi_file *file, int *n_slots);
    int adi_n_labchart_channels(const adi_file *file, int *n);
    int adi_slot_channel(const adi_file *file, int slot, int *channel);
    int adi_slot_name(const adi_file *file, int slot, char *buf, int buf_len,
                      int *len_needed);

    int adi_n_calc_settings(const adi_file *file, int *n);
    int adi_calc_setting_info(const adi_file *file, int index, int *slot, int *kind);
    int adi_calc_setting_text(const adi_file *file, int index, int what, char *buf,
                              int buf_len, int *len_needed);
    int adi_n_modules(const adi_file *file, int *n);
    int adi_module_name(const adi_file *file, int index, char *buf, int buf_len,
                        int *len_needed);

    int adi_channel_name(const adi_file *file, int channel, char *buf,
                         int buf_len, int *len_needed);
    int adi_channel_slot(const adi_file *file, int channel, int *slot);
    int adi_channel_record_info(const adi_file *file, int channel, int record,
                                int64_t *n_samples, double *dt, int *data_type);
    int adi_channel_units(const adi_file *file, int channel, int record,
                          int scaling, char *buf, int buf_len, int *len_needed);
    int adi_channel_scaling(const adi_file *file, int channel, int record,
                            int scaling, double *gain, double *offset);

    int adi_n_comments(const adi_file *file, int record, int *n_comments);
    int adi_comment_info(const adi_file *file, int record, int comment,
                         int *id, int *channel, int64_t *tick);
    int adi_comment_text(const adi_file *file, int record, int comment,
                         char *buf, int buf_len, int *len_needed);
    int adi_comment_slot(const adi_file *file, int record, int comment, int *slot);

    int adi_read_samples(const adi_file *file, int channel, int record,
                         int64_t first_sample, int64_t n_samples,
                         int out_type, int scaling, int n_threads, int simd,
                         void *out, int64_t *n_read);
    int adi_read_minmax(const adi_file *file, int channel, int record,
                        int64_t first_sample, int64_t n_samples, int64_t bin_size,
                        int out_type, int scaling, int n_threads, int simd,
                        void *mins, void *maxs, int64_t *n_bins);
"""

_LIB_FILES = ["io.c", "cfile.c", "map.c", "decode.c", "decode_sse2.c", "file.c", "read.c", "settings.c"]


def _is_x64():
    return platform.machine().lower() in ("amd64", "x86_64", "x64")


def _flags(msvc, avx2=False):
    if msvc:
        f = ["/O2", "/fp:precise"]
        if avx2:
            f.append("/arch:AVX2")
    else:
        # no fused multiply-add: identical values on every build
        f = ["-O3", "-ffp-contract=off"]
        if avx2:
            f.append("-mavx2")
    return f


def _compiler_is_msvc():
    try:
        from setuptools._distutils.ccompiler import new_compiler
    except ImportError:  # pragma: no cover
        from distutils.ccompiler import new_compiler
    return new_compiler().compiler_type == "msvc"


def _compile_avx2(csrc, tmpdir):
    """Compile decode_avx2.c with AVX2 enabled; returns the object files."""
    try:
        from setuptools._distutils.ccompiler import new_compiler
        from setuptools._distutils.sysconfig import customize_compiler
    except ImportError:  # pragma: no cover
        from distutils.ccompiler import new_compiler
        from distutils.sysconfig import customize_compiler
    cc = new_compiler()
    customize_compiler(cc)
    msvc = cc.compiler_type == "msvc"
    extra = _flags(msvc, avx2=True)
    if not msvc:
        extra.append("-fPIC")
    return cc.compile([os.path.join(csrc, "src", "decode_avx2.c")],
                      output_dir=os.path.join(tmpdir, "avx2"),
                      macros=[("ADI_STATIC", None)],
                      include_dirs=[os.path.join(csrc, "include"), os.path.join(csrc, "src")],
                      extra_postargs=extra)


def _make_builder(csrc, extra_objects=(), use_avx2=False):
    """Return an FFI configured to compile the library from ``csrc``."""
    msvc = _compiler_is_msvc()
    macros = [("ADI_STATIC", None)]
    if msvc:
        macros.append(("_CRT_SECURE_NO_WARNINGS", None))
    if not use_avx2:
        macros.append(("ADI_NO_AVX2", None))
    libraries = [] if os.name == "nt" else ["pthread", "m"]
    ffi = FFI()
    ffi.cdef(_CDEF)
    ffi.set_source(
        "adic._adic_cffi",
        '#include "adi_reader.h"',
        sources=[os.path.join(csrc, "src", f) for f in _LIB_FILES],
        include_dirs=[os.path.join(csrc, "include"), os.path.join(csrc, "src")],
        define_macros=macros,
        extra_compile_args=_flags(msvc),
        extra_objects=list(extra_objects),
        libraries=libraries,
    )
    return ffi


def build(verbose: bool = True) -> str:
    """Compile the extension and install only the ``.pyd``/``.so`` here.

    Returns the path of the installed extension.
    """
    if not os.path.isfile(os.path.join(_C_SRC, "include", "adi_reader.h")):
        raise RuntimeError(f"adi_reader C library not found in {_C_SRC} "
                           "(set ADI_C_READER_DIR)")
    tmpdir = tempfile.mkdtemp(prefix="adic_build_")
    try:
        # Compile a copy so no object files land in the C repository
        csrc = os.path.join(tmpdir, "csrc")
        for sub in ("include", "src"):
            os.makedirs(os.path.join(csrc, sub))
            for f in glob.glob(os.path.join(_C_SRC, sub, "*.[ch]")):
                shutil.copy2(f, os.path.join(csrc, sub))

        use_avx2 = _is_x64()
        objs = _compile_avx2(csrc, tmpdir) if use_avx2 else []
        out = _make_builder(csrc, objs, use_avx2).compile(tmpdir=tmpdir, verbose=verbose)
        if not out or not os.path.exists(out):
            hits = (glob.glob(os.path.join(tmpdir, "**", "_adic_cffi*.pyd"), recursive=True)
                    + glob.glob(os.path.join(tmpdir, "**", "_adic_cffi*.so"), recursive=True))
            if not hits:
                raise RuntimeError("CFFI build produced no extension module")
            out = hits[0]
        dst = os.path.join(_HERE, os.path.basename(out))
        shutil.copy2(out, dst)
        return dst
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    print("Installed:", build(verbose=True))
