# adi_python_reader_all_os

Read LabChart files (`.adicht`, `.adidat`) in Python on Windows, macOS or
Linux, without the ADInstruments SDK.

The interface mirrors the ADInstruments SDK Python package
([adinstruments_sdk_python](https://github.com/JimHokanson/adinstruments_sdk_python),
imported as `adi`). This package is imported as `adic`, so both can be
installed at once.

## Usage

```python
import adic

f = adic.read_file(r'C:\data\test_file.adicht')
# All id numbering is 1 based (channels, records, samples), as in the SDK package

data = f.channels[1].get_data(1)            # channel 2, record 1 (float32)

p = f.get_channel_by_name('pres')           # partial, case-insensitive
time, data = p.get_data(1, start_time=10, return_time=True)

comments = f.get_comments()                         # list of Comment
c = f.get_comments(return_as='object')              # CommentCollection
pairs = c.get_comment_pairs("start pump", "stop pump", unmatched='ignore')  # needs pandas
time, data = p.get_data(-1, comment_range=[pairs.id1[0], pairs.id2[0]], return_time=True)

f.close()        # or: with adic.read_file(path) as f: ...
```

### Objects

Same as the SDK package:

| object | attributes | methods |
|--------|------------|---------|
| `File` | `file_path`, `n_records`, `n_channels`, `records`, `channels`, `channel_names` | `get_channel_by_name`, `get_comments`, `is_channel_in_record`, `channels_in_record`, `close` |
| `Record` | `id`, `n_ticks`, `tick_dt`, `tick_fs`, `duration`, `comments`, `record_time` | |
| `RecordTime` | `rec_datetime`, `rec_datestr`, `rec_stop_datetime`, `trig_datetime`, `trig_start_delta`, `tz_offset` | |
| `Channel` | `id`, `name`, `units`, `n_samples`, `dt`, `fs`, `max_time` (per record), `base_units`, `data_type`, `slot` | `get_data` |
| `Comment` | `text`, `id`, `tick_position`, `channel`, `record_id`, `time` | |
| `CommentCollection` | `comments`, `record_times` | `to_dataframe`, `get_comment_pairs` |

### get_data

Options from the SDK package: `start_sample`, `stop_sample`, `sample_range`,
`start_time`, `stop_time`, `time_range`, `start_comment`, `stop_comment`,
`comment_range`, `return_time`.

Added here:

| option | values |
|--------|--------|
| `dtype` | `np.float32` (default, as the SDK), `np.float64`, `np.int16` (stored ADC counts; needs `scaling='raw'`) |
| `scaling` | `'display'` (default: units as LabChart shows them, e.g. mV), `'base'` (no SI prefix, e.g. V), `'raw'` (stored values) |
| `n_threads` | `0` = all cores (default), `1` = this thread only (use when reading from several threads/processes) |
| `simd` | `'auto'` (default), `'none'`, `'sse2'`, `'avx2'` |

`read_file(path, remove_empty_channels=False, n_threads=0, simd='auto')`:
with `remove_empty_channels=True` only channels with samples are listed (as the
SDK's MATLAB wrapper does).

### Differences from the SDK package

- `Comment.channel` is the 1 based channel id (the SDK package gives the
  SDK's 0 based number); `-1` still means all channels.
- float32 values are computed in double precision and then rounded, so they
  can differ from the SDK's in the last bit (relative difference < 1e-7).
- Trigger delays aren't read yet: `trig_start_delta` is 0 and
  `rec_datetime` equals `trig_datetime`.
- `get_data` releases the GIL while reading, so other Python threads keep
  running.

## Speed

Every channel of a 323 MB LabChart 8.1 file (447 M samples) as float32: 0.50 s
with `adic`, 4.95 s with the SDK package (Windows laptop, 8 cores, file in the
OS cache).

## How it works

`adic/_adic_cffi.<abi>.pyd` (`.so` on Linux/macOS) contains the compiled
adi_reader C library, which reads the file format directly. The C library is
separate and not part of this repository.

- `adic/_build_ffi.py`: builds the extension for the running Python from the C
  library's source (`python adic/_build_ffi.py`; set `ADI_C_READER_DIR` to the
  library's folder). Needs `cffi`, `setuptools` and a C compiler (on Windows,
  the MSVC Build Tools with "Desktop development with C++").
- `build_all_pyds.py`: the same for every Python found by the `py` launcher.

## Tests

`tests/compare_to_sdk.py FILE ...` compares `adic` with the SDK package
(Windows only): metadata, comments, every sample, time outputs, ranges, and
the extra options. `--time` also times both on the last file.
