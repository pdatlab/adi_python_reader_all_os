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
| `File` (added here) | `labchart_channels`, `calc_settings`, `modules` | |
| `Record` | `id`, `n_ticks`, `tick_dt`, `tick_fs`, `duration`, `comments`, `record_time` | |
| `RecordTime` | `rec_datetime`, `rec_datestr`, `rec_stop_datetime`, `trig_datetime`, `trig_start_delta`, `tz_offset` | |
| `Channel` | `id`, `name`, `units`, `n_samples`, `dt`, `fs`, `max_time` (per record), `base_units`, `data_type`, `slot`, `calculations` | `get_data`, `get_pyramid` |
| `LabChartChannel` | `number`, `name`, `channel`, `has_samples`, `calculations` | |
| `CalcSetting` | `channel_number`, `kind`, `name`, `class_id`, `expression`, `units` | |
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

### Min/max overviews ("pyramids")

To draw a long recording at any zoom, `get_pyramid(record_id)` returns the
minimum and maximum of each bin of samples at several resolutions. The samples
are decoded and scanned in the C library and never returned, so no sample
array is allocated:

```python
pyr = f.channels[0].get_pyramid(1)          # channel 1, record 1
lv = pyr.levels[0]                          # finest level
print(pyr.first, lv.bmin[:3], lv.bmax[:3])
print([lv.factor for lv in pyr.levels])     # e.g. [256, 1024, 4096, 16384]
lo, hi = pyr.extent()                       # over the whole record
```

- The finest level holds the min and max of each run of `first` samples;
  each coarser level combines `factor` (4) bins of the one below, until a level
  would have fewer than `min_level_bins` (512) bins. The last bin of a level
  can be shorter.
- `first` defaults to `max(256, ceil(n / 131072))` for `n` samples.
- `sample_range` / `time_range` select part of the record; `dtype`,
  `scaling`, `n_threads` and `simd` work as in `get_data`, and the bins equal
  the min / max of `get_data`'s values. A bin holding a NaN is NaN.
- `progress(done, total)` and `should_stop()` callbacks report and cancel
  long scans.

### Calculated channels

LabChart can compute channels from others (Arithmetic, Cyclic Measurements,
Digital Filter, Smoothing, ...). Such channels usually have no samples in the
file, and those after the last channel with samples aren't in
`File.channels` at all. `File.labchart_channels` lists every channel LabChart
defines, with the calculation settings stored for it:

```python
for c in f.labchart_channels:
    print(c.number, c.name, c.has_samples, c.calculations)
# 4 Pseudo CO 1 False [<CalcSetting Ch4: Arithmetic 'Ch2*Ch3' [V]>]
# 2 BPM 1 False [<CalcSetting Ch2: Cyclic Measurements>]
print(f.modules)          # e.g. ['CyclicMeasurements(8)', 'Arithmetic(8)', ...]
```

- `CalcSetting.kind`: `'arithmetic'` (with `expression` and `units`),
  `'cyclic_measurements'`, `'digital_filter'`, `'smoothing'`, or `'other'`
  (a settings class not identified yet: see `class_id`). Only Arithmetic's
  parameters are decoded so far.
- LabChart keeps a channel's settings when its calculation is changed or
  turned off, so a setting doesn't prove the calculation is in use, and a
  channel can have several. A channel without samples is the reliable sign
  that it is calculated.

### Differences from the SDK package

- `Comment.channel` is the 1 based channel id (the SDK package gives the
  SDK's 0 based number); `-1` still means all channels.
- float32 values are computed in double precision and then rounded, so they
  can differ from the SDK's in the last bit (relative difference < 1e-7).
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

`tests/test_pyramid.py FILE ...` compares `get_pyramid` with pyramids built
with NumPy from `get_data` (every channel and record, each dtype and scaling,
several bin sizes, a random sample range) and, if the SDK package is
installed, the record trigger offsets and times with the SDK's.
