"""
Read LabChart (.adicht / .adidat) files on any OS.

Mirrors the ADInstruments SDK Python package (``adi``), but reads the file
format directly through the adi_reader C library (compiled into
``adic._adic_cffi``), so it doesn't need the SDK or Windows.

    import adic
    f = adic.read_file(r'C:\\data\\file.adicht')
    p = f.get_channel_by_name('pres')
    time, data = p.get_data(1, start_time=10, return_time=True)

All ids are 1 based (records, channels, samples), as in the SDK package.
"""
from __future__ import annotations

#Standard
#------------------------
from dataclasses import dataclass
from typing import overload, Literal, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import inspect
from datetime import datetime, timedelta

#Third Party
#------------------------
import numpy as np

#Local
#-------------------------
try:
    from adic._adic_cffi import ffi, lib
except ImportError as exc:  # pragma: no cover - build guidance
    raise ImportError(
        "The compiled extension adic._adic_cffi is missing for this Python "
        "version. Build it (needs the adi_reader C library) with:\n"
        "    python adic/_build_ffi.py") from exc


def read_file(file_path, remove_empty_channels: bool = False,
              n_threads: int = 0, simd: str = "auto") -> File:
    """
    This is the preferred entry point for working with this module.

    Parameters
    ----------
    file_path : str or path-like
    remove_empty_channels : bool, optional
        False (default, as the SDK package): every channel LabChart lists,
        including channels with no samples in any record, so channel ids are
        LabChart's channel numbers. True: only channels with samples (as the
        SDK's MATLAB wrapper does by default).
    n_threads : int, optional
        Threads used to decode a channel: 0 (default) = one per logical
        processor, 1 = the calling thread only (use 1 when reading files
        from several threads or processes at once). Can be changed per call
        in Channel.get_data.
    simd : {'auto', 'none', 'sse2', 'avx2'}, optional
    """
    return File(file_path, remove_empty_channels=remove_empty_channels,
                n_threads=n_threads, simd=simd)


class AdiError(Exception):
    """Error from the adi_reader C library."""

    def __init__(self, status, message):
        self.status = status
        name = ffi.string(lib.adi_status_name(status)).decode("ascii")
        super().__init__(f"{name}: {message}")


def _check(status):
    if status != 0:
        buf = ffi.new("char[1024]")
        lib.adi_last_error(buf, 1024, ffi.NULL)
        raise AdiError(status, ffi.string(buf).decode("utf-8", "replace"))


def _get_string(fn, *args):
    n = ffi.new("int *")
    _check(fn(*args, ffi.NULL, 0, n))
    buf = ffi.new("char[]", n[0])
    _check(fn(*args, buf, n[0], n))
    return ffi.string(buf).decode("utf-8", "replace")


_DTYPES = {
    np.dtype(np.float64): lib.ADI_OUT_DOUBLE,
    np.dtype(np.float32): lib.ADI_OUT_SINGLE,
    np.dtype(np.int16): lib.ADI_OUT_INT16,
}
_SCALING = {"display": lib.ADI_SCALE_DISPLAY, "base": lib.ADI_SCALE_BASE,
            "raw": lib.ADI_SCALE_RAW}
_SIMD = {"auto": lib.ADI_SIMD_AUTO, "none": lib.ADI_SIMD_NONE,
         "sse2": lib.ADI_SIMD_SSE2, "avx2": lib.ADI_SIMD_AVX2}


def simd_best() -> str:
    """SIMD level 'auto' uses on this CPU ('avx2', 'sse2' or 'none')."""
    return {lib.ADI_SIMD_AVX2: "avx2", lib.ADI_SIMD_SSE2: "sse2"}.get(lib.adi_simd_best(), "none")


class _FileHandle:
    """Owns the C file handle; closed when the last user goes away."""

    def __init__(self, file_path, n_threads, simd):
        self.ptr = ffi.NULL
        p = ffi.new("adi_file **")
        _check(lib.adi_open(str(file_path).encode("utf-8"), p))
        self.ptr = p[0]
        self.n_threads = n_threads
        self.simd = simd

    def close(self):
        if self.ptr != ffi.NULL:
            lib.adi_close(self.ptr)
            self.ptr = ffi.NULL

    def __del__(self):
        self.close()

    def get(self):
        if self.ptr == ffi.NULL:
            raise ValueError("The file has been closed")
        return self.ptr


def print_object(obj,keys_hide=[]):
    """
    Goal is to eventually mimic Matlab's default display behavior for objects
    Example output from Matlab
    morphology: [1x1 seg_worm.features.morphology]
       posture: [1x1 seg_worm.features.posture]
    locomotion: [1x1 seg_worm.features.locomotion]
          path: [1x1 seg_worm.features.path]
          info: [1x1 seg_worm.info]
    #TODO: For ndarrays we should implement size displays instead of length
    #TODO: The @property hack doesn't work for @property values from parent
    classes, I would need to look at __bases__
    """

    # TODO - have some way of indicating nested function and not doing fancy
    # print for nested objects ...

    MAX_WIDTH = 70

    dict_local = obj.__dict__

    key_names = [k for k in dict_local if k not in keys_hide]

    try:
        # TODO: Also include __bases__
        names_of_prop_methods = [
            name for name, value in vars(
                obj.__class__).items() if isinstance(
                value, property)]
        prop_code_ok = True
    except:
        prop_code_ok = False

    is_prop = [False] * len(key_names)
    if prop_code_ok:
        is_prop += [True] * len(names_of_prop_methods)
        key_names += names_of_prop_methods

    key_lengths = [len(x) for x in key_names]

    if len(key_lengths) == 0:
        return ""

    max_key_length = max(key_lengths)
    key_padding = [max_key_length - x for x in key_lengths]

    max_leadin_length = max_key_length + 2
    max_value_length = MAX_WIDTH - max_leadin_length

    lead_strings = [' ' * x + y + ': ' for x, y in zip(key_padding, key_names)]

    # TODO: Alphabatize the results ????
    # Could pass in as a option
    # TODO: It might be better to test for built in types
    #   Class::Bio.Entrez.Parser.DictionaryElement
    #   => show actual dictionary, not what is above

    value_strings = []
    for key, is_prop_local in zip(key_names, is_prop):
        if is_prop_local:
            temp_str = '@property method'
        else:
            run_extra_code = False
            value = dict_local[key]
            if hasattr(value, '__dict__'):
                try:  # Not sure how to test for classes :/
                    class_name = value.__class__.__name__
                    module_name = inspect.getmodule(value).__name__
                    temp_str = 'Class::' + module_name + '.' + class_name
                except:
                    run_extra_code = True
            else:
                run_extra_code = True

            if run_extra_code:
                # TODO: Change length to shape if available
                if isinstance(value, list) and len(value) > max_value_length:
                    len_value = len(value)
                    temp_str = 'Type::List, Len %d' % len_value
                else:
                    # Perhaps we want str instead?
                    # Changed from repr to str because things Python was not
                    # happy with lists of numpy arrays
                    temp_str = str(value)
                    if len(temp_str) > max_value_length:
                        #type_str = str(type(value))
                        #type_str = type_str[7:-2]
                        try:
                            len_value = len(value)
                        except:
                            len_value = 1
                        temp_str = str.format(
                            'Type::{}, Len: {}', type(value).__name__, len_value)

        value_strings.append(temp_str)

    class_name = obj.__class__.__name__
    final_str = class_name + '\n' + '-' * len(class_name) + '\n'
    for cur_lead_str, cur_value in zip(lead_strings, value_strings):
        final_str += (cur_lead_str + cur_value + '\n')
    return final_str

@dataclass
class CommentCollection:
    comments: list[Comment]
    record_times : list[RecordTime]

    def to_dataframe(self) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame([
            {
                "record_id": c.record_id,
                "text": c.text,
                "id": c.id,
                "channel": c.channel,
                "time": c.time,
                "tick_position": c.tick_position,
            }
            for c in self.comments
        ])
    
    """
    def get_comment_pairs(self, start, stop, n_max=None, 
                          record_split: Literal["allow", "skip", "error"] = "skip"):
        
        #.record_id1
        #.record_id2
        #.id1
        #.id2
        #.start_time
        #.stop_time
        #.start_comment
        #.stop_comment
        #.duration
        pass
    """
    
    def get_comment_pairs(self, start, stop, n_max=None, 
                          record_split: Literal["allow", "skip", "error"] = "error",
                          unmatched: Literal["ignore", "warn", "error"] = "error"):
        """
        Find sequential pairs of comments whose text matches `start` and
        `stop` (case-insensitive partial match).

        Parameters
        ----------
        start : str
            Substring to match for the opening comment.
        stop : str
            Substring to match for the closing comment.
        n_max : int, optional
            Maximum number of pairs to return.
        record_split : {"allow", "skip", "error"}, default "error"
            How to handle pairs that span different records.
            - "skip": both comments must be in the same record; a record
              boundary orphans any open start.
            - "allow": a stop comment in a later record can close a start
              from an earlier record (duration will be None for
              cross-record pairs since times are record-relative).
            - "error": raise ValueError if any cross-record pairs are found.
        unmatched : {"ignore", "warn", "error"}, default "error"
            What to do when start or stop comments have no partner.
            - "ignore": silently discard them.
            - "warn": emit a warnings.warn with details.
            - "error": raise ValueError with details.

        Returns
        -------
        pd.DataFrame
            Columns: rid1, rid2, id1, id2, start_time,
            stop_time, start_comment, stop_comment, duration
            - rid1, rid2 - records IDs (shortened for table display)
            - duration (numeric, in seconds)
            
        """
        
        """
        Implementation details:
            - comments are merged into a single timeline and walked 
              linearly so that a second start before a stop orphans the 
              first start (rather than letting an early orphan steal a 
              later pair's stop)
        """
        
        import pandas as pd

        start_lower = start.lower()
        stop_lower = stop.lower()
        restrict_to_record = record_split == "skip"

        # Merge and tag all matching comments
        tagged = []
        for c in self.comments:
            text_lower = c.text.lower()
            is_start = start_lower in text_lower
            is_stop = stop_lower in text_lower
            if is_start or is_stop:
                tagged.append((c, is_start, is_stop))

        tagged.sort(key=lambda x: (x[0].record_id, x[0].time))

        # Linear scan: walk the timeline and pair start/stop sequentially.
        # If a new start appears before the current start is closed, the
        # current start is orphaned.
        pairs = []
        orphan_starts = []
        orphan_stops = []
        current_start = None

        for c, is_start, is_stop in tagged:
            # A record boundary orphans any open start when restricted
            if restrict_to_record and current_start is not None:
                if c.record_id != current_start.record_id:
                    orphan_starts.append(current_start)
                    current_start = None

            if is_stop and current_start is not None:
                # Close the pair
                pairs.append((current_start, c))
                current_start = None
                if n_max is not None and len(pairs) >= n_max:
                    break
            elif is_start:
                # A new start orphans any unclosed previous start
                if current_start is not None:
                    orphan_starts.append(current_start)
                current_start = c
            else:
                # is_stop with no open start
                orphan_stops.append(c)

        # Anything still open at the end is orphaned
        if current_start is not None:
            orphan_starts.append(current_start)

        # Enforce the "error" mode for cross-record pairs
        if record_split == "error":
            cross_record = [
                (c1, c2) for c1, c2 in pairs
                if c1.record_id != c2.record_id
            ]
            if cross_record:
                details = ", ".join(
                    f"comment {c1.id} (record {c1.record_id}) → "
                    f"comment {c2.id} (record {c2.record_id})"
                    for c1, c2 in cross_record
                )
                raise ValueError(
                    f"Found {len(cross_record)} cross-record pair(s): {details}")

        # Handle unmatched comments
        if unmatched != "ignore" and (orphan_starts or orphan_stops):
            parts = []
            if orphan_starts:
                details = ", ".join(
                    f"id {c.id} (record {c.record_id}, t={c.time:.3f})"
                    for c in orphan_starts)
                parts.append(f"{len(orphan_starts)} unmatched start(s): {details}")
            if orphan_stops:
                details = ", ".join(
                    f"id {c.id} (record {c.record_id}, t={c.time:.3f})"
                    for c in orphan_stops)
                parts.append(f"{len(orphan_stops)} unmatched stop(s): {details}")

            msg = "; ".join(parts)

            if unmatched == "error":
                raise ValueError(msg)
            else:
                import warnings
                warnings.warn(msg)

        # Build the output DataFrame
        rows = []
        for c1, c2 in pairs:
            same_record = c1.record_id == c2.record_id
            if same_record:
                duration = c2.time - c1.time
            else:
                r1 = self.record_times[c1.record_id-1]
                r2 = self.record_times[c2.record_id-1]
                #start2 = r2.rec_datetime
                #stop1 = r1.rec_stop_datetime
                c2_time_datetime = r2.rec_datetime + timedelta(seconds=c2.time)
                c1_time_datetime = r1.rec_datetime + timedelta(seconds=c1.time)
                duration = (c2_time_datetime - c1_time_datetime).total_seconds()
                
                #elapsed time = 
                # duration - c1.time
                #+ time between records ()
                #duration = 1
                
                
            rows.append(
                {
                    "rid1": c1.record_id,
                    "rid2": c2.record_id,
                    "id1": c1.id,
                    "id2": c2.id,
                    "start_time": c1.time,
                    "stop_time": c2.time,
                    "start_comment": c1.text,
                    "stop_comment": c2.text,
                    "duration": duration,
                }
            )

        return pd.DataFrame(rows)

    def __repr__(self):
        return print_object(self)

class Comment():

    """
    This is currently created in the Record constructor.

    A comment consists primarily of a string and the time that it applies to.

    Attributes
    ----------
    text : string
        The comment string
    tick_position :
        Ticks from the start of the record
    channel : numeric
        1 based channel id (as in File.channels); -1 indicates all channels.
        None if that channel was left out (remove_empty_channels=True).
        (The SDK package gives the SDK's 0 based number here.)
    id :
        This starts at 1 and increments but may have gaps if comments are
        deleted.
    record_id :
    tick_dt :
    time :
        Seconds since start of recording (based on tick position)

    """
    def __init__(self, text, tick_pos, channel_id, comment_id) -> None:
        self.text = text
        self.tick_position = tick_pos
        self.channel_ = channel_id
        self.channel = self.channel_
        self.id = comment_id

    def _add_info(self, record_id, tick_dt) -> None:
        self.record_id = record_id
        self.tick_dt = tick_dt
        self.time = self.tick_position*self.tick_dt

    def __repr__(self):
        return print_object(self, keys_hide=['channel_'])


class Channel():

    """
    Attributes
    ----------
    h :
        Handle to the open file.
    id : numeric
        Channel ID, starts at 1
    n_records : numeric
    tick_dt : list, length: n_records
        Highest sampling rate of any channel for that record. Not really
        critical for this object (use dt/fs instead)
    records : [Record]
        Provides acccess to the relevant record
    name :
        This is fixed across all records
    units : list, length: n_records
    n_samples : list, length: n_records
    dt : list, length: n_records
        Time between samples
    fs : list, length: n_records
        Sampling rate
    max_time :
        Last time point where data was collected.
    base_units : list, length: n_records
        Units without the SI prefix (e.g. 'V' for 'mV')
    data_type : list, length: n_records
        How samples are stored: 'int16', 'float32', or None (no samples)
    calculations : [CalcSetting]
        Calculation settings stored for the channel (see CalcSetting)

    Notes
    ------
    1. Data for a channel may not exist for a given record.

    Methods
    -------
    get_data : retrieves data

    """

    def __init__(self, h, channel_id, records, h_file, slot, c_index):
        self.h = h
        self.h_file = h_file
        self.id = channel_id #1 based
        self.n_records = len(records)
        self.tick_dt = [x.tick_dt for x in records]
        self.records = records
        self.slot = slot          #LabChart channel slot, 0 based
        self._c = c_index         #index in the C library, -1 = no samples
        self.calculations = []    #set by File

        ptr = h.get()
        self.name = _get_string(lib.adi_slot_name, ptr, slot)
        if c_index < 0:
            # no samples in any record
            self.units = [''] * self.n_records
            self.base_units = [''] * self.n_records
            self.n_samples = [0] * self.n_records
            self.dt = [float('nan')] * self.n_records
            self.data_type = [None] * self.n_records
            self.fs = [float('nan')] * self.n_records
            self.max_time = [None] * self.n_records
            return
        c = c_index

        n = ffi.new("int64_t *")
        dt = ffi.new("double *")
        t = ffi.new("int *")
        self.units = []
        self.base_units = []
        self.n_samples = []
        self.dt = []
        self.data_type = []
        for r in range(self.n_records):
            _check(lib.adi_channel_record_info(ptr, c, r, n, dt, t))
            self.n_samples.append(int(n[0]))
            #The SDK reports a sample period even for records without samples
            self.dt.append(dt[0])
            self.data_type.append({1: "int16", 2: "float32"}.get(t[0]))
            self.units.append(_get_string(lib.adi_channel_units, ptr, c, r, lib.ADI_SCALE_DISPLAY))
            self.base_units.append(_get_string(lib.adi_channel_units, ptr, c, r, lib.ADI_SCALE_BASE))
        self.fs = [1/x for x in self.dt]
        self.max_time = [(self.n_samples[x]-1)*self.dt[x] if self.n_samples[x] > 0 else None
                         for x in range(self.n_records)]

    @overload
    def get_data(self, record_id: int, *, return_time: Literal[False] = ..., **kw) -> np.ndarray: ...

    @overload
    def get_data(self, record_id: int, *, return_time: Literal[True], **kw) -> tuple[np.ndarray, np.ndarray]: ...

    def get_data(self, record_id: int,
                 start_sample: int | None = None,
                 stop_sample: int | None = None,
                 sample_range: Sequence[int] | None = None,
                 start_time: float | None = None,
                 stop_time: float | None = None,
                 time_range: Sequence[float] | None = None,
                 start_comment: int | None = None,
                 stop_comment: int | None = None,
                 comment_range: Sequence[int] | None = None,
                 return_time: bool = False,
                 dtype=np.float32,
                 scaling: Literal["display", "base", "raw"] = "display",
                 n_threads: int | None = None,
                 simd: str | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:

        """
        Calling Forms
        -------------
        data = chan.get_data(record_id, **options)

        time, data = chan.get_data(record_id, return_time=True, **options)


        Parameters
        ----------
        record_id : numeric
            Which record to retrieve data from. This is ignored if using
            comment IDs
        start_sample : numeric, optional
            Default 1. Ignored if start_time or start_comment is provided.
        stop_sample : numeric, optional
            Default last sample. Ignored if stop_time or stop_comment is provided.
        sample_range : [start, stop], optional
            Convenience alternative to start_sample/stop_sample. Unpacks into
            those parameters, so the same override rules apply.
        start_time : numeric, optional
            Start time in seconds. Overrides start_sample if provided.
        stop_time : numeric, optional
            Stop time in seconds. Overrides stop_sample if provided.
        time_range : [start, stop], optional
            Convenience alternative to start_time/stop_time. Unpacks into
            those parameters, so the same override rules apply.
        start_comment : numeric, optional
            Comment ID. Overrides start_time and start_sample if provided.
        stop_comment : numeric, optional
            Comment ID. Overrides stop_time and stop_sample if provided.
        comment_range : [start, stop], optional
            Convenience alternative to start_comment/stop_comment. Unpacks
            into those parameters, so the same override rules apply.
        return_time : boolean, optional
            If True, returns a tuple of (time_array, data_array).
            The default is False.

        Additional options (not in the SDK package)
        -------------------------------------------
        dtype : np.float32 (default, as the SDK), np.float64, or np.int16
            np.int16 gives the stored ADC counts and needs scaling='raw'
            (and int16 data).
        scaling : 'display' (default), 'base', or 'raw'
            display: units as LabChart and the SDK show them (e.g. mV)
            base: without the SI prefix (e.g. V)
            raw: the stored values (ADC counts for int16 data)
        n_threads : int, optional
            0 = all cores, 1 = this thread only. Default from read_file.
        simd : 'auto', 'none', 'sse2', 'avx2'. Default from read_file.


        Returns
        -------
        numpy array, or tuple of (numpy array, numpy array) if return_time=True


        Priority
        --------
        comment > time > sample. Ranges unpack into their respective
        start/stop before the cascade runs.

        """

        # Unpack ranges into individual start/stop
        #------------------------------------------
        if comment_range is not None:
            start_comment, stop_comment = comment_range

        if time_range is not None:
            start_time, stop_time = time_range

        if sample_range is not None:
            start_sample, stop_sample = sample_range

        # Comment → time resolution
        #------------------------------------------
        if start_comment is not None or stop_comment is not None:
            all_comments = self.h_file.get_comments()
            comments_by_id = {c.id: c for c in all_comments}

            if start_comment is not None:
                if start_comment not in comments_by_id:
                    raise ValueError(f"No comment found with id {start_comment}")
                c1 = comments_by_id[start_comment]
                start_time = c1.time
                start_record = c1.record_id

            if stop_comment is not None:
                if stop_comment not in comments_by_id:
                    raise ValueError(f"No comment found with id {stop_comment}")
                c2 = comments_by_id[stop_comment]
                stop_time = c2.time
                stop_record = c2.record_id

            if start_comment is not None and stop_comment is not None:
                if start_record != stop_record:
                    raise ValueError(
                        f"start_comment (record {start_record}) and "
                        f"stop_comment (record {stop_record}) are in different records")
                record_id = start_record
            elif start_comment is not None:
                record_id = start_record
            else:
                record_id = stop_record

        if not 1 <= record_id <= self.n_records:
            raise ValueError(f"record_id {record_id} out of range [1, {self.n_records}]")
        dt = self.dt[record_id - 1]

        # Time → sample resolution
        #------------------------------------------
        if start_time is not None:
            start_sample = int(start_time / dt) + 1
        elif start_sample is None:
            start_sample = 1

        if stop_time is not None:
            stop_sample = int(stop_time / dt) + 1
        elif stop_sample is None:
            stop_sample = self.n_samples[record_id - 1]

        # Validate computed sample bounds
        #------------------------------------------
        if start_sample < 1:
            raise Exception('Computed start_sample is out of range')
        if stop_sample > self.n_samples[record_id - 1]:
            raise Exception('Computed stop_sample is out of range')

        n_elements = stop_sample - start_sample + 1
        if n_elements <= 0:
            raise ValueError("# of samples requested is less than 1")

        dt_out = np.dtype(dtype)
        if dt_out not in _DTYPES:
            raise ValueError(f"dtype must be float32, float64 or int16, not {dt_out}")
        if scaling not in _SCALING:
            raise ValueError(f"scaling must be 'display', 'base' or 'raw', not {scaling!r}")
        simd = self.h.simd if simd is None else simd
        if simd not in _SIMD:
            raise ValueError(f"simd must be one of {list(_SIMD)}, not {simd!r}")
        n_threads = self.h.n_threads if n_threads is None else n_threads

        # The C library writes straight into the numpy array
        data = np.empty(n_elements, dtype=dt_out)
        _check(lib.adi_read_samples(self.h.get(), self._c, record_id - 1,
                                    start_sample - 1, n_elements, _DTYPES[dt_out],
                                    _SCALING[scaling], n_threads, _SIMD[simd],
                                    ffi.from_buffer(data), ffi.NULL))

        if return_time:
            time = np.arange(start_sample - 1, stop_sample) * dt
            return time, data
        else:
            return data

    def get_pyramid(self, record_id: int, first: int | None = None,
                    sample_range: Sequence[int] | None = None,
                    time_range: Sequence[float] | None = None,
                    dtype=np.float32,
                    scaling: Literal["display", "base", "raw"] = "display",
                    factor: int = 4, min_level_bins: int = 512,
                    max_levels: int = 24, target_max_bins: int = 131_072,
                    n_threads: int | None = None, simd: str | None = None,
                    progress=None, should_stop=None) -> "Pyramid | None":
        """
        Min/max overview ("pyramid") of the channel in one record, for
        drawing long recordings at any zoom. The samples are decoded and
        scanned in C and never returned, so no sample array is allocated.

        The finest level holds the minimum and maximum of each run of
        ``first`` consecutive samples; each coarser level combines
        ``factor`` bins of the one below, until a level would have fewer
        than ``min_level_bins`` bins. The last bin of a level may cover
        fewer samples.

        Parameters
        ----------
        record_id : int
            1 based record.
        first : int, optional
            Samples per bin of the finest level. Default
            max(256, ceil(n / target_max_bins)) for n samples.
        sample_range : [start, stop], optional
            1 based, inclusive (as get_data).
        time_range : [start, stop], optional
            Seconds (as get_data). Overrides sample_range.
        dtype, scaling, n_threads, simd :
            As get_data. The bins equal the min / max of get_data's values;
            a bin holding a NaN is NaN.
        factor, min_level_bins, max_levels, target_max_bins :
            Shape of the pyramid (see above).
        progress : callable(done_samples, total_samples), optional
        should_stop : callable() -> bool, optional
            Checked between pieces of ~4 M samples; True abandons the scan
            and returns None.

        Returns
        -------
        Pyramid, or None if the record has no samples of this channel.
        """
        if not 1 <= record_id <= self.n_records:
            raise ValueError(f"record_id {record_id} out of range [1, {self.n_records}]")
        n_rec = self.n_samples[record_id - 1]
        if n_rec == 0:
            return None
        start, stop = 1, n_rec
        if sample_range is not None:
            start, stop = sample_range
        if time_range is not None:
            dt = self.dt[record_id - 1]
            start, stop = int(time_range[0] / dt) + 1, int(time_range[1] / dt) + 1
        if start < 1 or stop > n_rec:
            raise ValueError(f"samples [{start}, {stop}] out of range [1, {n_rec}]")
        n = stop - start + 1
        if n <= 0:
            raise ValueError("# of samples requested is less than 1")
        if factor < 2:
            raise ValueError("factor must be at least 2")
        if first is None:
            first = max(256, -(-n // int(target_max_bins)))
        first = int(first)
        if first <= 0:
            raise ValueError("first must be positive")
        dt_out = np.dtype(dtype)
        if dt_out not in _DTYPES:
            raise ValueError(f"dtype must be float32, float64 or int16, not {dt_out}")
        if scaling not in _SCALING:
            raise ValueError(f"scaling must be 'display', 'base' or 'raw', not {scaling!r}")
        simd = self.h.simd if simd is None else simd
        if simd not in _SIMD:
            raise ValueError(f"simd must be one of {list(_SIMD)}, not {simd!r}")
        n_threads = self.h.n_threads if n_threads is None else n_threads

        # In pieces of whole bins (~4 M samples) for progress / cancelling;
        # the bins are the same as from one call
        nb = -(-n // first)
        bmin = np.empty(nb, dt_out)
        bmax = np.empty(nb, dt_out)
        piece = max(1, 4_000_000 // first) * first
        done = 0
        while done < n:
            if should_stop is not None and should_stop():
                return None
            m = min(piece, n - done)
            b0 = done // first
            _check(lib.adi_read_minmax(self.h.get(), self._c, record_id - 1,
                                       start - 1 + done, m, first, _DTYPES[dt_out],
                                       _SCALING[scaling], n_threads, _SIMD[simd],
                                       ffi.from_buffer(bmin[b0:]), ffi.from_buffer(bmax[b0:]),
                                       ffi.NULL))
            done += m
            if progress is not None:
                progress(done, n)

        levels = [PyramidLevel(first, bmin, bmax)]
        f = first
        while bmin.size // factor >= min_level_bins and len(levels) < max_levels:
            bmin = _reduce_bins(bmin, factor, np.minimum)
            bmax = _reduce_bins(bmax, factor, np.maximum)
            f *= factor
            levels.append(PyramidLevel(f, bmin, bmax))
        return Pyramid(first=first, levels=levels, n_samples=n)

    def __repr__(self):
        return print_object(self)


def _reduce_bins(arr, r, ufunc):
    """Combine every r adjacent bins with ufunc (np.minimum / np.maximum);
    a shorter group at the end still makes one bin."""
    nb = arr.size // r
    head = ufunc.reduce(arr[:nb * r].reshape(nb, r), axis=1)
    if nb * r < arr.size:
        head = np.append(head, ufunc.reduce(arr[nb * r:]))
    return head


@dataclass
class PyramidLevel:
    """One level of a Pyramid: the min and max of each bin of ``factor``
    samples (bin k holds samples k*factor ... (k+1)*factor - 1)."""
    factor: int
    bmin: np.ndarray
    bmax: np.ndarray

    def __repr__(self):
        return print_object(self)


@dataclass
class Pyramid:
    """
    Min/max overview of a channel in one record, finest level first (see
    Channel.get_pyramid).

    Attributes
    ----------
    first : int
        Samples per bin of the finest level, ``levels[0]``.
    levels : list of PyramidLevel
        Finest first; each level's ``factor`` is its samples per bin.
    n_samples : int
        Samples summarised.
    """
    first: int
    levels: list
    n_samples: int

    def extent(self) -> tuple:
        """(min, max) over all samples, ignoring NaN; (0.0, 0.0) if there is
        no finite value."""
        import warnings
        lv = self.levels[-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lo = float(np.nanmin(lv.bmin))
            hi = float(np.nanmax(lv.bmax))
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return 0.0, 0.0
        return lo, hi

    def __repr__(self):
        return print_object(self)


class RecordTime():

    """
    Describes when a record started (in real/wall time).

    Attributes
    ----------
    trig_datetime : datetime
        Time of the trigger. Local time (as LabChart stores it, without a
        time zone)
    trig_start_delta : numeric
        Difference between trigger and data collection start, in ticks. If
        positive then data started later; if negative (a pre-trigger), before.
    trig_datestr : string
    rec_datetime : datetime
        Time when the first sample was collected
    rec_datestr : string
    rec_stop_datetime : datetime
    tz_offset : float
        UTC minus local time in seconds (nan if not stored). Not in the SDK.

    """
    def __init__(self, tick_dt, start, tz_offset, trig_start_delta=0):

        # the C library gives the record start; the trigger is delta ticks
        # before it
        self.rec_datetime = datetime(1970, 1, 1) + timedelta(seconds=start)
        self.trig_start_delta = trig_start_delta
        self.trig_datetime = self.rec_datetime - timedelta(
            seconds=trig_start_delta * tick_dt)
        self.trig_datestr = self.trig_datetime.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip('0')
        self.rec_datestr = self.rec_datetime.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip('0')
        self.tz_offset = tz_offset

    def add_info(self, duration):
        self.rec_stop_datetime = self.rec_datetime + timedelta(seconds=duration)

    def __repr__(self):
        return print_object(self)


class Record():

    """
    Holder of comments and record timing information.

    Attributes
    ----------
    h : handle to the open file
    id : numeric
        Record indicator, starts at 1
    n_ticks : numeric
        # of samples in the record for the channel sampled at the highest rate
    tick_dt : numeric
        Time between "ticks"
    tick_fs : numeric
        Sampling rate of ticks
    duration : numeric (seconds)
    comments : [Comment]
    record_time : RecordTime

    """
    def __init__(self, h, record_id, slot_to_id, all_slots=True):
        """
        Parameters
        ----------
        h :
            Handle to the open file
        record_id : numeric
            1 based record
        slot_to_id : dict
            channel slot -> 1 based channel id (for comment channels)
        all_slots : bool
            True if channel ids are LabChart's channel numbers (slot + 1),
            i.e. channels without samples are listed too
        """
        self.h = h
        self.id = record_id
        ptr = h.get()
        r = record_id - 1

        n_ticks = ffi.new("int64_t *")
        tick_dt = ffi.new("double *")
        start = ffi.new("double *")
        tz = ffi.new("double *")
        trig = ffi.new("int64_t *")
        _check(lib.adi_record_info(ptr, r, n_ticks, tick_dt, start, tz))
        _check(lib.adi_record_trigger(ptr, r, trig))
        self.n_ticks = int(n_ticks[0])
        self.tick_dt = tick_dt[0]
        self.tick_fs = 1.0/self.tick_dt
        self.duration = self.tick_dt*self.n_ticks

        n_com = ffi.new("int *")
        _check(lib.adi_n_comments(ptr, r, n_com))
        cid = ffi.new("int *")
        chan = ffi.new("int *")
        slot = ffi.new("int *")
        tick = ffi.new("int64_t *")
        self.comments = []
        for i in range(n_com[0]):
            _check(lib.adi_comment_info(ptr, r, i, cid, chan, tick))
            _check(lib.adi_comment_slot(ptr, r, i, slot))
            text = _get_string(lib.adi_comment_text, ptr, r, i)
            # 1 based channel id, -1 = all channels. A comment can belong to
            # a LabChart channel that is in no record (e.g. a calculated
            # channel); its id is then LabChart's channel number, as the SDK
            # reports it (or None when channel ids are renumbered).
            if slot[0] < 0:
                ch = -1
            else:
                ch = slot_to_id.get(slot[0], slot[0] + 1 if all_slots else None)
            self.comments.append(Comment(text, int(tick[0]), ch, cid[0]))
        for c in self.comments:
            c._add_info(self.id, self.tick_dt)

        self.record_time = RecordTime(self.tick_dt, start[0], tz[0], int(trig[0]))
        self.record_time.add_info(self.duration)

    def __repr__(self):
        return print_object(self)


_CALC_KINDS = {
    lib.ADI_CALC_OTHER: "other",
    lib.ADI_CALC_ARITHMETIC: "arithmetic",
    lib.ADI_CALC_CYCLIC_MEASUREMENTS: "cyclic_measurements",
    lib.ADI_CALC_DIGITAL_FILTER: "digital_filter",
    lib.ADI_CALC_SMOOTHING: "smoothing",
}


class CalcSetting():

    """
    Settings of a channel calculation stored in the file.

    LabChart keeps a channel's calculation settings when the calculation is
    changed or turned off, so a setting doesn't prove the calculation is in
    use: a channel can have several, or have samples recorded from an input.

    Attributes
    ----------
    channel_number : int
        LabChart's channel number (1 based)
    kind : str
        'arithmetic', 'cyclic_measurements', 'digital_filter', 'smoothing' or
        'other' (a settings class not identified yet; see class_id)
    name : str
        e.g. 'Arithmetic' ('' for 'other')
    class_id : str
        GUID of the settings class
    expression : str or None
        Arithmetic only, e.g. 'Ch2*Ch3'
    units : str or None
        Arithmetic only
    """

    def __init__(self, channel_number, kind, name, class_id, expression, units):
        self.channel_number = channel_number
        self.kind = kind
        self.name = name
        self.class_id = class_id
        self.expression = expression
        self.units = units

    def __repr__(self):
        s = f"Ch{self.channel_number}: {self.name or 'class ' + self.class_id}"
        if self.expression is not None:
            s += f" {self.expression!r} [{self.units}]"
        return f"<CalcSetting {s}>"


class LabChartChannel():

    """
    A channel as LabChart numbers it, including calculated channels that have
    no samples in the file.

    Attributes
    ----------
    number : int
        LabChart's channel number (1 based)
    name : str
    channel : Channel or None
        The File.channels entry for this channel (None if it isn't listed,
        e.g. a channel without samples after the last channel with samples,
        or with remove_empty_channels=True)
    has_samples : bool
        True if any record stores samples for it
    calculations : [CalcSetting]
        Calculation settings stored for the channel (see CalcSetting)
    """

    def __init__(self, number, name, channel, has_samples, calculations):
        self.number = number
        self.name = name
        self.channel = channel
        self.has_samples = has_samples
        self.calculations = calculations

    def __repr__(self):
        c = ", ".join(x.name or x.class_id for x in self.calculations)
        return (f"<LabChartChannel {self.number} {self.name!r} "
                f"samples={self.has_samples} calculations=[{c}]>")


class File():

    """
    Attributes
    ----------
    file_path
    file_loaded
    h
    n_records
    n_channels
    records
    channels
    channel_names
    labchart_channels : [LabChartChannel]
        Every channel LabChart defines, including calculated channels without
        samples, with the calculation settings stored for each
    calc_settings : [CalcSetting]
        All calculation settings, by channel number
    modules : [str]
        LabChart modules (extensions) with settings in the file, e.g.
        'CyclicMeasurements(8)'

    Methods
    -------
    get_channel_by_name - returns a specific channel
    get_comments - all comments
    is_channel_in_record - whether a channel has samples in a record
    channels_in_record - names of the channels with samples in a record
    close - close the file (also done when the object is deleted)

    """
    def __init__(self, file_path, remove_empty_channels: bool = False,
                 n_threads: int = 0, simd: str = "auto"):
        self.file_path = str(file_path)
        self.file_loaded = False
        if simd not in _SIMD:
            raise ValueError(f"simd must be one of {list(_SIMD)}, not {simd!r}")
        self.h = _FileHandle(file_path, n_threads, simd)
        self.file_loaded = True

        ptr = self.h.get()
        n = ffi.new("int *")
        _check(lib.adi_n_records(ptr, n))
        self.n_records = n[0]

        # Channels by LabChart slot (as the SDK), optionally only those
        # with samples
        _check(lib.adi_n_slots(ptr, n))
        slots = []
        for s in range(n[0]):
            _check(lib.adi_slot_channel(ptr, s, n))
            if n[0] >= 0 or not remove_empty_channels:
                slots.append((s, n[0]))
        self.n_channels = len(slots)
        slot_to_id = {s: k + 1 for k, (s, _) in enumerate(slots)}

        self.records = [Record(self.h, x+1, slot_to_id, not remove_empty_channels)
                        for x in range(self.n_records)]

        self.channels = [Channel(self.h, k+1, self.records, self, s, c)
                         for k, (s, c) in enumerate(slots)]
        self.channel_names = [x.name for x in self.channels]
        self._read_calculations(ptr)

    def _read_calculations(self, ptr):
        n = ffi.new("int *")
        kind = ffi.new("int *")
        _check(lib.adi_n_calc_settings(ptr, n))
        calcs = []
        for i in range(n[0]):
            slot = ffi.new("int *")
            _check(lib.adi_calc_setting_info(ptr, i, slot, kind))
            arith = kind[0] == lib.ADI_CALC_ARITHMETIC
            text = lambda w: _get_string(lib.adi_calc_setting_text, ptr, i, w)
            calcs.append(CalcSetting(
                slot[0] + 1, _CALC_KINDS.get(kind[0], "other"),
                text(lib.ADI_CALC_TEXT_NAME), text(lib.ADI_CALC_TEXT_CLASS),
                text(lib.ADI_CALC_TEXT_EXPRESSION) if arith else None,
                text(lib.ADI_CALC_TEXT_UNITS) if arith else None))
        self.calc_settings = calcs

        _check(lib.adi_n_modules(ptr, n))
        self.modules = [_get_string(lib.adi_module_name, ptr, i) for i in range(n[0])]

        by_slot = {c.slot: c for c in self.channels}
        _check(lib.adi_n_labchart_channels(ptr, n))
        self.labchart_channels = []
        for s in range(n[0]):
            _check(lib.adi_slot_channel(ptr, s, kind))
            self.labchart_channels.append(LabChartChannel(
                s + 1, _get_string(lib.adi_slot_name, ptr, s), by_slot.get(s),
                kind[0] >= 0, [c for c in calcs if c.channel_number == s + 1]))
        for c in self.channels:
            c.calculations = self.labchart_channels[c.slot].calculations

    @overload
    def get_comments(self, return_as: Literal["list"] = ...) -> list[Comment]: ...

    @overload
    def get_comments(self, return_as: Literal["table"]) -> pd.DataFrame: ...

    @overload
    def get_comments(self, return_as: Literal["object"]) -> CommentCollection: ...

    def get_comments(
        self,
        return_as: Literal["list", "table", "object"] = "list",
    ) -> list[Comment] | pd.DataFrame | CommentCollection:
        """
        Retrieve all comments associated with this file.

        Parameters
        ----------
        return_as : {"list", "table", "object"}, optional
            The format in which to return the comments (default is "list").

            - "list": a plain list of Comment objects.
            - "table": a pandas DataFrame with one row per comment.
            - "object": a CommentCollection wrapping the comments, with
              helper methods (e.g. get_comment_pairs).
        """
        comments: list[Comment] = []
        record_times: list[RecordTime] = []

        for record in self.records:
            comments.extend(record.comments)
            record_times.append(record.record_time)

        if return_as == "list":
            return comments
        if return_as == "object":
            return CommentCollection(comments, record_times)
        if return_as == "table":
            return CommentCollection(comments, record_times).to_dataframe()

        raise ValueError(f"Unknown return_as value: {return_as!r}")

    def get_channel_by_name(self, chan_name, case_sensitive=False, partial_match=True) -> Channel:
        """

        Parameters
        ----------
        chan_name : string
            Name of the channel to match
        case_sensitive : boolean, optional
            Whether to require case matching. The default is False.
        partial_match : boolean, optional
            Whether to allow partial matching. For example 'pres' could
            be used to match 'Bladder Pressure'. The default is True.

        Returns
        -------
        channel object

        """
        search_name = chan_name if case_sensitive else chan_name.lower()

        matched_names = []
        matched_channels = []

        for name, channel in zip(self.channel_names, self.channels):
            candidate = name if case_sensitive else name.lower()

            if partial_match:
                is_match = search_name in candidate
            else:
                is_match = search_name == candidate

            if is_match:
                matched_names.append(name)
                matched_channels.append(channel)

        if len(matched_channels) == 0:
            raise ValueError(
                f"No channel found matching '{chan_name}'. "
                f"Available channels: {self.channel_names}"
            )
        elif len(matched_channels) > 1:
            raise ValueError(
                f"Multiple channels found matching '{chan_name}': {matched_names}. "
                f"Please provide a more specific name."
            )

        return matched_channels[0]

    def is_channel_in_record(self, channel_id, record_id) -> bool:
        """True if the channel (1 based) has samples in the record (1 based)."""
        return self.channels[channel_id - 1].n_samples[record_id - 1] > 0

    def channels_in_record(self, record_id) -> list[str]:
        """Names of the channels with samples in the record (1 based)."""
        return [c.name for c in self.channels if c.n_samples[record_id - 1] > 0]

    def close(self):
        """Close the file. Channel data can't be read afterwards."""
        self.h.close()
        self.file_loaded = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return print_object(self)
