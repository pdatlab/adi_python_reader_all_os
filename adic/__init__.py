# -*- coding: utf-8 -*-
"""Read LabChart (.adicht / .adidat) files on any OS. See adic.read_file."""

from .read import (read_file, File, Channel, Record, RecordTime, Comment,
                   CommentCollection, AdiError, simd_best)

__all__ = ["read_file", "File", "Channel", "Record", "RecordTime", "Comment",
           "CommentCollection", "AdiError", "simd_best"]
