"""The TVQ1 container (model, test vectors, phonemizer data, dumps), as engine/tv.h reads it.

Little-endian:
    header   "TVQ1", u32 version (1), u32 count, u32 reserved
    entries  count x { char name[64]; u32 dtype; u32 ndim; u32 shape[4]; u32 offset; u32 nbytes }
    data     each tensor 16-byte aligned; offset counts from the start of the file
    dtype    0 f32, 1 i8, 2 i32

write_raw needs only the standard library (serial_dump.py runs under ESP-IDF's Python, which has
no numpy); write and read use numpy.
"""
import struct

HEADER, ENTRY = "<4sIII", "<64sII4III"
F32, I8, I32 = 0, 1, 2


def _align(n):
    return (n + 15) & ~15


def write_raw(path, tensors):
    """tensors: list of (name, dtype, shape, data bytes). Returns the end of the last tensor."""
    offset = _align(struct.calcsize(HEADER) + struct.calcsize(ENTRY) * len(tensors))
    entries, blobs = [], []
    for name, dtype, shape, data in tensors:
        if len(name) >= 64 or len(shape) > 4:
            raise ValueError(name)
        dims = list(shape) + [1] * (4 - len(shape))
        entries.append(struct.pack(ENTRY, name.encode(), dtype, len(shape), *dims, offset, len(data)))
        blobs.append((offset, data))
        offset = _align(offset + len(data))
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER, b"TVQ1", 1, len(tensors), 0) + b"".join(entries))
        for off, data in blobs:
            f.write(b"\0" * (off - f.tell()))
            f.write(data)
    return offset


def _numpy_dtypes():
    import numpy as np
    return np, {np.float32: F32, np.int8: I8, np.int32: I32}


def write(path, tensors):
    """tensors: {name: numpy array of float32, int8 or int32}. Returns the end of the last tensor."""
    np, codes = _numpy_dtypes()
    raw = []
    for name, value in tensors.items():
        a = np.ascontiguousarray(value)
        if a.dtype.type not in codes:
            raise TypeError(f"{name}: unsupported dtype {a.dtype}")
        raw.append((name, codes[a.dtype.type], a.shape, a.tobytes()))
    return write_raw(path, raw)


def read(path):
    """{name: numpy array} (views of the file's bytes)."""
    np, codes = _numpy_dtypes()
    types = {code: t for t, code in codes.items()}
    data = open(path, "rb").read()
    magic, version, count, _ = struct.unpack_from(HEADER, data)
    if magic != b"TVQ1" or version != 1:
        raise ValueError(f"{path}: not a TVQ1 file")
    out = {}
    for i in range(count):
        name, dtype, ndim, *rest = struct.unpack_from(ENTRY, data, struct.calcsize(HEADER) + struct.calcsize(ENTRY) * i)
        shape, offset, nbytes = rest[:ndim], rest[4], rest[5]
        t = types[dtype]
        out[name.rstrip(b"\0").decode()] = np.frombuffer(data, t, nbytes // np.dtype(t).itemsize, offset).reshape(shape)
    return out
