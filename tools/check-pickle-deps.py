#!/usr/bin/env python3
"""Report the modules/classes needed to unpickle every ``.pkl`` file in this repository.

The data set is read-only here, so this never *loads* a payload (the largest files are
several gigabytes).  It walks the pickle opcode stream of each file, resolves the
``STACK_GLOBAL`` / ``GLOBAL`` operands - including the ones that come back through the
pickle memo - and prints the set of referenced globals.  Raw byte probes are printed as
a cross-check.

Usage:
    python tools/check-pickle-deps.py [repo-root]
"""

import collections
import json
import os
import pickletools
import sys

# Strings pushed by these opcodes are candidates for module/class names.
_STRING_OPS = {
    "SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8",
    "SHORT_BINSTRING", "BINSTRING", "STRING", "BINBYTES",
    "SHORT_BINBYTES", "BINBYTES8", "BYTEARRAY8",
}
_PUSH_MEMO = {"MEMOIZE"}
_PUT_MEMO = {"BINPUT", "LONG_BINPUT", "PUT"}
_GET_MEMO = {"BINGET", "LONG_BINGET", "GET"}
_MEMO_FREE = {"POP", "POP_MARK", "DUP"}  # do not change the two-value history

# Raw byte probes used as an independent cross-check of the parsed globals.
PROBES = [
    b"numpy._core.multiarray",   # written by NumPy >= 2.0
    b"numpy.core.multiarray",    # written by NumPy 1.x
    b"numpy.ndarray",
    b"numpy.dtype",
    b"numpy.ma.",
    b"scipy",
    b"pandas",
    b"sklearn",
    b"matplotlib",
    b"torch",
    b"sympy",
    b"phcpy",
    b"builtins",
    b"copyreg",
    b"_codecs",
    b"dill",
    b"cloudpickle",
    b"h5py",
    b"networkx",
]


def count_occurrences(mm, needle):
    n, start, ln = 0, 0, len(needle)
    while True:
        i = mm.find(needle, start)
        if i < 0:
            return n
        n += 1
        start = i + ln


def parse_globals(path):
    """Return (protocol, sorted globals, opcode count) for one pickle file."""
    memo = {}
    next_index = 0
    v_last = v_prev = None
    globals_ = set()
    proto = None
    nops = 0

    with open(path, "rb") as f:
        for opcode, arg, _pos in pickletools.genops(f):
            nops += 1
            name = opcode.name
            value = None

            if name == "PROTO":
                proto = arg
                continue

            if name in _STRING_OPS:
                value = arg.decode("utf-8", "replace") if isinstance(arg, bytes) else arg
            elif name == "GLOBAL":
                mod, _, nm = arg.partition(" ")
                globals_.add((mod, nm))
                value = None  # the pushed object is not a string constant
            elif name == "STACK_GLOBAL":
                if isinstance(v_prev, str) and isinstance(v_last, str):
                    globals_.add((v_prev, v_last))
                value = None
            elif name in _GET_MEMO:
                value = memo.get(arg)
            elif name in _PUT_MEMO:
                memo[arg] = v_last
                next_index = max(next_index, (arg or 0) + 1)
                continue                      # memo ops do not touch the stack history
            elif name in _PUSH_MEMO:
                memo[next_index] = v_last
                next_index += 1
                continue
            elif name in _MEMO_FREE:
                continue

            v_prev, v_last = v_last, value

    return proto, sorted(f"{m}.{n}" for m, n in globals_), nops


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    files = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "/.git" in dirpath:
            continue
        files += [os.path.join(dirpath, fn) for fn in filenames if fn.endswith(".pkl")]
    files.sort()

    report = {}
    for path in files:
        proto, globs, nops = parse_globals(path)
        probes = {}
        with open(path, "rb") as f:
            mm = __import__("mmap").mmap(f.fileno(), 0, access=__import__("mmap").ACCESS_READ)
            for p in PROBES:
                c = count_occurrences(mm, p)
                if c:
                    probes[p.decode()] = c
            mm.close()
        report[path] = {"size": os.path.getsize(path), "protocol": proto,
                        "opcodes": nops, "globals": globs, "probes": probes}
        print(f"{path}  size={report[path]['size']}  proto={proto}  ops={nops}")
        print(f"    globals={globs}")
        print(f"    probes={probes}")

    all_globs = collections.Counter()
    all_probes = collections.Counter()
    protocols = collections.Counter()
    for r in report.values():
        all_globs.update(r["globals"])
        all_probes.update(r["probes"])
        protocols[r["protocol"]] += 1

    print("\n=== globals referenced by the corpus (files) ===")
    for k, v in sorted(all_globs.items()):
        print(f"{v:6d}  {k}")
    print("\n=== raw byte probes (total occurrences) ===")
    for k, v in sorted(all_probes.items()):
        print(f"{v:6d}  {k}")
    print("\n=== pickle protocols ===")
    for k, v in sorted(protocols.items()):
        print(f"{v:6d}  protocol {k}")
    print(f"\nTOTAL FILES: {len(files)}  TOTAL BYTES: {sum(r['size'] for r in report.values())}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickle-dep-report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
