"""Check that a macOS payload runs on every Mac the installer lets it onto.

Usage: python3 check_payload.py <payload-dir> <arch> <min-macos>

Walks <payload-dir>, reads the header of every Mach-O file (executables,
.so and .dylib), and fails when one has no <arch> slice or needs a newer
macOS than <min-macos>. The installer's Distribution.xml promises those two
things, so a dependency that drops an architecture or raises its macOS floor
fails the build instead of reaching a Mac it cannot run on.

Reads the headers itself (no lipo or otool), so it runs anywhere Python does.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path

CPU_TYPES = {0x01000007: "x86_64", 0x0100000C: "arm64"}
FAT_MAGIC = 0xCAFEBABE
FAT_MAGIC_64 = 0xCAFEBABF
MH_MAGIC = 0xFEEDFACE
MH_MAGIC_64 = 0xFEEDFACF
LC_VERSION_MIN_MACOSX = 0x24
LC_BUILD_VERSION = 0x32
PLATFORM_MACOS = 1
# Java class files share FAT_MAGIC; a real fat header lists only a few slices.
MAX_FAT_SLICES = 16


@dataclass(frozen=True)
class Slice:
    """One architecture inside a Mach-O file and the macOS it needs."""

    arch: str
    min_macos: tuple[int, int] | None


def _version(encoded: int) -> tuple[int, int]:
    """Decode a Mach-O xxxx.yy.zz version into (major, minor)."""
    return encoded >> 16, (encoded >> 8) & 0xFF


def _thin_slice(data: bytes, offset: int) -> Slice | None:
    """Read the architecture and macOS floor of the thin Mach-O at ``offset``."""
    if len(data) < offset + 28:
        return None
    magic = struct.unpack_from("<I", data, offset)[0]
    if magic == MH_MAGIC_64:
        header_size = 32
    elif magic == MH_MAGIC:
        header_size = 28
    else:
        return None
    cputype, _subtype, _filetype, ncmds, _sizeofcmds = struct.unpack_from("<iiiII", data, offset + 4)
    arch = CPU_TYPES.get(cputype & 0xFFFFFFFF, f"cputype-{cputype:#x}")
    min_macos = None
    cursor = offset + header_size
    for _ in range(ncmds):
        if len(data) < cursor + 8:
            break
        cmd, cmdsize = struct.unpack_from("<II", data, cursor)
        if cmd == LC_BUILD_VERSION:
            platform, minos = struct.unpack_from("<II", data, cursor + 8)
            if platform == PLATFORM_MACOS:
                min_macos = _version(minos)
        elif cmd == LC_VERSION_MIN_MACOSX:
            min_macos = _version(struct.unpack_from("<I", data, cursor + 8)[0])
        if cmdsize == 0:
            break
        cursor += cmdsize
    return Slice(arch, min_macos)


def read_slices(path: Path) -> list[Slice]:
    """Return the slices of a Mach-O file, or an empty list for anything else."""
    with path.open("rb") as handle:
        head = handle.read(8)
    if len(head) < 8:
        return []
    big_magic, count = struct.unpack(">II", head)
    if big_magic in (FAT_MAGIC, FAT_MAGIC_64) and 0 < count <= MAX_FAT_SLICES:
        data = path.read_bytes()
        entry_size = 32 if big_magic == FAT_MAGIC_64 else 20
        slices = []
        for index in range(count):
            entry = 8 + index * entry_size
            if big_magic == FAT_MAGIC_64:
                offset = struct.unpack_from(">Q", data, entry + 8)[0]
            else:
                offset = struct.unpack_from(">I", data, entry + 8)[0]
            thin = _thin_slice(data, offset)
            if thin:
                slices.append(thin)
        return slices
    if struct.unpack("<I", head[:4])[0] in (MH_MAGIC, MH_MAGIC_64):
        thin = _thin_slice(path.read_bytes(), 0)
        return [thin] if thin else []
    return []


def check_payload(root: Path, arch: str, min_macos: tuple[int, int]) -> tuple[int, list[str]]:
    """Return how many Mach-O files were checked and every problem found."""
    checked = 0
    problems = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        slices = read_slices(path)
        if not slices:
            continue
        checked += 1
        name = path.relative_to(root)
        match = next((s for s in slices if s.arch == arch), None)
        if match is None:
            found = ", ".join(s.arch for s in slices)
            problems.append(f"{name}: no {arch} code (has {found})")
        elif match.min_macos and match.min_macos > min_macos:
            needed = ".".join(map(str, match.min_macos))
            problems.append(f"{name}: needs macOS {needed}")
    return checked, problems


def main(argv: list[str]) -> int:
    """Check one payload directory; print a summary and any problems."""
    if len(argv) != 4:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    root, arch, floor_text = Path(argv[1]), argv[2], argv[3]
    major, _, minor = floor_text.partition(".")
    floor = (int(major), int(minor or 0))
    checked, problems = check_payload(root, arch, floor)
    if checked == 0:
        print(f"ERROR: no Mach-O files found under {root}", file=sys.stderr)
        return 1
    if problems:
        print(f"ERROR: {len(problems)} of {checked} binaries cannot run on {arch} macOS {floor_text}:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"  {checked} binaries checked: all have {arch} code and run on macOS {floor_text}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
