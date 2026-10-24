#!/usr/bin/env python3
"""Helpers for scripts/seed_build_tree.sh: path rewrite, RUNPATH fix-up, source mtimes.

  seed_build_tree_fix.py rewrite DST_BUILD OLD=NEW...
  seed_build_tree_fix.py runpath DST_BUILD OLD_BUILD NEW_BUILD [OLD_SRC NEW_SRC] [--check]
  seed_build_tree_fix.py mtimes SEED_SRC DST_SRC CHANGED_LIST UNCHANGED_LIST
  seed_build_tree_fix.py keep-outputs|restore-outputs DST_BUILD WORKDIR
  seed_build_tree_fix.py --self-test

rewrite: every text file under DST_BUILD gets each OLD path replaced by its NEW path in one
pass (longest OLD first, so a build dir inside its source dir and a destination inside the
seed's source both map correctly). Byte-level; files holding a NUL byte are left alone; mode
and mtime are kept, so make sees no regenerated input. Except: a flags.make whose -D values
changed (BATCHLAS_TUNED_SOURCE_DIR="/abs/tuned") gets mtime now, so that target recompiles;
its objects hold the seed's path as a string literal and would read the seed's files.

runpath: an ELF whose RUNPATH/RPATH names the seed would load the SEED's libraries until it is
relinked. Entries under OLD_BUILD are rewritten in place to $ORIGIN-relative form (what
CMAKE_BUILD_RPATH_USE_ORIGIN emits; always shorter than an absolute path here). An ELF whose
fixed string does not fit, or whose string shares bytes with another .dynstr name, is deleted
so make relinks it. --check only reports and exits 1 on any seed path.

mtimes: unchanged files get the seed checkout's mtime for the same path (the seed tree was
built against those), changed files get now.

keep-outputs / restore-outputs: around the regenerating cmake run, which deletes every
custom-command output whose rule hash changed (all of them, the hash covers absolute paths).
"""

import filecmp
import os
import re
import shutil
import struct
import sys
import tempfile
import time

SKIP_SUFFIXES = (".o", ".a", ".so", ".bin", ".cubin", ".fatbin", ".ptx", ".gch", ".pcm", ".png",
                 ".gz", ".xz", ".zst", ".pdf")
MAX_TEXT_BYTES = 64 << 20
NOT_PATH_CHAR = rb"(?![A-Za-z0-9_.+\-])"


def build_pattern(pairs):
    olds = sorted({old for old, _ in pairs}, key=len, reverse=True)
    table = dict(pairs)
    regex = re.compile(b"(" + b"|".join(re.escape(o) for o in olds) + b")" + NOT_PATH_CHAR)
    return regex, table


def is_skipped_name(name):
    return name.endswith(SKIP_SUFFIXES) or ".so." in name


def defines(flags_make):
    """The *_DEFINES lines of a flags.make: a path there is compiled into the object."""
    return [ln for ln in flags_make.splitlines() if re.match(rb"\w+_DEFINES\s*=", ln)]


def rewrite_tree(root, pairs):
    regex, table = build_pattern([(o.encode(), n.encode()) for o, n in pairs])
    files = links = stale = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                target = os.readlink(path).encode()
                fixed = regex.sub(lambda m: table[m.group(1)], target)
                if fixed != target:
                    os.unlink(path)
                    os.symlink(fixed, path)
                    links += 1
                continue
            if is_skipped_name(name):
                continue
            st = os.stat(path)
            if st.st_size > MAX_TEXT_BYTES or st.st_size == 0:
                continue
            with open(path, "rb") as fh:
                data = fh.read()
            if data.startswith(b"\x7fELF") or b"\0" in data:
                continue
            fixed = regex.sub(lambda m: table[m.group(1)], data)
            if fixed == data:
                continue
            with open(path, "wb") as fh:
                fh.write(fixed)
            if name == "flags.make" and defines(fixed) != defines(data):
                stale += 1
            else:
                os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
            files += 1
    print("rewrite: %d file(s), %d symlink(s) updated; %d target(s) with a path in a -D recompile"
          % (files, links, stale))
    return 0


class Elf:
    """Just enough ELF64 little-endian parsing to find and patch DT_RUNPATH / DT_RPATH."""

    DT_NEEDED, DT_STRTAB, DT_SONAME, DT_RPATH, DT_RUNPATH = 1, 5, 14, 15, 29

    def __init__(self, data):
        if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
            raise ValueError("not ELF64 LE")
        self.data = data
        (self.phoff, self.shoff) = struct.unpack_from("<QQ", data, 0x20)
        (self.phentsize, self.phnum, self.shentsize, self.shnum) = struct.unpack_from("<HHHH", data, 0x36)

    def segments(self):
        for i in range(self.phnum):
            yield struct.unpack_from("<IIQQQQQQ", self.data, self.phoff + i * self.phentsize)

    def vaddr_to_off(self, addr):
        for p_type, _, p_off, p_vaddr, _, p_filesz, _, _ in self.segments():
            if p_type == 1 and p_vaddr <= addr < p_vaddr + p_filesz:
                return addr - p_vaddr + p_off
        raise ValueError("address not in a PT_LOAD")

    def dynamic(self):
        for p_type, _, p_off, _, _, p_filesz, _, _ in self.segments():
            if p_type == 2:
                return [struct.unpack_from("<qQ", self.data, p_off + 16 * i) for i in range(p_filesz // 16)]
        return []

    def dynsym_name_offsets(self, strtab_off):
        names = []
        for i in range(self.shnum):
            sh = struct.unpack_from("<IIQQQQIIQQ", self.data, self.shoff + i * self.shentsize)
            sh_type, sh_offset, sh_size, sh_link, sh_entsize = sh[1], sh[4], sh[5], sh[6], sh[9]
            if sh_type != 11 or not sh_entsize:
                continue
            link = struct.unpack_from("<IIQQQQIIQQ", self.data, self.shoff + sh_link * self.shentsize)
            if link[4] != strtab_off:
                continue
            names += [struct.unpack_from("<I", self.data, sh_offset + j * sh_entsize)[0]
                      for j in range(sh_size // sh_entsize)]
        return names

    def runpath(self):
        """(file offset of the string, string bytes, set of other .dynstr offsets) or None."""
        dyn = self.dynamic()
        strtab = [v for t, v in dyn if t == self.DT_STRTAB]
        rp = [v for t, v in dyn if t in (self.DT_RUNPATH, self.DT_RPATH)]
        if not strtab or not rp:
            return None
        base = self.vaddr_to_off(strtab[0])
        start = base + rp[0]
        end = self.data.index(b"\0", start)
        others = {v for t, v in dyn if t in (self.DT_NEEDED, self.DT_SONAME)}
        others.update(self.dynsym_name_offsets(base))
        return start, self.data[start:end], {base + o for o in others}


def under(entry, prefix):
    return bool(prefix) and (entry == prefix or entry.startswith(prefix + "/"))


def seed_entry(entry, olds, news):
    """True for an entry under a seed path; a destination may itself sit below the seed source."""
    return any(under(entry, o) for o in olds) and not any(under(entry, n) for n in news)


def fixed_runpath(value, elf_dir, old_build, new_build, old_src, new_src):
    out = []
    for entry in value.split(":"):
        if not seed_entry(entry, [old_build, old_src], [new_build, new_src]):
            out.append(entry)
        elif under(entry, old_build):
            rel = os.path.relpath(new_build + entry[len(old_build):], elf_dir)
            out.append("$ORIGIN" if rel == "." else "$ORIGIN/" + rel)
        else:
            out.append(new_src + entry[len(old_src):])
    return ":".join(out)


def names_seed(value, olds, news=()):
    return any(seed_entry(e, olds, news) for e in value.split(":"))


def runpath_tree(root, old_build, new_build, old_src=None, new_src=None, check=False):
    patched = deleted = bad = 0
    olds = [old_build, old_src]
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if os.path.islink(path) or name.endswith((".o", ".a")):
                continue
            try:
                with open(path, "rb") as fh:
                    if fh.read(4) != b"\x7fELF":
                        continue
                with open(path, "rb") as fh:
                    elf = Elf(fh.read())
                found = elf.runpath()
            except (OSError, ValueError, struct.error):
                continue
            if not found:
                continue
            start, raw, others = found
            value = raw.decode("utf-8", "surrogateescape")
            if not names_seed(value, olds, [new_build, new_src]):
                continue
            if check:
                print("runpath: %s still names the seed: %s" % (path, value))
                bad += 1
                continue
            new = fixed_runpath(value, dirpath, old_build, new_build, old_src, new_src).encode(
                "utf-8", "surrogateescape")
            shared = any(start < o <= start + len(raw) for o in others)
            if len(new) <= len(raw) and not shared:
                st = os.stat(path)
                with open(path, "r+b") as fh:
                    fh.seek(start)
                    fh.write(new + b"\0" * (len(raw) - len(new)))
                os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
                patched += 1
            else:
                os.unlink(path)
                deleted += 1
                print("runpath: deleted %s (fixed RUNPATH does not fit); make will relink it" % path)
    if check:
        print("runpath check: %d ELF file(s) still name the seed" % bad)
        return 1 if bad else 0
    print("runpath: %d ELF file(s) patched to $ORIGIN form, %d deleted for relink" % (patched, deleted))
    return 0


def rule_outputs(build):
    """Custom-command outputs CMake tracks in CMakeRuleHashes.txt (relative to the build dir)."""
    try:
        with open(os.path.join(build, "CMakeFiles", "CMakeRuleHashes.txt"), encoding="utf-8") as fh:
            rels = [line.split(None, 1)[1].strip() for line in fh if line[:1] not in ("#", "") and " " in line]
    except OSError:
        return []
    return [r for r in rels if os.path.isfile(os.path.join(build, r))]


def rule_makefiles(build, rel):
    """The build.make files that hold the rule producing REL."""
    pattern = re.compile(rb"^" + re.escape(rel.encode()) + rb":", re.M)
    found = []
    for dirpath, _, filenames in os.walk(build):
        if "build.make" in filenames:
            path = os.path.join(dirpath, "build.make")
            with open(path, "rb") as fh:
                if pattern.search(fh.read()):
                    found.append(path)
    return found


def keep_outputs(build, work):
    """Save custom-command outputs before regeneration.

    A moved tree changes every rule hash (the commands carry absolute paths), so cmake deletes
    those outputs, e.g. generated/select/tuned_tables.cc, and make would regenerate, recompile
    and relink them.
    """
    saved = 0
    for rel in rule_outputs(build):
        for mk in rule_makefiles(build, rel):
            dst = os.path.join(work, "makefiles", os.path.relpath(mk, build))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(mk, dst)
        dst = os.path.join(work, "outputs", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(build, rel), dst)
        saved += 1
    print("outputs: saved %d custom-command output(s)" % saved)
    return 0


def restore_outputs(build, work):
    """Put back an output cmake deleted when its rule is unchanged apart from the path rewrite."""
    restored = kept_deleted = 0
    root = os.path.join(work, "outputs")
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            target = os.path.join(build, rel)
            if os.path.exists(target):
                continue
            mks = rule_makefiles(build, rel)
            saved = [os.path.join(work, "makefiles", os.path.relpath(mk, build)) for mk in mks]
            same = bool(mks) and all(os.path.exists(sv) and filecmp.cmp(mk, sv, shallow=False)
                                     for mk, sv in zip(mks, saved))
            if same:
                shutil.copy2(os.path.join(dirpath, name), target)
                restored += 1
            else:
                kept_deleted += 1
                print("outputs: %s: its rule changed, left for make to regenerate" % rel)
    print("outputs: restored %d, %d left to regenerate" % (restored, kept_deleted))
    return 0


def read_list(path):
    with open(path, encoding="utf-8", errors="surrogateescape") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


def mtimes(seed_src, dst_src, changed_list, unchanged_list):
    now = time.time_ns()
    synced = touched = 0
    for rel in read_list(unchanged_list):
        seed, dst = os.path.join(seed_src, rel), os.path.join(dst_src, rel)
        try:
            st = os.stat(seed)
            if os.path.isfile(dst) and not os.path.islink(dst):
                os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))
                synced += 1
        except OSError:
            pass
    for rel in read_list(changed_list):
        dst = os.path.join(dst_src, rel)
        if os.path.isfile(dst) and not os.path.islink(dst):
            os.utime(dst, ns=(now, now))
            touched += 1
    print("mtimes: %d unchanged file(s) given the seed's mtime, %d changed file(s) touched" % (synced, touched))
    return 0


def self_test():
    bad = 0

    def expect(name, got, want):
        nonlocal bad
        ok = got == want
        bad += not ok
        print("  %-4s %s%s" % ("ok" if ok else "FAIL", name, "" if ok else ": got %r want %r" % (got, want)))

    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "flags.make")
        with open(f, "wb") as fh:
            fh.write(b"/r/B/build/src /r/B/src /r/B/build-acpp /r/B2 /r/B\n")
        os.utime(f, ns=(1, 1_000_000_000))
        rewrite_tree(tmp, [("/r/B/build", "/r/B/.wt/x/build"), ("/r/B", "/r/B/.wt/x")])
        with open(f, "rb") as fh:
            expect("one pass, longest first, path boundary", fh.read(),
                   b"/r/B/.wt/x/build/src /r/B/.wt/x/src /r/B/.wt/x/build-acpp /r/B2 /r/B/.wt/x\n")
        expect("mtime kept", os.stat(f).st_mtime_ns, 1_000_000_000)
        d = os.path.join(tmp, "t.dir")
        os.mkdir(d)
        for sub, body in (("inc", b"CXX_INCLUDES = -I/r/B/src\nCXX_DEFINES = -DX=1\n"),
                          ("def", b"CXX_INCLUDES = -I/r/B/src\nCXX_DEFINES = -DP=\\\"/r/B/tuned\\\"\n")):
            os.makedirs(os.path.join(d, sub))
            g = os.path.join(d, sub, "flags.make")
            with open(g, "wb") as fh:
                fh.write(body)
            os.utime(g, ns=(1, 1_000_000_000))
        rewrite_tree(d, [("/r/B", "/r/W")])
        expect("include-only rewrite keeps mtime", os.stat(os.path.join(d, "inc", "flags.make")).st_mtime_ns,
               1_000_000_000)
        expect("path in a -D bumps mtime", os.stat(os.path.join(d, "def", "flags.make")).st_mtime_ns > 1_000_000_000,
               True)
    expect("origin form", fixed_runpath("/opt/l:/r/B/build/src", "/r/W/build/tests", "/r/B/build",
                                        "/r/W/build", "/r/B", "/r/W"), "/opt/l:$ORIGIN/../src")
    expect("same dir", fixed_runpath("/r/B/build/src", "/r/W/build/src", "/r/B/build", "/r/W/build",
                                     None, None), "$ORIGIN")
    expect("seed named", names_seed("/opt/l:/r/B/build/src", ["/r/B/build"]), True)
    expect("prefix is not a path", names_seed("/r/B/build-x/src", ["/r/B/build"]), False)
    expect("destination below the seed source", names_seed("/r/B/.wt/x/build/src", ["/r/B/build", "/r/B"],
                                                           ["/r/B/.wt/x/build", "/r/B/.wt/x"]), False)
    print("seed_build_tree_fix self-test: %d failure(s)" % bad)
    return 1 if bad else 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "--self-test":
        return self_test()
    if len(argv) >= 4 and argv[1] == "rewrite":
        return rewrite_tree(argv[2], [a.split("=", 1) for a in argv[3:]])
    if len(argv) >= 5 and argv[1] == "runpath":
        check = "--check" in argv
        args = [a for a in argv[2:] if a != "--check"] + [None, None]
        return runpath_tree(args[0], args[1], args[2], args[3], args[4], check)
    if len(argv) == 4 and argv[1] in ("keep-outputs", "restore-outputs"):
        return (keep_outputs if argv[1] == "keep-outputs" else restore_outputs)(argv[2], argv[3])
    if len(argv) == 6 and argv[1] == "mtimes":
        return mtimes(*argv[2:])
    print(__doc__, file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))
