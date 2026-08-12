"""Per-kernel PTX instruction census for tile-complex-split.cpp.

NVPTX emits untyped mnemonics: a 128-bit shared load is ld.shared.v4.b32 for
every scalar type (4 x b32 = 16 bytes), so that one counter is the number of
128-bit shared loads regardless of float/double/complex.
"""
import re, sys

txt = ''
for f in sys.argv[1].split(','):
    txt += open(f).read()
entries = re.split(r'\n(?=\.(?:visible|weak) \.entry )', txt)
tmap = {'f': 'float', 'd': 'double', 'St7complexIfE': 'cfloat', 'St7complexIdE': 'cdouble'}
want = sys.argv[2] if len(sys.argv) > 2 else ''

COUNTERS = [
    ('lds128', r'\bld\.shared\.v4\.b32\b'),
    ('lds_oth', r'\bld\.shared\.(?!v4\.b32)\S+'),
    ('sts128', r'\bst\.shared\.v4\.b32\b'),
    ('sts64', r'\bst\.shared\.b64\b'),
    ('sts32', r'\bst\.shared\.b32\b'),
    ('ldg128', r'\bld\.global\.v4\.b32\b'),
    ('ldg_oth', r'\bld\.global\.(?!v4\.b32)\S+'),
    ('fma32', r'\bfma\.rn\.f32\b'),
    ('fma64', r'\bfma\.rn\.f64\b'),
    ('mul32', r'\bmul\.rn\.f32\b'),
    ('mul64', r'\bmul\.rn\.f64\b'),
    ('calls', r'\bcall\.uni\b'),
    ('mulc3', r'__mul[sd]c3'),
]

rows = []
for e in entries:
    m = re.match(r'\.(?:visible|weak) \.entry (\S+)\(', e)
    if not m:
        continue
    nm = m.group(1)
    if '_with_offset' in nm:
        continue
    d = re.match(r"_ZTS15SplitGemmKernelI(.+)7TileCfgILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)EELb(\d)ELb(\d)E", nm)
    if not d:
        continue
    key = "%-8s %-14s %-10s %-4s" % (
        tmap.get(d.group(1), d.group(1)),
        "%sx%sx%s/%sx%s" % d.group(2, 3, 4, 5, 6),
        'planar' if d.group(8) == '1' else 'interleav',
        'pred' if d.group(7) == '1' else 'fast')
    if want and want not in key:
        continue
    rows.append((key, [len(re.findall(p, e)) for _, p in COUNTERS]))

if not rows:
    print("no matching kernels")
    sys.exit(0)
print("%-40s" % 'kernel' + ''.join("%8s" % n for n, _ in COUNTERS))
for k, v in sorted(rows):
    print("%-40s" % k + ''.join("%8d" % x for x in v))
