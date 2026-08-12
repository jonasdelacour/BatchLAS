import re, sys
txt = open(sys.argv[1]).read()
blocks = re.findall(
    r"Compiling entry function '([^']+)' for 'sm_89'\n.*?\n\s+(\d+) bytes stack frame, "
    r"(\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s+: Used (\d+) registers",
    txt)
tmap = {'f': 'float', 'd': 'double', 'St7complexIfE': 'cfloat', 'St7complexIdE': 'cdouble'}
rows = []
for nm, stack, ss, sl, regs in blocks:
    if '_with_offset' in nm:
        continue
    m = re.match(r"_ZTS15SplitGemmKernelI(.+)7TileCfgILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)EELb(\d)ELb(\d)E", nm)
    if not m:
        continue
    tile = 'x'.join(m.group(2, 3, 4)) + '/' + m.group(5) + 'x' + m.group(6)
    rows.append((tmap.get(m.group(1), m.group(1)), tile,
                 'planar' if m.group(8) == '1' else 'interleav',
                 'pred' if m.group(7) == '1' else 'fast',
                 int(regs), int(stack), int(ss), int(sl)))
order = {'cfloat': 0, 'cdouble': 1, 'double': 2, 'float': 3}
rows.sort(key=lambda r: (order.get(r[0], 9), r[1], r[2], r[3]))
print("%-8s %-14s %-10s %-5s %5s %6s %8s %8s" %
      ('dtype', 'tile', 'layout', 'path', 'regs', 'stack', 'spillst', 'spillld'))
for r in rows:
    print("%-8s %-14s %-10s %-5s %5d %6d %8d %8d" % r)
