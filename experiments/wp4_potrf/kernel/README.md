# WP4 steps 1.1-1.4 / 1.6 — register-probe evidence for the native POTRF CTA kernel

Three normalized ptxas reports of `batchlas_extensions_cta`, the device-link unit
`src/extensions/potrf_cta.cc` lands in. Taken with
`experiments/wp4_potrf/regbaseline/regprobe_any.sh batchlas_extensions_cta <abs.log>`
and normalized with `summarize_ptxas.awk` from the same directory —
**not** `scripts/register_probe.sh`, which hardcodes `batchlas_sycl.dir/link.txt`
and would report a clean result for code it never compiled.

Columns: `name, regs, own_stack, own_spill_st, own_spill_ld, callee_spill_st_max,
callee_spill_ld_max, cum_stack, barriers`.

| file | configuration |
|---|---|
| `regprobe_nb16_notunrolled.tsv` | first working kernel: `NB=16` for float/double/cfloat, `(P1)` loop with `break` |
| `regprobe_nb16_unrolled.tsv` | `(P1)` predicated instead of `break` (unrolls), still `NB=16` |
| `regprobe_extensions_cta_potrf_shipped.tsv` | **shipped**: predicated `(P1)`, `NB=8` for every type |

Eight kernel instantiations, four scalar types x two `PotrfScope` values
(`E0` = `SubGroup`, `E1` = `WorkGroup`); each also appears as `_with_offset`,
which is the same kernel and carries identical numbers.

| T | NB | TS | (P1) loop | regs SG/WG | stack frame | spill |
|---|---|---|---|---|---|---|
| float | 16 | 4 | break | 110 / 106 | 0 | 0 |
| double | 16 | 4 | break | 144 / 128 | **128** | 0 |
| complex\<float\> | 16 | 4 | break | 201 / 188 | **128** | 0 |
| complex\<double\> | 8 | 2 | break | 128 / 112 | 0 | 0 |
| float | 16 | 4 | predicated | 156 / 154 | 0 | 0 |
| double | 16 | 4 | predicated | 172 / 150 | 0 | 0 |
| complex\<float\> | 16 | 4 | predicated | 206 / 167 | 0 | 0 |
| **float** | **8** | **4** | predicated | **64 / 56** | **0** | **0** |
| **double** | **8** | **4** | predicated | **94 / 80** | **0** | **0** |
| **complex\<float\>** | **8** | **4** | predicated | **102 / 92** | **0** | **0** |
| **complex\<double\>** | **8** | **2** | predicated | **128 / 109** | **0** | **0** |

Additional cells measured and discarded along the way: `double NB=8 TS=4` with the
`break` (94 / 80, frame 0) and `double NB=16 TS=2` with the `break`
(142 / 128, frame **128** — which is what refuted the "it is `acc[TS][TS]`"
hypothesis, since `acc[2][2]` for a double is 32 B).

`regs x work_group_size` at the ladder's ceiling of 128 work-items: worst cell is
`complex<double>` at 128 x 128 = 16,384, against the hard 65,536-per-block limit.
The third gate condition is not reachable for any work-group size this kernel
launches.

**No non-potrf row changed in any of the three reports** — `diff` against
`../regbaseline/batchlas_extensions_cta.tsv` filtered of potrf rows is empty, so
the 16 pre-existing `complex<double>` spillers in this library
(`OrmQxCTAKernel`, `GesvdjCTAKernel`, `SyevCtaFusedKernel`) are untouched and the
whole-unit spill count is still 16. A whole-unit "zero spill" assertion is red
before potrf exists; the gate here is scoped to the eight potrf entries and
their callees, all of which report `0 0 0 0`.
