# Engineering Compromises

Deviations from `docs/task.md` that are intentional and approved.

## Parameter Space Reduction

| Parameter | task.md | Actual | Reason |
|-----------|---------|--------|--------|
| MS dims `d` | {50, 100, 200, 500} | {50, 60, 70} | Iteration speed for debugging; scale-up is E15 |
| MS rank `r` | {5, d/2, d} | 5 (fixed) | Focus on low-rank regime first; rank scan is E7 |
| Measurements `m` | 3d² | 2dr | Matches RIP lower bound, faster per-iteration |
| Reps `R` | 50 | 8–30 | Practical runtime; power analysis deferred to E20 |
| Noise levels | {0, 0.01} | {0.0} default | Noise sensitivity is E6 |
| MF layers `L` | 2,3,4 | TBD | MF not yet fully implemented in notebooks |

## Rationale

These reductions keep single-experiment runtime manageable (~minutes) while
preserving the qualitative comparison between Muon and SGD. The full parameter
space from task.md can be restored by editing the notebook constants.

*Approved by: user*
*Date: 2026-04-26*
