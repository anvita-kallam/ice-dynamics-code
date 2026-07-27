# more_sliding with A = 80 (superseded)

**Superseded:** production spin-ups now use **A=40** via the same script with
`case_id=more_sliding_A40`. See [`../more_sliding_A40/`](../more_sliding_A40/).

This directory keeps the earlier A=80 attempt notes (CG1→CG2 / fine_low handoff
blow-ups that motivated C re-ramp).

| | Baseline | This case (historical) |
|--|--|--|
| A | 20 | **80** |
| C | 1e-3 | 1e-3 |

Soft ice (A=80) diverged on projection handoffs without a C re-ramp. The script
now re-ramps C; prefer A=40 for the next production run.
