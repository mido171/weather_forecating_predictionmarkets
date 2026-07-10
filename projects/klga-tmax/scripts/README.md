# KLGA scripts

Scripts are thin bounded entry points. Reusable logic belongs in
`src/klga_tmax`. New scripts must be finite, default to one worker, provide
`--help`, have focused offline tests, and make zero provider calls unless the
operator passes `--execute` with explicit date/source/request/runtime budgets.
