# 04 ASOS METAR HF And Minute Observations

Source spec:

```text
04_asos_metar_hf_and_minute_observations.md
```

Execution role:

This task follows IEM MOS because surface observations are needed for target-day state, diagnostics, station context, and later cutoff-safe feature construction.

Persistence target:

```text
postgresql://postgres:root@127.0.0.1:5432/klga_tmax_research
```

Start condition:

Begin with the regular IEM ASOS/METAR archive for all canonical non-pseudo stations. Keep one-minute delayed archive and optional low-latency feeds distinct from standard METAR history.
