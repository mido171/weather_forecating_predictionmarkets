# 03 IEM MOS Station Guidance

Source spec:

```text
02_iem_mos_station_guidance.md
```

Execution role:

This task is early because IEM MOS provides long-history station-specific forecast guidance with manageable request size and no GribStream bulk allowance dependency.

Persistence target:

```text
postgresql://<user>:<password>@127.0.0.1:5432/klga_tmax_research
```

Start condition:

Fetch the mandatory MOS product families separately by product and station: `MAV`, `MET`, `MEX`, `LAV`, `NBS`, and `NBE`. Do not collapse `MAV`, `MEX`, and `LAV` into one GFS value.
