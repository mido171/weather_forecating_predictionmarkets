# 01 Station Universe And Coordinates

Source spec:

```text
10_station_universe_and_coordinates.md
```

Execution role:

This task comes before provider fetches because all station-based sources need the canonical station ids, MOS station suffixes, coordinates, region groupings, and pseudo-point definitions.

Persistence target:

```text
postgresql://<user>:<password>@127.0.0.1:5432/klga_tmax_research
```

Start condition:

Load or materialize the canonical KLGA station registry before Wunderground, IEM MOS, ASOS/METAR, GribStream point extraction, or Open-Meteo requests.
