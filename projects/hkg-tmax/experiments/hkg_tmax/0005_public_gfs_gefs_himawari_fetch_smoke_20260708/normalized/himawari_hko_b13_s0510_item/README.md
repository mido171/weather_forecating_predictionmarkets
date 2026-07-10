# Himawari HKG B13 S0510 Item

Generated: `2026-07-08T08:04:15.373043Z`

This is one decoded Himawari-9 item for HKG inspection: B13 infrared full-disk segment `S0510`, observed at `2026-07-08T06:20:00Z`. It contains the projected Hong Kong Observatory pixel.

## Fastest Files

| File | What it contains |
|---|---|
| `hko_b13_s0510_item_summary.json` | Header, calibration, HKO pixel, segment stats, and output inventory. |
| `hko_b13_s0510_hko_21x21_window.csv` | Readable 441-pixel local window centered on HKO. |
| `hko_b13_s0510_all_pixels_first_5000_rows.csv` | First rows of the full pixel table for quick viewing. |
| `hko_b13_s0510_all_pixels.parquet` | Full decoded 3,025,000-pixel table. |
| `hko_b13_s0510_all_pixels.csv.gz` | Full decoded 3,025,000-pixel table as compressed CSV. |
| `hko_b13_s0510_header_full.json` | Complete parsed HSD header fields. |
| `hko_b13_s0510_calibration.json` | Count-to-radiance and radiance-to-brightness-temperature coefficients. |

## HKO Pixel

| Field | Value |
|---|---:|
| global line | 2730.213 |
| global column | 2772.788 |
| local row | 529 |
| local column | 2772 |
| count | 1552 |
| radiance | 9.373766 |
| B13 brightness temp C | 23.883 |

## Segment Summary

| Metric | Value |
|---|---:|
| pixels | 3025000 |
| valid pixels | 3014000 |
| outside-scan pixels | 11000 |
| brightness temp C min | -203.152 |
| brightness temp C median | 5.475 |
| brightness temp C max | 27.000 |
| HKO 21x21 mean C | 23.496 |

`quality_code`: `0 = valid`, `1 = outside scan`, `2 = error`.
