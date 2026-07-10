# GribStream Support Backfill Allowance Email Draft

Subject: Request for temporary allowance increase for HKG Tmax research backfill

Hi Alejandro / GribStream Support,

Thank you again for unblocking my IP and for clarifying the rate-limit guidance.

I have adjusted my client to follow your advice: one worker, fewer larger requests, exact `/api/v2/{dataset}/runs` calls, respectful spacing, `Retry-After` handling, and no automated crawling of documentation/model pages.

I would like to request a temporary allowance increase for a one-time research backfill for a Hong Kong Observatory daily maximum temperature forecasting project.

The objective is not to build a general-purpose NWP warehouse. The backfill is a small tactical extraction:

- one exact model run per model per target date;
- exact `timesList` run selection;
- only the lead windows needed for the Hong Kong local-calendar Tmax window;
- 12 point coordinates for deterministic and ensemble-mean models;
- HKO center only for full-member ensemble pulls;
- a compact variable set only.

Core datasets/ranges:

- `gfs`: 2021-03-22T00:00:00Z through 2026-06-22T00:00:00Z
- `gefsatmosmean`: 2020-10-01T18:00:00Z through 2026-06-21T18:00:00Z
- `gefsatmos`: 2020-10-01T18:00:00Z through 2026-06-21T18:00:00Z
- `ifsoper`: 2024-02-28T18:00:00Z through 2026-06-21T18:00:00Z
- `ifsenfo`: 2024-03-01T18:00:00Z through 2026-06-21T18:00:00Z
- `cwawrf15`: rolling last-three-day/prospective collection only

Optional/shadow datasets/ranges:

- `aifsoper`: 2025-02-25T18:00:00Z through 2026-06-21T18:00:00Z
- `aifsenfo`: 2025-07-02T18:00:00Z through 2026-06-21T18:00:00Z
- `aigfssfc`: 2026-04-16T18:00:00Z through 2026-06-21T18:00:00Z
- `aigfspres`: 2026-04-16T18:00:00Z through 2026-06-21T18:00:00Z
- `aigefssfc`: 2025-06-01T18:00:00Z through 2026-06-21T18:00:00Z
- `graphcast`: 2024-04-25T18:00:00Z through 2026-05-05T00:00:00Z
- `fourcastnetgfs`: 2024-05-02T18:00:00Z through 2026-03-01T12:00:00Z
- `nbmoc`: tiny 3-point probe only, not a full backfill

Using the published GribStream credit formula, my estimate for the core plus optional/shadow backfill is approximately 1.90 million credits before any cache discounts. My current plan is about 98,000 credits per day, so even the full tactical backfill is below 30 x 98,000 credits in total.

Would you be able to temporarily increase my allowance or advise the safest way to run this one-time backfill without triggering rate limits or firewall protection?

I am happy to follow any chunk-size, request-spacing, daily-cap, or time-window guidance you recommend.

Best,
Ahmad
