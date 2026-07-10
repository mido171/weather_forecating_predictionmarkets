# Immutable HKG T+24 Point-in-Time Constitution

## Forecast target and cutoff

- Target: daily maximum temperature at the Hong Kong Observatory target station on local calendar date T.
- Default decision cutoff: 15:00:00 Asia/Hong_Kong on T−1, equivalent to 07:00:00 UTC on T−1.
- Hong Kong uses UTC+08:00 without daylight saving time.
- The cutoff must exist in one versioned configuration and one tested database/application function.

## Three different notions that must never be confused

1. `valid_time`: when a meteorological value describes the atmosphere.
2. `archive_presence`: whether a retrospective service contains that value today.
3. `available_at`: whether the operational system could actually have obtained that value before the historical cutoff.

Only item 3 permits strict deployable use.

## Availability proof grades

- `A_EXACT_FIRST_SEEN`: a collector recorded the provider/API response before cutoff.
- `B_PROVIDER_SCHEDULE_PROVEN`: an authoritative provider schedule plus a conservative approved latency buffer proves availability before cutoff.
- `C_RUN_TIME_ONLY`: only model initialization/run time is known; diagnostic until promoted.
- `D_RETROSPECTIVE_ONLY`: best track, finalized climate, reanalysis, or archive without operational publication proof.
- `E_REJECTED`: timestamp, parser, coverage, or source integrity failure.

A historical GribStream `asOf` value is a run-time filter, not availability proof. It may support B only when combined with provider dissemination evidence and a conservative buffer.

## Selection rule

For target date T, source s, and required valid time v:

```text
eligible = available_at_utc <= cutoff_utc(T)
latest_eligible = maximum available_at_utc among eligible rows
```

If `available_at_utc` is not provable, the row cannot enter strict production-style scoring.

## Target-memory rule

A finalized daily target value for T−1 is not automatically known at 15:00 on T−1. Target-memory features must be generated from labels whose own availability is proven before cutoff. Without publication proof, use the repository’s approved conservative lag contract, such as lag 7, rather than casually using lag 1.

## Online-state rule

For target T:

1. compute prediction using state updated only through previously settled targets;
2. issue and freeze the prediction;
3. after T settles, score T;
4. then update residual states.

## Sealed outcomes

Pre-2024 rows are the default development evidence base. Opening 2024, 2025, or 2026 outcomes requires the one-time protocol in `design/VALIDATION_PROTOCOL.md`. Data may be stored while labels remain inaccessible to development roles.
