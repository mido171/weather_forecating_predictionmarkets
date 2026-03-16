import test from "node:test";
import assert from "node:assert/strict";

import { toFiniteNumber } from "./liveOrderbookUtils.js";
import {
  buildAllOpportunitiesFromStations,
  calculateFractionalKellyPositionSize,
  chooseAvailableTargetDates,
  retainRecordKeys,
  shouldRevalidateSnapshotMeta,
  snapshotMetaFromData,
  stationWithSortedBuckets,
} from "./liveTradingData.js";

test("toFiniteNumber preserves missing values instead of converting them to zero", () => {
  assert.equal(toFiniteNumber(null), null);
  assert.equal(toFiniteNumber(undefined), null);
  assert.equal(toFiniteNumber(""), null);
  assert.equal(toFiniteNumber("   "), null);
  assert.equal(toFiniteNumber(0), 0);
  assert.equal(toFiniteNumber("42"), 42);
});

test("buildAllOpportunitiesFromStations excludes buckets with missing prices or EV", () => {
  const rows = buildAllOpportunitiesFromStations([
    {
      stationId: "KMIA",
      buckets: [
        {
          marketTicker: "A",
          bucketLabel: "84 to 85",
          yesModelWinProbability: 0.71,
          yesEv: 0.11,
          yesAskCents: 34,
          noModelWinProbability: 0.29,
          noEv: -0.09,
          noAskCents: 68,
        },
        {
          marketTicker: "B",
          bucketLabel: "86 or above",
          yesModelWinProbability: 0.8,
          yesEv: 0.2,
          yesAskCents: null,
          noModelWinProbability: 0.2,
          noEv: null,
          noAskCents: 80,
        },
      ],
    },
  ]);

  assert.equal(rows.length, 2);
  assert.deepEqual(rows.map((row) => row.marketTicker), ["A", "A"]);
});

test("stationWithSortedBuckets drops malformed buckets and normalizes station defaults", () => {
  const station = stationWithSortedBuckets({
    stationId: "KLAX",
    buckets: [
      null,
      {
        marketTicker: "T2",
        bucketLabel: "77 or above",
      },
      {
        marketTicker: "T1",
        bucketLabel: "75 to 76",
      },
    ],
  });

  assert.equal(station.stationId, "KLAX");
  assert.equal(station.displayName, "KLAX");
  assert.deepEqual(station.buckets.map((bucket) => bucket.marketTicker), ["T1", "T2"]);
});

test("buildAllOpportunitiesFromStations ignores malformed station and bucket entries", () => {
  const rows = buildAllOpportunitiesFromStations([
    null,
    {
      stationId: "KMIA",
      buckets: [
        null,
        {
          marketTicker: "A",
          bucketLabel: "84 to 85",
          yesModelWinProbability: 0.71,
          yesEv: 0.11,
          yesAskCents: 34,
        },
      ],
    },
  ]);

  assert.equal(rows.length, 1);
  assert.equal(rows[0].marketTicker, "A");
});

test("chooseAvailableTargetDates prefers backend-provided dates over browser-local fallbacks", () => {
  const dates = chooseAvailableTargetDates({
    backendDates: ["2026-03-07", "2026-03-08"],
    cachedDates: ["2026-03-08"],
    fallbackDates: ["2026-03-08", "2026-03-09"],
  });

  assert.deepEqual(dates, ["2026-03-07", "2026-03-08"]);
});

test("shouldRevalidateSnapshotMeta refreshes stale or inference-empty cache entries", () => {
  const nowMillis = 1_000_000;
  const maxAgeMs = 15_000;

  assert.equal(shouldRevalidateSnapshotMeta(null, nowMillis, maxAgeMs), true);
  assert.equal(
    shouldRevalidateSnapshotMeta(
      snapshotMetaFromData([], [], nowMillis),
      nowMillis,
      maxAgeMs,
    ),
    true,
  );
  assert.equal(
    shouldRevalidateSnapshotMeta(
      {
        fetchedAtMillis: nowMillis,
        stationCount: 4,
        opportunityCount: 0,
        hasInference: false,
      },
      nowMillis,
      maxAgeMs,
    ),
    true,
  );
  assert.equal(
    shouldRevalidateSnapshotMeta(
      {
        fetchedAtMillis: nowMillis - 20_000,
        stationCount: 4,
        opportunityCount: 3,
        hasInference: true,
      },
      nowMillis,
      maxAgeMs,
    ),
    true,
  );
  assert.equal(
    shouldRevalidateSnapshotMeta(
      {
        fetchedAtMillis: nowMillis - 5_000,
        stationCount: 4,
        opportunityCount: 3,
        hasInference: true,
      },
      nowMillis,
      maxAgeMs,
    ),
    false,
  );
});

test("retainRecordKeys prunes stale keys from bounded UI caches", () => {
  const source = {
    "2026-03-07": { rows: 7 },
    "2026-03-08": { rows: 8 },
    "2026-03-09": { rows: 9 },
  };

  const retained = retainRecordKeys(source, ["2026-03-08", "2026-03-09"]);

  assert.deepEqual(retained, {
    "2026-03-08": { rows: 8 },
    "2026-03-09": { rows: 9 },
  });
});

test("retainRecordKeys preserves object identity when no pruning is needed", () => {
  const source = {
    KXHIGHNY: { expanded: true },
    KXHIGHMIA: { expanded: false },
  };

  const retained = retainRecordKeys(source, ["KXHIGHNY", "KXHIGHMIA"]);

  assert.equal(retained, source);
});

test("calculateFractionalKellyPositionSize uses the configured fractional kelly stake formula", () => {
  const sizing = calculateFractionalKellyPositionSize({
    balanceUsd: 1_000,
    modelWinProbability: 0.72,
    entryPriceCents: 40,
    ev: 0.32,
    kellyFraction: 0.2,
    stakeCapUsd: 700,
  });

  assert.equal(sizing.marketPriceProbability, 0.4);
  assert.ok(Math.abs(sizing.fullKelly - 0.5333333333333333) < 1e-12);
  assert.ok(Math.abs(sizing.riskFractionUsed - 0.10666666666666667) < 1e-12);
  assert.ok(Math.abs(sizing.stakeUsd - 106.66666666666667) < 1e-9);
  assert.equal(sizing.isCapped, false);
});

test("calculateFractionalKellyPositionSize respects the stake cap", () => {
  const sizing = calculateFractionalKellyPositionSize({
    balanceUsd: 10_000,
    modelWinProbability: 0.8,
    entryPriceCents: 25,
    ev: 0.55,
    kellyFraction: 0.2,
    stakeCapUsd: 700,
  });

  assert.ok(Math.abs(sizing.uncappedStakeUsd - 1466.6666666666665) < 1e-9);
  assert.equal(sizing.stakeUsd, 700);
  assert.equal(sizing.isCapped, true);
});

test("calculateFractionalKellyPositionSize returns zero risk for non-executable prices", () => {
  const sizing = calculateFractionalKellyPositionSize({
    balanceUsd: 2_700,
    modelWinProbability: 0.9,
    entryPriceCents: 0,
    ev: 0.9,
    kellyFraction: 0.2,
    stakeCapUsd: 700,
  });

  assert.equal(sizing.fullKelly, 0);
  assert.equal(sizing.riskFractionUsed, 0);
  assert.equal(sizing.stakeUsd, 0);
});
