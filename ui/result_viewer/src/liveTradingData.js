import { bucketSortKey, toFiniteNumber } from "./liveOrderbookUtils.js";

function isRecord(value) {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

function normalizedKeySet(keys) {
  return new Set(
    (Array.isArray(keys) ? keys : [])
      .map((value) => String(value ?? "").trim())
      .filter(Boolean),
  );
}

function normalizeBucket(bucket) {
  if (!isRecord(bucket)) return null;
  const marketTicker = String(bucket.marketTicker ?? "").trim();
  const bucketLabel = String(bucket.bucketLabel ?? "").trim() || marketTicker || "--";
  return {
    ...bucket,
    marketTicker,
    bucketLabel,
    yesTopLevels: Array.isArray(bucket.yesTopLevels) ? bucket.yesTopLevels.filter(isRecord) : [],
    noTopLevels: Array.isArray(bucket.noTopLevels) ? bucket.noTopLevels.filter(isRecord) : [],
  };
}

export function stationWithSortedBuckets(station) {
  if (!isRecord(station)) return null;
  const bucketsRaw = Array.isArray(station.buckets) ? station.buckets : [];
  const buckets = bucketsRaw
    .map(normalizeBucket)
    .filter(Boolean)
    .sort((a, b) => bucketSortKey(a?.bucketLabel).localeCompare(bucketSortKey(b?.bucketLabel)));
  return {
    ...station,
    stationId: String(station.stationId ?? "").trim() || "UNKNOWN",
    displayName: String(station.displayName ?? "").trim() || String(station.stationId ?? "").trim() || "Unknown",
    targetDateLocal: String(station.targetDateLocal ?? "").trim(),
    zoneId: String(station.zoneId ?? "").trim(),
    buckets,
  };
}

export function compareOpportunityRows(left, right) {
  const leftEv = toFiniteNumber(left?.ev);
  const rightEv = toFiniteNumber(right?.ev);
  const byEv = (rightEv ?? Number.NEGATIVE_INFINITY) - (leftEv ?? Number.NEGATIVE_INFINITY);
  if (byEv !== 0) return byEv;

  const leftWin = toFiniteNumber(left?.modelWinProbability);
  const rightWin = toFiniteNumber(right?.modelWinProbability);
  const byWin = (rightWin ?? Number.NEGATIVE_INFINITY) - (leftWin ?? Number.NEGATIVE_INFINITY);
  if (byWin !== 0) return byWin;

  const leftPrice = toFiniteNumber(left?.entryPriceCents);
  const rightPrice = toFiniteNumber(right?.entryPriceCents);
  const byPrice = (leftPrice ?? Number.POSITIVE_INFINITY) - (rightPrice ?? Number.POSITIVE_INFINITY);
  if (byPrice !== 0) return byPrice;

  const byStation = String(left?.stationId ?? "").localeCompare(String(right?.stationId ?? ""));
  if (byStation !== 0) return byStation;
  const byBucket = String(left?.bucketLabel ?? "").localeCompare(String(right?.bucketLabel ?? ""));
  if (byBucket !== 0) return byBucket;
  return String(left?.side ?? "").localeCompare(String(right?.side ?? ""));
}

export function buildAllOpportunitiesFromStations(stations) {
  if (!Array.isArray(stations) || stations.length === 0) return [];
  const rows = [];
  for (const station of stations) {
    if (!isRecord(station)) continue;
    const stationId = String(station?.stationId ?? "").trim();
    const buckets = Array.isArray(station?.buckets) ? station.buckets : [];
    for (const bucket of buckets) {
      if (!isRecord(bucket)) continue;
      const yesWin = toFiniteNumber(bucket?.yesModelWinProbability);
      const yesEv = toFiniteNumber(bucket?.yesEv);
      const yesEntry = toFiniteNumber(bucket?.yesAskCents);
      if (Number.isFinite(yesWin) && Number.isFinite(yesEv) && Number.isFinite(yesEntry)) {
        rows.push({
          stationId,
          marketTicker: bucket?.marketTicker,
          bucketLabel: bucket?.bucketLabel,
          side: "YES",
          modelWinProbability: yesWin,
          marketPriceProbability: yesEntry / 100,
          entryPriceCents: yesEntry,
          ev: yesEv,
        });
      }

      const noWin = toFiniteNumber(bucket?.noModelWinProbability);
      const noEv = toFiniteNumber(bucket?.noEv);
      const noEntry = toFiniteNumber(bucket?.noAskCents);
      if (Number.isFinite(noWin) && Number.isFinite(noEv) && Number.isFinite(noEntry)) {
        rows.push({
          stationId,
          marketTicker: bucket?.marketTicker,
          bucketLabel: bucket?.bucketLabel,
          side: "NO",
          modelWinProbability: noWin,
          marketPriceProbability: noEntry / 100,
          entryPriceCents: noEntry,
          ev: noEv,
        });
      }
    }
  }
  return rows.sort(compareOpportunityRows);
}

export function hasStationInference(station) {
  if (Number.isFinite(toFiniteNumber(station?.predictionPointTmaxF))) {
    return true;
  }
  const quantiles = station?.predictionQuantiles;
  if (!quantiles || typeof quantiles !== "object") {
    return false;
  }
  return Object.values(quantiles).some((value) => Number.isFinite(toFiniteNumber(value)));
}

export function snapshotMetaFromData(stations, opportunities, fetchedAtMillis = Date.now()) {
  const normalizedStations = Array.isArray(stations) ? stations.filter(isRecord) : [];
  const normalizedOpportunities = Array.isArray(opportunities) ? opportunities : [];
  return {
    fetchedAtMillis,
    stationCount: normalizedStations.length,
    opportunityCount: normalizedOpportunities.length,
    hasInference: normalizedStations.some(hasStationInference),
  };
}

export function shouldRevalidateSnapshotMeta(meta, nowMillis, maxAgeMs) {
  if (!meta) return true;
  if (!Number.isFinite(meta.fetchedAtMillis)) return true;
  if (!Number.isFinite(nowMillis) || !Number.isFinite(maxAgeMs)) return true;
  if (nowMillis - meta.fetchedAtMillis > maxAgeMs) return true;
  if (meta.stationCount <= 0) return true;
  if (!meta.hasInference) return true;
  return false;
}

export function retainRecordKeys(record, keepKeys = []) {
  if (!isRecord(record)) return {};
  const keepSet = normalizedKeySet(keepKeys);
  let changed = false;
  const next = {};

  for (const [key, value] of Object.entries(record)) {
    if (keepSet.has(key)) {
      next[key] = value;
    } else {
      changed = true;
    }
  }

  return changed ? next : record;
}

export function calculateFractionalKellyPositionSize({
  balanceUsd,
  modelWinProbability,
  entryPriceCents,
  ev,
  kellyFraction,
  stakeCapUsd,
} = {}) {
  const balance = toFiniteNumber(balanceUsd);
  const marketPriceCents = toFiniteNumber(entryPriceCents);
  const marketPriceProbability = Number.isFinite(marketPriceCents) ? marketPriceCents / 100 : null;
  const modelWin = toFiniteNumber(modelWinProbability);
  const explicitEdge = toFiniteNumber(ev);
  const edge = Number.isFinite(explicitEdge)
    ? explicitEdge
    : (Number.isFinite(modelWin) && Number.isFinite(marketPriceProbability)
      ? modelWin - marketPriceProbability
      : null);
  const kelly = Math.max(0, toFiniteNumber(kellyFraction) ?? 0);
  const stakeCap = toFiniteNumber(stakeCapUsd);

  let fullKelly = 0;
  if (Number.isFinite(marketPriceProbability) && marketPriceProbability > 0 && marketPriceProbability < 1 && Number.isFinite(edge)) {
    fullKelly = Math.max(0, Math.min(1, edge / (1 - marketPriceProbability)));
  }

  const riskFractionUsed = kelly * fullKelly;
  const uncappedStakeUsd = Number.isFinite(balance) ? Math.max(0, balance) * riskFractionUsed : null;
  const effectiveCapUsd = Number.isFinite(stakeCap) && stakeCap >= 0 ? stakeCap : null;
  const stakeUsd = Number.isFinite(uncappedStakeUsd)
    ? Math.max(0, effectiveCapUsd == null ? uncappedStakeUsd : Math.min(uncappedStakeUsd, effectiveCapUsd))
    : null;

  return {
    balanceUsd: Number.isFinite(balance) ? balance : null,
    marketPriceProbability,
    edge,
    fullKelly,
    riskFractionUsed,
    uncappedStakeUsd,
    stakeUsd,
    isCapped: Number.isFinite(stakeUsd) && Number.isFinite(uncappedStakeUsd) && stakeUsd + 1e-9 < uncappedStakeUsd,
  };
}

export function chooseAvailableTargetDates({
  backendDates = [],
  frameStations = [],
  cachedDates = [],
  fallbackDates = [],
  maxCount = 2,
} = {}) {
  const stationDates = Array.isArray(frameStations)
    ? frameStations.map((station) => String(station?.targetDateLocal ?? "").trim()).filter(Boolean)
    : [];
  const combined = [
    ...(Array.isArray(backendDates) ? backendDates : []),
    ...stationDates,
    ...(Array.isArray(cachedDates) ? cachedDates : []),
    ...(Array.isArray(fallbackDates) ? fallbackDates : []),
  ];
  const deduped = [...new Set(combined.map((value) => String(value ?? "").trim()).filter(Boolean))];
  return deduped.slice(0, Math.max(1, maxCount));
}
