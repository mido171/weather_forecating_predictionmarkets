import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Area,
  CartesianGrid,
  LabelList,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./liveTrading.css";
import {
  buildAllOpportunitiesFromStations,
  calculateFractionalKellyPositionSize,
  chooseAvailableTargetDates,
  retainRecordKeys,
  shouldRevalidateSnapshotMeta,
  snapshotMetaFromData,
  stationWithSortedBuckets,
} from "./liveTradingData.js";
import {
  AGE_THRESHOLDS,
  ageBand,
  ageSeconds,
  classifyDelta,
  depthBarPct,
  formatAgeFromIso,
  formatPriceCents,
  formatQtyExact,
  formatTimestamp,
  getSpreadSeverity,
  signedCents,
  signedQty,
  toFiniteNumber,
  toneFromClassification,
} from "./liveOrderbookUtils";

const DEFAULT_WS_PATH = "/ws/live-orderbooks";
const DEFAULT_SNAPSHOT_PATH = "/api/live-trading/orderbooks/snapshot";
const DEFAULT_INFERENCE_RUN_PATH = "/api/live-trading/inference/run";
const DEFAULT_ACCOUNT_BALANCE_PATH = "/api/live-trading/account/balance";
const DEV_FALLBACK_WS_URL = "ws://localhost:8080/ws/live-orderbooks";
const DEV_FALLBACK_SNAPSHOT_URL = "http://localhost:8080/api/live-trading/orderbooks/snapshot";
const DEV_FALLBACK_INFERENCE_RUN_URL = "http://localhost:8080/api/live-trading/inference/run";
const DEV_FALLBACK_ACCOUNT_BALANCE_URL = "http://localhost:8080/api/live-trading/account/balance";
const RECENT_LIVE_BALANCE_CURVE_PATH = "/data/live_trade_balance_recent_ex_haywire_20260314.json";
const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 10000;
const STRONG_PULSE_MS = 950;
const SOFT_PULSE_MS = 700;
const RECENT_EVENT_PULSE_MS = 1200;
const RECENT_EVENT_RETENTION_MS = 30000;
const HOT_WINDOW_MS = 3000;
const GLOBAL_DELAYED_SECONDS = 2.5;
const TOP_DEPTH_ROWS = 3;
const DEFAULT_OPPORTUNITY_ROWS = 14;
const OPPORTUNITIES_CUTOFF_HOUR = 17;
const OPPORTUNITIES_CUTOFF_MINUTE = 45;
const OPPORTUNITIES_POLL_CONNECTED_MS = 10000;
const OPPORTUNITIES_POLL_DISCONNECTED_MS = 3000;
const INFERENCE_AUTO_REFRESH_INTERVAL_MS = 10000;
const ACCOUNT_BALANCE_POLL_MS = 15000;
const SNAPSHOT_REVALIDATE_MAX_AGE_MS = 15000;
const LIVE_CLOCK_TICK_MS = 1000;
const PCT_FORMATTER = new Intl.NumberFormat(undefined, {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});
const USD_COMPACT_FORMATTER = new Intl.NumberFormat(undefined, {
  minimumFractionDigits: 0,
  maximumFractionDigits: 2,
});
const TEMP_FORMATTER = new Intl.NumberFormat(undefined, {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});
const OPPORTUNITIES_MIN_WIN_PROBABILITY = 0.6;
const DEFAULT_LIVE_CONFIG = {
  referenceLabel: "2024-2025 | Top #3",
  periodLabel: "2024-10-01 -> 2025-12-31",
  stationIds: ["KNYC", "KMIA", "KMDW", "KLAX"],
  minWinProbability: OPPORTUNITIES_MIN_WIN_PROBABILITY,
  minEv: 0.3,
  minSidePriceProbability: 0.25,
  sizingMode: "fractional_kelly",
  kellyFraction: 0.2,
  stakeCapUsd: 700,
  entryRule: "Entry >= max(T-1 12:00Z, open+30m)",
  predictionSource: "live-script replay",
};

function formatProbabilityPct(probability, missing = "--") {
  const value = toFiniteNumber(probability);
  if (!Number.isFinite(value)) return missing;
  return `${PCT_FORMATTER.format(value * 100)}%`;
}

function formatEvCents(ev, missing = "--") {
  const value = toFiniteNumber(ev);
  if (!Number.isFinite(value)) return missing;
  const cents = value * 100;
  const sign = cents > 0 ? "+" : "";
  return `${sign}${PCT_FORMATTER.format(cents)}c`;
}

function formatTempF(value, missing = "--") {
  const numeric = toFiniteNumber(value);
  if (!Number.isFinite(numeric)) return missing;
  return `${TEMP_FORMATTER.format(numeric)}F`;
}

function formatKellyFraction(value, missing = "--") {
  const numeric = toFiniteNumber(value);
  if (!Number.isFinite(numeric)) return missing;
  return numeric.toFixed(2);
}

function formatUsdWhole(value, missing = "--") {
  const numeric = toFiniteNumber(value);
  if (!Number.isFinite(numeric)) return missing;
  return `$${Math.round(numeric).toLocaleString()}`;
}

function formatUsdCompact(value, missing = "--") {
  const numeric = toFiniteNumber(value);
  if (!Number.isFinite(numeric)) return missing;
  return `$${USD_COMPACT_FORMATTER.format(numeric)}`;
}

function formatUsdSigned(value, missing = "--") {
  const numeric = toFiniteNumber(value);
  if (!Number.isFinite(numeric)) return missing;
  const sign = numeric > 0 ? "+" : numeric < 0 ? "-" : "";
  return `${sign}$${USD_COMPACT_FORMATTER.format(Math.abs(numeric))}`;
}

function formatUsdAxis(value, missing = "--") {
  const numeric = toFiniteNumber(value);
  if (!Number.isFinite(numeric)) return missing;
  const sign = numeric < 0 ? "-" : "";
  return `${sign}$${USD_COMPACT_FORMATTER.format(Math.abs(numeric))}`;
}

function formatThresholdDecimal(value, missing = "--") {
  const numeric = toFiniteNumber(value);
  if (!Number.isFinite(numeric)) return missing;
  return numeric.toFixed(2);
}

function normalizeLiveConfig(raw) {
  const fallback = DEFAULT_LIVE_CONFIG;
  const stationIds = Array.isArray(raw?.stationIds) && raw.stationIds.length
    ? raw.stationIds.map((value) => String(value ?? "").trim()).filter(Boolean)
    : fallback.stationIds;
  return {
    referenceLabel: String(raw?.referenceLabel ?? fallback.referenceLabel).trim() || fallback.referenceLabel,
    periodLabel: String(raw?.periodLabel ?? fallback.periodLabel).trim() || fallback.periodLabel,
    stationIds,
    minWinProbability: Number.isFinite(toFiniteNumber(raw?.minWinProbability))
      ? toFiniteNumber(raw?.minWinProbability)
      : fallback.minWinProbability,
    minEv: Number.isFinite(toFiniteNumber(raw?.minEv)) ? toFiniteNumber(raw?.minEv) : fallback.minEv,
    minSidePriceProbability: Number.isFinite(toFiniteNumber(raw?.minSidePriceProbability))
      ? toFiniteNumber(raw?.minSidePriceProbability)
      : fallback.minSidePriceProbability,
    sizingMode: String(raw?.sizingMode ?? fallback.sizingMode).trim() || fallback.sizingMode,
    kellyFraction: Number.isFinite(toFiniteNumber(raw?.kellyFraction))
      ? toFiniteNumber(raw?.kellyFraction)
      : fallback.kellyFraction,
    stakeCapUsd: Number.isFinite(toFiniteNumber(raw?.stakeCapUsd))
      ? toFiniteNumber(raw?.stakeCapUsd)
      : fallback.stakeCapUsd,
    entryRule: String(raw?.entryRule ?? fallback.entryRule).trim() || fallback.entryRule,
    predictionSource: String(raw?.predictionSource ?? fallback.predictionSource).trim() || fallback.predictionSource,
  };
}

function meetsOpportunityFilters(row, config) {
  const modelWin = toFiniteNumber(row?.modelWinProbability);
  const ev = toFiniteNumber(row?.ev);
  const entryPriceProb = toFiniteNumber(row?.entryPriceCents) / 100;
  return Number.isFinite(modelWin)
    && Number.isFinite(ev)
    && Number.isFinite(entryPriceProb)
    && modelWin >= config.minWinProbability
    && ev >= config.minEv
    && entryPriceProb >= config.minSidePriceProbability;
}

function meetsOpportunityWinFloor(row, config) {
  const modelWin = toFiniteNumber(row?.modelWinProbability);
  return Number.isFinite(modelWin) && modelWin >= config.minWinProbability;
}

function quantileFromStation(station, key) {
  return toFiniteNumber(station?.predictionQuantiles?.[key]);
}

function isoLocalDateFromMillis(millis) {
  const date = new Date(millis);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function shiftIsoDate(isoDate, dayDelta) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(isoDate ?? "").trim());
  if (!match) return null;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const date = new Date(year, month - 1, day);
  if (!Number.isFinite(date.getTime())) return null;
  date.setDate(date.getDate() + dayDelta);
  return isoLocalDateFromMillis(date.getTime());
}

function formatTargetDateLabel(isoDate) {
  const text = String(isoDate ?? "").trim();
  const date = new Date(`${text}T00:00:00`);
  if (!Number.isFinite(date.getTime())) return text || "--";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function describeOpportunityDate(isoDate, referenceMillis) {
  const label = formatTargetDateLabel(isoDate);
  const referenceDate = isoLocalDateFromMillis(referenceMillis);
  const tomorrowDate = shiftIsoDate(referenceDate, 1);
  if (isoDate === referenceDate) return `Today · ${label}`;
  if (isoDate === tomorrowDate) return `Tomorrow · ${label}`;
  return label;
}

function shouldDefaultToTomorrow(nowMillis) {
  const now = new Date(nowMillis);
  const hour = now.getHours();
  const minute = now.getMinutes();
  return hour > OPPORTUNITIES_CUTOFF_HOUR
    || (hour === OPPORTUNITIES_CUTOFF_HOUR && minute >= OPPORTUNITIES_CUTOFF_MINUTE);
}

function displayOpportunityDate(isoDate) {
  return formatTargetDateLabel(isoDate);
}

function isAbortError(error) {
  return error instanceof DOMException && error.name === "AbortError";
}

function snapshotUrlForTargetDate(snapshotUrl, targetDateLocal) {
  if (!targetDateLocal) return snapshotUrl;
  const separator = snapshotUrl.includes("?") ? "&" : "?";
  return `${snapshotUrl}${separator}targetDateLocal=${encodeURIComponent(targetDateLocal)}`;
}

function resolveWsUrl() {
  const envUrl = String(import.meta.env.VITE_LIVE_TRADING_WS_URL ?? "").trim();
  if (envUrl) return envUrl;
  if (typeof window === "undefined") return DEV_FALLBACK_WS_URL;

  const backendPort = String(import.meta.env.VITE_LIVE_TRADING_BACKEND_PORT ?? "8080").trim() || "8080";
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const currentPort = String(window.location.port ?? "").trim();
  if (currentPort && currentPort !== backendPort) {
    const host = window.location.hostname || "localhost";
    return `${protocol}//${host}:${backendPort}${DEFAULT_WS_PATH}`;
  }
  return `${protocol}//${window.location.host}${DEFAULT_WS_PATH}`;
}

function resolveSnapshotUrl() {
  const envUrl = String(import.meta.env.VITE_LIVE_TRADING_SNAPSHOT_URL ?? "").trim();
  if (envUrl) return envUrl;
  if (typeof window === "undefined") return DEV_FALLBACK_SNAPSHOT_URL;

  const backendPort = String(import.meta.env.VITE_LIVE_TRADING_BACKEND_PORT ?? "8080").trim() || "8080";
  const currentPort = String(window.location.port ?? "").trim();
  if (currentPort && currentPort !== backendPort) {
    const protocol = window.location.protocol === "https:" ? "https:" : "http:";
    const host = window.location.hostname || "localhost";
    return `${protocol}//${host}:${backendPort}${DEFAULT_SNAPSHOT_PATH}`;
  }
  return DEFAULT_SNAPSHOT_PATH;
}

function resolveInferenceRunUrl() {
  const envUrl = String(import.meta.env.VITE_LIVE_TRADING_INFERENCE_RUN_URL ?? "").trim();
  if (envUrl) return envUrl;
  if (typeof window === "undefined") return DEV_FALLBACK_INFERENCE_RUN_URL;

  const backendPort = String(import.meta.env.VITE_LIVE_TRADING_BACKEND_PORT ?? "8080").trim() || "8080";
  const currentPort = String(window.location.port ?? "").trim();
  if (currentPort && currentPort !== backendPort) {
    const protocol = window.location.protocol === "https:" ? "https:" : "http:";
    const host = window.location.hostname || "localhost";
    return `${protocol}//${host}:${backendPort}${DEFAULT_INFERENCE_RUN_PATH}`;
  }
  return DEFAULT_INFERENCE_RUN_PATH;
}

function resolveAccountBalanceUrl() {
  const envUrl = String(import.meta.env.VITE_LIVE_TRADING_ACCOUNT_BALANCE_URL ?? "").trim();
  if (envUrl) return envUrl;
  if (typeof window === "undefined") return DEV_FALLBACK_ACCOUNT_BALANCE_URL;

  const backendPort = String(import.meta.env.VITE_LIVE_TRADING_BACKEND_PORT ?? "8080").trim() || "8080";
  const currentPort = String(window.location.port ?? "").trim();
  if (currentPort && currentPort !== backendPort) {
    const protocol = window.location.protocol === "https:" ? "https:" : "http:";
    const host = window.location.hostname || "localhost";
    return `${protocol}//${host}:${backendPort}${DEFAULT_ACCOUNT_BALANCE_PATH}`;
  }
  return DEFAULT_ACCOUNT_BALANCE_PATH;
}

function resolveBalanceUsd(payload) {
  const balanceDollars = toFiniteNumber(payload?.balanceDollars);
  if (Number.isFinite(balanceDollars)) return balanceDollars;

  const balanceCents = toFiniteNumber(payload?.balanceCents);
  if (Number.isFinite(balanceCents)) return balanceCents / 100;

  const portfolioValueDollars = toFiniteNumber(payload?.portfolioValueDollars);
  if (Number.isFinite(portfolioValueDollars)) return portfolioValueDollars;

  const portfolioValueCents = toFiniteNumber(payload?.portfolioValueCents);
  if (Number.isFinite(portfolioValueCents)) return portfolioValueCents / 100;

  return null;
}

function levelToModel(level) {
  if (!level) return null;
  const price = toFiniteNumber(level.priceCents ?? level.price);
  const qty = toFiniteNumber(level.quantity ?? level.qty);
  if (!Number.isFinite(price) || !Number.isFinite(qty)) return null;
  return { price, qty };
}

function normalizeLevels(levels) {
  if (!Array.isArray(levels)) return [];
  return levels.map(levelToModel).filter(Boolean);
}

function firstLevelQty(levels) {
  const first = normalizeLevels(levels)[0];
  return first ? first.qty : null;
}

function complementDepthLevels(levels) {
  const normalized = normalizeLevels(levels);
  return normalized.map((level) => ({
    price: Math.max(0, Math.min(100, 100 - level.price)),
    qty: level.qty,
  }));
}

function describeNumericChange(label, prev, next, upWord, downWord, formatter = signedCents) {
  if (!Number.isFinite(prev) || !Number.isFinite(next)) return `${label} changed`;
  if (prev === next) return `${label} unchanged`;
  const directionWord = next > prev ? upWord : downWord;
  return `${label} ${directionWord} ${formatter(prev, next)}`;
}

function compareScalarField({
  prevValue,
  nextValue,
  deltaField,
  markerField,
  priority,
  labelBuilder,
  markers,
  events,
  defaultTone = "neutral",
}) {
  if (prevValue == null && nextValue == null) return;
  if (prevValue != null && nextValue != null && prevValue === nextValue) return;

  if (prevValue == null || nextValue == null) {
    const action = nextValue == null ? "removed" : "appeared";
    const suffix = nextValue == null ? "" : ` ${formatPriceCents(nextValue, "")}`;
    markers[markerField] = {
      tone: defaultTone,
      strength: "strong",
    };
    events.push({
      priority,
      tone: defaultTone,
      label: `${labelBuilder("", "", true).replace(/\s+/g, " ").trim()} ${action}${suffix}`,
    });
    return;
  }

  const classification = classifyDelta(deltaField, prevValue, nextValue);
  const tone = toneFromClassification(classification) || defaultTone;
  markers[markerField] = {
    tone,
    strength: "strong",
  };
  events.push({
    priority,
    tone,
    label: labelBuilder(prevValue, nextValue, false),
  });
}
function evaluateBucketDelta(prevBucket, nextBucket) {
  const markers = {};
  const events = [];

  const prevYesBid = toFiniteNumber(prevBucket?.yesBidCents);
  const nextYesBid = toFiniteNumber(nextBucket?.yesBidCents);
  compareScalarField({
    prevValue: prevYesBid,
    nextValue: nextYesBid,
    deltaField: "bestBid",
    markerField: "yes.bid.price",
    priority: 10,
    markers,
    events,
    labelBuilder: (prev, next, isMissingTransition) =>
      isMissingTransition
        ? "YES bid"
        : describeNumericChange("YES bid", prev, next, "up", "down"),
  });

  const prevYesAsk = toFiniteNumber(prevBucket?.yesAskCents);
  const nextYesAsk = toFiniteNumber(nextBucket?.yesAskCents);
  compareScalarField({
    prevValue: prevYesAsk,
    nextValue: nextYesAsk,
    deltaField: "bestAsk",
    markerField: "yes.ask.price",
    priority: 11,
    markers,
    events,
    labelBuilder: (prev, next, isMissingTransition) =>
      isMissingTransition
        ? "YES ask"
        : describeNumericChange("YES ask", prev, next, "up", "down"),
  });

  const prevYesSpread = toFiniteNumber(prevBucket?.yesSpreadCents);
  const nextYesSpread = toFiniteNumber(nextBucket?.yesSpreadCents);
  compareScalarField({
    prevValue: prevYesSpread,
    nextValue: nextYesSpread,
    deltaField: "spread",
    markerField: "yes.spread",
    priority: 20,
    markers,
    events,
    defaultTone: "warn",
    labelBuilder: (prev, next, isMissingTransition) => {
      if (isMissingTransition) return "YES spread";
      return describeNumericChange("YES spread", prev, next, "wider", "tighter");
    },
  });

  const prevNoBid = toFiniteNumber(prevBucket?.noBidCents);
  const nextNoBid = toFiniteNumber(nextBucket?.noBidCents);
  compareScalarField({
    prevValue: prevNoBid,
    nextValue: nextNoBid,
    deltaField: "bestBid",
    markerField: "no.bid.price",
    priority: 12,
    markers,
    events,
    labelBuilder: (prev, next, isMissingTransition) =>
      isMissingTransition
        ? "NO bid"
        : describeNumericChange("NO bid", prev, next, "up", "down"),
  });

  const prevNoAsk = toFiniteNumber(prevBucket?.noAskCents);
  const nextNoAsk = toFiniteNumber(nextBucket?.noAskCents);
  compareScalarField({
    prevValue: prevNoAsk,
    nextValue: nextNoAsk,
    deltaField: "bestAsk",
    markerField: "no.ask.price",
    priority: 13,
    markers,
    events,
    labelBuilder: (prev, next, isMissingTransition) =>
      isMissingTransition
        ? "NO ask"
        : describeNumericChange("NO ask", prev, next, "up", "down"),
  });

  const prevNoSpread = toFiniteNumber(prevBucket?.noSpreadCents);
  const nextNoSpread = toFiniteNumber(nextBucket?.noSpreadCents);
  compareScalarField({
    prevValue: prevNoSpread,
    nextValue: nextNoSpread,
    deltaField: "spread",
    markerField: "no.spread",
    priority: 21,
    markers,
    events,
    defaultTone: "warn",
    labelBuilder: (prev, next, isMissingTransition) => {
      if (isMissingTransition) return "NO spread";
      return describeNumericChange("NO spread", prev, next, "wider", "tighter");
    },
  });

  const prevMid = toFiniteNumber(prevBucket?.midYesCents);
  const nextMid = toFiniteNumber(nextBucket?.midYesCents);
  compareScalarField({
    prevValue: prevMid,
    nextValue: nextMid,
    deltaField: "mid",
    markerField: "mid",
    priority: 30,
    markers,
    events,
    labelBuilder: (prev, next, isMissingTransition) =>
      isMissingTransition
        ? "Mid"
        : describeNumericChange("Mid", prev, next, "up", "down"),
  });

  const prevYesBidSize = firstLevelQty(prevBucket?.yesTopLevels);
  const nextYesBidSize = firstLevelQty(nextBucket?.yesTopLevels);
  if (prevYesBid != null && nextYesBid != null && prevYesBid === nextYesBid && prevYesBidSize != null && nextYesBidSize != null && prevYesBidSize !== nextYesBidSize) {
    const classification = classifyDelta("bidSize", prevYesBidSize, nextYesBidSize);
    const tone = toneFromClassification(classification);
    markers["yes.bid.size"] = { tone, strength: "soft" };
    events.push({
      priority: 60,
      tone,
      label: describeNumericChange("YES bid size", prevYesBidSize, nextYesBidSize, "up", "down", signedQty),
    });
  }

  const prevNoBidSize = firstLevelQty(prevBucket?.noTopLevels);
  const nextNoBidSize = firstLevelQty(nextBucket?.noTopLevels);
  if (prevNoBid != null && nextNoBid != null && prevNoBid === nextNoBid && prevNoBidSize != null && nextNoBidSize != null && prevNoBidSize !== nextNoBidSize) {
    const classification = classifyDelta("bidSize", prevNoBidSize, nextNoBidSize);
    const tone = toneFromClassification(classification);
    markers["no.bid.size"] = { tone, strength: "soft" };
    events.push({
      priority: 61,
      tone,
      label: describeNumericChange("NO bid size", prevNoBidSize, nextNoBidSize, "up", "down", signedQty),
    });
  }

  const prevYesAskSize = firstLevelQty(prevBucket?.noTopLevels);
  const nextYesAskSize = firstLevelQty(nextBucket?.noTopLevels);
  if (prevYesAsk != null && nextYesAsk != null && prevYesAsk === nextYesAsk && prevYesAskSize != null && nextYesAskSize != null && prevYesAskSize !== nextYesAskSize) {
    const classification = classifyDelta("askSize", prevYesAskSize, nextYesAskSize);
    const tone = toneFromClassification(classification);
    markers["yes.ask.size"] = { tone, strength: "soft" };
    events.push({
      priority: 62,
      tone,
      label: describeNumericChange("YES ask size", prevYesAskSize, nextYesAskSize, "up", "down", signedQty),
    });
  }

  const prevNoAskSize = firstLevelQty(prevBucket?.yesTopLevels);
  const nextNoAskSize = firstLevelQty(nextBucket?.yesTopLevels);
  if (prevNoAsk != null && nextNoAsk != null && prevNoAsk === nextNoAsk && prevNoAskSize != null && nextNoAskSize != null && prevNoAskSize !== nextNoAskSize) {
    const classification = classifyDelta("askSize", prevNoAskSize, nextNoAskSize);
    const tone = toneFromClassification(classification);
    markers["no.ask.size"] = { tone, strength: "soft" };
    events.push({
      priority: 63,
      tone,
      label: describeNumericChange("NO ask size", prevNoAskSize, nextNoAskSize, "up", "down", signedQty),
    });
  }

  const depthSpecs = [
    { sideLabel: "YES", markerPrefix: "yes.depth", previous: normalizeLevels(prevBucket?.yesTopLevels), next: normalizeLevels(nextBucket?.yesTopLevels), priorityBase: 80 },
    { sideLabel: "NO", markerPrefix: "no.depth", previous: normalizeLevels(prevBucket?.noTopLevels), next: normalizeLevels(nextBucket?.noTopLevels), priorityBase: 90 },
  ];

  for (const spec of depthSpecs) {
    for (let idx = 0; idx <= TOP_DEPTH_ROWS; idx += 1) {
      const previousLevel = spec.previous[idx] ?? null;
      const nextLevel = spec.next[idx] ?? null;
      if (!previousLevel && !nextLevel) continue;
      const markerField = `${spec.markerPrefix}.${idx}`;

      if (!previousLevel || !nextLevel) {
        markers[markerField] = {
          tone: "neutral",
          strength: "strong",
        };
        events.push({
          priority: spec.priorityBase + idx,
          tone: "neutral",
          label: `${spec.sideLabel} depth shifted`,
        });
        continue;
      }

      if (previousLevel.price !== nextLevel.price) {
        const classification = classifyDelta("bestBid", previousLevel.price, nextLevel.price);
        const tone = toneFromClassification(classification);
        markers[markerField] = { tone, strength: "strong" };
        events.push({
          priority: spec.priorityBase + idx,
          tone,
          label: `${spec.sideLabel} depth shifted ${signedCents(previousLevel.price, nextLevel.price)}`,
        });
        continue;
      }

      if (previousLevel.qty !== nextLevel.qty) {
        const classification = classifyDelta("bidSize", previousLevel.qty, nextLevel.qty);
        const tone = toneFromClassification(classification);
        markers[markerField] = { tone, strength: "soft" };
        events.push({
          priority: spec.priorityBase + idx,
          tone,
          label: `${spec.sideLabel} depth size ${nextLevel.qty > previousLevel.qty ? "up" : "down"} ${signedQty(previousLevel.qty, nextLevel.qty)}`,
        });
      }
    }
  }

  if (!events.length && !Object.keys(markers).length) {
    return null;
  }

  const hasStrong = Object.values(markers).some((marker) => marker.strength === "strong");
  const hasSoft = Object.values(markers).some((marker) => marker.strength === "soft");
  const sortedEvents = [...events].sort((a, b) => a.priority - b.priority);
  const event = sortedEvents[0] ?? null;

  return {
    markers,
    event: event
      ? {
          label: event.label,
          tone: event.tone,
          kind: "quote",
          emphasis: hasStrong ? "strong" : hasSoft ? "soft" : "neutral",
        }
      : null,
  };
}

function buildFrameDiff(previousFrame, nextFrame) {
  if (!previousFrame || !nextFrame) {
    return { markers: {}, events: {} };
  }

  const previousByTicker = new Map();
  const previousStations = Array.isArray(previousFrame?.stations) ? previousFrame.stations : [];
  for (const station of previousStations) {
    const buckets = Array.isArray(station?.buckets) ? station.buckets : [];
    for (const bucket of buckets) {
      if (bucket?.marketTicker) {
        previousByTicker.set(bucket.marketTicker, bucket);
      }
    }
  }

  const markers = {};
  const events = {};
  const nextStations = Array.isArray(nextFrame?.stations) ? nextFrame.stations : [];

  for (const station of nextStations) {
    const buckets = Array.isArray(station?.buckets) ? station.buckets : [];
    for (const bucket of buckets) {
      const ticker = bucket?.marketTicker;
      if (!ticker) continue;
      const previousBucket = previousByTicker.get(ticker);
      if (!previousBucket) continue;

      const delta = evaluateBucketDelta(previousBucket, bucket);
      if (!delta) continue;

      for (const [field, marker] of Object.entries(delta.markers)) {
        markers[`${ticker}:${field}`] = marker;
      }
      if (delta.event) {
        events[ticker] = delta.event;
      }
    }
  }

  return { markers, events };
}

function markerClass(markers, ticker, field, nowMillis) {
  const marker = markers[`${ticker}:${field}`];
  if (!marker) return "";
  if (marker.expiresAt <= nowMillis) return "";
  return ["lt-delta", `lt-delta-${marker.strength}`, `lt-delta-${marker.tone}`].join(" ");
}

function deriveGlobalConnectionState(wsStatus, frame, nowMillis) {
  const frameAge = ageSeconds(frame?.asOfUtc, nowMillis);
  if (wsStatus === "error") return "ERROR";
  if (wsStatus === "connecting" && !frame) return "CONNECTING";
  if (wsStatus === "reconnecting") return "RECONNECTING";
  if (Number.isFinite(frameAge) && frameAge > GLOBAL_DELAYED_SECONDS) return "DELAYED";
  if (wsStatus === "live") return "LIVE";
  if (!frame) return "CONNECTING";
  return "DELAYED";
}

function recentEventModel(recentEvents, ticker, nowMillis) {
  const event = recentEvents[ticker];
  if (!event) {
    return {
      label: "No recent move",
      tone: "neutral",
      kind: "none",
      emphasis: "neutral",
      pulse: false,
      hot: false,
    };
  }
  const ageMillis = nowMillis - event.atMillis;
  if (ageMillis > RECENT_EVENT_RETENTION_MS) {
    return {
      label: "No recent move",
      tone: "neutral",
      kind: "none",
      emphasis: "neutral",
      pulse: false,
      hot: false,
    };
  }
  return {
    ...event,
    pulse: event.pulseUntil > nowMillis,
    hot: ageMillis < HOT_WINDOW_MS,
  };
}

function bucketState(bucket, nowMillis, recentEvents) {
  const ticker = bucket.marketTicker;
  const yesLevels = normalizeLevels(bucket.yesTopLevels);
  const noLevels = normalizeLevels(bucket.noTopLevels);
  const ageValue = ageSeconds(bucket.bookAsOfUtc, nowMillis);
  const band = ageBand(ageValue, AGE_THRESHOLDS);

  const yesSeverity = getSpreadSeverity(bucket.yesSpreadCents);
  const noSeverity = getSpreadSeverity(bucket.noSpreadCents);
  const wide = yesSeverity.level === "wide" || yesSeverity.level === "danger" || noSeverity.level === "wide" || noSeverity.level === "danger";

  const hasYesQuote = bucket.yesBidCents != null || bucket.yesAskCents != null;
  const hasNoQuote = bucket.noBidCents != null || bucket.noAskCents != null;
  const empty = !hasYesQuote && !hasNoQuote && yesLevels.length === 0 && noLevels.length === 0;
  const stale = band === "stale" || band === "frozen";

  const statusText = String(bucket.marketStatus ?? "").toLowerCase();
  const error = statusText.includes("error");
  const recentEvent = recentEventModel(recentEvents, ticker, nowMillis);
  const statusLabel = error
    ? "ERROR"
    : empty
      ? "EMPTY"
      : stale
        ? "STALE"
        : wide
          ? "WIDE"
          : recentEvent.hot
            ? "HOT"
            : "LIVE";

  const bestYesBidSize = yesLevels[0]?.qty ?? null;
  const bestNoBidSize = noLevels[0]?.qty ?? null;
  const bestYesAskSize = noLevels[0]?.qty ?? null;
  const bestNoAskSize = yesLevels[0]?.qty ?? null;

  return {
    ticker,
    yesLevels,
    noLevels,
    ageBand: band,
    yesSeverity,
    noSeverity,
    wide,
    empty,
    stale,
    error,
    recentEvent,
    statusLabel,
    bestYesBidSize,
    bestYesAskSize,
    bestNoBidSize,
    bestNoAskSize,
  };
}
const LiveTopBar = memo(function LiveTopBar({
  connectionState,
  frame,
  nowMillis,
  freshnessSummary,
  liveConfig,
}) {
  const stationLabel = Array.isArray(liveConfig?.stationIds) && liveConfig.stationIds.length
    ? liveConfig.stationIds.join(" / ")
    : "KNYC / KMIA / KMDW / KLAX";
  return (
    <header className="lt-topBar">
      <div className="lt-topBarTitleWrap">
        <h1 className="lt-topBarTitle">Live Orderbook Monitor</h1>
        <p className="lt-topBarSub">{stationLabel} · {liveConfig?.referenceLabel ?? DEFAULT_LIVE_CONFIG.referenceLabel}</p>
      </div>
      <div className="lt-topBarMeta">
        <span className={`lt-stateChip state-${connectionState.toLowerCase()}`}>{connectionState}</span>
        <span className="lt-metaChip">Fresh {freshnessSummary.freshCount}/{freshnessSummary.total}</span>
        <span className="lt-metaChip">
          Last {formatTimestamp(frame?.asOfUtc)} - {formatAgeFromIso(frame?.asOfUtc, nowMillis)} old
        </span>
      </div>
    </header>
  );
});

const LiveStrategyBar = memo(function LiveStrategyBar({ liveConfig }) {
  const config = normalizeLiveConfig(liveConfig);
  const chips = [
    `Reference backtest: ${config.referenceLabel}`,
    `Period: ${config.periodLabel}`,
    `Stations: ${config.stationIds.join(" + ")}`,
    `EV >= ${formatThresholdDecimal(config.minEv)}`,
    `Win >= ${formatThresholdDecimal(config.minWinProbability)}`,
    `Side price >= ${Math.round(config.minSidePriceProbability * 100)}c`,
    config.sizingMode === "fractional_kelly"
      ? `Fractional Kelly ${formatKellyFraction(config.kellyFraction)}`
      : `Sizing: ${config.sizingMode}`,
    `Stake cap ${formatUsdWhole(config.stakeCapUsd)}`,
    config.entryRule,
    `Prediction source: ${config.predictionSource}`,
  ];
  return (
    <section className="lt-strategyBar">
      <div className="lt-strategyHeader">
        <h2 className="lt-strategyTitle">Active Live Config</h2>
      </div>
      <div className="lt-strategyChips">
        {chips.map((chip) => (
          <span key={chip} className="lt-strategyChip">{chip}</span>
        ))}
      </div>
    </section>
  );
});

const OpportunitiesPanel = memo(function OpportunitiesPanel({
  liveConfig,
  accountBalanceUsd,
  accountBalanceStatus,
  filteredRowsByDate,
  allRowsByDate,
  showAllRows,
  onToggleShowAllRows,
  dateOptions,
  selectedDate,
  onSelectDate,
  isLoading,
  isAutoInferenceActive,
  onToggleAutoInference,
  autoInferenceStatus,
  nowMillis,
}) {
  const config = normalizeLiveConfig(liveConfig);
  const dates = Array.isArray(dateOptions) ? dateOptions : [];
  const modeLabel = showAllRows
    ? `All opportunities with Win >= ${formatThresholdDecimal(config.minWinProbability)} sorted by EV for each target date`
    : `Eligible by live config: EV >= ${formatThresholdDecimal(config.minEv)}, Win >= ${formatThresholdDecimal(config.minWinProbability)}, Side >= ${Math.round(config.minSidePriceProbability * 100)}c · sorted by EV (top ${DEFAULT_OPPORTUNITY_ROWS})`;
  const toggleLabel = showAllRows ? "Show Eligible Only" : "Show All by EV";
  const activeDateLabel = selectedDate ? displayOpportunityDate(selectedDate) : "--";
  const hasBalance = Number.isFinite(toFiniteNumber(accountBalanceUsd));
  const balanceLabel = hasBalance
    ? formatUsdCompact(accountBalanceUsd)
    : accountBalanceStatus === "error"
    ? "unavailable"
    : "loading...";
  const balanceChipClassName = [
    "lt-opportunitiesBalanceChip",
    hasBalance ? "is-ready" : accountBalanceStatus === "error" ? "is-error" : "is-loading",
  ].join(" ");
  const sections = dates.map((date) => {
    const filtered = Array.isArray(filteredRowsByDate?.[date]) ? filteredRowsByDate[date] : [];
    const all = Array.isArray(allRowsByDate?.[date]) ? allRowsByDate[date] : [];
    const modeRows = showAllRows ? all : filtered;
    const rows = showAllRows ? modeRows : modeRows.slice(0, DEFAULT_OPPORTUNITY_ROWS);
    return {
      date,
      title: displayOpportunityDate(date),
      filteredCount: filtered.length,
      totalCount: all.length,
      modeCount: modeRows.length,
      rows,
    };
  });

  return (
    <section className="lt-opportunitiesPanel">
      <div className="lt-opportunitiesHeader">
        <h2 className="lt-opportunitiesTitle">Top Opportunities</h2>
        <div className="lt-opportunitiesActions">
          {dates.length ? (
            <div className="lt-opportunitiesDateSwitch">
              {dates.map((date) => (
                <button
                  key={date}
                  type="button"
                  className={["lt-opportunitiesDateBtn", date === selectedDate ? "is-active" : ""].join(" ")}
                  onClick={() => onSelectDate(date)}
                >
                  {displayOpportunityDate(date)}
                </button>
              ))}
            </div>
          ) : null}
          <span className="lt-opportunitiesMeta">{modeLabel}</span>
          <span className="lt-opportunitiesCount">Active refresh target: {activeDateLabel}</span>
          <span className={balanceChipClassName}>Balance: {balanceLabel}</span>
          <button
            type="button"
            className={["lt-opportunitiesAutoBtn", isAutoInferenceActive ? "is-active" : ""].join(" ")}
            onClick={onToggleAutoInference}
          >
            {isAutoInferenceActive ? "Stop Auto Refresh" : "Auto Refresh Inference (10s)"}
          </button>
          <button type="button" className="lt-opportunitiesToggle" onClick={onToggleShowAllRows}>
            {toggleLabel}
          </button>
        </div>
      </div>

      {autoInferenceStatus ? <div className="lt-opportunitiesAutoStatus">{autoInferenceStatus}</div> : null}

      <div className="lt-opportunitiesGrid">
        {sections.map((section) => (
          <article
            key={section.date}
            className={["lt-opportunitySection", section.date === selectedDate ? "is-active" : ""].join(" ")}
          >
            <button
              type="button"
              className="lt-opportunitySectionHeader"
              onClick={() => onSelectDate(section.date)}
            >
              <div>
                <h3 className="lt-opportunitySectionTitle">{section.title}</h3>
                <p className="lt-opportunitySectionMeta">
                  Showing {section.rows.length} / {section.modeCount}
                  {showAllRows
                    ? ` · total ${section.totalCount}`
                    : ` · eligible: ${section.filteredCount}`}
                </p>
              </div>
              {section.date === selectedDate ? <span className="lt-opportunitySectionBadge">Active</span> : null}
            </button>

            {section.rows.length === 0 ? (
              <div className="lt-opportunitiesEmpty">
                {isLoading
                  ? `Loading opportunities for ${section.title}...`
                  : showAllRows
                  ? "No opportunities available from current station frame."
                  : "No opportunities currently meet the active live config."}
              </div>
            ) : (
              <div className="lt-opportunitiesTableWrap">
                <table className="lt-opportunitiesTable">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Station</th>
                      <th>Bucket</th>
                      <th>Side</th>
                      <th>Model win</th>
                      <th>Entry</th>
                      <th>EV</th>
                      <th className="lt-opportunitySizeCol">
                        <div className="lt-opportunitySizeHeadValue">Size</div>
                        <div className="lt-opportunitySizeHeadMeta">@ {balanceLabel}</div>
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {section.rows.map((row, idx) => {
                      const positionSize = calculateFractionalKellyPositionSize({
                        balanceUsd: accountBalanceUsd,
                        modelWinProbability: row.modelWinProbability,
                        entryPriceCents: row.entryPriceCents,
                        ev: row.ev,
                        kellyFraction: config.kellyFraction,
                        stakeCapUsd: config.stakeCapUsd,
                      });
                      const sizeMeta = `${formatProbabilityPct(positionSize.riskFractionUsed)} Kelly${positionSize.isCapped ? " · cap" : ""}`;

                      return (
                        <tr key={`${section.date}-${row.marketTicker}-${row.side}-${idx}`}>
                          <td>{idx + 1}</td>
                          <td>{row.stationId}</td>
                          <td>
                            <div className="lt-opportunityBucket">{row.bucketLabel}</div>
                            <div className="lt-opportunityTicker">{row.marketTicker}</div>
                          </td>
                          <td>
                            <span className={["lt-opportunitySide", row.side === "YES" ? "yes" : "no"].join(" ")}>
                              {row.side}
                            </span>
                          </td>
                          <td>{formatProbabilityPct(row.modelWinProbability)}</td>
                          <td>{formatPriceCents(row.entryPriceCents)}</td>
                          <td className={["lt-opportunityEv", toFiniteNumber(row.ev) >= 0 ? "positive" : "negative"].join(" ")}>
                            {formatEvCents(row.ev)}
                          </td>
                          <td className="lt-opportunitySizeCell">
                            <div className="lt-opportunitySizeValue">{formatUsdCompact(positionSize.stakeUsd)}</div>
                            <div className={["lt-opportunitySizeMeta", positionSize.isCapped ? "is-capped" : ""].join(" ")}>
                              {sizeMeta}
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </article>
        ))}
      </div>
    </section>
  );
});

const StationHeader = memo(function StationHeader({ station, summary }) {
  const q10 = quantileFromStation(station, "q_0.10");
  const q50 = quantileFromStation(station, "q_0.50");
  const q90 = quantileFromStation(station, "q_0.90");
  const hasInference = Number.isFinite(toFiniteNumber(station?.predictionPointTmaxF));

  return (
    <header className="lt-stationHeader">
      <div className="lt-stationTitleRow">
        <h2 className="lt-stationCode">{station.stationId}</h2>
        <p className="lt-stationName">{station.displayName}</p>
      </div>
      <div className="lt-stationContext">
        <span className="lt-stationContextItem">Trade date {station.targetDateLocal ?? "--"}</span>
        <span className="lt-stationContextItem">Zone {station.zoneId ?? "--"}</span>
      </div>
      <div className="lt-stationKpis">
        <span className="lt-stationKpi">
          <strong>{summary.active}/{summary.total}</strong> Active
        </span>
        <span className="lt-stationKpi">
          <strong>{summary.wide}</strong> Wide
        </span>
        <span className="lt-stationKpi">
          <strong>{summary.stale}</strong> Stale
        </span>
      </div>
      <div className="lt-stationInference">
        {hasInference ? (
          <>
            <span className="lt-stationInferenceChip">
              Mean <strong>{formatTempF(station.predictionPointTmaxF)}</strong>
            </span>
            <span className="lt-stationInferenceChip">
              Q10 <strong>{formatTempF(q10)}</strong>
            </span>
            <span className="lt-stationInferenceChip">
              Q50 <strong>{formatTempF(q50)}</strong>
            </span>
            <span className="lt-stationInferenceChip">
              Q90 <strong>{formatTempF(q90)}</strong>
            </span>
          </>
        ) : (
          <span className="lt-stationInferenceChip muted">Inference not loaded</span>
        )}
      </div>
    </header>
  );
});

function sideTitleClass(side) {
  return side === "yes" ? "side-yes" : "side-no";
}

function statusChipClass(status) {
  switch (status) {
    case "LIVE":
      return "chip-live";
    case "HOT":
      return "chip-hot";
    case "WIDE":
      return "chip-wide";
    case "STALE":
      return "chip-stale";
    case "EMPTY":
      return "chip-empty";
    case "ERROR":
      return "chip-error";
    default:
      return "";
  }
}

function eventToneClass(tone) {
  switch (tone) {
    case "buy":
      return "tone-buy";
    case "sell":
      return "tone-sell";
    case "warn":
      return "tone-warn";
    default:
      return "tone-neutral";
  }
}

const QuoteTile = memo(function QuoteTile({
  label,
  priceText,
  sizeText,
  subLabel,
  marker,
}) {
  const textValue = String(priceText ?? "");
  const isTextValue = textValue.includes(" ") || textValue.length > 6;
  return (
    <article className={["lt-quoteTile", marker].filter(Boolean).join(" ")}>
      <div className="lt-quoteLabel">{label}</div>
      <div className={["lt-quoteValue", isTextValue ? "is-text" : ""].join(" ")}>{priceText}</div>
      <div className="lt-quoteSub">{sizeText}</div>
      {subLabel ? <div className="lt-quoteHint">{subLabel}</div> : null}
    </article>
  );
});

const DepthLadder = memo(function DepthLadder({
  side,
  bidLevels,
  askLevels,
  expanded,
  onToggleExpand,
  ticker,
  markerLookup,
}) {
  const oppositeSide = side === "yes" ? "no" : "yes";
  const bidQueue = normalizeLevels(bidLevels).slice(1);
  const askQueue = normalizeLevels(askLevels).slice(1);

  const bidHasMore = bidQueue.length > TOP_DEPTH_ROWS;
  const askHasMore = askQueue.length > TOP_DEPTH_ROWS;
  const canExpand = bidHasMore || askHasMore;
  const visibleCount = expanded ? Math.max(bidQueue.length, askQueue.length) : TOP_DEPTH_ROWS;
  const renderCount = Math.max(TOP_DEPTH_ROWS, visibleCount);
  const visibleBid = bidQueue.slice(0, visibleCount);
  const visibleAsk = askQueue.slice(0, visibleCount);
  const bidMaxQty = visibleBid.reduce((max, level) => Math.max(max, level.qty), 0);
  const askMaxQty = visibleAsk.reduce((max, level) => Math.max(max, level.qty), 0);
  const hiddenRows = Math.max(bidQueue.length, askQueue.length) - TOP_DEPTH_ROWS;

  const renderDepthRow = (level, index, lane) => {
    const isPlaceholder = !level;
    const markerField = lane === "bid"
      ? `${side}.depth.${index + 1}`
      : `${oppositeSide}.depth.${index + 1}`;
    const marker = markerLookup(markerField);
    const maxQty = lane === "bid" ? bidMaxQty : askMaxQty;
    const width = level ? depthBarPct(level.qty, maxQty) : 0;

    return (
      <div
        key={`${ticker}-${side}-${lane}-depth-${index}`}
        className={[
          "lt-depthRow",
          marker,
          sideTitleClass(side),
          lane === "ask" ? "is-ask" : "",
          isPlaceholder ? "is-placeholder" : "",
        ].filter(Boolean).join(" ")}
      >
        <div className="lt-depthBar" style={{ width: `${width}%` }} />
        <div className="lt-depthText">
          <span className="lt-depthPrice">{level ? formatPriceCents(level.price, "--") : "--"}</span>
          <span className="lt-depthQty">{level ? formatQtyExact(level.qty) : "--"}</span>
        </div>
      </div>
    );
  };

  return (
    <div className="lt-depthLadder">
      <div className="lt-depthColumns">
        <div className="lt-depthColumn">
          <div className="lt-depthColumnTitle">Bid queue (after best)</div>
          <div className="lt-depthRows">
            {Array.from({ length: renderCount }).map((_, index) => renderDepthRow(visibleBid[index] ?? null, index, "bid"))}
          </div>
        </div>

        <div className="lt-depthColumn">
          <div className="lt-depthColumnTitle">Ask queue (after best)</div>
          <div className="lt-depthRows">
            {Array.from({ length: renderCount }).map((_, index) => renderDepthRow(visibleAsk[index] ?? null, index, "ask"))}
          </div>
        </div>
      </div>

      {canExpand ? (
        <button type="button" className="lt-depthToggle" onClick={onToggleExpand}>
          {expanded ? "Show top 3" : `+${hiddenRows} more levels`}
        </button>
      ) : null}
    </div>
  );
});

const SidePanel = memo(function SidePanel({
  side,
  ticker,
  bidCents,
  askCents,
  spreadCents,
  bidSize,
  askSize,
  spreadSeverity,
  bidLevels,
  askLevels,
  expanded,
  onToggleExpand,
  markerLookup,
}) {
  const sideLabel = side.toUpperCase();
  const panelClass = ["lt-sidePanel", sideTitleClass(side)].join(" ");

  const bidText = bidCents == null ? "No bid" : formatPriceCents(bidCents);
  const askText = askCents == null ? "No ask" : formatPriceCents(askCents);
  const spreadText = spreadCents == null ? "Spread unavailable" : formatPriceCents(spreadCents);

  return (
    <section className={panelClass}>
      <div className="lt-sideHeader">
        <span className="lt-sideDot" aria-hidden="true" />
        <span className="lt-sideName">{sideLabel}</span>
      </div>
      <div className="lt-quoteRow">
        <QuoteTile
          label="Bid"
          priceText={bidText}
          sizeText={bidCents == null ? "--" : formatQtyExact(bidSize)}
          marker={[markerLookup(`${side}.bid.price`), markerLookup(`${side}.bid.size`)].filter(Boolean).join(" ")}
        />
        <QuoteTile
          label="Ask"
          priceText={askText}
          sizeText={askCents == null ? "--" : formatQtyExact(askSize)}
          marker={[markerLookup(`${side}.ask.price`), markerLookup(`${side}.ask.size`)].filter(Boolean).join(" ")}
        />
        <QuoteTile
          label="Spread"
          priceText={spreadText}
          sizeText={spreadSeverity.label}
          marker={markerLookup(`${side}.spread`)}
        />
      </div>
      <DepthLadder
        side={side}
        bidLevels={bidLevels}
        askLevels={askLevels}
        expanded={expanded}
        onToggleExpand={onToggleExpand}
        ticker={ticker}
        markerLookup={markerLookup}
      />
    </section>
  );
});
const BucketCard = memo(function BucketCard({
  bucket,
  state,
  nowMillis,
  markers,
  expanded,
  onToggleExpand,
}) {
  const cardClass = [
    "lt-bucketCard",
    state.stale ? "is-stale" : "",
    state.empty ? "is-empty" : "",
  ].filter(Boolean).join(" ");

  const markerLookup = useCallback((field) => markerClass(markers, bucket.marketTicker, field, nowMillis), [
    markers,
    bucket.marketTicker,
    nowMillis,
  ]);

  const eventClass = [
    "lt-eventChip",
    eventToneClass(state.recentEvent.tone),
    state.recentEvent.pulse ? "is-pulse" : "",
    state.recentEvent.kind === "none" ? "is-muted" : "",
  ].join(" ");
  const hasModelMetrics =
    Number.isFinite(toFiniteNumber(bucket.yesModelWinProbability)) ||
    Number.isFinite(toFiniteNumber(bucket.noModelWinProbability));

  return (
    <article className={cardClass} data-ticker={bucket.marketTicker}>
      <header className="lt-bucketHeader">
        <div>
          <div className="lt-bucketName">{bucket.bucketLabel}</div>
          <div className="lt-bucketTicker">{bucket.marketTicker}</div>
        </div>
        <div className="lt-statusChips">
          <span className={["lt-statusChip", statusChipClass(state.statusLabel)].filter(Boolean).join(" ")}>
            {state.statusLabel}
          </span>
        </div>
      </header>

      <div className={eventClass}>
        {state.recentEvent.label}
      </div>

      <div className="lt-modelLine">
        {hasModelMetrics ? (
          <>
            <span className="lt-modelChip yes">
              YES {formatProbabilityPct(bucket.yesModelWinProbability)} - EV {formatEvCents(bucket.yesEv)}
            </span>
            <span className="lt-modelChip no">
              NO {formatProbabilityPct(bucket.noModelWinProbability)} - EV {formatEvCents(bucket.noEv)}
            </span>
          </>
        ) : (
          <span className="lt-modelChip muted">Model probabilities unavailable</span>
        )}
      </div>

      <div className="lt-sidesGrid">
        <SidePanel
          side="yes"
          ticker={bucket.marketTicker}
          bidCents={bucket.yesBidCents}
          askCents={bucket.yesAskCents}
          spreadCents={bucket.yesSpreadCents}
          bidSize={state.bestYesBidSize}
          askSize={state.bestYesAskSize}
          spreadSeverity={state.yesSeverity}
          bidLevels={state.yesLevels}
          askLevels={complementDepthLevels(state.noLevels)}
          expanded={expanded.yes}
          onToggleExpand={() => onToggleExpand(bucket.marketTicker, "yes")}
          markerLookup={markerLookup}
        />

        <SidePanel
          side="no"
          ticker={bucket.marketTicker}
          bidCents={bucket.noBidCents}
          askCents={bucket.noAskCents}
          spreadCents={bucket.noSpreadCents}
          bidSize={state.bestNoBidSize}
          askSize={state.bestNoAskSize}
          spreadSeverity={state.noSeverity}
          bidLevels={state.noLevels}
          askLevels={complementDepthLevels(state.yesLevels)}
          expanded={expanded.no}
          onToggleExpand={() => onToggleExpand(bucket.marketTicker, "no")}
          markerLookup={markerLookup}
        />
      </div>

      <footer className="lt-bucketFooter">
        <span className={["lt-footerChip", markerLookup("mid")].filter(Boolean).join(" ")}>
          Mid {bucket.midYesCents == null ? "unavailable" : formatPriceCents(bucket.midYesCents)}
        </span>
        <span className={["lt-footerChip", `age-${state.ageBand}`].filter(Boolean).join(" ")}>
          Age {formatAgeFromIso(bucket.bookAsOfUtc, nowMillis)}
        </span>
      </footer>
    </article>
  );
});

const StationColumn = memo(function StationColumn({
  station,
  nowMillis,
  markers,
  recentEvents,
  expandedByTicker,
  onToggleExpand,
}) {
  const bucketStates = useMemo(() => {
    const byTicker = {};
    const buckets = Array.isArray(station?.buckets) ? station.buckets : [];
    for (const bucket of buckets) {
      if (!bucket?.marketTicker) continue;
      byTicker[bucket.marketTicker] = bucketState(bucket, nowMillis, recentEvents);
    }
    return byTicker;
  }, [station?.buckets, nowMillis, recentEvents]);

  const summary = useMemo(() => {
    const buckets = Array.isArray(station?.buckets) ? station.buckets : [];
    let active = 0;
    let wide = 0;
    let stale = 0;
    let empty = 0;

    for (const bucket of buckets) {
      const state = bucketStates[bucket.marketTicker];
      if (!state) continue;
      if (!state.empty && !state.stale && !state.error) active += 1;
      if (state.wide) wide += 1;
      if (state.stale) stale += 1;
      if (state.empty) empty += 1;
    }

    return {
      total: buckets.length,
      active,
      wide,
      stale,
      empty,
    };
  }, [station?.buckets, bucketStates]);

  return (
    <section className="lt-stationColumn" data-station={station.stationId}>
      <div className="lt-stationScroll">
        <StationHeader station={station} summary={summary} />

        <div className="lt-bucketList">
          {station.buckets.map((bucket) => {
            const state = bucketStates[bucket.marketTicker];
            const expanded = expandedByTicker[bucket.marketTicker] ?? { yes: false, no: false };
            return (
              <BucketCard
                key={bucket.marketTicker}
                bucket={bucket}
                state={state}
                nowMillis={nowMillis}
                markers={markers}
                expanded={expanded}
                onToggleExpand={onToggleExpand}
              />
            );
          })}
        </div>
      </div>
    </section>
  );
});

const RecentLiveBalanceTooltip = memo(function RecentLiveBalanceTooltip({
  active,
  payload,
  hasEstimatedBalance,
}) {
  const point = Array.isArray(payload) && payload.length > 0 ? payload[0]?.payload : null;
  if (!active || !point) return null;
  return (
    <div className="lt-recentBalanceTooltip">
      <div className="lt-recentBalanceTooltipDate">{formatTargetDateLabel(point.targetDateLocal)}</div>
      <div className="lt-recentBalanceTooltipRow">
        <span>Balance</span>
        <strong>{formatUsdAxis(point.balanceUsd)}</strong>
      </div>
      <div className="lt-recentBalanceTooltipRow">
        <span>Day P&amp;L</span>
        <strong className={point.dailyPnlUsd >= 0 ? "positive" : "negative"}>
          {formatUsdSigned(point.dailyPnlUsd)}
        </strong>
      </div>
      <div className="lt-recentBalanceTooltipRow">
        <span>Cumulative</span>
        <strong className={point.cumulativePnlUsd >= 0 ? "positive" : "negative"}>
          {formatUsdSigned(point.cumulativePnlUsd)}
        </strong>
      </div>
      <div className="lt-recentBalanceTooltipRow">
        <span>Markets</span>
        <strong>{point.marketCount}</strong>
      </div>
      <div className="lt-recentBalanceTooltipMode">
        {hasEstimatedBalance ? "Estimated absolute balance" : "Normalized from $0 start"}
      </div>
    </div>
  );
});

function RecentLiveBalanceValueLabel(props) {
  const {
    x,
    y,
    value,
    index,
  } = props ?? {};
  const numericValue = toFiniteNumber(value);
  if (!Number.isFinite(numericValue) || !Number.isFinite(x) || !Number.isFinite(y)) {
    return null;
  }

  const labelY = Number(y) + (Number(index) % 2 === 0 ? -14 : 20);
  return (
    <text
      x={x}
      y={labelY}
      textAnchor="middle"
      fontSize="11"
      fontWeight="800"
      fill="#e9f7ff"
      stroke="#0f1b2d"
      strokeWidth="3"
      paintOrder="stroke"
      className="lt-recentBalancePointLabel"
    >
      {formatUsdAxis(numericValue)}
    </text>
  );
}

const RecentLiveBalancePanel = memo(function RecentLiveBalancePanel({
  report,
  accountBalanceUsd,
}) {
  const model = useMemo(() => {
    const daily = Array.isArray(report?.daily) ? report.daily : [];
    if (!daily.length) return null;

    const summary = report?.summary ?? {};
    const finalCumulativePnlUsd = Number.isFinite(toFiniteNumber(summary?.netPnlUsd))
      ? toFiniteNumber(summary?.netPnlUsd)
      : toFiniteNumber(daily[daily.length - 1]?.cumulativePnlUsd);
    const currentBalanceUsd = toFiniteNumber(accountBalanceUsd);
    const hasEstimatedBalance = Number.isFinite(currentBalanceUsd) && Number.isFinite(finalCumulativePnlUsd);
    const estimatedStartBalanceUsd = hasEstimatedBalance
      ? currentBalanceUsd - finalCumulativePnlUsd
      : 0;

    const series = daily.map((row) => {
      const cumulativePnlUsd = toFiniteNumber(row?.cumulativePnlUsd);
      return {
        targetDateLocal: String(row?.targetDateLocal ?? "").trim(),
        dailyPnlUsd: toFiniteNumber(row?.dailyPnlUsd),
        cumulativePnlUsd,
        marketCount: Number.isFinite(toFiniteNumber(row?.marketCount)) ? toFiniteNumber(row?.marketCount) : 0,
        balanceUsd: estimatedStartBalanceUsd + cumulativePnlUsd,
      };
    }).filter((row) => row.targetDateLocal);

    if (!series.length) return null;

    const endBalanceUsd = series[series.length - 1].balanceUsd;
    const peakBalanceUsd = series.reduce(
      (best, row) => (Number.isFinite(row.balanceUsd) ? Math.max(best, row.balanceUsd) : best),
      Number.NEGATIVE_INFINITY,
    );
    const troughBalanceUsd = series.reduce(
      (best, row) => (Number.isFinite(row.balanceUsd) ? Math.min(best, row.balanceUsd) : best),
      Number.POSITIVE_INFINITY,
    );

    return {
      series,
      hasEstimatedBalance,
      estimatedStartBalanceUsd,
      endBalanceUsd,
      peakBalanceUsd,
      troughBalanceUsd,
      periodStart: String(report?.periodStart ?? "").trim(),
      periodEnd: String(report?.periodEnd ?? "").trim(),
      excludedTradeLabel: String(report?.excludedTrade?.label ?? report?.excludedTrade?.marketTicker ?? "").trim(),
      tradeCount: Number.isFinite(toFiniteNumber(summary?.tradeCount)) ? toFiniteNumber(summary?.tradeCount) : series.length,
      winRate: toFiniteNumber(summary?.winRate),
      netPnlUsd: finalCumulativePnlUsd,
    };
  }, [accountBalanceUsd, report]);

  if (!model) return null;

  const baselineValue = model.hasEstimatedBalance ? model.estimatedStartBalanceUsd : 0;
  const baselineLabel = model.hasEstimatedBalance ? "Est. start" : "Zero base";
  const subtitleParts = [];
  if (model.periodStart && model.periodEnd) {
    subtitleParts.push(`${model.periodStart} -> ${model.periodEnd}`);
  }
  if (model.excludedTradeLabel) {
    subtitleParts.push(`Excludes ${model.excludedTradeLabel}`);
  }

  return (
    <section className="lt-recentBalancePanel">
      <div className="lt-recentBalanceHead">
        <div className="lt-recentBalanceTitleWrap">
          <h2 className="lt-recentBalanceTitle">Recent Live Balance Evolution</h2>
          <p className="lt-recentBalanceSub">{subtitleParts.join(" · ")}</p>
        </div>
        <div className="lt-recentBalanceChips">
          <span className="lt-strategyChip">{baselineLabel}: {formatUsdCompact(baselineValue)}</span>
          <span className="lt-strategyChip">End: {formatUsdCompact(model.endBalanceUsd)}</span>
          <span className="lt-strategyChip">Net: {formatUsdSigned(model.netPnlUsd)}</span>
          <span className="lt-strategyChip">Trades: {model.tradeCount}</span>
          <span className="lt-strategyChip">Win: {formatProbabilityPct(model.winRate)}</span>
        </div>
      </div>

      <div className="lt-recentBalanceNote">
        {model.hasEstimatedBalance
          ? "Curve is anchored to the current account balance and backsolves an estimated start-of-window balance from realized P&L in this sample."
          : "Current account balance is unavailable, so the curve is shown as a normalized realized P&L path starting from $0."}
      </div>

      <div className="lt-recentBalanceChartWrap">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={model.series} margin={{ top: 18, right: 18, left: 8, bottom: 6 }}>
            <defs>
              <linearGradient id="ltRecentBalanceFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#2fd4bc" stopOpacity={0.32} />
                <stop offset="100%" stopColor="#2fd4bc" stopOpacity={0.04} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#294062" />
            <XAxis
              dataKey="targetDateLocal"
              stroke="#9aa9da"
              tick={{ fontSize: 12 }}
              tickFormatter={formatTargetDateLabel}
            />
            <YAxis
              stroke="#9aa9da"
              tick={{ fontSize: 12 }}
              width={88}
              tickFormatter={(value) => formatUsdAxis(value, "$0")}
            />
            <Tooltip content={<RecentLiveBalanceTooltip hasEstimatedBalance={model.hasEstimatedBalance} />} />
            <ReferenceLine y={baselineValue} stroke="#5a759f" strokeDasharray="4 4" />
            <Area type="monotone" dataKey="balanceUsd" stroke="none" fill="url(#ltRecentBalanceFill)" />
            <Line
              type="monotone"
              dataKey="balanceUsd"
              stroke="#2fd4bc"
              strokeWidth={2.6}
              dot={{ r: 3, fill: "#2fd4bc", strokeWidth: 0 }}
              activeDot={{ r: 5, fill: "#dffef7", stroke: "#0f1b2d", strokeWidth: 2 }}
            >
              <LabelList dataKey="balanceUsd" content={<RecentLiveBalanceValueLabel />} />
            </Line>
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="lt-recentBalanceFooter">
        <span>Peak: <strong>{formatUsdAxis(model.peakBalanceUsd)}</strong></span>
        <span>Trough: <strong>{formatUsdAxis(model.troughBalanceUsd)}</strong></span>
        <span>Window result: <strong className={model.netPnlUsd >= 0 ? "positive" : "negative"}>{formatUsdSigned(model.netPnlUsd)}</strong></span>
      </div>
    </section>
  );
});

export default function LiveTradingPage() {
  const [frame, setFrame] = useState(null);
  const [wsStatus, setWsStatus] = useState("connecting");
  const [wsError, setWsError] = useState("");
  const [showAllOpportunities, setShowAllOpportunities] = useState(false);
  const [selectedOpportunitiesDate, setSelectedOpportunitiesDate] = useState("");
  const [opportunityRowsByDate, setOpportunityRowsByDate] = useState({});
  const [stationSnapshotsByDate, setStationSnapshotsByDate] = useState({});
  const [snapshotMetaByDate, setSnapshotMetaByDate] = useState({});
  const [knownTargetDates, setKnownTargetDates] = useState([]);
  const [opportunityRowsLoading, setOpportunityRowsLoading] = useState(false);
  const [isAutoInferenceActive, setIsAutoInferenceActive] = useState(false);
  const [autoInferenceStatus, setAutoInferenceStatus] = useState("");
  const [accountBalanceUsd, setAccountBalanceUsd] = useState(null);
  const [accountBalanceStatus, setAccountBalanceStatus] = useState("loading");
  const [recentBalanceReport, setRecentBalanceReport] = useState(null);
  const [nowMillis, setNowMillis] = useState(() => Date.now());
  const [markers, setMarkers] = useState({});
  const [recentEvents, setRecentEvents] = useState({});
  const [expandedByTicker, setExpandedByTicker] = useState({});

  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const rafRef = useRef(null);
  const pendingFrameRef = useRef(null);
  const closedByUnmountRef = useRef(false);
  const previousFrameRef = useRef(null);
  const hasReceivedFrameRef = useRef(false);
  const hasLoadedOpportunityDatesRef = useRef(false);
  const autoInferenceInFlightRef = useRef(false);
  const wsUrl = useMemo(() => resolveWsUrl(), []);
  const snapshotUrl = useMemo(() => resolveSnapshotUrl(), []);
  const inferenceRunUrl = useMemo(() => resolveInferenceRunUrl(), []);
  const accountBalanceUrl = useMemo(() => resolveAccountBalanceUrl(), []);

  const fetchSnapshotForDate = useCallback(async (targetDateLocal, options = {}) => {
    if (!targetDateLocal) return null;
    const url = snapshotUrlForTargetDate(snapshotUrl, targetDateLocal);
    const response = await fetch(url, {
      cache: "no-store",
      signal: options.signal,
    });
    if (!response.ok) {
      throw new Error(`Failed snapshot for ${targetDateLocal}: HTTP ${response.status}`);
    }
    const payload = await response.json();
    const payloadStations = Array.isArray(payload?.stations)
      ? payload.stations.map(stationWithSortedBuckets).filter(Boolean)
      : [];
    return {
      stations: payloadStations,
      opportunities: buildAllOpportunitiesFromStations(payloadStations),
      availableTargetDates: Array.isArray(payload?.availableTargetDates)
        ? payload.availableTargetDates.map((value) => String(value ?? "").trim()).filter(Boolean)
        : [],
    };
  }, [snapshotUrl]);

  const storeSnapshotForDate = useCallback((targetDateLocal, snapshot, fetchedAtMillis = Date.now()) => {
    const rows = Array.isArray(snapshot?.opportunities) ? snapshot.opportunities : [];
    const stations = Array.isArray(snapshot?.stations) ? snapshot.stations : [];
    setOpportunityRowsByDate((current) => ({
      ...current,
      [targetDateLocal]: rows,
    }));
    setStationSnapshotsByDate((current) => ({
      ...current,
      [targetDateLocal]: stations,
    }));
    setSnapshotMetaByDate((current) => ({
      ...current,
      [targetDateLocal]: snapshotMetaFromData(stations, rows, fetchedAtMillis),
    }));
    if (Array.isArray(snapshot?.availableTargetDates) && snapshot.availableTargetDates.length > 0) {
      setKnownTargetDates((current) => chooseAvailableTargetDates({
        backendDates: [...current, ...snapshot.availableTargetDates],
        cachedDates: [targetDateLocal],
      }));
    }
  }, []);

  const invokeInferenceForDate = useCallback(async (targetDateLocal, options = {}) => {
    if (!targetDateLocal) {
      throw new Error("target date is required");
    }
    const url = snapshotUrlForTargetDate(inferenceRunUrl, targetDateLocal);
    const response = await fetch(url, {
      method: "POST",
      cache: "no-store",
      signal: options.signal,
    });
    if (!response.ok) {
      throw new Error(`Inference invoke failed for ${targetDateLocal}: HTTP ${response.status}`);
    }
    return response.json();
  }, [inferenceRunUrl]);

  const fetchAccountBalance = useCallback(async (options = {}) => {
    const response = await fetch(accountBalanceUrl, {
      cache: "no-store",
      signal: options.signal,
    });
    if (!response.ok) {
      throw new Error(`Failed account balance fetch: HTTP ${response.status}`);
    }
    const payload = await response.json();
    return resolveBalanceUsd(payload);
  }, [accountBalanceUrl]);

  useEffect(() => {
    const timer = setInterval(() => setNowMillis(Date.now()), LIVE_CLOCK_TICK_MS);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    let stopped = false;
    let intervalId = null;
    let inFlight = false;
    let activeController = null;

    const refreshAccountBalance = async () => {
      if (stopped || inFlight) return;
      inFlight = true;
      const controller = new AbortController();
      activeController = controller;
      try {
        const nextBalance = await fetchAccountBalance({ signal: controller.signal });
        if (stopped) return;
        if (!Number.isFinite(nextBalance)) {
          setAccountBalanceStatus("error");
          return;
        }
        setAccountBalanceUsd(nextBalance);
        setAccountBalanceStatus("ready");
      } catch (error) {
        if (!isAbortError(error)) {
          // Keep the last known balance if the account endpoint is temporarily unavailable.
          setAccountBalanceStatus("error");
        }
      } finally {
        if (activeController === controller) {
          activeController = null;
        }
        inFlight = false;
      }
    };

    refreshAccountBalance();
    intervalId = setInterval(refreshAccountBalance, ACCOUNT_BALANCE_POLL_MS);

    return () => {
      stopped = true;
      if (intervalId) clearInterval(intervalId);
      if (activeController) {
        activeController.abort();
        activeController = null;
      }
    };
  }, [fetchAccountBalance]);

  useEffect(() => {
    const controller = new AbortController();

    fetch(RECENT_LIVE_BALANCE_CURVE_PATH, {
      cache: "no-store",
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed recent live balance fetch: HTTP ${response.status}`);
        }
        return response.json();
      })
      .then((payload) => {
        setRecentBalanceReport(payload);
      })
      .catch((error) => {
        if (!isAbortError(error)) {
          setRecentBalanceReport(null);
        }
      });

    return () => {
      controller.abort();
    };
  }, []);

  const queueFrame = useCallback((payload) => {
    pendingFrameRef.current = payload;
    if (rafRef.current) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      const next = pendingFrameRef.current;
      pendingFrameRef.current = null;
      if (next) {
        setFrame(next);
      }
    });
  }, []);

  useEffect(() => {
    closedByUnmountRef.current = false;
    let reconnectAttempts = 0;

    const clearReconnect = () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };

    const closeSocket = () => {
      if (!wsRef.current) return;
      try {
        wsRef.current.close();
      } catch (_) {
        // ignore close errors
      }
      wsRef.current = null;
    };

    const scheduleReconnect = () => {
      if (closedByUnmountRef.current) return;
      clearReconnect();
      reconnectAttempts += 1;
      const delay = Math.min(RECONNECT_MAX_MS, RECONNECT_BASE_MS * 2 ** Math.max(0, reconnectAttempts - 1));
      setWsStatus("reconnecting");
      reconnectTimerRef.current = setTimeout(() => {
        reconnectTimerRef.current = null;
        connect();
      }, delay);
    };

    const connect = () => {
      if (closedByUnmountRef.current) return;
      clearReconnect();
      closeSocket();
      setWsStatus(hasReceivedFrameRef.current ? "reconnecting" : "connecting");

      let ws;
      try {
        ws = new WebSocket(wsUrl);
      } catch (error) {
        setWsError(error instanceof Error ? error.message : String(error));
        setWsStatus("error");
        scheduleReconnect();
        return;
      }

      wsRef.current = ws;

      ws.onopen = () => {
        if (closedByUnmountRef.current) return;
        reconnectAttempts = 0;
        setWsError("");
        setWsStatus("live");
      };

      ws.onmessage = (event) => {
        if (closedByUnmountRef.current) return;
        try {
          const parsed = JSON.parse(String(event.data ?? ""));
          queueFrame(parsed);
        } catch (error) {
          setWsStatus("error");
          setWsError(error instanceof Error ? error.message : "Failed to parse websocket payload");
        }
      };

      ws.onerror = () => {
        if (closedByUnmountRef.current) return;
        setWsStatus("error");
      };

      ws.onclose = () => {
        if (closedByUnmountRef.current) return;
        scheduleReconnect();
      };
    };

    connect();

    return () => {
      closedByUnmountRef.current = true;
      clearReconnect();
      closeSocket();
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [queueFrame, wsUrl]);

  useEffect(() => {
    let stopped = false;
    let intervalId = null;
    let inFlight = false;
    let activeController = null;

    const pullSnapshot = async () => {
      if (stopped || inFlight) return;
      inFlight = true;
      const controller = new AbortController();
      activeController = controller;
      try {
        const response = await fetch(snapshotUrl, {
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok) return;
        const payload = await response.json();
        if (payload && Array.isArray(payload.stations) && payload.stations.length > 0) {
          queueFrame(payload);
        }
      } catch (error) {
        if (!isAbortError(error)) {
        // Snapshot fallback is best-effort when websocket is unavailable.
        }
      } finally {
        if (activeController === controller) {
          activeController = null;
        }
        inFlight = false;
      }
    };

    pullSnapshot();
    intervalId = setInterval(pullSnapshot, wsStatus === "live" ? 10000 : 1500);

    return () => {
      stopped = true;
      if (intervalId) clearInterval(intervalId);
      if (activeController) {
        activeController.abort();
        activeController = null;
      }
    };
  }, [snapshotUrl, wsStatus, queueFrame]);

  useEffect(() => {
    if (!frame) return;
    hasReceivedFrameRef.current = true;
    const now = Date.now();
    const delta = buildFrameDiff(previousFrameRef.current, frame);
    previousFrameRef.current = frame;

    if (Object.keys(delta.markers).length) {
      setMarkers((current) => {
        const next = {};
        for (const [key, value] of Object.entries(current)) {
          if (value.expiresAt > now) {
            next[key] = value;
          }
        }
        for (const [key, marker] of Object.entries(delta.markers)) {
          next[key] = {
            ...marker,
            expiresAt: now + (marker.strength === "strong" ? STRONG_PULSE_MS : SOFT_PULSE_MS),
          };
        }
        return next;
      });
    } else {
      setMarkers((current) => {
        const next = {};
        for (const [key, value] of Object.entries(current)) {
          if (value.expiresAt > now) next[key] = value;
        }
        return next;
      });
    }

    if (Object.keys(delta.events).length) {
      setRecentEvents((current) => {
        const next = {};
        for (const [ticker, event] of Object.entries(current)) {
          if (now - event.atMillis <= RECENT_EVENT_RETENTION_MS) {
            next[ticker] = event;
          }
        }
        for (const [ticker, event] of Object.entries(delta.events)) {
          next[ticker] = {
            ...event,
            atMillis: now,
            pulseUntil: now + RECENT_EVENT_PULSE_MS,
          };
        }
        return next;
      });
    } else {
      setRecentEvents((current) => {
        const next = {};
        for (const [ticker, event] of Object.entries(current)) {
          if (now - event.atMillis <= RECENT_EVENT_RETENTION_MS) {
            next[ticker] = event;
          }
        }
        return next;
      });
    }
  }, [frame]);

  const frameStations = useMemo(() => {
    const rawStations = Array.isArray(frame?.stations) ? frame.stations : [];
    return rawStations.map(stationWithSortedBuckets).filter(Boolean);
  }, [frame]);
  const liveConfig = useMemo(() => normalizeLiveConfig({
    ...DEFAULT_LIVE_CONFIG,
    ...(frame?.config ?? {}),
    minWinProbability: OPPORTUNITIES_MIN_WIN_PROBABILITY,
  }), [frame]);

  const liveFrameOpportunities = useMemo(() => buildAllOpportunitiesFromStations(frameStations), [frameStations]);
  const opportunitiesMinuteBucket = Math.floor(nowMillis / 60000);

  const opportunitiesDateOptions = useMemo(() => {
    const clockMillis = opportunitiesMinuteBucket * 60000;
    const localToday = isoLocalDateFromMillis(clockMillis);
    const localTomorrow = shiftIsoDate(localToday, 1);
    const preferredDate = shouldDefaultToTomorrow(clockMillis) ? localTomorrow : localToday;
    const alternateDate = preferredDate === localToday ? localTomorrow : localToday;
    return chooseAvailableTargetDates({
      backendDates: frame?.availableTargetDates,
      frameStations,
      cachedDates: [
        ...Object.keys(opportunityRowsByDate),
        ...Object.keys(stationSnapshotsByDate),
        ...knownTargetDates,
      ],
      fallbackDates: [preferredDate, alternateDate],
    });
  }, [frame?.availableTargetDates, frameStations, knownTargetDates, opportunitiesMinuteBucket, opportunityRowsByDate, stationSnapshotsByDate]);

  useEffect(() => {
    if (!opportunitiesDateOptions.length) return;
    if (!selectedOpportunitiesDate || !opportunitiesDateOptions.includes(selectedOpportunitiesDate)) {
      setSelectedOpportunitiesDate(opportunitiesDateOptions[0]);
    }
  }, [opportunitiesDateOptions, selectedOpportunitiesDate]);

  useEffect(() => {
    const frameTargetDate = frameStations.find((station) => station?.targetDateLocal)?.targetDateLocal;
    if (!frameTargetDate) return;
    storeSnapshotForDate(frameTargetDate, {
      stations: frameStations,
      opportunities: liveFrameOpportunities,
      availableTargetDates: frame?.availableTargetDates,
    }, Date.now());
  }, [frame?.availableTargetDates, frameStations, liveFrameOpportunities, storeSnapshotForDate]);

  const retainedTargetDateKeys = useMemo(() => {
    const keys = new Set();
    for (const date of opportunitiesDateOptions) {
      const value = String(date ?? "").trim();
      if (value) keys.add(value);
    }
    const selected = String(selectedOpportunitiesDate ?? "").trim();
    if (selected) keys.add(selected);
    for (const station of frameStations) {
      const value = String(station?.targetDateLocal ?? "").trim();
      if (value) keys.add(value);
    }
    return [...keys];
  }, [frameStations, opportunitiesDateOptions, selectedOpportunitiesDate]);

  useEffect(() => {
    setOpportunityRowsByDate((current) => retainRecordKeys(current, retainedTargetDateKeys));
    setStationSnapshotsByDate((current) => retainRecordKeys(current, retainedTargetDateKeys));
    setSnapshotMetaByDate((current) => retainRecordKeys(current, retainedTargetDateKeys));
  }, [retainedTargetDateKeys]);

  useEffect(() => {
    let stopped = false;
    let intervalId = null;
    let inFlight = false;
    let activeController = null;
    const dates = opportunitiesDateOptions;
    if (!dates.length) return undefined;

    const fetchOpportunitiesForDates = async () => {
      if (stopped || inFlight) return;
      inFlight = true;
      const controller = new AbortController();
      activeController = controller;
      if (!hasLoadedOpportunityDatesRef.current) {
        setOpportunityRowsLoading(true);
      }
      try {
        const fetched = await Promise.all(dates.map(async (targetDateLocal) => ([
          targetDateLocal,
          await fetchSnapshotForDate(targetDateLocal, { signal: controller.signal }),
        ])));
        if (stopped) return;
        const fetchedAtMillis = Date.now();
        for (const [date, snapshot] of fetched) {
          storeSnapshotForDate(date, snapshot, fetchedAtMillis);
        }
      } catch (error) {
        if (!isAbortError(error)) {
        // Keep previous opportunities if one snapshot pull fails.
        }
      } finally {
        if (activeController === controller) {
          activeController = null;
        }
        inFlight = false;
        if (!stopped) {
          hasLoadedOpportunityDatesRef.current = true;
          setOpportunityRowsLoading(false);
        }
      }
    };

    fetchOpportunitiesForDates();
    intervalId = setInterval(
      fetchOpportunitiesForDates,
      wsStatus === "live" ? OPPORTUNITIES_POLL_CONNECTED_MS : OPPORTUNITIES_POLL_DISCONNECTED_MS,
    );

    return () => {
      stopped = true;
      if (intervalId) clearInterval(intervalId);
      if (activeController) {
        activeController.abort();
        activeController = null;
      }
    };
  }, [fetchSnapshotForDate, opportunitiesDateOptions, storeSnapshotForDate, wsStatus]);

  const thresholdRowsByDate = useMemo(() => {
    const next = {};
    for (const date of opportunitiesDateOptions) {
      const rows = Array.isArray(opportunityRowsByDate[date]) ? opportunityRowsByDate[date] : [];
      next[date] = rows.filter((row) => meetsOpportunityFilters(row, liveConfig));
    }
    return next;
  }, [liveConfig, opportunitiesDateOptions, opportunityRowsByDate]);

  const winFloorRowsByDate = useMemo(() => {
    const next = {};
    for (const date of opportunitiesDateOptions) {
      const rows = Array.isArray(opportunityRowsByDate[date]) ? opportunityRowsByDate[date] : [];
      next[date] = rows.filter((row) => meetsOpportunityWinFloor(row, liveConfig));
    }
    return next;
  }, [liveConfig, opportunitiesDateOptions, opportunityRowsByDate]);

  const selectedDateEligibleOpportunities = useMemo(() => {
    if (selectedOpportunitiesDate) {
      const selected = thresholdRowsByDate[selectedOpportunitiesDate];
      return Array.isArray(selected) ? selected : [];
    }
    return liveFrameOpportunities.filter((row) => meetsOpportunityFilters(row, liveConfig));
  }, [liveConfig, liveFrameOpportunities, selectedOpportunitiesDate, thresholdRowsByDate]);

  useEffect(() => {
    if (!isAutoInferenceActive || !selectedOpportunitiesDate) {
      return undefined;
    }
    if (selectedDateEligibleOpportunities.length > 0) {
      setAutoInferenceStatus(
        `Target ${formatTargetDateLabel(selectedOpportunitiesDate)} populated. Auto refresh stopped.`,
      );
      setIsAutoInferenceActive(false);
      return undefined;
    }

    let stopped = false;
    let intervalId = null;
    let activeController = null;

    const runInferenceTick = async () => {
      if (stopped || autoInferenceInFlightRef.current) return;
      autoInferenceInFlightRef.current = true;
      const controller = new AbortController();
      activeController = controller;
      const dateLabel = formatTargetDateLabel(selectedOpportunitiesDate);
      setAutoInferenceStatus(`Invoking live inference for ${dateLabel}...`);
      try {
        const runResult = await invokeInferenceForDate(selectedOpportunitiesDate, { signal: controller.signal });
        const refreshedSnapshot = await fetchSnapshotForDate(selectedOpportunitiesDate, { signal: controller.signal });
        if (stopped) return;
        const refreshedRows = Array.isArray(refreshedSnapshot?.opportunities) ? refreshedSnapshot.opportunities : [];
        storeSnapshotForDate(selectedOpportunitiesDate, refreshedSnapshot, Date.now());
        if (refreshedRows.some((row) => meetsOpportunityFilters(row, liveConfig))) {
          setAutoInferenceStatus(`Target ${dateLabel} populated. Auto refresh stopped.`);
          setIsAutoInferenceActive(false);
          return;
        }
        const status = String(runResult?.status ?? "").toLowerCase();
        if (status === "busy") {
          setAutoInferenceStatus(`Inference runner busy for ${dateLabel}. Retrying in 10s.`);
        } else if (status === "success") {
          setAutoInferenceStatus(`Inference refreshed for ${dateLabel}. Waiting for opportunities...`);
        } else if (status === "disabled") {
          setAutoInferenceStatus("Inference invoke is disabled in backend configuration.");
          setIsAutoInferenceActive(false);
        } else {
          const message = String(runResult?.message ?? "").trim();
          setAutoInferenceStatus(
            message
              ? `Inference invoke status: ${status || "unknown"} (${message})`
              : `Inference invoke status: ${status || "unknown"}`,
          );
        }
      } catch (error) {
        if (stopped) return;
        if (isAbortError(error)) return;
        const message = error instanceof Error ? error.message : String(error);
        setAutoInferenceStatus(`Inference invoke failed: ${message}`);
      } finally {
        if (activeController === controller) {
          activeController = null;
        }
        autoInferenceInFlightRef.current = false;
      }
    };

    runInferenceTick();
    intervalId = setInterval(runInferenceTick, INFERENCE_AUTO_REFRESH_INTERVAL_MS);

    return () => {
      stopped = true;
      if (activeController) {
        activeController.abort();
        activeController = null;
      }
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [
    fetchSnapshotForDate,
    liveConfig,
    invokeInferenceForDate,
    isAutoInferenceActive,
    selectedDateEligibleOpportunities.length,
    selectedOpportunitiesDate,
    storeSnapshotForDate,
  ]);

  const displayedStations = useMemo(() => {
    if (selectedOpportunitiesDate) {
      const selected = stationSnapshotsByDate[selectedOpportunitiesDate];
      if (Array.isArray(selected) && selected.length > 0) {
        return selected.filter(Boolean);
      }
    }
    return frameStations.filter(Boolean);
  }, [frameStations, selectedOpportunitiesDate, stationSnapshotsByDate]);

  const displayedTickers = useMemo(() => {
    const keys = new Set();
    for (const station of displayedStations) {
      const buckets = Array.isArray(station?.buckets) ? station.buckets : [];
      for (const bucket of buckets) {
        const ticker = String(bucket?.marketTicker ?? "").trim();
        if (ticker) keys.add(ticker);
      }
    }
    return [...keys];
  }, [displayedStations]);

  useEffect(() => {
    setExpandedByTicker((current) => retainRecordKeys(current, displayedTickers));
  }, [displayedTickers]);

  const freshnessSummary = useMemo(() => {
    const buckets = displayedStations.flatMap((station) => station.buckets || []);
    let freshCount = 0;
    let staleCount = 0;

    for (const bucket of buckets) {
      const ageValue = ageSeconds(bucket.bookAsOfUtc, nowMillis);
      const band = ageBand(ageValue, AGE_THRESHOLDS);
      if (band === "fresh" || band === "aging") {
        freshCount += 1;
      }
      if (band === "stale" || band === "frozen") {
        staleCount += 1;
      }
    }

    return {
      total: buckets.length,
      freshCount,
      staleCount,
    };
  }, [displayedStations, nowMillis]);

  const connectionState = deriveGlobalConnectionState(wsStatus, frame, nowMillis);

  const toggleDepthExpand = useCallback((ticker, side) => {
    setExpandedByTicker((current) => {
      const previous = current[ticker] ?? { yes: false, no: false };
      return {
        ...current,
        [ticker]: {
          ...previous,
          [side]: !previous[side],
        },
      };
    });
  }, []);
  const toggleShowAllOpportunities = useCallback(() => {
    setShowAllOpportunities((current) => !current);
  }, []);
  const toggleAutoInference = useCallback(() => {
    setIsAutoInferenceActive((current) => {
      const next = !current;
      if (next) {
        const label = formatTargetDateLabel(selectedOpportunitiesDate);
        setAutoInferenceStatus(`Auto refresh started for ${label}.`);
      } else {
        setAutoInferenceStatus("Auto refresh stopped.");
      }
      return next;
    });
  }, [selectedOpportunitiesDate]);
  const selectOpportunitiesDate = useCallback((targetDateLocal) => {
    setSelectedOpportunitiesDate(targetDateLocal);
    setAutoInferenceStatus("");
    if (!targetDateLocal) {
      return;
    }
    if (!shouldRevalidateSnapshotMeta(snapshotMetaByDate[targetDateLocal], Date.now(), SNAPSHOT_REVALIDATE_MAX_AGE_MS)) {
      return;
    }
    setOpportunityRowsLoading(true);
    fetchSnapshotForDate(targetDateLocal)
      .then((snapshot) => {
        storeSnapshotForDate(targetDateLocal, snapshot, Date.now());
      })
      .catch(() => {
        // Keep prior date rows; selected date will show empty until next poll succeeds.
      })
      .finally(() => {
        setOpportunityRowsLoading(false);
      });
  }, [fetchSnapshotForDate, snapshotMetaByDate, storeSnapshotForDate]);

  return (
    <section className="lt-workstation">
      <LiveTopBar
        connectionState={connectionState}
        frame={frame}
        nowMillis={nowMillis}
        freshnessSummary={freshnessSummary}
        liveConfig={liveConfig}
      />

      <LiveStrategyBar liveConfig={liveConfig} />

      <OpportunitiesPanel
        liveConfig={liveConfig}
        accountBalanceUsd={accountBalanceUsd}
        accountBalanceStatus={accountBalanceStatus}
        filteredRowsByDate={thresholdRowsByDate}
        allRowsByDate={winFloorRowsByDate}
        showAllRows={showAllOpportunities}
        onToggleShowAllRows={toggleShowAllOpportunities}
        dateOptions={opportunitiesDateOptions}
        selectedDate={selectedOpportunitiesDate}
        onSelectDate={selectOpportunitiesDate}
        isLoading={opportunityRowsLoading}
        isAutoInferenceActive={isAutoInferenceActive}
        onToggleAutoInference={toggleAutoInference}
        autoInferenceStatus={autoInferenceStatus}
        nowMillis={nowMillis}
      />

      {wsError ? <div className="lt-globalError">Feed error: {wsError}</div> : null}

      <div className="lt-stationGrid">
        {displayedStations.length === 0 ? (
          <article className="lt-emptyState">
            <h2>No live station frame yet</h2>
            <p>Waiting for KNYC, KMIA, KMDW, and KLAX orderbook frames.</p>
          </article>
        ) : null}

        {displayedStations.map((station) => (
          <StationColumn
            key={station.stationId}
            station={station}
            nowMillis={nowMillis}
            markers={markers}
            recentEvents={recentEvents}
            expandedByTicker={expandedByTicker}
            onToggleExpand={toggleDepthExpand}
          />
        ))}
      </div>

      <RecentLiveBalancePanel
        report={recentBalanceReport}
        accountBalanceUsd={accountBalanceUsd}
      />
    </section>
  );
}
