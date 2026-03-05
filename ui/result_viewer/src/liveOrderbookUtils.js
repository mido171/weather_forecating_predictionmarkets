const NUMERIC_FORMATTER = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 3,
});

export const AGE_THRESHOLDS = {
  freshSeconds: 2,
  agingSeconds: 5,
  staleSeconds: 10,
};

export const SPREAD_THRESHOLDS = {
  tight: 1,
  normal: 3,
  wide: 6,
};

export function toFiniteNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function parseTimestampMillis(isoText) {
  if (!isoText) return null;
  const parsed = Date.parse(String(isoText).trim());
  return Number.isFinite(parsed) ? parsed : null;
}

export function ageSeconds(isoText, nowMillis) {
  const ts = parseTimestampMillis(isoText);
  if (!Number.isFinite(ts)) return null;
  const deltaMillis = Math.max(0, nowMillis - ts);
  return deltaMillis / 1000;
}

export function formatAgeSeconds(ageValue) {
  if (!Number.isFinite(ageValue)) return "--";
  if (ageValue < 10) return `${ageValue.toFixed(1)}s`;
  return `${Math.round(ageValue)}s`;
}

export function formatAgeFromIso(isoText, nowMillis) {
  const seconds = ageSeconds(isoText, nowMillis);
  return formatAgeSeconds(seconds);
}

export function ageBand(ageValue, thresholds = AGE_THRESHOLDS) {
  if (!Number.isFinite(ageValue)) return "unknown";
  if (ageValue < thresholds.freshSeconds) return "fresh";
  if (ageValue < thresholds.agingSeconds) return "aging";
  if (ageValue < thresholds.staleSeconds) return "stale";
  return "frozen";
}

export function formatTimestamp(isoText) {
  if (!isoText) return "--";
  const millis = parseTimestampMillis(isoText);
  if (!Number.isFinite(millis)) return String(isoText);
  const date = new Date(millis);
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

export function formatPriceCents(cents, missingLabel = "--") {
  const value = toFiniteNumber(cents);
  if (!Number.isFinite(value)) return missingLabel;
  return `${Math.round(value)}c`;
}

export function formatQtyExact(quantity) {
  const value = toFiniteNumber(quantity);
  if (!Number.isFinite(value)) return "--";
  return NUMERIC_FORMATTER.format(value);
}

export function getSpreadSeverity(spreadCents, thresholds = SPREAD_THRESHOLDS) {
  const spread = toFiniteNumber(spreadCents);
  if (!Number.isFinite(spread)) {
    return { level: "na", label: "Unavailable" };
  }
  if (spread <= thresholds.tight) {
    return { level: "tight", label: "Tight" };
  }
  if (spread <= thresholds.normal) {
    return { level: "normal", label: "Normal" };
  }
  if (spread <= thresholds.wide) {
    return { level: "wide", label: "Wide" };
  }
  return { level: "danger", label: "Danger" };
}

export function depthBarPct(quantity, maxQuantity) {
  const qty = toFiniteNumber(quantity);
  const maxQty = toFiniteNumber(maxQuantity);
  if (!Number.isFinite(qty) || !Number.isFinite(maxQty) || qty <= 0 || maxQty <= 0) return 0;
  const raw = (Math.log1p(qty) / Math.log1p(maxQty)) * 100;
  return Math.max(8, Math.min(100, raw));
}

export function bucketSortKey(label) {
  const text = String(label ?? "").toLowerCase();
  const numbers = [...text.matchAll(/\d+/g)].map((match) => Number(match[0]));
  const first = numbers.length ? numbers[0] : 9999;
  const second = numbers.length > 1 ? numbers[1] : first;

  if (text.includes("or below") || text.includes("or less")) {
    return `a-${String(first).padStart(4, "0")}-${text}`;
  }
  if (text.includes("or above") || text.includes("or higher")) {
    return `c-${String(first).padStart(4, "0")}-${text}`;
  }
  if (numbers.length >= 2) {
    const lo = Math.min(first, second);
    const hi = Math.max(first, second);
    return `b-${String(lo).padStart(4, "0")}-${String(hi).padStart(4, "0")}-${text}`;
  }
  return `d-${String(first).padStart(4, "0")}-${String(second).padStart(4, "0")}-${text}`;
}

export function classifyDelta(field, prev, next) {
  if (prev == null || next == null || prev === next) return null;
  switch (field) {
    case "bestBid":
      return next > prev ? "buy" : "sell";
    case "bestAsk":
      return next < prev ? "buy" : "sell";
    case "bidSize":
      return next > prev ? "buySoft" : "sellSoft";
    case "askSize":
      return next < prev ? "buySoft" : "sellSoft";
    case "spread":
      return next < prev ? "tighten" : "widen";
    case "mid":
      return next > prev ? "buy" : "sell";
    default:
      return null;
  }
}

export function toneFromClassification(classification) {
  switch (classification) {
    case "buy":
    case "buySoft":
    case "tighten":
      return "buy";
    case "sell":
    case "sellSoft":
      return "sell";
    case "widen":
      return "warn";
    default:
      return "neutral";
  }
}

export function signedCents(prev, next) {
  if (!Number.isFinite(prev) || !Number.isFinite(next)) return "";
  const delta = Math.round(next - prev);
  if (delta === 0) return "0c";
  return `${delta > 0 ? "+" : ""}${delta}c`;
}

export function signedQty(prev, next) {
  if (!Number.isFinite(prev) || !Number.isFinite(next)) return "";
  const delta = next - prev;
  if (delta === 0) return "0";
  return `${delta > 0 ? "+" : ""}${NUMERIC_FORMATTER.format(delta)}`;
}
