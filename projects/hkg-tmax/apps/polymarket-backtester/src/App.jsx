import {
  Activity,
  ArrowLeft,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  CalendarDays,
  CheckCircle2,
  Clock3,
  CircleDollarSign,
  Loader2,
  RefreshCcw,
  RotateCcw,
  Search,
  ShieldCheck,
  TrendingUp,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const DEFAULT_START = "2026-07-01";
const DEFAULT_END = "2026-07-10";
const FALLBACK_SELECTED_DATE = "2026-07-08";
const DEFAULT_PROFILE = "t_minus_1_2359_hkt";
const AUTO_REFRESH_MS = 20_000;
const MIN_TRADE_EDGE_PP = 15.0;
const MIN_TRADE_WIN_PROBABILITY = 0.7;

const FALLBACK_PROFILES = [
  {
    id: "t_minus_1_1800_hkt",
    label: "12:00 Stockholm",
    stockholmEntry: "12:00",
    hktCutoff: "18:00",
    qualityScore: 62,
    validationStatus: "validated_apples_to_apples",
    tradeable: true,
    operational: { n: 5050, raw_mae: 0.9648, b4_rps: 0.04479, b4_nll: 1.11268 },
    strictCommon: { n: 5050, raw_mae: 0.9648, b4_rps: 0.04479 },
    warning: "strict threshold only",
  },
  {
    id: "t_minus_1_1900_hkt",
    label: "13:00 Stockholm",
    stockholmEntry: "13:00",
    hktCutoff: "19:00",
    qualityScore: 63,
    validationStatus: "validated_apples_to_apples",
    tradeable: true,
    operational: { n: 5056, raw_mae: 0.9645, b4_rps: 0.04479, b4_nll: 1.11274 },
    strictCommon: { n: 5050, raw_mae: 0.964, b4_rps: 0.04474 },
    warning: "strict threshold only",
  },
  {
    id: "t_minus_1_2000_hkt",
    label: "14:00 Stockholm",
    stockholmEntry: "14:00",
    hktCutoff: "20:00",
    qualityScore: 65,
    validationStatus: "validated_apples_to_apples",
    tradeable: true,
    operational: { n: 5086, raw_mae: 0.9599, b4_rps: 0.04442, b4_nll: 1.10603 },
    strictCommon: { n: 5050, raw_mae: 0.9605, b4_rps: 0.04447 },
    warning: "strict threshold only",
  },
  {
    id: "t_minus_1_2100_hkt",
    label: "15:00 Stockholm",
    stockholmEntry: "15:00",
    hktCutoff: "21:00",
    qualityScore: 67,
    validationStatus: "validated_apples_to_apples",
    tradeable: true,
    operational: { n: 5088, raw_mae: 0.9595, b4_rps: 0.04442, b4_nll: 1.1058 },
    strictCommon: { n: 5050, raw_mae: 0.96, b4_rps: 0.04446 },
    warning: "acceptable with strict edge",
  },
  {
    id: "t_minus_1_2200_hkt",
    label: "16:00 Stockholm",
    stockholmEntry: "16:00",
    hktCutoff: "22:00",
    qualityScore: 68,
    validationStatus: "validated_apples_to_apples",
    tradeable: true,
    operational: { n: 5092, raw_mae: 0.9584, b4_rps: 0.04433, b4_nll: 1.1047 },
    strictCommon: { n: 5050, raw_mae: 0.9591, b4_rps: 0.0444 },
    warning: "acceptable with strict edge",
  },
  {
    id: "t_minus_1_2300_hkt",
    label: "17:00 Stockholm",
    stockholmEntry: "17:00",
    hktCutoff: "23:00",
    qualityScore: 70,
    validationStatus: "validated_apples_to_apples",
    tradeable: true,
    operational: { n: 5099, raw_mae: 0.9566, b4_rps: 0.04424, b4_nll: 1.10328 },
    strictCommon: { n: 5050, raw_mae: 0.9578, b4_rps: 0.04435 },
    warning: "preferred late-window",
  },
  {
    id: "t_minus_1_2359_hkt",
    label: "17:59 Stockholm",
    stockholmEntry: "17:59",
    hktCutoff: "23:59",
    qualityScore: 82,
    validationStatus: "validated_apples_to_apples",
    tradeable: true,
    operational: { n: 5629, raw_mae: 0.9309, b4_rps: 0.04193, b4_nll: 1.05096 },
    strictCommon: { n: 5050, raw_mae: 0.9444, b4_rps: 0.04401 },
    warning: "strongest validated profile",
  },
  {
    id: "live_now",
    label: "Live exploratory",
    stockholmEntry: "Now",
    hktCutoff: "live",
    qualityScore: null,
    validationStatus: "not_apples_to_apples",
    tradeable: false,
    operational: null,
    strictCommon: null,
    warning: "display only",
  },
];

function dollars(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `$${Number(value).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function number(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function percent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${number(Number(value) * 100, digits)}%`;
}

function cents(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${number(value, digits)}c`;
}

function decimalMetric(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function defaultSelectedDate() {
  try {
    const parts = new Intl.DateTimeFormat("en-CA", {
      timeZone: "Europe/Stockholm",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    }).formatToParts(new Date());
    const year = Number(parts.find((part) => part.type === "year")?.value);
    const month = Number(parts.find((part) => part.type === "month")?.value);
    const day = Number(parts.find((part) => part.type === "day")?.value);
    if (year === 2026 && month === 7 && day >= 1 && day <= 9) {
      return `2026-07-${String(day + 1).padStart(2, "0")}`;
    }
  } catch (_exc) {
    return FALLBACK_SELECTED_DATE;
  }
  return FALLBACK_SELECTED_DATE;
}

function defaultAsOfProfile() {
  try {
    const hour = Number(
      new Intl.DateTimeFormat("en-GB", {
        timeZone: "Europe/Stockholm",
        hour: "2-digit",
        hourCycle: "h23",
      }).format(new Date()),
    );
    if (hour >= 18) return "t_minus_1_2359_hkt";
    if (hour >= 17) return "t_minus_1_2300_hkt";
    if (hour >= 16) return "t_minus_1_2200_hkt";
    if (hour >= 15) return "t_minus_1_2100_hkt";
    if (hour >= 14) return "t_minus_1_2000_hkt";
    if (hour >= 13) return "t_minus_1_1900_hkt";
    return "t_minus_1_1800_hkt";
  } catch (_exc) {
    return DEFAULT_PROFILE;
  }
}

function profileById(profiles, id) {
  return profiles.find((profile) => profile.id === id) || FALLBACK_PROFILES.find((profile) => profile.id === id);
}

function isProfileTradeable(profile, market) {
  const resolved = market?.profile || profile;
  return Boolean(resolved?.tradeable && resolved?.forecastAnchorAvailable !== false && market?.status === "ok");
}

function isExecutableSource(value) {
  return value === "clob_ask";
}

function priceSourceLabel(value) {
  if (!value) return "Market snapshot";
  if (value === "clob_ask") return "CLOB ask";
  if (value === "gamma_outcome_price_fallback") return "Gamma snapshot";
  if (value === "missing_token" || value === "missing_market") return "No market quote";
  if (String(value).startsWith("clob_error")) return "Snapshot fallback unavailable";
  return String(value).replaceAll("_", " ");
}

function tradeCurrentPriceCents(trade) {
  if (trade?.current_price_cents !== null && trade?.current_price_cents !== undefined) {
    return Number(trade.current_price_cents);
  }
  const markedValue = Number(trade?.marked_value_usd);
  const shares = Number(trade?.shares);
  if (!Number.isNaN(markedValue) && !Number.isNaN(shares) && shares > 0) {
    return (markedValue / shares) * 100;
  }
  return null;
}

function tradeLossFractionOfMaxLoss(trade) {
  if (trade?.manual_loss_fraction_of_max_loss !== null && trade?.manual_loss_fraction_of_max_loss !== undefined) {
    const value = Number(trade.manual_loss_fraction_of_max_loss);
    return Number.isNaN(value) ? null : value;
  }
  const unrealized = Number(trade?.unrealized_pnl_usd);
  const stake = Number(trade?.stake_usd);
  if (!Number.isNaN(unrealized) && !Number.isNaN(stake) && stake > 0 && unrealized < 0) {
    return Math.min(1, Math.abs(unrealized) / stake);
  }
  return null;
}

function tradeManualSettlement(trade) {
  const currentPrice = tradeCurrentPriceCents(trade);
  const winThreshold = Number(trade?.manual_win_settle_threshold_cents ?? trade?.manual_settle_threshold_cents ?? 98);
  const lossThreshold = Number(trade?.manual_loss_settle_threshold_fraction ?? 0.97);
  const lossFraction = tradeLossFractionOfMaxLoss(trade);
  const winEligible =
    trade?.manual_win_settle_eligible === true || (currentPrice !== null && currentPrice >= winThreshold);
  const lossEligible =
    trade?.manual_loss_settle_eligible === true ||
    (lossFraction !== null && !Number.isNaN(lossFraction) && lossFraction >= lossThreshold);
  if (winEligible) {
    return {
      eligible: true,
      result: "win",
      title: "Settle this demo trade as a win",
      label: "Settle win",
    };
  }
  if (lossEligible) {
    return {
      eligible: true,
      result: "loss",
      title: "Settle this demo trade as a loss",
      label: "Settle loss",
    };
  }
  return {
    eligible: false,
    result: null,
    title:
      currentPrice === null
        ? "Requires a current contract price"
        : `Requires ${cents(winThreshold)}+ for win settlement or at least ${percent(lossThreshold)} of max loss for loss settlement`,
    label: "Settle",
  };
}

function valueTone(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "";
  return Number(value) >= 0 ? "positive" : "negative";
}

function tradeStatusTone(trade) {
  if (trade?.status === "settled" && trade?.result === "win") return "good";
  if (trade?.status === "settled" && trade?.result === "loss") return "bad";
  if (trade?.status === "open") return "open";
  return "neutral";
}

function tradeStatusLabel(trade) {
  if (trade?.status === "settled" && trade?.result) return trade.result;
  return trade?.status || "unknown";
}

function formatZonedTime(value, timeZone) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return new Intl.DateTimeFormat(undefined, {
    timeZone,
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
    timeZoneName: "short",
  }).format(date);
}

function formatExactZonedTime(value, timeZone) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return new Intl.DateTimeFormat(undefined, {
    timeZone,
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
    timeZoneName: "short",
  }).format(date);
}

function formatUtcIsoTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toISOString();
}

function formatStoredTimestamp(value) {
  if (!value) return "-";
  if (typeof value === "string") return value;
  return formatUtcIsoTime(value);
}

function formatEntryTimestamp(value) {
  return formatExactZonedTime(value, "Europe/Stockholm");
}

function formatCadenceHour(value, timeZone) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  const hourStart = new Date(Math.floor(date.getTime() / 3_600_000) * 3_600_000);
  return new Intl.DateTimeFormat(undefined, {
    timeZone,
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
    timeZoneName: "short",
  }).format(hourStart);
}

function formatTimeOnly(value) {
  if (!value) return "pending";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "pending";
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
  }).format(date);
}

function edgeTone(edge) {
  if (edge === null || edge === undefined || Number.isNaN(Number(edge))) return "neutral";
  if (Number(edge) >= 8) return "elite";
  if (Number(edge) >= 3) return "good";
  if (Number(edge) <= -3) return "bad";
  return "neutral";
}

function probabilityTone(probability) {
  const value = Number(probability || 0);
  if (value >= 0.25) return "hot";
  if (value >= 0.12) return "warm";
  if (value >= 0.04) return "cool";
  return "cold";
}

function modelMethodLabel(value) {
  if (!value) return "-";
  return String(value)
    .replace(/^B4_/, "B4 ")
    .replaceAll("_", " ")
    .replace("pmf", "PMF");
}

function dateLabel(value) {
  const [year, month, day] = value.split("-").map(Number);
  return new Date(Date.UTC(year, month - 1, day)).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function fullDateLabel(value) {
  const [year, month, day] = value.split("-").map(Number);
  return new Date(Date.UTC(year, month - 1, day)).toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function shiftDate(value, days) {
  const date = new Date(`${value}T00:00:00Z`);
  date.setUTCDate(date.getUTCDate() + days);
  return date.toISOString().slice(0, 10);
}

async function apiFetch(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed with ${response.status}`);
  }
  return payload;
}

function StatusPill({ children, tone = "neutral" }) {
  return <span className={`status-pill ${tone}`}>{children}</span>;
}

function IconButton({ title, children, className = "", ...props }) {
  return (
    <button className={`icon-button ${className}`} title={title} aria-label={title} {...props}>
      {children}
    </button>
  );
}

function Kpi({ icon: Icon, label, value, trend }) {
  return (
    <div className="kpi">
      <div className="kpi-icon">
        <Icon size={18} />
      </div>
      <div>
        <p>{label}</p>
        <strong className={trend ? (Number(trend) >= 0 ? "positive" : "negative") : ""}>{value}</strong>
      </div>
    </div>
  );
}

function EmptyState({ title, body }) {
  return (
    <div className="empty-state">
      <Search size={20} />
      <strong>{title}</strong>
      <span>{body}</span>
    </div>
  );
}

function AsOfProfileSelector({ profiles, selectedProfile, setSelectedProfile, market }) {
  return (
    <section className="profile-control-panel" aria-label="Trading as-of profile">
      <div className="profile-control-heading">
        <div>
          <div className="eyebrow">
            <Clock3 size={15} />
            Trading as-of profile
          </div>
          <h3>Validated cutoff for 12-18 Stockholm entries</h3>
        </div>
        <StatusPill tone={isProfileTradeable(profileById(profiles, selectedProfile), market) ? "good" : "warn"}>
          {isProfileTradeable(profileById(profiles, selectedProfile), market) ? "tradeable" : "blocked"}
        </StatusPill>
      </div>
      <div className="profile-strip" role="tablist" aria-label="Validated cutoff profiles">
        {profiles.map((profile) => {
          const active = profile.id === selectedProfile;
          const validated = profile.validationStatus === "validated_apples_to_apples";
          const metrics = profile.strictCommon || profile.operational;
          return (
            <button
              className={active ? "profile-card active" : "profile-card"}
              key={profile.id}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => setSelectedProfile(profile.id)}
            >
              <span className="profile-time">{profile.stockholmEntry}</span>
              <strong>{profile.hktCutoff} HKT</strong>
              <small>{validated ? `score ${profile.qualityScore}/100` : "display only"}</small>
              {metrics ? (
                <span className="profile-metrics">
                  MAE {decimalMetric(metrics.raw_mae, 4)} · RPS {decimalMetric(metrics.b4_rps, 5)}
                </span>
              ) : (
                <span className="profile-metrics muted">not validated</span>
              )}
            </button>
          );
        })}
      </div>
      <div className="profile-rule-row">
        <span>Entry guard</span>
        <strong>edge &gt;= {number(MIN_TRADE_EDGE_PP, 1)} pp · win &gt;= {percent(MIN_TRADE_WIN_PROBABILITY)}</strong>
        <small>Only executable CLOB ask prices can create demo trades.</small>
      </div>
    </section>
  );
}

function EngineSnapshotPanel({ market }) {
  const forecast = market?.forecast || null;
  const model = market?.model || null;
  const profile = market?.profile || model?.profile || forecast?.profile || null;
  const rows = market?.marketRows || [];
  const peakRow = rows.reduce(
    (best, row) => (Number(row.modelProbability || 0) > Number(best?.modelProbability || -1) ? row : best),
    null,
  );
  const updateTime = forecast?.update_time_hkt ?? forecast?.updateTimeHkt;
  const forecastMin = forecast?.forecast_min_c ?? forecast?.forecastMinC;
  const forecastMax = forecast?.forecast_max_c ?? forecast?.forecastMaxC;
  const forecastRange =
    forecastMin === null || forecastMin === undefined || forecastMax === null || forecastMax === undefined
      ? null
      : Number(forecastMax) - Number(forecastMin);
  const forecastMaxTenths =
    forecastMax === null || forecastMax === undefined || Number.isNaN(Number(forecastMax))
      ? null
      : Math.round(Number(forecastMax) * 10);
  const officialMaxRound =
    forecastMax === null || forecastMax === undefined || Number.isNaN(Number(forecastMax))
      ? null
      : Math.round(Number(forecastMax));
  const targetMonth = market?.target_date ? Number(String(market.target_date).slice(5, 7)) : null;

  return (
    <section className="analysis-panel engine-panel">
      <div className="analysis-header">
        <div>
          <div className="eyebrow">
            <BrainCircuit size={15} />
            Probability engine
          </div>
          <h3>B4 distribution calculation</h3>
        </div>
        <StatusPill tone={isProfileTradeable(profile, market) ? "good" : "warn"}>
          {isProfileTradeable(profile, market) ? "apples-to-apples" : "not tradeable"}
        </StatusPill>
      </div>

      <div className="engine-hero">
        <div className="engine-forecast-block">
          <div className="engine-block-label">Forecast used</div>
          <div className="engine-temp-readout">
            <strong>{forecastMax === null || forecastMax === undefined ? "-" : number(forecastMax, 1)}</strong>
            <span>deg C max</span>
          </div>
          <div className="engine-temp-meta">
            <span>{forecastMin === null || forecastMin === undefined ? "Min not supplied" : `${number(forecastMin, 1)} deg C min`}</span>
            <span>{forecastRange === null ? "Range unavailable" : `${number(forecastRange, 1)} deg range`}</span>
            <span>{peakRow ? `${peakRow.label} peak bucket` : "Peak bucket unavailable"}</span>
          </div>
          <small>{forecast?.source || "No forecast source"}</small>
        </div>

        <div className="engine-time-block">
          <div className="engine-time-item is-primary">
            <span>
              <Clock3 size={14} />
              Stockholm forecast time
            </span>
            <strong>{formatZonedTime(updateTime, "Europe/Stockholm")}</strong>
            <small>Cadence hour {formatCadenceHour(updateTime, "Europe/Stockholm")}</small>
          </div>
          <div className="engine-time-item">
            <span>Source update</span>
            <strong>{formatZonedTime(updateTime, "Asia/Hong_Kong")}</strong>
            <small>Hong Kong Observatory release time</small>
          </div>
        </div>
      </div>

      <div className="engine-model-grid">
        <div className="engine-model-block profile-contract-block">
          <div className="engine-section-title">
            <ShieldCheck size={15} />
            Active profile
          </div>
          <dl className="engine-kv-list">
            <div>
              <dt>Stockholm entry</dt>
              <dd>{profile?.stockholmEntry || "-"}</dd>
            </div>
            <div>
              <dt>HKT cutoff</dt>
              <dd>{profile?.hktCutoff || "-"}</dd>
            </div>
            <div>
              <dt>Validation</dt>
              <dd>{profile?.validationStatus === "validated_apples_to_apples" ? "Apples-to-apples" : "Blocked"}</dd>
            </div>
            <div>
              <dt>Score</dt>
              <dd>{profile?.qualityScore ? `${profile.qualityScore}/100` : "-"}</dd>
            </div>
          </dl>
        </div>

        <div className="engine-model-block">
          <div className="engine-section-title">
            <Activity size={15} />
            Model contract
          </div>
          <dl className="engine-kv-list">
            <div>
              <dt>Method</dt>
              <dd>{modelMethodLabel(model?.method)}</dd>
            </div>
            <div>
              <dt>Training window</dt>
              <dd>{model ? `${model.train_start} to ${model.train_end}` : "-"}</dd>
            </div>
            <div>
              <dt>Train rows</dt>
              <dd>{model?.train_rows ?? "-"}</dd>
            </div>
            <div>
              <dt>Cutoff rows</dt>
              <dd>{model?.cutoff_profile || market?.as_of_profile || "-"}</dd>
            </div>
            <div>
              <dt>Smoothing</dt>
              <dd>{model ? `month ${number(model.month_alpha, 2)} / cell ${number(model.cell_alpha, 2)}` : "-"}</dd>
            </div>
          </dl>
        </div>

        <div className="engine-calculation-block">
          <div className="engine-section-title">
            <BarChart3 size={15} />
            Calculation path
          </div>
          <div className="formula-line">
            <span className="formula-token">month {targetMonth || "-"}</span>
            <span className="formula-token">max tenths {forecastMaxTenths ?? "-"}</span>
            <span className="formula-token">rounded {officialMaxRound === null ? "-" : `${officialMaxRound}C`}</span>
            <span className="formula-token">{profile?.hktCutoff || "-"} cutoff</span>
          </div>
          <strong>P(bucket) = B4 residual PMF mass inside the exact Polymarket bucket</strong>
          <small>{profile?.asOfRule || "Market prices are used for edge, not blended into model probability."}</small>
        </div>
      </div>

      <div className="probability-stack engine-distribution">
        <div className="probability-stack-header">
          <span>Bucket distribution</span>
          <span>Model PMF</span>
          <span>Fair YES</span>
        </div>
        {rows.map((row) => {
          const probability = Number(row.modelProbability || 0);
          const width = Math.max(2, probability * 100);
          return (
            <div className="probability-row" key={row.bucket}>
              <span className="probability-bucket">{row.label}</span>
              <div className="probability-bar-track" aria-hidden="true">
                <span
                  className={`probability-bar ${probabilityTone(probability)}`}
                  style={{ width: `${width}%` }}
                />
              </div>
              <strong>{percent(row.modelProbability)}</strong>
              <small>fair YES {cents(row.modelYesPct)}</small>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function EdgeStackPanel({ market }) {
  const edges = [...(market?.edges || [])].sort((left, right) => Number(right.edgePp) - Number(left.edgePp));
  const executableCount = edges.filter((edge) => edge.executable).length;

  return (
    <section className="analysis-panel edge-panel">
      <div className="analysis-header">
        <div>
          <div className="eyebrow">
            <TrendingUp size={15} />
            Edge stack
          </div>
          <h3>Best edges high to low</h3>
        </div>
        <StatusPill tone={executableCount ? "good" : "warn"}>{executableCount} CLOB</StatusPill>
      </div>

      {edges.length ? (
        <div className="edge-list">
          {edges.map((edge, index) => (
            <div
              className={`edge-row ${edgeTone(edge.edgePp)} ${edge.executable ? "" : "display-only"}`}
              key={`${edge.side}-${edge.bucket}-${index}`}
            >
              <div className="edge-rank">{index + 1}</div>
              <div className="edge-contract">
                <strong>
                  {edge.side.toUpperCase()} {edge.label || edge.bucket}
                </strong>
                <span>{edge.executable ? "Executable CLOB ask" : priceSourceLabel(edge.priceSource)}</span>
              </div>
              <div className="edge-probs">
                <div>
                  <span>Market</span>
                  <strong>{number(edge.marketPriceC, 1)}%</strong>
                </div>
                <div>
                  <span>Our engine</span>
                  <strong>{number(edge.modelFairC, 1)}%</strong>
                </div>
              </div>
              <div className="edge-score">
                <strong>{number(edge.edgePp, 1)} pp</strong>
                <span>{edge.classification}</span>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <EmptyState title="No priced edges" body="Refresh once model probabilities and market quotes are both available." />
      )}
    </section>
  );
}

function TradeLedgerPanel({
  account,
  loading,
  ledgerTab,
  setLedgerTab,
  selectedTradeId,
  setSelectedTradeId,
  settleTradeAsWin,
  settleTradeAsLoss,
}) {
  const ledgerTrades = Array.from(
    new Map([...(account?.trades || []), ...(account?.openTrades || [])].map((trade) => [trade.id, trade])).values(),
  ).sort((left, right) => {
    const rightTime = new Date(right.opened_at_utc || 0).getTime();
    const leftTime = new Date(left.opened_at_utc || 0).getTime();
    return rightTime - leftTime;
  });
  const openTrades = account?.openTrades || [];
  const selectedTrade =
    ledgerTrades.find((trade) => String(trade.id) === String(selectedTradeId)) || ledgerTrades[0] || null;

  function tradeProfileLabel(trade) {
    const profile = trade?.metadata_json?.entry?.profile;
    if (!profile) return "profile not stored";
    return `${profile.stockholmEntry || "-"} / ${profile.hktCutoff || "-"} HKT`;
  }

  function showTradeDetails(trade) {
    setSelectedTradeId(trade.id);
    setLedgerTab("details");
  }

  function handleTradeKeyDown(event, trade) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      showTradeDetails(trade);
    }
  }

  function renderOpenTrade(trade) {
    const currentPrice = tradeCurrentPriceCents(trade);
    const settlement = tradeManualSettlement(trade);
    const lossFraction = tradeLossFractionOfMaxLoss(trade);
    const isSettling = loading.settlingTradeId === trade.id;
    const isSelected = selectedTrade && String(selectedTrade.id) === String(trade.id);
    const statusTone = tradeStatusTone(trade);
    return (
      <div
        className={[
          "trade-row",
          `status-${statusTone}`,
          "is-clickable",
          settlement.eligible ? "settle-ready" : "",
          settlement.result ? `settle-${settlement.result}-ready` : "",
          isSelected ? "selected" : "",
        ]
          .filter(Boolean)
          .join(" ")}
        key={trade.id}
        role="button"
        tabIndex={0}
        onClick={() => showTradeDetails(trade)}
        onKeyDown={(event) => handleTradeKeyDown(event, trade)}
      >
        <div className="trade-main">
          <strong>
            {trade.side.toUpperCase()} {trade.bucket_key}
          </strong>
          <span>
            {trade.target_date} entry {cents(trade.entry_price_cents)} · {priceSourceLabel(trade.price_source)}
          </span>
          <span className={`ledger-result-badge ${statusTone}`}>{tradeStatusLabel(trade)}</span>
          <span className="trade-entry-time">Entry time {formatEntryTimestamp(trade.opened_at_utc)}</span>
          <span className="trade-entry-time">Profile {tradeProfileLabel(trade)}</span>
        </div>
        <div className="trade-mark">
          <span>Contract</span>
          <strong>{currentPrice === null ? "-" : cents(currentPrice)}</strong>
        </div>
        <div className="trade-pnl">
          <strong className={valueTone(trade.unrealized_pnl_usd)}>{dollars(trade.unrealized_pnl_usd)}</strong>
          <span>
            {number(trade.shares, 2)} shares
            {lossFraction !== null && lossFraction > 0 ? ` · ${percent(lossFraction)} max loss` : ""}
          </span>
        </div>
        <button
          className={`settle-button ${settlement.result || ""}`}
          type="button"
          disabled={!settlement.eligible || isSettling}
          onClick={(event) => {
            event.stopPropagation();
            if (settlement.result === "loss") {
              settleTradeAsLoss(trade);
            } else {
              settleTradeAsWin(trade);
            }
          }}
          title={settlement.title}
        >
          {isSettling ? <Loader2 size={14} className="spin" /> : <CheckCircle2 size={14} />}
          {settlement.label}
        </button>
      </div>
    );
  }

  function renderHistoryTrade(trade) {
    const isSelected = selectedTrade && String(selectedTrade.id) === String(trade.id);
    const statusTone = tradeStatusTone(trade);
    return (
      <button
        className={["ledger-row", `status-${statusTone}`, isSelected ? "selected" : ""].filter(Boolean).join(" ")}
        key={trade.id}
        type="button"
        onClick={() => showTradeDetails(trade)}
      >
        <div className="ledger-contract">
          <strong>
            {trade.side.toUpperCase()} {trade.bucket_key}
          </strong>
          <span>
            {trade.target_date} · trade #{trade.id}
          </span>
          <span className={`ledger-result-badge ${statusTone}`}>{tradeStatusLabel(trade)}</span>
        </div>
        <div>
          <span>Entry</span>
          <strong>{cents(trade.entry_price_cents)}</strong>
          <small>{priceSourceLabel(trade.price_source)}</small>
        </div>
        <div className="ledger-profile">
          <span>Profile</span>
          <strong>{tradeProfileLabel(trade)}</strong>
          <small>{trade.metadata_json?.entry?.profile?.validationStatus || "legacy"}</small>
        </div>
        <div className="ledger-entry-time">
          <span>Entry time</span>
          <strong>{formatEntryTimestamp(trade.opened_at_utc)}</strong>
          <small>Stockholm exact</small>
        </div>
        <div>
          <span>Win prob</span>
          <strong>{percent(trade.model_win_probability)}</strong>
          <small>{number(trade.shares, 2)} shares</small>
        </div>
        <div>
          <span>Edge / EV</span>
          <strong className={valueTone(trade.edge_pp)}>{number(trade.edge_pp, 1)} pp</strong>
          <small className={valueTone(trade.ev_usd)}>{dollars(trade.ev_usd)}</small>
        </div>
      </button>
    );
  }

  function renderTradeDetails() {
    if (!selectedTrade) {
      return <EmptyState title="No trade selected" body="Open or history rows will show full persisted trade metadata here." />;
    }

    const currentPrice = tradeCurrentPriceCents(selectedTrade);
    const lossFraction = tradeLossFractionOfMaxLoss(selectedTrade);
    const metadata = selectedTrade.metadata_json || {};
    const bucketSnapshot = metadata.bucket_snapshot || {};
    const entryProfile = metadata.entry?.profile || null;

    return (
      <div className="trade-detail-view">
        <div className="trade-detail-header">
          <div>
            <div className={`side-chip ${selectedTrade.side}`}>{selectedTrade.side.toUpperCase()}</div>
            <h4>
              {selectedTrade.bucket_key} bucket · {selectedTrade.target_date}
            </h4>
            <p>
              Trade #{selectedTrade.id} · snapshot #{selectedTrade.snapshot_id ?? "-"} · entry{" "}
              {formatEntryTimestamp(selectedTrade.opened_at_utc)}
            </p>
          </div>
          <StatusPill tone={tradeStatusTone(selectedTrade)}>
            {selectedTrade.status}
            {selectedTrade.result ? ` · ${selectedTrade.result}` : ""}
          </StatusPill>
        </div>

        <div className="detail-card-grid">
          <div className="detail-card entry">
            <span>Market entry</span>
            <strong>{cents(selectedTrade.entry_price_cents)}</strong>
            <small>{priceSourceLabel(selectedTrade.price_source)}</small>
          </div>
          <div className="detail-card entry-time">
            <span>Entry timestamp</span>
            <strong>{formatEntryTimestamp(selectedTrade.opened_at_utc)}</strong>
            <small>Stored {formatStoredTimestamp(selectedTrade.opened_at_utc)}</small>
          </div>
          <div className="detail-card profile">
            <span>As-of profile</span>
            <strong>{entryProfile ? `${entryProfile.stockholmEntry} / ${entryProfile.hktCutoff} HKT` : "Not stored"}</strong>
            <small>{entryProfile?.validationStatus || "Legacy trade metadata"}</small>
          </div>
          <div className="detail-card">
            <span>Stake / shares</span>
            <strong>
              {dollars(selectedTrade.stake_usd)} · {number(selectedTrade.shares, 2)}
            </strong>
            <small>Shares = stake divided by contract price</small>
          </div>
          <div className="detail-card">
            <span>Model win probability</span>
            <strong>{percent(selectedTrade.model_win_probability)}</strong>
            <small>Bucket probability {percent(selectedTrade.model_probability_bucket)}</small>
          </div>
          <div className="detail-card">
            <span>Edge / EV</span>
            <strong className={valueTone(selectedTrade.edge_pp)}>{number(selectedTrade.edge_pp, 1)} pp</strong>
            <small className={valueTone(selectedTrade.ev_usd)}>{dollars(selectedTrade.ev_usd)} expected value</small>
          </div>
          <div className="detail-card">
            <span>Current mark</span>
            <strong>{currentPrice === null ? "-" : cents(currentPrice)}</strong>
            <small>
              Marked {dollars(selectedTrade.marked_value_usd)} · unrealized{" "}
              <span className={valueTone(selectedTrade.unrealized_pnl_usd)}>
                {dollars(selectedTrade.unrealized_pnl_usd)}
              </span>
              {lossFraction !== null && lossFraction > 0 ? ` · ${percent(lossFraction)} max loss` : ""}
            </small>
          </div>
          <div className="detail-card">
            <span>Settlement</span>
            <strong className={valueTone(selectedTrade.realized_pnl_usd)}>{dollars(selectedTrade.realized_pnl_usd)}</strong>
            <small>
              {selectedTrade.settled_at_utc
                ? formatZonedTime(selectedTrade.settled_at_utc, "Europe/Stockholm")
                : "Open position"}
            </small>
          </div>
        </div>

        <div className="metadata-summary">
          <div>
            <span>Stored entry</span>
            <strong>
              {metadata.entry?.manual_price_cents ? `${metadata.entry.manual_price_cents}c manual override` : "Market quote"}
            </strong>
          </div>
          <div>
            <span>Entry timestamp</span>
            <strong>{formatEntryTimestamp(selectedTrade.opened_at_utc)}</strong>
          </div>
          <div>
            <span>As-of profile</span>
            <strong>{entryProfile ? entryProfile.label : "Not stored"}</strong>
          </div>
          <div>
            <span>Bucket snapshot</span>
            <strong>{bucketSnapshot.marketSlug || selectedTrade.event_slug}</strong>
          </div>
          <div>
            <span>Execution guard</span>
            <strong>{metadata.no_real_order ? "No real order signed" : "Demo record"}</strong>
          </div>
        </div>

        <div className="metadata-block">
          <div className="metadata-title">
            <ShieldCheck size={15} />
            Raw recorded metadata_json
          </div>
          <pre className="metadata-code">{JSON.stringify(metadata, null, 2)}</pre>
        </div>
      </div>
    );
  }

  return (
    <div className="panel trade-ledger-panel">
      <div className="section-title">
        <div>
          <h3>Trade ledger</h3>
          <span>
            {openTrades.length} open · {ledgerTrades.length} total
          </span>
        </div>
        <div className="ledger-tabs" role="tablist" aria-label="Trade ledger sections">
          <button
            className={ledgerTab === "open" ? "ledger-tab active" : "ledger-tab"}
            type="button"
            onClick={() => setLedgerTab("open")}
          >
            Open <span>{openTrades.length}</span>
          </button>
          <button
            className={ledgerTab === "history" ? "ledger-tab active" : "ledger-tab"}
            type="button"
            onClick={() => setLedgerTab("history")}
          >
            History <span>{ledgerTrades.length}</span>
          </button>
          <button
            className={ledgerTab === "details" ? "ledger-tab active" : "ledger-tab"}
            type="button"
            onClick={() => setLedgerTab("details")}
          >
            Metadata
          </button>
        </div>
      </div>

      {loading.account ? (
        <div className="table-loading">
          <Loader2 size={18} className="spin" /> Loading account
        </div>
      ) : ledgerTab === "open" ? (
        openTrades.length ? (
          <div className="trade-list">{openTrades.map(renderOpenTrade)}</div>
        ) : (
          <EmptyState title="No open trades" body="Use History or Metadata to inspect settled demo records." />
        )
      ) : ledgerTab === "history" ? (
        ledgerTrades.length ? (
          <div className="ledger-list">{ledgerTrades.map(renderHistoryTrade)}</div>
        ) : (
          <EmptyState title="No trade history" body="Stage a bucket trade to start the demo ledger." />
        )
      ) : (
        renderTradeDetails()
      )}
    </div>
  );
}

function App() {
  const [windowStart, setWindowStart] = useState(DEFAULT_START);
  const [windowEnd, setWindowEnd] = useState(DEFAULT_END);
  const [since, setSince] = useState(DEFAULT_START);
  const [profiles, setProfiles] = useState(FALLBACK_PROFILES);
  const [asOfProfile, setAsOfProfile] = useState(defaultAsOfProfile);
  const [markets, setMarkets] = useState([]);
  const [selectedDate, setSelectedDate] = useState(defaultSelectedDate);
  const [market, setMarket] = useState(null);
  const [account, setAccount] = useState(null);
  const [ticket, setTicket] = useState({
    bucket: "32",
    side: "yes",
    stakeUsd: "25",
  });
  const [loading, setLoading] = useState({
    markets: true,
    market: true,
    account: true,
    trade: false,
    settlingTradeId: null,
  });
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [lastRefreshAt, setLastRefreshAt] = useState(null);
  const [ledgerTab, setLedgerTab] = useState("open");
  const [selectedTradeId, setSelectedTradeId] = useState(null);
  const autoRefreshInFlight = useRef(false);
  const marketSelectionRef = useRef({ asOfProfile, selectedDate });

  const selectedBucket = useMemo(
    () => market?.marketRows?.find((row) => row.bucket === ticket.bucket) || null,
    [market, ticket.bucket],
  );
  const forecastMaxC = market?.forecast?.forecastMaxC ?? market?.forecast?.forecast_max_c;
  const modelTrainRows = market?.model?.trainRows ?? market?.model?.train_rows;
  const statusReason = market?.statusReason ?? market?.status_reason;
  const selectedProfile = useMemo(() => profileById(profiles, asOfProfile), [asOfProfile, profiles]);
  const activeProfile = market?.profile || selectedProfile;
  const activeProfileTradeable = isProfileTradeable(activeProfile, market);

  const selectedPrice = useMemo(() => {
    if (!selectedBucket) return null;
    return selectedBucket[ticket.side === "yes" ? "marketBuyYesC" : "marketBuyNoC"];
  }, [selectedBucket, ticket.side]);

  const selectedPriceSource = selectedBucket?.[ticket.side === "yes" ? "marketBuyYesSource" : "marketBuyNoSource"];
  const selectedPriceExecutable = isExecutableSource(selectedPriceSource);
  const marketPriceAvailable =
    selectedPrice !== null && selectedPrice !== undefined && !Number.isNaN(Number(selectedPrice)) && Number(selectedPrice) > 0;

  const effectivePrice = useMemo(() => {
    return Number(selectedPrice || 0);
  }, [selectedPrice]);

  const estimated = useMemo(() => {
    const stake = Number(ticket.stakeUsd || 0);
    const price = effectivePrice;
    if (!selectedBucket || !stake || !price) return null;
    if (selectedBucket.modelProbability === null || selectedBucket.modelProbability === undefined) return null;
    const modelProbability = Number(selectedBucket.modelProbability);
    if (Number.isNaN(modelProbability)) return null;
    const winProbability = ticket.side === "yes" ? modelProbability : 1 - modelProbability;
    const shares = stake / (price / 100);
    const grossPayout = shares;
    const profitIfRight = grossPayout - stake;
    return {
      shares,
      grossPayout,
      profitIfRight,
      lossIfWrong: stake,
      winProbability,
      edge: winProbability * 100 - price,
      ev: shares * winProbability - stake,
    };
  }, [effectivePrice, selectedBucket, ticket.side, ticket.stakeUsd]);
  const strategyGatePass =
    Boolean(estimated) &&
    estimated.winProbability >= MIN_TRADE_WIN_PROBABILITY &&
    estimated.edge >= MIN_TRADE_EDGE_PP;
  const tradeBlockedReason = !activeProfileTradeable
    ? "Validated as-of profile is unavailable."
    : !marketPriceAvailable
    ? "No market entry price."
    : !selectedPriceExecutable
    ? "Displayed price is not an executable CLOB ask."
    : !estimated
    ? "Ticket estimate unavailable."
    : !strategyGatePass
    ? `Needs edge >= ${number(MIN_TRADE_EDGE_PP, 1)} pp and win >= ${percent(MIN_TRADE_WIN_PROBABILITY)}.`
    : "";

  useEffect(() => {
    marketSelectionRef.current = { asOfProfile, selectedDate };
  }, [asOfProfile, selectedDate]);

  const isCurrentMarketSelection = useCallback((requestedDate, requestedProfile) => {
    const current = marketSelectionRef.current;
    return current.selectedDate === requestedDate && current.asOfProfile === requestedProfile;
  }, []);

  const resetMarketForSelectionChange = useCallback(() => {
    setMarket(null);
    setMessage("");
    setError("");
    setLoading((prev) => ({ ...prev, market: true }));
  }, []);

  const selectMarketDate = useCallback(
    (nextDate) => {
      resetMarketForSelectionChange();
      setSelectedDate(nextDate);
    },
    [resetMarketForSelectionChange],
  );

  const selectAsOfProfile = useCallback(
    (profileId) => {
      if (profileId === asOfProfile) return;
      resetMarketForSelectionChange();
      setAsOfProfile(profileId);
    },
    [asOfProfile, resetMarketForSelectionChange],
  );

  const loadProfiles = useCallback(async () => {
    try {
      const data = await apiFetch("/api/profiles");
      if (Array.isArray(data.profiles) && data.profiles.length) {
        setProfiles(data.profiles);
      }
    } catch (_exc) {
      setProfiles(FALLBACK_PROFILES);
    }
  }, []);

  const loadMarkets = useCallback(async () => {
    setLoading((prev) => ({ ...prev, markets: true }));
    setError("");
    try {
      const data = await apiFetch(`/api/markets?start=${windowStart}&end=${windowEnd}`);
      setMarkets(data.markets || []);
      if (!data.markets?.some((item) => item.targetDate === selectedDate) && data.markets?.[0]) {
        setSelectedDate(data.markets[0].targetDate);
      }
    } catch (exc) {
      setError(exc.message);
    } finally {
      setLoading((prev) => ({ ...prev, markets: false }));
    }
  }, [selectedDate, windowEnd, windowStart]);

  const loadMarket = useCallback(async () => {
    if (!selectedDate) return;
    const requestedDate = selectedDate;
    const requestedProfile = asOfProfile;
    setLoading((prev) => ({ ...prev, market: true }));
    setError("");
    try {
      const data = await apiFetch(`/api/markets/${requestedDate}?asOfProfile=${encodeURIComponent(requestedProfile)}`);
      if (!isCurrentMarketSelection(requestedDate, requestedProfile)) return;
      setMarket(data);
      if (data.marketRows?.length && !data.marketRows.some((row) => row.bucket === ticket.bucket)) {
        setTicket((prev) => ({ ...prev, bucket: data.marketRows[0].bucket }));
      }
    } catch (exc) {
      if (!isCurrentMarketSelection(requestedDate, requestedProfile)) return;
      setError(exc.message);
      setMarket(null);
    } finally {
      if (isCurrentMarketSelection(requestedDate, requestedProfile)) {
        setLoading((prev) => ({ ...prev, market: false }));
      }
    }
  }, [asOfProfile, isCurrentMarketSelection, selectedDate, ticket.bucket]);

  const loadAccount = useCallback(async () => {
    setLoading((prev) => ({ ...prev, account: true }));
    try {
      const data = await apiFetch(`/api/account?since=${since}`);
      setAccount(data);
    } catch (exc) {
      setError(exc.message);
    } finally {
      setLoading((prev) => ({ ...prev, account: false }));
    }
  }, [since]);

  useEffect(() => {
    loadProfiles();
  }, [loadProfiles]);

  useEffect(() => {
    loadMarkets();
  }, [loadMarkets]);

  useEffect(() => {
    loadMarket();
  }, [loadMarket]);

  useEffect(() => {
    loadAccount();
  }, [loadAccount]);

  const refreshEverything = useCallback(async () => {
    if (!selectedDate || autoRefreshInFlight.current) return;
    const requestedDate = selectedDate;
    const requestedProfile = asOfProfile;
    autoRefreshInFlight.current = true;
    setError("");
    try {
      const [marketData, marketsData, accountData] = await Promise.all([
        apiFetch(`/api/markets/${requestedDate}/refresh`, {
          method: "POST",
          body: JSON.stringify({ asOfProfile: requestedProfile }),
        }),
        apiFetch(`/api/markets?start=${windowStart}&end=${windowEnd}`),
        apiFetch(`/api/account?since=${since}`),
      ]);
      if (isCurrentMarketSelection(requestedDate, requestedProfile)) {
        setMarket(marketData);
      }
      setMarkets(marketsData.markets || []);
      if (!marketsData.markets?.some((item) => item.targetDate === selectedDate) && marketsData.markets?.[0]) {
        setMarket(null);
        setSelectedDate(marketsData.markets[0].targetDate);
      }
      setAccount(accountData);
      setLastRefreshAt(new Date().toISOString());
    } catch (exc) {
      setError(`Auto refresh failed: ${exc.message}`);
    } finally {
      autoRefreshInFlight.current = false;
    }
  }, [asOfProfile, isCurrentMarketSelection, selectedDate, since, windowEnd, windowStart]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      refreshEverything();
    }, AUTO_REFRESH_MS);
    return () => window.clearInterval(interval);
  }, [refreshEverything]);

  async function refreshMarket() {
    const requestedDate = selectedDate;
    const requestedProfile = asOfProfile;
    setLoading((prev) => ({ ...prev, market: true }));
    setMessage("");
    setError("");
    try {
      const data = await apiFetch(`/api/markets/${requestedDate}/refresh`, {
        method: "POST",
        body: JSON.stringify({ asOfProfile: requestedProfile }),
      });
      if (!isCurrentMarketSelection(requestedDate, requestedProfile)) return;
      setMarket(data);
      setMessage("Snapshot refreshed.");
      loadMarkets();
    } catch (exc) {
      if (!isCurrentMarketSelection(requestedDate, requestedProfile)) return;
      setError(exc.message);
    } finally {
      if (isCurrentMarketSelection(requestedDate, requestedProfile)) {
        setLoading((prev) => ({ ...prev, market: false }));
      }
    }
  }

  async function submitTrade(event) {
    event.preventDefault();
    setLoading((prev) => ({ ...prev, trade: true }));
    setMessage("");
    setError("");
    try {
      const payload = {
        targetDate: selectedDate,
        bucketKey: ticket.bucket,
        side: ticket.side,
        stakeUsd: Number(ticket.stakeUsd),
        asOfProfile,
      };
      const data = await apiFetch("/api/trades", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setAccount(data.account);
      if (data.trade?.id) {
        setSelectedTradeId(data.trade.id);
        setLedgerTab("details");
      }
      const entryTime = data.trade?.opened_at_utc ? ` Entry time ${formatEntryTimestamp(data.trade.opened_at_utc)}.` : "";
      setMessage(
        `Recorded demo ${ticket.side.toUpperCase()} trade on ${ticket.bucket} using ${activeProfile?.label || asOfProfile}.${entryTime}`,
      );
      loadMarkets();
      loadMarket();
    } catch (exc) {
      setError(exc.message);
    } finally {
      setLoading((prev) => ({ ...prev, trade: false }));
    }
  }

  async function settleTrades() {
    setMessage("");
    setError("");
    try {
      const data = await apiFetch("/api/settle", { method: "POST", body: JSON.stringify({}) });
      setAccount(data.account);
      if (data.settled?.[0]?.id) {
        setSelectedTradeId(data.settled[0].id);
        setLedgerTab("details");
      }
      setMessage(`Settlement pass complete. ${data.settled?.length || 0} trade(s) settled.`);
      loadMarkets();
    } catch (exc) {
      setError(exc.message);
    }
  }

  async function settleTradeAsWin(trade) {
    setLoading((prev) => ({ ...prev, settlingTradeId: trade.id }));
    setMessage("");
    setError("");
    try {
      const data = await apiFetch(`/api/trades/${trade.id}/settle-win`, {
        method: "POST",
        body: JSON.stringify({}),
      });
      setAccount(data.account);
      setSelectedTradeId(data.trade?.id || trade.id);
      setLedgerTab("details");
      setMessage(`Settled ${trade.side.toUpperCase()} ${trade.bucket_key} as a win.`);
      loadMarkets();
    } catch (exc) {
      setError(exc.message);
    } finally {
      setLoading((prev) => ({ ...prev, settlingTradeId: null }));
    }
  }

  async function settleTradeAsLoss(trade) {
    setLoading((prev) => ({ ...prev, settlingTradeId: trade.id }));
    setMessage("");
    setError("");
    try {
      const data = await apiFetch(`/api/trades/${trade.id}/settle-loss`, {
        method: "POST",
        body: JSON.stringify({}),
      });
      setAccount(data.account);
      setSelectedTradeId(data.trade?.id || trade.id);
      setLedgerTab("details");
      setMessage(`Settled ${trade.side.toUpperCase()} ${trade.bucket_key} as a loss.`);
      loadMarkets();
    } catch (exc) {
      setError(exc.message);
    } finally {
      setLoading((prev) => ({ ...prev, settlingTradeId: null }));
    }
  }

  async function resetAccount() {
    setMessage("");
    setError("");
    try {
      const data = await apiFetch("/api/account/reset", { method: "POST", body: JSON.stringify({}) });
      setAccount(data.account);
      setSelectedTradeId(null);
      setLedgerTab("open");
      setMessage("Started a fresh $1000 demo account.");
      loadMarkets();
    } catch (exc) {
      setError(exc.message);
    }
  }

  function selectTrade(row, side) {
    setTicket((prev) => ({
      ...prev,
      bucket: row.bucket,
      side,
    }));
  }

  const bestEdge = market?.bestEdge || null;
  const suggestedStakeUsd = account?.cashUsd ? Math.max(1, Number(account.cashUsd) * 0.05) : null;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">
            <ShieldCheck size={15} />
            Local demo ledger
          </div>
          <h1>HKG Tmax Polymarket backtester</h1>
        </div>
        <div className="topbar-actions">
          <label className="since-control">
            Since
            <input type="date" value={since} onChange={(event) => setSince(event.target.value)} />
          </label>
          <button className="ghost-button" onClick={settleTrades}>
            <CheckCircle2 size={16} />
            Settle
          </button>
          <button className="ghost-button danger" onClick={resetAccount}>
            <RotateCcw size={16} />
            Reset
          </button>
        </div>
      </header>

      <section className="kpi-strip" aria-label="Account summary">
        <Kpi icon={CircleDollarSign} label="Equity" value={dollars(account?.equityUsd)} trend={account?.totalPnlUsd} />
        <Kpi icon={Activity} label="Cash" value={dollars(account?.cashUsd)} />
        <Kpi icon={BarChart3} label="Total PnL" value={dollars(account?.totalPnlUsd)} trend={account?.totalPnlUsd} />
        <Kpi icon={CalendarDays} label="Open exposure" value={dollars(account?.openExposureUsd)} />
      </section>

      <section className="market-window">
        <div className="window-controls">
          <div className="date-inputs">
            <label>
              Start
              <input type="date" value={windowStart} onChange={(event) => setWindowStart(event.target.value)} />
            </label>
            <label>
              End
              <input type="date" value={windowEnd} onChange={(event) => setWindowEnd(event.target.value)} />
            </label>
            <button className="primary-button compact" onClick={loadMarkets}>
              Load
            </button>
          </div>
          <div className="pager">
            <IconButton title="Previous day" onClick={() => selectMarketDate((value) => shiftDate(value, -1))}>
              <ArrowLeft size={16} />
            </IconButton>
            <IconButton title="Next day" onClick={() => selectMarketDate((value) => shiftDate(value, 1))}>
              <ArrowRight size={16} />
            </IconButton>
          </div>
        </div>

        <div className="date-tabs" aria-label="Market dates">
          {loading.markets ? (
            <span className="inline-loading"><Loader2 size={15} /> Loading dates</span>
          ) : (
            markets.map((item) => (
              <button
                key={item.targetDate}
                className={item.targetDate === selectedDate ? "active" : ""}
                onClick={() => selectMarketDate(item.targetDate)}
              >
                <span>{dateLabel(item.targetDate)}</span>
                {item.settlement ? <small>settled</small> : item.openTradeCount ? <small>{item.openTradeCount} open</small> : <small>{item.snapshot?.status || "new"}</small>}
              </button>
            ))
          )}
        </div>

        <AsOfProfileSelector
          profiles={profiles}
          selectedProfile={asOfProfile}
          setSelectedProfile={selectAsOfProfile}
          market={market}
        />
      </section>

      {message ? <div className="toast success">{message}</div> : null}
      {error ? <div className="toast error">{error}</div> : null}

      <section className="market-grid">
        <div className="main-column">
          <section className="event-header">
            <div>
              <div className="eyebrow">
                <CalendarDays size={15} />
                {fullDateLabel(selectedDate)}
              </div>
              <h2>{market?.event?.title || "Highest temperature in Hong Kong"}</h2>
              <p>
                HKO Daily Extract settlement, one-decimal bucket mapping, cutoff-specific B4 probabilities, and
                executable CLOB ask checks.
              </p>
            </div>
            <div className="event-actions">
              <span className="auto-refresh-indicator" title="Market, account, and PnL refresh every 20 seconds">
                <Clock3 size={14} />
                Auto 20s
                <small>{formatTimeOnly(lastRefreshAt)}</small>
              </span>
              <StatusPill tone={market?.status === "ok" ? "good" : "warn"}>{market?.status || "loading"}</StatusPill>
              <button className="ghost-button" onClick={refreshMarket} disabled={loading.market}>
                {loading.market ? <Loader2 size={16} className="spin" /> : <RefreshCcw size={16} />}
                Refresh
              </button>
            </div>
          </section>

          <section className="info-strip">
            <div>
              <span>As-of profile</span>
              <strong>{activeProfile?.label || selectedProfile?.label || asOfProfile}</strong>
              <small>
                {activeProfile?.hktCutoff ? `${activeProfile.hktCutoff} HKT cutoff · ${activeProfile.warning || ""}` : "Awaiting profile"}
              </small>
            </div>
            <div>
              <span>Forecast</span>
              <strong>
                {market?.forecast ? `${number(forecastMaxC, 1)} deg C max` : "Unavailable"}
              </strong>
              <small>{market?.forecast?.source || statusReason || "Awaiting snapshot"}</small>
            </div>
            <div>
              <span>Model</span>
              <strong>{market?.model?.method || "B4 residual PMF"}</strong>
              <small>{market?.model ? `${modelTrainRows} train rows` : "No probability snapshot"}</small>
            </div>
            <div>
              <span>Best edge</span>
              <strong>
                {bestEdge ? `${bestEdge.side.toUpperCase()} ${bestEdge.bucket} ${number(bestEdge.edgePp, 1)} pp` : "-"}
              </strong>
              <small>{bestEdge?.executable ? bestEdge.classification : "No executable CLOB edge"}</small>
            </div>
          </section>

          {market ? (
            <section className="analysis-grid">
              <EngineSnapshotPanel market={market} />
              <EdgeStackPanel market={market} />
            </section>
          ) : null}

          <section className="bucket-section">
            <div className="section-title">
              <h3>Bucket ladder</h3>
              <span>Click YES or NO to stage a demo trade.</span>
            </div>
            {loading.market ? (
              <div className="table-loading"><Loader2 size={20} className="spin" /> Loading market snapshot</div>
            ) : market?.marketRows?.length ? (
              <div className="bucket-table-wrap">
                <table className="bucket-table">
                  <thead>
                    <tr>
                      <th>Bucket</th>
                      <th>Market</th>
                      <th>Model</th>
                      <th>YES</th>
                      <th>NO</th>
                      <th>Volume</th>
                    </tr>
                  </thead>
                  <tbody>
                    {market.marketRows.map((row) => (
                      <tr key={row.bucket} className={ticket.bucket === row.bucket ? "selected-row" : ""}>
                        <td>
                          <strong>{row.label}</strong>
                          <small>{row.status}</small>
                        </td>
                        <td>
                          <span>{row.active ? "Active" : row.closed ? "Closed" : "Listed"}</span>
                          <small>{row.acceptingOrders ? "accepting orders" : "manual ok"}</small>
                        </td>
                        <td>
                          <span>{percent(row.modelProbability)}</span>
                          <small>fair YES {cents(row.modelYesPct)}</small>
                        </td>
                        <td>
                          <button
                            className="trade-button yes"
                            disabled={!activeProfileTradeable || !row.marketBuyYesExecutable}
                            onClick={() => selectTrade(row, "yes")}
                          >
                            YES {cents(row.marketBuyYesC)}
                          </button>
                          <small className={Number(row.yesEdgePp) >= 0 ? "positive" : "negative"}>
                            {row.yesEdgePp === null ? "edge -" : `${number(row.yesEdgePp, 1)} pp`}
                          </small>
                          <small>{priceSourceLabel(row.marketBuyYesSource)}</small>
                        </td>
                        <td>
                          <button
                            className="trade-button no"
                            disabled={!activeProfileTradeable || !row.marketBuyNoExecutable}
                            onClick={() => selectTrade(row, "no")}
                          >
                            NO {cents(row.marketBuyNoC)}
                          </button>
                          <small className={Number(row.noEdgePp) >= 0 ? "positive" : "negative"}>
                            {row.noEdgePp === null ? "edge -" : `${number(row.noEdgePp, 1)} pp`}
                          </small>
                          <small>{priceSourceLabel(row.marketBuyNoSource)}</small>
                        </td>
                        <td>
                          <span>{dollars(row.volume)}</span>
                          <small>liq {dollars(row.liquidity)}</small>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <EmptyState title="No bucket rows" body="Refresh the market after source data is available." />
            )}
          </section>
        </div>

        <aside className="ticket-column">
          <form className="trade-ticket" onSubmit={submitTrade}>
            <div className="section-title">
              <h3>Trade ticket</h3>
              <StatusPill tone={ticket.side === "yes" ? "good" : "bad"}>{ticket.side.toUpperCase()}</StatusPill>
            </div>
            <label>
              Bucket
              <select value={ticket.bucket} onChange={(event) => setTicket((prev) => ({ ...prev, bucket: event.target.value }))}>
                {(market?.marketRows || []).map((row) => (
                  <option key={row.bucket} value={row.bucket}>
                    {row.label}
                  </option>
                ))}
              </select>
            </label>
            <div className="segmented">
              <button type="button" className={ticket.side === "yes" ? "active yes" : ""} onClick={() => setTicket((prev) => ({ ...prev, side: "yes" }))}>
                YES
              </button>
              <button type="button" className={ticket.side === "no" ? "active no" : ""} onClick={() => setTicket((prev) => ({ ...prev, side: "no" }))}>
                NO
              </button>
            </div>
            <label>
              Stake USD
              <div className="stake-control">
                <input
                  type="number"
                  min="1"
                  step="0.01"
                  value={ticket.stakeUsd}
                  onChange={(event) => setTicket((prev) => ({ ...prev, stakeUsd: event.target.value }))}
                />
                <button
                  type="button"
                  className="stake-quick-button"
                  disabled={!suggestedStakeUsd}
                  onClick={() =>
                    setTicket((prev) => ({
                      ...prev,
                      stakeUsd: suggestedStakeUsd ? suggestedStakeUsd.toFixed(2) : prev.stakeUsd,
                    }))
                  }
                >
                  5%
                </button>
              </div>
            </label>
            <div className={`profile-entry-banner ${activeProfileTradeable ? "" : "warn"}`}>
              <span>As-of profile</span>
              <strong>{activeProfile?.label || asOfProfile}</strong>
              <small>
                {activeProfile?.hktCutoff || "-"} HKT cutoff · {activeProfile?.validationStatus || "unknown"}
              </small>
            </div>
            <div className={`market-entry-banner ${marketPriceAvailable && selectedPriceExecutable ? "" : "warn"}`}>
              <span>Market entry</span>
              <strong>
                {marketPriceAvailable && selectedPriceExecutable
                  ? `BUY ${ticket.side.toUpperCase()} @ ${cents(effectivePrice)}`
                  : `BUY ${ticket.side.toUpperCase()} unavailable`}
              </strong>
              <small>{priceSourceLabel(selectedPriceSource)}</small>
            </div>
            <div className={`strategy-gate ${strategyGatePass ? "pass" : "warn"}`}>
              <span>Strategy gate</span>
              <strong>
                {estimated
                  ? `${percent(estimated.winProbability)} win · ${number(estimated.edge, 1)} pp edge`
                  : "Awaiting estimate"}
              </strong>
              <small>{strategyGatePass ? "Threshold met" : tradeBlockedReason || "Threshold not met"}</small>
            </div>
            <div className="ticket-meta">
              <div>
                <span>Market price</span>
                <strong>{effectivePrice ? cents(effectivePrice) : "-"}</strong>
              </div>
              <div>
                <span>Shares</span>
                <strong>{estimated ? number(estimated.shares, 2) : "-"}</strong>
              </div>
              <div>
                <span>Payout if right</span>
                <strong>{estimated ? dollars(estimated.grossPayout) : "-"}</strong>
              </div>
              <div>
                <span>Profit if right</span>
                <strong className="positive">{estimated ? dollars(estimated.profitIfRight) : "-"}</strong>
              </div>
              <div>
                <span>Loss if wrong</span>
                <strong className="negative">{estimated ? dollars(-estimated.lossIfWrong) : "-"}</strong>
              </div>
              <div>
                <span>Win prob</span>
                <strong>{estimated ? percent(estimated.winProbability) : "-"}</strong>
              </div>
              <div>
                <span>Edge</span>
                <strong className={estimated?.edge >= 0 ? "positive" : "negative"}>
                  {estimated ? `${number(estimated.edge, 1)} pp` : "-"}
                </strong>
              </div>
              <div>
                <span>EV</span>
                <strong className={estimated?.ev >= 0 ? "positive" : "negative"}>{estimated ? dollars(estimated.ev) : "-"}</strong>
              </div>
            </div>
            <button
              className="primary-button"
              type="submit"
              disabled={
                loading.trade ||
                market?.status !== "ok" ||
                !marketPriceAvailable ||
                !selectedPriceExecutable ||
                !estimated ||
                !activeProfileTradeable ||
                !strategyGatePass
              }
              title={tradeBlockedReason || "Record demo trade"}
            >
              {loading.trade ? <Loader2 size={16} className="spin" /> : <CircleDollarSign size={16} />}
              Buy {ticket.side.toUpperCase()} at market
            </button>
            <p className="ticket-note">
              Entries remain fictitious. The server records the exact entry timestamp, cutoff profile, forecast issue,
              model rows, CLOB quote, and strategy gate.
            </p>
          </form>
        </aside>
      </section>

      <section className="bottom-grid">
        <TradeLedgerPanel
          account={account}
          loading={loading}
          ledgerTab={ledgerTab}
          setLedgerTab={setLedgerTab}
          selectedTradeId={selectedTradeId}
          setSelectedTradeId={setSelectedTradeId}
          settleTradeAsWin={settleTradeAsWin}
          settleTradeAsLoss={settleTradeAsLoss}
        />

        <div className="panel">
          <div className="section-title">
            <h3>Performance</h3>
            <span>Since {since}</span>
          </div>
          <div className="performance-grid">
            <Kpi icon={BarChart3} label="Realized" value={dollars(account?.realizedPnlUsd)} trend={account?.realizedPnlUsd} />
            <Kpi icon={Activity} label="Unrealized" value={dollars(account?.unrealizedPnlUsd)} trend={account?.unrealizedPnlUsd} />
            <Kpi icon={CalendarDays} label="Change since" value={dollars(account?.changeSinceUsd)} trend={account?.changeSinceUsd} />
          </div>
          <div className="curve">
            {(account?.balanceCurve || []).map((point, index) => (
              <div className="curve-point" key={`${point.date}-${index}`}>
                <span>{point.date}</span>
                <strong>{dollars(point.equityUsd)}</strong>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}

export default App;
