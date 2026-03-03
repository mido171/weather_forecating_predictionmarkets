import { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";
import {
  Area,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const DATASET_OPTIONS = [
  {
    key: "2024-2025",
    label: "2024-2025",
    csv: "/data/all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_fractionalkellyceiling_risk7p5_kelly0p12_cap700_2024_2025_best_highreturn_smooth_highwin_stable_meanprice65_with_balance.csv",
    params: [
      "EV >= 0.25",
      "Win >= 0.85",
      "Risk ceiling 7.5%",
      "Kelly fraction 0.12",
      "Stake cap $700",
      "Side price >= 25c",
      "Mean entry <= 65c",
      "Entry >= max(T-1 12:00Z, open+30m)",
    ],
  },
  {
    key: "2026",
    label: "2026",
    csv: "/data/all_trades_sideaware_cojoined_blend12_knyc_kmia_tminus1_1200z_openplus30m_ev0p25_win85_minprice25c_fractionalkellyceiling_risk7p5_kelly0p12_cap700_live_script_2025oct_to_2026feb_with_balance.csv",
    params: [
      "Period: 2025-10-01 -> 2026-02-28",
      "EV >= 0.25",
      "Win >= 0.85",
      "Risk ceiling 7.5%",
      "Kelly fraction 0.12",
      "Stake cap $700",
      "Side price >= 25c",
      "Entry >= max(T-1 12:00Z, open+30m)",
      "Prediction source: live-script replay",
    ],
  },
];

const DEFAULT_DATASET_KEY = "2024-2025";

function resolveCsvOverride() {
  const envCsv = String(import.meta.env.VITE_TRADES_CSV_FILE ?? "").trim();
  if (typeof window === "undefined") {
    return envCsv || "";
  }
  const queryCsv = new URLSearchParams(window.location.search).get("csv");
  if (queryCsv && queryCsv.trim()) {
    return queryCsv.trim();
  }
  return envCsv || "";
}

function resolveInitialDatasetKey() {
  if (typeof window === "undefined") return DEFAULT_DATASET_KEY;
  const queryDataset = String(new URLSearchParams(window.location.search).get("dataset") ?? "").trim();
  if (queryDataset && DATASET_OPTIONS.some((x) => x.key === queryDataset)) {
    return queryDataset;
  }
  return DEFAULT_DATASET_KEY;
}

function resolveDatasetCsv(datasetKey) {
  return DATASET_OPTIONS.find((x) => x.key === datasetKey)?.csv ?? DATASET_OPTIONS[0].csv;
}

const COLUMNS = [
  "Target date (Local)",
  "Market file day (Local)",
  "Entry date (Stockholm)",
  "Entry time (Stockholm)",
  "Station",
  "Bucket",
  "Bucket raw (market)",
  "Side",
  "Market win % (side)",
  "Model win %",
  "EV",
  "Amount invested ($)",
  "Profit made ($)",
  "Result",
  "Balance after trade ($)",
  "Market open (UTC)",
  "Gate cutoff (UTC)",
  "Effective cutoff (UTC)",
];

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function toNumber(value) {
  if (value == null || value === "") return NaN;
  const parsed = Number(String(value).replace(/,/g, ""));
  return Number.isFinite(parsed) ? parsed : NaN;
}

function prettyNumber(value, digits = 2) {
  if (!Number.isFinite(value)) return "0.00";
  return value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function prettyCompact(value) {
  if (!Number.isFinite(value)) return "$0";
  const sign = value < 0 ? "-" : "";
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(1)}m`;
  if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(1)}k`;
  return `${sign}$${abs.toFixed(0)}`;
}

function pct(value, digits = 1) {
  if (!Number.isFinite(value)) return "0%";
  return `${(value * 100).toFixed(digits)}%`;
}

function monthLabel(monthKey) {
  if (!monthKey || !/^\d{4}-\d{2}$/.test(monthKey)) return monthKey ?? "";
  const [year, month] = monthKey.split("-");
  const monthIndex = Number(month) - 1;
  return `${MONTHS[monthIndex] ?? month} '${String(year).slice(2)}`;
}

function parseEntryMonth(entryTime) {
  if (!entryTime || entryTime.length < 7) return "";
  return entryTime.slice(0, 7);
}

function parseTradeDate(entryTime) {
  if (!entryTime || entryTime.length < 10) return "";
  return entryTime.slice(0, 10);
}

function parseStockholmEntryToUtcMillis(entryTime) {
  const s = String(entryTime ?? "").trim();
  const m = s.match(/^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}) (CET|CEST)$/);
  if (m) {
    const year = Number(m[1]);
    const month = Number(m[2]);
    const day = Number(m[3]);
    const hour = Number(m[4]);
    const minute = Number(m[5]);
    const second = Number(m[6]);
    const zone = m[7];
    const offsetMinutes = zone === "CEST" ? 120 : 60;
    return Date.UTC(year, month - 1, day, hour, minute, second) - offsetMinutes * 60 * 1000;
  }
  const fallback = Date.parse(s);
  return Number.isFinite(fallback) ? fallback : NaN;
}

function minutesAfterOpen(entryTimeStockholm, marketOpenUtc) {
  const entryMillis = parseStockholmEntryToUtcMillis(entryTimeStockholm);
  const openMillis = Date.parse(String(marketOpenUtc ?? "").trim());
  if (!Number.isFinite(entryMillis) || !Number.isFinite(openMillis)) return NaN;
  return (entryMillis - openMillis) / 60000;
}

function normalizeTargetDate(value) {
  const s = String(value ?? "").trim();
  return /^\d{4}-\d{2}-\d{2}$/.test(s) ? s : "";
}

function median(values) {
  const finite = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (!finite.length) return 0;
  const mid = Math.floor(finite.length / 2);
  if (finite.length % 2 === 1) return finite[mid];
  return (finite[mid - 1] + finite[mid]) / 2;
}

function computeMaxDrawdown(balanceSeries, startBalance) {
  let peak = startBalance;
  let maxDrawdown = 0;

  for (const value of balanceSeries) {
    if (!Number.isFinite(value)) continue;
    peak = Math.max(peak, value);
    if (peak <= 0) continue;
    const dd = (peak - value) / peak;
    maxDrawdown = Math.max(maxDrawdown, dd);
  }
  return maxDrawdown;
}

function computeStreaks(rows) {
  let maxWin = 0;
  let maxLoss = 0;
  let currentType = "";
  let currentLen = 0;

  for (const row of rows) {
    const type = String(row.Result ?? "").toLowerCase() === "win" ? "win" : "loss";
    if (type === currentType) {
      currentLen += 1;
    } else {
      currentType = type;
      currentLen = 1;
    }
    if (type === "win") maxWin = Math.max(maxWin, currentLen);
    if (type === "loss") maxLoss = Math.max(maxLoss, currentLen);
  }
  return { maxWin, maxLoss };
}

function buildDrawdownSeries(rows) {
  if (!rows.length) {
    return { series: [], peakIndex: null, troughIndex: null, troughPct: 0 };
  }

  const firstPnl = rows[0]["Profit made ($)"] || 0;
  const startBalance = (rows[0]["Balance after trade ($)"] || 0) - firstPnl;

  let runningPeak = startBalance;
  let runningPeakIndex = 0;
  let troughPct = 0;
  let troughIndex = 0;
  let troughPeakIndex = 0;

  const series = rows.map((row, idx) => {
    const balance = row["Balance after trade ($)"] || 0;
    if (balance >= runningPeak) {
      runningPeak = balance;
      runningPeakIndex = idx + 1;
    }

    const ddPct = runningPeak > 0 ? ((balance - runningPeak) / runningPeak) * 100 : 0;
    if (ddPct < troughPct) {
      troughPct = ddPct;
      troughIndex = idx + 1;
      troughPeakIndex = runningPeakIndex;
    }

    return {
      idx: idx + 1,
      drawdownPct: ddPct,
      balance,
      peakBalance: runningPeak,
    };
  });

  return {
    series,
    peakIndex: troughPeakIndex || null,
    troughIndex: troughIndex || null,
    troughPct,
  };
}

function App() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedMonth, setSelectedMonth] = useState("ALL");
  const [selectedDataset, setSelectedDataset] = useState(resolveInitialDatasetKey);

  const csvOverride = useMemo(() => resolveCsvOverride(), []);
  const csvFile = useMemo(() => {
    if (csvOverride) return csvOverride;
    return resolveDatasetCsv(selectedDataset);
  }, [csvOverride, selectedDataset]);
  const datasetLabel = useMemo(
    () => DATASET_OPTIONS.find((x) => x.key === selectedDataset)?.label ?? selectedDataset,
    [selectedDataset],
  );
  const datasetParams = useMemo(
    () => DATASET_OPTIONS.find((x) => x.key === selectedDataset)?.params ?? [],
    [selectedDataset],
  );

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError("");
      setSelectedMonth("ALL");
      try {
        const text = await fetch(csvFile).then((r) => {
          if (!r.ok) {
            throw new Error(`Failed to fetch CSV (${r.status}) at ${csvFile}`);
          }
          return r.text();
        });
        const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
        if (parsed.errors?.length) {
          throw new Error(parsed.errors[0].message);
        }
        const normalized = parsed.data.map((row, idx) => {
          const targetDate = normalizeTargetDate(row["Target date (Local)"]);
          const monthKey = targetDate ? targetDate.slice(0, 7) : parseEntryMonth(row["Entry time (Stockholm)"]);
          const tradeDate = targetDate || parseTradeDate(row["Entry time (Stockholm)"]);
          return {
            id: idx + 1,
            ...row,
            monthKey,
            monthLabel: monthLabel(monthKey),
            tradeDate,
            "Market win % (side)": toNumber(row["Market win % (side)"]),
            "Model win %": toNumber(row["Model win %"]),
            EV: toNumber(row["EV"]),
            "Amount invested ($)": toNumber(row["Amount invested ($)"]),
            "Profit made ($)": toNumber(row["Profit made ($)"]),
            "Balance after trade ($)": toNumber(row["Balance after trade ($)"]),
          };
        });
        setRows(normalized);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [csvFile]);

  const summary = useMemo(() => {
    if (!rows.length) {
      return {
        trades: 0,
        wins: 0,
        losses: 0,
        winRate: 0,
        totalPnl: 0,
        totalStake: 0,
        finalBalance: 0,
        startBalance: 0,
        peakBalance: 0,
        roi: 0,
        profitFactor: 0,
        avgRR: 0,
        meanEntryPricePct: 0,
        mmeMinutes: 0,
        avgEv: 0,
        medianEv: 0,
        bestWin: 0,
        maxDrawdown: 0,
        stationCounts: {},
        streaks: { maxWin: 0, maxLoss: 0 },
      };
    }

    const wins = rows.filter((r) => String(r.Result).toLowerCase() === "win").length;
    const losses = rows.filter((r) => String(r.Result).toLowerCase() === "loss").length;
    const winRate = rows.length ? wins / rows.length : 0;
    const totalPnl = rows.reduce((sum, r) => sum + (r["Profit made ($)"] || 0), 0);
    const totalStake = rows.reduce((sum, r) => sum + (r["Amount invested ($)"] || 0), 0);
    const finalBalance = rows[rows.length - 1]["Balance after trade ($)"] || 0;
    const firstPnl = rows[0]["Profit made ($)"] || 0;
    const startBalance = (rows[0]["Balance after trade ($)"] || 0) - firstPnl;
    const peakBalance = rows.reduce(
      (max, r) => Math.max(max, r["Balance after trade ($)"] || 0),
      Number.NEGATIVE_INFINITY
    );
    const roi = startBalance > 0 ? (finalBalance - startBalance) / startBalance : 0;
    const evValues = rows.map((r) => r.EV);
    const avgEv = evValues.reduce((sum, v) => sum + (Number.isFinite(v) ? v : 0), 0) / rows.length;
    const medianEv = median(evValues);
    const grossProfit = rows.reduce((sum, r) => sum + Math.max(0, r["Profit made ($)"] || 0), 0);
    const grossLossAbs = rows.reduce((sum, r) => sum + Math.max(0, -(r["Profit made ($)"] || 0)), 0);
    const profitFactor = grossLossAbs > 0 ? grossProfit / grossLossAbs : 0;
    const avgWin = wins > 0 ? grossProfit / wins : 0;
    const avgLossAbs = losses > 0 ? grossLossAbs / losses : 0;
    const avgRR = avgLossAbs > 0 ? avgWin / avgLossAbs : 0;
    const meanEntryPricePct =
      rows.reduce((sum, r) => sum + (Number.isFinite(r["Market win % (side)"]) ? r["Market win % (side)"] : 0), 0) /
      rows.length;
    const mmeValues = rows
      .map((r) => minutesAfterOpen(r["Entry time (Stockholm)"], r["Market open (UTC)"]))
      .filter((v) => Number.isFinite(v) && v >= 0);
    const mmeMinutes = median(mmeValues);
    const bestWin = rows.reduce((max, r) => Math.max(max, r["Profit made ($)"] || 0), Number.NEGATIVE_INFINITY);
    const maxDrawdown = computeMaxDrawdown(
      rows.map((r) => r["Balance after trade ($)"]),
      startBalance
    );
    const stationCounts = rows.reduce((acc, row) => {
      const key = row.Station || "Unknown";
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    }, {});
    const streaks = computeStreaks(rows);

    return {
      trades: rows.length,
      wins,
      losses,
      winRate,
      totalPnl,
      totalStake,
      finalBalance,
      startBalance,
      peakBalance,
      roi,
      profitFactor,
      avgRR,
      meanEntryPricePct,
      mmeMinutes,
      avgEv,
      medianEv,
      bestWin,
      maxDrawdown,
      stationCounts,
      streaks,
    };
  }, [rows]);

  const monthlyData = useMemo(() => {
    const bucket = new Map();
    for (const row of rows) {
      const key = row.monthKey || "Unknown";
      if (!bucket.has(key)) {
        bucket.set(key, {
          monthKey: key,
          monthLabel: row.monthLabel || key,
          pnl: 0,
          wins: 0,
          losses: 0,
          trades: 0,
        });
      }
      const item = bucket.get(key);
      item.pnl += row["Profit made ($)"] || 0;
      item.trades += 1;
      if (String(row.Result).toLowerCase() === "win") item.wins += 1;
      else item.losses += 1;
    }
    return [...bucket.values()]
      .sort((a, b) => a.monthKey.localeCompare(b.monthKey))
      .map((m) => ({
        ...m,
        winRate: m.trades ? m.wins / m.trades : 0,
      }));
  }, [rows]);

  const filteredRows = useMemo(() => {
    if (selectedMonth === "ALL") return rows;
    return rows.filter((r) => r.monthKey === selectedMonth);
  }, [rows, selectedMonth]);

  const filteredSummary = useMemo(() => {
    const wins = filteredRows.filter((r) => String(r.Result).toLowerCase() === "win").length;
    const losses = filteredRows.filter((r) => String(r.Result).toLowerCase() === "loss").length;
    const pnl = filteredRows.reduce((sum, r) => sum + (r["Profit made ($)"] || 0), 0);
    return {
      trades: filteredRows.length,
      wins,
      losses,
      winRate: filteredRows.length ? wins / filteredRows.length : 0,
      pnl,
    };
  }, [filteredRows]);

  const tradePnlSeries = useMemo(
    () =>
      filteredRows.map((row, idx) => ({
        idx: idx + 1,
        pnl: row["Profit made ($)"] || 0,
      })),
    [filteredRows]
  );

  const equitySeries = useMemo(
    () =>
      filteredRows.map((row, idx) => ({
        idx: idx + 1,
        tradeDate: row.tradeDate || "",
        balance: row["Balance after trade ($)"] || 0,
      })),
    [filteredRows]
  );

  const drawdownChart = useMemo(() => buildDrawdownSeries(rows), [rows]);

  const monthRangeText = useMemo(() => {
    if (!monthlyData.length) return "";
    const first = monthlyData[0].monthLabel;
    const last = monthlyData[monthlyData.length - 1].monthLabel;
    return `${first} -> ${last}`;
  }, [monthlyData]);

  if (loading) {
    return <div className="page loading">Loading trade results...</div>;
  }

  if (error) {
    return <div className="page error">Failed to load data: {error}</div>;
  }

  return (
    <div className="page">
      <div className="ambient ambientOne" />
      <div className="ambient ambientTwo" />

      <header className="topHeader">
        <div>
          <p className="microTitle">Strategy B - Filtered Portfolio</p>
          <div className="datasetToggleBar" role="tablist" aria-label="Backtest window">
            {DATASET_OPTIONS.map((option) => (
              <button
                key={option.key}
                type="button"
                role="tab"
                aria-selected={selectedDataset === option.key}
                className={`datasetToggleBtn ${selectedDataset === option.key ? "active" : ""}`}
                onClick={() => setSelectedDataset(option.key)}
                disabled={Boolean(csvOverride)}
              >
                {option.label}
              </button>
            ))}
          </div>
          {csvOverride ? <p className="datasetOverrideNote">CSV override active via query/env.</p> : null}
          <h1>
            Equity <span>Tracker</span>
          </h1>
          <p className="subLine">
            KNYC + KMIA · Window {datasetLabel} · {monthRangeText} · {summary.trades} trades
          </p>
          <div className="paramChips">
            {datasetParams.map((item) => (
              <span key={item} className="paramChip">
                {item}
              </span>
            ))}
          </div>
        </div>
        <div className="headlineStat">
          <div className="headlineValue">{prettyCompact(summary.finalBalance)}</div>
          <div className="headlineSub">
            from ${prettyNumber(summary.startBalance, 0)} · {pct(summary.roi)} return
          </div>
        </div>
      </header>

      <section className="heroGrid">
        <article className="winCard panel">
          <p className="cardLabel">Win Rate</p>
          <div
            className="ring"
            style={{
              "--ring-percent": `${Math.min(100, Math.max(0, summary.winRate * 100))}%`,
            }}
          >
            <div className="ringInner">{pct(summary.winRate)}</div>
          </div>
          <div className="wlText">
            {summary.wins}W - {summary.losses}L
          </div>
          <div className="tradeCount">of {summary.trades} trades</div>
          <div className="wlBar">
            <div className="wlWin" style={{ width: `${summary.winRate * 100}%` }} />
          </div>
          <div className="wlLegend">
            <span className="winText">{summary.wins} wins</span>
            <span className="lossText">{summary.losses} losses</span>
          </div>
        </article>

        <article className="metricGrid">
          <div className="panel metricCard">
            <p className="cardLabel">Net P&L</p>
            <h3 className="green">{prettyCompact(summary.totalPnl)}</h3>
            <p>staked ${prettyNumber(summary.totalStake, 0)}</p>
          </div>

          <div className="panel metricCard">
            <p className="cardLabel">MME</p>
            <h3>{prettyNumber(summary.mmeMinutes, 1)}m</h3>
            <p>median mins after open</p>
          </div>

          <div className="panel metricCard">
            <p className="cardLabel">Peak Balance</p>
            <h3 className="violet">{prettyCompact(summary.peakBalance)}</h3>
            <p>equity high-water mark</p>
          </div>

          <div className="panel metricCard">
            <p className="cardLabel">Total ROI</p>
            <h3 className="green">{pct(summary.roi)}</h3>
            <p>{monthlyData.length} months</p>
          </div>

          <div className="panel metricCard">
            <p className="cardLabel">Profit Factor</p>
            <h3>{summary.profitFactor.toFixed(2)}</h3>
            <p>gross profit / gross loss</p>
          </div>

          <div className="panel metricCard">
            <p className="cardLabel">Avg R/R</p>
            <h3>{summary.avgRR.toFixed(2)}</h3>
            <p>avg win / avg loss</p>
          </div>

          <div className="panel metricCard">
            <p className="cardLabel">Avg EV</p>
            <h3 className="amber">{summary.avgEv.toFixed(3)}</h3>
            <p>edge {(summary.avgEv * 100).toFixed(1)}pp</p>
          </div>

          <div className="panel metricCard">
            <p className="cardLabel">Median EV</p>
            <h3 className="amber">{summary.medianEv.toFixed(3)}</h3>
            <p>50th percentile edge</p>
          </div>

          <div className="panel metricCard">
            <p className="cardLabel">Mean Entry Price</p>
            <h3 className="amber">{summary.meanEntryPricePct.toFixed(2)}%</h3>
            <p>{summary.meanEntryPricePct.toFixed(2)}c on $1</p>
          </div>

          <div className="panel metricCard">
            <p className="cardLabel">Best Win</p>
            <h3 className="green">{prettyCompact(summary.bestWin)}</h3>
            <p>single trade</p>
          </div>

          <div className="panel metricCard">
            <p className="cardLabel">Streaks</p>
            <h3>
              {summary.streaks.maxWin}W / {summary.streaks.maxLoss}L
            </h3>
            <p>best / worst</p>
          </div>

          <div className="panel drawdownCard">
            <p className="cardLabel">Max Drawdown</p>
            <div className="drawdownRow">
              <div className="drawdownValue">-{pct(summary.maxDrawdown)}</div>
              <div className="drawdownTrack">
                <div className="drawdownFill" style={{ width: `${summary.maxDrawdown * 100}%` }} />
              </div>
            </div>
            <p>peak-to-trough</p>
          </div>
        </article>
      </section>

      <section className="panel breakdownPanel fullWidthPanel">
        <p className="cardLabel">Monthly Breakdown (click to filter)</p>
        <div className="breakdownList">
          <button
            type="button"
            className={`monthRow ${selectedMonth === "ALL" ? "selected" : ""}`}
            onClick={() => setSelectedMonth("ALL")}
          >
            <div className="monthTitle">All Months</div>
            <div className="monthMeta">{summary.trades} trades</div>
            <div className="monthBadge">{pct(summary.winRate)}</div>
            <div className={`monthPnl ${summary.totalPnl >= 0 ? "greenText" : "redText"}`}>
              {prettyCompact(summary.totalPnl)}
            </div>
          </button>

          {monthlyData.map((month) => (
            <button
              type="button"
              key={month.monthKey}
              className={`monthRow ${selectedMonth === month.monthKey ? "selected" : ""}`}
              onClick={() => setSelectedMonth(month.monthKey)}
            >
              <div className="monthTitle">{month.monthLabel}</div>
              <div className="monthMeta">
                {month.wins}W-{month.losses}L
              </div>
              <div className="monthBarTrack">
                <div className="monthBarFill" style={{ width: `${month.winRate * 100}%` }} />
              </div>
              <div className="monthBadge">{pct(month.winRate)}</div>
              <div className={`monthPnl ${month.pnl >= 0 ? "greenText" : "redText"}`}>
                {prettyCompact(month.pnl)}
              </div>
            </button>
          ))}
        </div>
      </section>

      <section className="chartStack">
        <article className="panel chartPanel chartPanelWide">
          <p className="cardLabel">Monthly P&L Trend</p>
          <div className="chartWrap">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={monthlyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a2555" />
                <XAxis dataKey="monthLabel" stroke="#9aa9da" tick={{ fontSize: 12 }} />
                <YAxis stroke="#9aa9da" tick={{ fontSize: 12 }} />
                <Tooltip
                  cursor={{ stroke: "#7b86b8", strokeDasharray: "4 3" }}
                  contentStyle={{
                    background: "#08112f",
                    border: "1px solid #1d2d6d",
                    borderRadius: "10px",
                    color: "#b9c4f2",
                  }}
                  formatter={(value) => `$${prettyNumber(Number(value))}`}
                />
                <ReferenceLine y={0} stroke="#3e4a7a" />
                <Line type="monotone" dataKey="pnl" stroke="#82a0ff" strokeWidth={2.25} dot={{ r: 2.5 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel chartPanel chartPanelWide">
          <p className="cardLabel">Equity Curve (After Each Trade)</p>
          <div className="chartWrap equityCurveWrap">
            <ResponsiveContainer width="100%" height={430}>
              <LineChart data={equitySeries}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a2555" />
                <XAxis
                  dataKey="tradeDate"
                  stroke="#9aa9da"
                  tick={{ fontSize: 12 }}
                  interval="preserveStartEnd"
                  minTickGap={30}
                  tickFormatter={(value) =>
                    typeof value === "string" && value.length >= 10 ? value.slice(5) : String(value ?? "")
                  }
                />
                <YAxis stroke="#9aa9da" tick={{ fontSize: 12 }} />
                <Tooltip
                  cursor={{ stroke: "#7b86b8", strokeDasharray: "4 3" }}
                  contentStyle={{
                    background: "#08112f",
                    border: "1px solid #1d2d6d",
                    borderRadius: "10px",
                    color: "#b9c4f2",
                  }}
                  formatter={(value) => `$${prettyNumber(Number(value))}`}
                  labelFormatter={(value) => `Date ${value}`}
                />
                <Line type="monotone" dataKey="balance" stroke="#20c997" strokeWidth={2.25} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel chartPanel chartPanelWide">
          <p className="cardLabel">Trade P&L (Per Trade)</p>
          <div className="chartWrap">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={tradePnlSeries}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a2555" />
                <XAxis dataKey="idx" stroke="#9aa9da" tick={{ fontSize: 12 }} />
                <YAxis stroke="#9aa9da" tick={{ fontSize: 12 }} />
                <Tooltip
                  cursor={{ stroke: "#7b86b8", strokeDasharray: "4 3" }}
                  contentStyle={{
                    background: "#08112f",
                    border: "1px solid #1d2d6d",
                    borderRadius: "10px",
                    color: "#b9c4f2",
                  }}
                  formatter={(value) => `$${prettyNumber(Number(value))}`}
                  labelFormatter={(value) => `Trade #${value}`}
                />
                <ReferenceLine y={0} stroke="#3e4a7a" />
                <Line type="monotone" dataKey="pnl" stroke="#f0b429" strokeWidth={2.2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel chartPanel chartPanelWide drawdownCurvePanel">
          <p className="cardLabel">Drawdown From Peak</p>
          <div className="chartWrap drawdownCurveWrap">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={drawdownChart.series}>
                <defs>
                  <linearGradient id="ddFillGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#ff4f7d" stopOpacity={0.22} />
                    <stop offset="100%" stopColor="#ff4f7d" stopOpacity={0.03} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a2555" />
                <XAxis dataKey="idx" stroke="#9aa9da" tick={{ fontSize: 12 }} />
                <YAxis
                  stroke="#9aa9da"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => `${Number(value).toFixed(0)}%`}
                />
                <Tooltip
                  cursor={{ stroke: "#7b86b8", strokeDasharray: "4 3" }}
                  contentStyle={{
                    background: "#08112f",
                    border: "1px solid #1d2d6d",
                    borderRadius: "10px",
                    color: "#b9c4f2",
                  }}
                  formatter={(value, key) => {
                    if (key === "drawdownPct") return [`${Number(value).toFixed(2)}%`, "Drawdown"];
                    if (key === "balance") return [`$${prettyNumber(Number(value))}`, "Balance"];
                    if (key === "peakBalance") return [`$${prettyNumber(Number(value))}`, "Peak"];
                    return [value, key];
                  }}
                  labelFormatter={(value) => `Trade #${value}`}
                />
                <ReferenceLine y={0} stroke="#54649d" />
                <Area type="monotone" dataKey="drawdownPct" stroke="none" fill="url(#ddFillGradient)" />
                <Line type="monotone" dataKey="drawdownPct" stroke="#ff4f7d" strokeWidth={2.2} dot={false} />
                {drawdownChart.peakIndex ? (
                  <ReferenceDot
                    x={drawdownChart.peakIndex}
                    y={0}
                    r={5}
                    fill="#22d3a2"
                    stroke="#0a102e"
                    strokeWidth={2}
                    label={{ value: "Peak", position: "top", fill: "#8de7cc", fontSize: 12 }}
                  />
                ) : null}
                {drawdownChart.troughIndex ? (
                  <ReferenceDot
                    x={drawdownChart.troughIndex}
                    y={drawdownChart.troughPct}
                    r={5}
                    fill="#ff4f7d"
                    stroke="#0a102e"
                    strokeWidth={2}
                    label={{ value: "Trough", position: "bottom", fill: "#ff86a8", fontSize: 12 }}
                  />
                ) : null}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="drawdownMeta">
            <span>
              Peak trade: <strong>#{drawdownChart.peakIndex ?? "-"}</strong>
            </span>
            <span>
              Trough trade: <strong>#{drawdownChart.troughIndex ?? "-"}</strong>
            </span>
            <span className="redText">
              Depth: <strong>{drawdownChart.troughPct.toFixed(2)}%</strong>
            </span>
          </div>
        </article>
      </section>

      <section className="panel tradeLogPanel">
        <div className="tradeLogTop">
          <p className="cardLabel">Trade Log</p>
          <div className="tradeLogStats">
            <span>{pct(filteredSummary.winRate)} WR</span>
            <span>
              {filteredSummary.wins}W {filteredSummary.losses}L
            </span>
            <span className={filteredSummary.pnl >= 0 ? "greenText" : "redText"}>
              ({prettyCompact(filteredSummary.pnl)})
            </span>
          </div>
        </div>
        <div className="tickStrip">
          {filteredRows.map((row) => (
            <span
              key={row.id}
              className={`tick ${String(row.Result).toLowerCase() === "win" ? "tickWin" : "tickLoss"}`}
              title={`${row["Entry time (Stockholm)"]} | ${row.Station} | ${row.Result} | $${prettyNumber(
                row["Profit made ($)"]
              )}`}
            />
          ))}
        </div>
      </section>

      <section className="panel tablePanel">
        <h2 className="tableTitle">
          Full Trade Table {selectedMonth === "ALL" ? "" : `(${monthLabel(selectedMonth)})`}
        </h2>
        <div className="tableWrap">
          <table>
            <thead>
              <tr>
                {COLUMNS.map((col) => (
                  <th key={col}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredRows.map((row) => {
                const isWin = String(row.Result).toLowerCase() === "win";
                return (
                  <tr key={row.id} className={isWin ? "rowWin" : "rowLoss"}>
                    {COLUMNS.map((col) => {
                      const value = row[col];
                      let cellClassName = "";

                      if (col === "Result") {
                        cellClassName = isWin ? "resultWin" : "resultLoss";
                      } else if (col === "Profit made ($)") {
                        cellClassName = Number(value) >= 0 ? "pnlWin" : "pnlLoss";
                      }

                      if (
                        col === "Market win % (side)" ||
                        col === "Model win %" ||
                        col === "EV" ||
                        col === "Amount invested ($)" ||
                        col === "Profit made ($)" ||
                        col === "Balance after trade ($)"
                      ) {
                        const digits = col === "EV" ? 4 : 2;
                        return (
                          <td key={col} className={cellClassName}>
                            {prettyNumber(Number(value), digits)}
                          </td>
                        );
                      }

                      return (
                        <td key={col} className={cellClassName}>
                          {String(value ?? "")}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

export default App;
