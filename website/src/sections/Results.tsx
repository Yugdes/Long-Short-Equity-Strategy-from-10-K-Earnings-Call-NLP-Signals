import { useState } from "react";
import Plot from "../components/Plot";
import { Section } from "./Static";
import { Row, Series, cssVar, fx, pct, useData, useTheme } from "../lib";

type Strategy = {
  cum_gross: Series; cum_net25: Series; cum_long_ex: Series; cum_short_ex: Series; cum_market_ex: Series;
  drawdown: Series; decile_bars: Record<string, number[]>; variants: Row[]; alphas: Row[];
  ic_primary: Series; ic_controls: Series; test_start: string; test_end: string; unlocked: boolean;
  variants_test?: Row[]; alphas_test?: Row[];
};
type Models = { ladder_dev: Row[]; ladder_final?: Row[]; single_sorts_dev: Row[]; fama_macbeth_dev: Row[]; vol_prediction_dev: Row[] };

const SET_LABEL: Record<string, string> = { C: "Controls only", "C+R": "+ risk text", "C+R+M": "+ marketing", ALL: "+ interactions (all)", TEXT: "Text only, no controls" };
const MODEL_LABEL: Record<string, string> = { enet: "Elastic net", lgbm: "LightGBM", xgb: "XGBoost", mlp: "Neural net", ens: "Ensemble" };
const VARIANT_LABEL = (v: string) => {
  const [q, w, n] = v.split("_");
  return `${q.startsWith("d") ? "Deciles" : "Quintiles"}, ${w === "vw" ? "value-weighted" : "equal-weighted"}${n ? ", sector-neutral" : ""}`;
};
const tCls = (t: unknown) => (typeof t === "number" && Math.abs(t) >= 3 ? "sig" : typeof t === "number" && Math.abs(t) < 2 ? "weak" : "");

function testShape(start: string, end: string, unlocked: boolean) {
  return unlocked ? [{ type: "rect", xref: "x", yref: "paper", x0: start, x1: end, y0: 0, y1: 1, fillcolor: cssVar("--paper-2"), line: { width: 0 }, layer: "below" }] : [];
}

function VariantTable({ rows, caption }: { rows: Row[]; caption: string }) {
  return (
    <div className="table-wrap">
      <table>
        <caption>{caption}</caption>
        <thead><tr><th>Construction</th><th>Ann. mean</th><th>Vol</th><th>Sharpe</th><th>Sortino</th><th>Max DD</th><th>Calmar</th><th>Turnover / mo</th>
          <th>α t</th><th>α t, + low-vol</th><th>Net 10 bps</th><th>Net 25 bps</th><th>Net 50 bps</th><th>Break-even</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.variant as string} className={r.variant === "d10_vw" ? "primary" : ""}>
              <td>{VARIANT_LABEL(r.variant as string)}</td><td>{pct(r.ann_mean as number)}</td><td>{pct(r.ann_vol as number)}</td><td>{fx(r.sharpe as number)}</td>
              <td>{fx(r.sortino as number)}</td><td>{pct(r.max_dd as number)}</td><td>{fx(r.calmar as number)}</td><td>{pct(r.turnover_m as number, 0)}</td>
              <td className={tCls(r.t_alpha)}>{fx(r.t_alpha as number)}</td><td className={tCls(r.t_alpha_lv)}>{fx(r.t_alpha_lv as number)}</td>
              <td>{fx(r.net10_sharpe as number)}</td><td>{fx(r.net25_sharpe as number)}</td><td>{fx(r.net50_sharpe as number)}</td>
              <td>{(r.breakeven_bps as number) > 0 ? `${fx(r.breakeven_bps as number, 0)} bps` : "none"}</td>
            </tr>))}
        </tbody>
      </table>
    </div>
  );
}

function AlphaTables({ rows, label }: { rows: Row[]; label: string }) {
  const ls = rows.filter((r) => (r.series === "LS net 25bps" || r.series === "LS gross") && ["CAPM", "FF5+Mom", "FF5+Mom+LowVol"].includes(String(r.model)));
  const legs = rows.filter((r) => String(r.series).includes("leg") && ["CAPM", "FF5+Mom", "FF5+Mom+LowVol"].includes(String(r.model)));
  return (
    <div className="cols-2">
      <div className="table-wrap">
        <table>
          <caption>{label}: primary long-short on factor returns, Newey-West (6 lags) t-statistics.</caption>
          <thead><tr><th>Series</th><th>Model</th><th>α / yr</th><th>t</th><th>Mkt β</th><th>Mom β</th><th>Low-vol β</th><th>adj. R²</th></tr></thead>
          <tbody>{ls.map((r, i) => (
            <tr key={i}><td>{String(r.series).replace("LS ", "").replace("25bps", "25 bps")}</td><td>{r.model as string}</td><td>{pct(r.alpha_ann as number)}</td>
              <td className={tCls(r.t_alpha)}>{fx(r.t_alpha as number)}</td><td>{fx(r.b_mktrf as number)}</td><td>{fx(r.b_mom as number)}</td><td>{fx(r.b_lowvol as number)}</td><td>{fx(r.r2_adj as number)}</td></tr>))}
          </tbody>
        </table>
      </div>
      <div className="table-wrap">
        <table>
          <caption>{label}: each leg's excess return. A negative short-leg alpha means shorting those stocks added value.</caption>
          <thead><tr><th>Leg</th><th>Model</th><th>α / yr</th><th>t</th><th>Mkt β</th></tr></thead>
          <tbody>{legs.map((r, i) => (
            <tr key={i}><td>{String(r.series).replace(" (excess)", "")}</td><td>{r.model as string}</td><td>{pct(r.alpha_ann as number)}</td>
              <td className={tCls(r.t_alpha)}>{fx(r.t_alpha as number)}</td><td>{fx(r.b_mktrf as number)}</td></tr>))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function LadderTable({ rows, caption }: { rows: Row[]; caption: string }) {
  const sorted = [...rows].sort((a, b) => (b.ic_mean as number) - (a.ic_mean as number));
  return (
    <div className="table-wrap">
      <table>
        <caption>{caption}</caption>
        <thead><tr><th>Features</th><th>Model</th><th>Mean IC</th><th>IC t (NW)</th><th>IC &gt; 0</th><th>OOS R²</th><th>Hit rate</th><th>L/S Sharpe</th><th>FF5+Mom α</th><th>α t</th></tr></thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={`${r.set}${r.model}`} className={r.set === "ALL" && r.model === "ens" ? "primary" : ""}>
              <td>{SET_LABEL[r.set as string]}</td><td>{MODEL_LABEL[r.model as string]}</td>
              <td>{fx(r.ic_mean as number, 4)}</td><td className={tCls(r.ic_t_nw)}>{fx(r.ic_t_nw as number)}</td>
              <td>{pct(r.ic_pos_months as number, 0)}</td><td>{pct(r.oos_r2_rank as number, 2)}</td><td>{pct(r.hit_rate as number, 1)}</td>
              <td>{fx(r.vw_sharpe as number)}</td><td>{pct(r.vw_alpha_ann as number)}</td><td className={tCls(r.vw_t_alpha)}>{fx(r.vw_t_alpha as number)}</td>
            </tr>))}
        </tbody>
      </table>
    </div>
  );
}

export function Ladder() {
  const m = useData<Models>("models");
  if (!m) return null;
  const note = "IC = monthly Spearman correlation of prediction with next-month return rank; R² against a zero forecast of the centred rank; long-short = capped value-weighted top minus bottom decile, gross.";
  return (
    <>
      <LadderTable rows={m.ladder_dev} caption={`Out-of-sample model ladder, development window 2014–2021 (${m.ladder_dev[0]?.n_months ?? "–"} months). ${note}`} />
      {m.ladder_final && <LadderTable rows={m.ladder_final} caption={`The frozen models on the locked test window (from 2022-01, ${m.ladder_final[0]?.n_months ?? "–"} months), run once after the design was frozen.`} />}
    </>
  );
}

export function SingleSignals() {
  const m = useData<Models>("models");
  if (!m) return null;
  const fm = Object.fromEntries(m.fama_macbeth_dev.filter((r) => r.spec === "with_controls").map((r) => [r.signal as string, r]));
  const vw = m.single_sorts_dev.filter((r) => r.weight === "vw");
  return (
    <div className="table-wrap">
      <table>
        <caption>Single signals, development window 2010–2021. Value-weighted decile long-short (top minus bottom) and Fama-MacBeth
          slope with ten controls. Green: |t| ≥ 3, the multiple-testing hurdle; grey: |t| &lt; 2.</caption>
        <thead><tr><th>Hypothesis</th><th>Signal</th><th>L/S mean / mo</th><th>t</th><th>FF5+Mom α / yr</th><th>α t</th><th>FM slope</th><th>FM t</th></tr></thead>
        <tbody>
          {vw.map((r) => {
            const f = fm[r.signal as string];
            return (
              <tr key={r.signal as string}>
                <td>{r.hyp === "explore_factors" ? "Explore" : r.hyp}</td><td>{(r.signal as string).replace(/^r_/, "")}</td>
                <td>{pct(r.ls_mean_m as number, 2)}</td><td className={tCls(r.t_mean)}>{fx(r.t_mean as number)}</td>
                <td>{pct(r.alpha_ann as number)}</td><td className={tCls(r.t_alpha)}>{fx(r.t_alpha as number)}</td>
                <td>{f ? fx(f.coef as number, 4) : "–"}</td><td className={f ? tCls(f.t_nw) : ""}>{f ? fx(f.t_nw as number) : "–"}</td>
              </tr>);
          })}
        </tbody>
      </table>
    </div>
  );
}

/** Drawdown narrative computed from the exported series (descriptive window 2019-12 to 2021-06 chosen after inspecting the chart). */
function ddText(s: Strategy) {
  const y = s.drawdown.y, x = s.drawdown.x, i = y.indexOf(Math.min(...y));
  const cum = s.cum_net25, peakI = cum.y.slice(0, cum.x.indexOf(x[i]) + 1).reduce((b, v, k, a) => (v > a[b] ? k : b), 0);
  const g = (ser: Series, a: string, b: string) => ser.y[ser.x.indexOf(b)] / ser.y[ser.x.indexOf(a)] - 1;
  return `Its deepest drawdown, ${pct(y[i], 0)}, ran from ${cum.x[peakI]} to ${x[i]}. Most of it came between 2019-12 and 2021-06, when the short leg
    (the stocks the model ranked lowest) gained ${pct(g(s.cum_short_ex, "2019-12", "2021-06"), 0)} in excess of T-bills against ${pct(g(s.cum_long_ex, "2019-12", "2021-06"), 0)} for the long leg.`;
}

export function Results() {
  const s = useData<Strategy>("strategy");
  const [bars, setBars] = useState("d10_vw");
  useTheme();
  if (!s) return <p className="loading">Loading results…</p>;
  const ink = cssVar("--ink"), ink3 = cssVar("--ink-3"), teal = cssVar("--teal"), amber = cssVar("--amber"), neg = cssVar("--neg");
  const shapes = testShape(s.test_start, s.test_end, s.unlocked);
  const icRoll = (x: Series) => ({ x: x.x.slice(11), y: x.y.map((_, i) => i < 11 ? NaN : x.y.slice(i - 11, i + 1).reduce((a, b) => a + b, 0) / 12).slice(11) });
  return (
    <Section id="results" title="Results">
      <div className="prose"><p>The primary strategy, fixed before any result was seen, sorts stocks each month on the ensemble
        forecast from all features and holds the top decile against the bottom decile, value-weighted with capped weights, after
        25 basis points of one-way trading costs.{s.unlocked ? " Shading marks the locked test window, opened once after the design was frozen." : ""}</p></div>
      <figure>
        <Plot label="Cumulative value of one dollar in the long-short strategy and its legs" height={380}
          data={[
            { x: s.cum_net25.x, y: s.cum_net25.y, name: "Long-short, net of 25 bps", line: { color: ink, width: 2.6 } },
            { x: s.cum_gross.x, y: s.cum_gross.y, name: "Long-short, gross", line: { color: ink, width: 1.2, dash: "dot" } },
            { x: s.cum_long_ex.x, y: s.cum_long_ex.y, name: "Long leg, excess of T-bills", line: { color: teal, width: 1.6 } },
            { x: s.cum_short_ex.x, y: s.cum_short_ex.y, name: "Short leg (held long), excess", line: { color: amber, width: 1.6 } },
            { x: s.cum_market_ex.x, y: s.cum_market_ex.y, name: "Market, excess", line: { color: ink3, width: 1.4, dash: "dash" } },
          ].map((t) => ({ ...t, type: "scatter", mode: "lines", hovertemplate: "%{x}: %{y:.2f}<extra>%{fullData.name}</extra>" }))}
          layout={{ yaxis: { type: "log", title: { text: "growth of $1 (log)" } }, shapes }} />
        <figcaption><b>Growth of a dollar.</b> The long and short legs are shown separately as excess returns; the strategy earns their
          difference. {ddText(s)}</figcaption>
      </figure>
      <div className="cols-2">
        <figure>
          <Plot label="Drawdown of the net long-short strategy" height={240}
            data={[{ type: "scatter", mode: "lines", x: s.drawdown.x, y: s.drawdown.y, fill: "tozeroy", line: { color: neg, width: 1.2 },
              fillcolor: neg + "33", hovertemplate: "%{x}: %{y:.1%}<extra></extra>" }]}
            layout={{ margin: { l: 52, r: 8, t: 8, b: 32 }, yaxis: { tickformat: ".0%" }, shapes }} />
          <figcaption><b>Drawdown</b> of the net strategy from its previous peak.</figcaption>
        </figure>
        <figure>
          <div className="controls">
            <select aria-label="Portfolio construction" value={bars} onChange={(e) => setBars(e.target.value)}>
              {Object.keys(s.decile_bars).map((k) => <option key={k} value={k}>{VARIANT_LABEL(k)}</option>)}
            </select>
          </div>
          <Plot label="Average annualised return by forecast portfolio" height={210}
            data={[{ type: "bar", x: s.decile_bars[bars].map((_, i) => `P${i + 1}`), y: s.decile_bars[bars], marker: { color: teal },
              hovertemplate: "%{x}: %{y:.1%} a year<extra></extra>" }]}
            layout={{ margin: { l: 52, r: 8, t: 8, b: 32 }, yaxis: { tickformat: ".0%" } }} />
          <figcaption><b>Returns by forecast portfolio</b> (P1 = lowest forecast), annualised, gross, development window 2014–2021. A useful forecast rises from left to right.</figcaption>
        </figure>
      </div>
      <figure>
        <Plot label="Rolling 12-month mean information coefficient" height={260}
          data={[
            { ...icRoll(s.ic_primary), name: "All features (ensemble)", line: { color: ink, width: 2 } },
            { ...icRoll(s.ic_controls), name: "Controls only (ensemble)", line: { color: ink3, width: 1.5, dash: "dash" } },
          ].map((t) => ({ ...t, type: "scatter", mode: "lines", hovertemplate: "%{x}: %{y:.3f}<extra>%{fullData.name}</extra>" }))}
          layout={{ yaxis: { title: { text: "12-month mean IC" }, zeroline: true }, shapes }} />
        <figcaption><b>Forecast quality over time.</b> Rank IC averaged over the trailing twelve months. The gap between the lines is what the text features add.</figcaption>
      </figure>
      <h3>Model ladder</h3>
      <Ladder />
      <h3>Strategy variants</h3>
      <VariantTable rows={s.variants} caption="Development window, 2014–2021. All construction variants over the same months. Return, risk and α columns are gross of costs; α t is from FF5 + momentum, and the next column adds a low-volatility factor built in this sample. Turnover is buys plus sells in both legs as a share of one leg's capital per month (divide by four for one-way turnover per leg); break-even is the one-way cost that would erase the gross return." />
      {s.variants_test && <VariantTable rows={s.variants_test} caption={`Locked test window, ${s.test_start} to ${s.test_end}, run once after the design was frozen.`} />}
      <p className="small prose">Net columns are Sharpe ratios after the stated one-way cost on every dollar traded.</p>
      <h3>Factor alphas and legs</h3>
      <AlphaTables rows={s.alphas} label="Development window, 2014–2021" />
      {s.alphas_test && <AlphaTables rows={s.alphas_test} label={`Locked test window, ${s.test_start} to ${s.test_end}`} />}
      <h3>The single signals behind the model</h3>
      <SingleSignals />
    </Section>
  );
}
