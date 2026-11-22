import Plot from "../components/Plot";
import { Section } from "./Static";
import { Row, Series, cap, cssVar, factorColor, fx, pct, useData, useTheme } from "../lib";

type Robust = {
  subperiods: Row[]; size_buckets: Row[]; industry_ic: Row[]; alt_lags: Row[]; horizon3: Row[];
  placebo_null_sharpes: number[]; placebo_real_sharpe: number; spec_count: number;
  extended_ew: Record<string, { cum: Series; stats: Record<string, number> }>;
};
type Summary = {
  dsr: { dsr: number; sr0_monthly: number; n_trials: number }; pbo: { pbo: number; n_splits: number; n_strategies: number };
  placebo: { p_value_sharpe: number; real_ic: number; null_ic_p95: number; p_value_ic: number; n_draws: number };
  shuffled_label: { ic_mean: number; ic_t_nw: number };
};
type Shap = { importance: Row[]; groups: Record<string, number>; dependence: Record<string, { x: number[]; shap: number[] }> };
type Models = { vol_prediction_dev: Row[] };

const LAG_LABEL: Record<string, string> = { mkt_lag0: "Call scores usable one month earlier (no buffer)", mkt_lag2: "Call scores one month later", risk_lag10bd: "10-K scores 10 business days after filing" };
const GROUP_LABEL: Record<string, string> = { controls: "Price and accounting controls", risk: "10-K risk text", marketing: "Earnings-call marketing", interactions: "Risk × marketing interactions" };

export const featureLabel = (f: string) => f.replace(/^r_/, "").replace(/^x_/, "× ").replace(/_prob$/, "").replace(/^d_/, "Δ ").replace(/^s_/, "surprise ").replace(/_/g, " ");

function factorOf(f: string) {
  const m = f.match(/(accounting|finance|international|legal|management|marketing|operations|technology)_prob/);
  return m ? m[1] : null;
}

export function Robustness() {
  const r = useData<Robust>("robustness");
  const s = useData<Summary>("summary");
  const mo = useData<Models>("models");
  useTheme();
  if (!r || !s || !mo) return <p className="loading">Loading robustness…</p>;
  const ink = cssVar("--ink"), ink3 = cssVar("--ink-3"), amber = cssVar("--amber");
  const vol = mo.vol_prediction_dev.filter((x) => x.spec === "with_controls" && factorOf(String(x.signal)));
  return (
    <Section id="robustness" title="Robustness and overfitting checks">
      <div className="cols-2">
        <div className="table-wrap">
          <table>
            <caption>Primary strategy by subperiod, net of 25 bps.</caption>
            <thead><tr><th>Period</th><th>Months</th><th>Ann. mean</th><th>Sharpe</th><th>t</th><th>FF5+Mom α t</th></tr></thead>
            <tbody>{r.subperiods.map((x) => (
              <tr key={String(x.period)}><td>{x.period as string}</td><td>{x.months as number}</td><td>{pct(x.ann_mean as number)}</td><td>{fx(x.sharpe as number)}</td><td>{fx(x.t_mean as number)}</td><td>{fx(x.t_alpha as number)}</td></tr>))}
            </tbody>
          </table>
        </div>
        <div className="table-wrap">
          <table>
            <caption>By size tercile (equal-weighted quintile long-short, gross) and mean IC.</caption>
            <thead><tr><th>Size</th><th>Names / mo</th><th>Ann. mean</th><th>Sharpe</th><th>Mean IC</th></tr></thead>
            <tbody>{r.size_buckets.map((x) => (
              <tr key={String(x.bucket)}><td>{cap(String(x.bucket))}</td><td>{fx(x.names_per_month as number, 0)}</td><td>{pct(x.ls_q5_ew_ann as number)}</td><td>{fx(x.sharpe as number)}</td><td>{fx(x.ic_mean as number, 4)}</td></tr>))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="cols-2">
        <figure>
          <Plot label="Placebo distribution of long-short Sharpe ratios from shuffled forecasts" height={250}
            data={[{ type: "histogram", x: r.placebo_null_sharpes, nbinsx: 40, marker: { color: ink3 }, name: "shuffled" }]}
            layout={{ margin: { l: 44, r: 8, t: 8, b: 36 }, showlegend: false, bargap: 0.05,
              shapes: [{ type: "line", x0: r.placebo_real_sharpe, x1: r.placebo_real_sharpe, yref: "paper", y0: 0, y1: 1, line: { color: amber, width: 2.5 } }],
              xaxis: { title: { text: "annualised Sharpe, equal-weighted decile long-short" } } }} />
          <figcaption><b>Placebo.</b> {s.placebo.n_draws} reshuffles of the forecast within each month (grey) against the real forecast
            (line): p = {fx(s.placebo.p_value_sharpe, 3)}. Retraining on shuffled labels gives a mean IC of {fx(s.shuffled_label.ic_mean, 4)}.</figcaption>
        </figure>
        <div className="prose">
          <h3>Multiple testing</h3>
          <p>{r.spec_count} strategy and model specifications were evaluated on the development window and every one is in the
            log. Correcting the primary Sharpe ratio for that many trials gives a deflated Sharpe probability of{" "}
            <strong>{fx(s.dsr.dsr, 2)}</strong> (0.95 would be convincing). Across {s.pbo.n_strategies} model and feature-set
            combinations, the probability that the in-sample winner underperforms the median out of sample is{" "}
            <strong>{fx(s.pbo.pbo, 2)}</strong> ({s.pbo.n_splits} combinatorial splits; below 0.5 is the goal).</p>
          <h3>Timing and horizon</h3>
          <div className="table-wrap"><table>
            <thead><tr><th>Variant (LightGBM, all features)</th><th>Mean IC</th><th>IC t</th></tr></thead>
            <tbody>{r.alt_lags.map((x) => (<tr key={String(x.variant)}><td>{LAG_LABEL[String(x.variant)]}</td><td>{fx(x.ic_mean as number, 4)}</td><td>{fx(x.ic_t_nw as number)}</td></tr>))}
              {r.horizon3.map((x) => (<tr key={`h3${x.set}${x.model}`}><td>3-month horizon: {String(x.model)} on {String(x.set)}</td><td>{fx(x.ic_mean as number, 4)}</td><td>{fx(x.ic_t_nw as number)}</td></tr>))}
            </tbody></table></div>
        </div>
      </div>
      <h3>Does disclosed risk predict risk?</h3>
      <figure>
        <Plot label="Fama-MacBeth t-statistics of each risk factor predicting next-year volatility" height={280}
          data={[{ type: "bar", orientation: "h", y: vol.map((x) => cap(factorOf(String(x.signal))!)), x: vol.map((x) => x.t_nw as number),
            marker: { color: vol.map((x) => factorColor(factorOf(String(x.signal))!)) }, hovertemplate: "%{y}: t = %{x:.1f}<extra></extra>" }]}
          layout={{ margin: { l: 104, r: 16, t: 8, b: 40 }, xaxis: { title: { text: "t-statistic, slope on next-12-month volatility rank (with controls)" }, zeroline: true, zerolinecolor: ink },
            shapes: [-3, 3].map((v) => ({ type: "line", x0: v, x1: v, yref: "paper", y0: 0, y1: 1, line: { color: ink3, dash: "dot", width: 1 } })) }} />
        <figcaption><b>Risk text forecasts volatility, in both directions.</b> Monthly cross-sectional regressions of realised volatility
          over the next twelve months on each factor's rank, controlling for past volatility, size, beta and seven other characteristics
          (formation months 2010–2020). Dotted lines: |t| = 3.</figcaption>
      </figure>
    </Section>
  );
}

export function Explain() {
  const sh = useData<Shap>("shap");
  useTheme();
  if (!sh) return <p className="loading">Loading explanations…</p>;
  const ink3 = cssVar("--ink-3"), teal = cssVar("--teal");
  const imp = sh.importance.slice(0, 15).reverse();
  const col = (f: string) => factorOf(f) ? factorColor(factorOf(f)!) : (String(f).includes("mkt") ? cssVar("--amber") : ink3);
  const totals = Object.entries(sh.groups); const sum = totals.reduce((a, [, v]) => a + v, 0);
  return (
    <Section id="explain" title="What the model uses">
      <div className="prose"><p>SHAP values for the frozen LightGBM model that makes the first out-of-sample forecast, explained on
        50,000 training stock-months. Bars are mean absolute contributions to the predicted rank.</p></div>
      <div className="cols-2">
        <figure>
          <Plot label="Top 15 features by mean absolute SHAP value" height={440}
            data={[{ type: "bar", orientation: "h", y: imp.map((r) => featureLabel(String(r.feature))), x: imp.map((r) => r.mean_abs_shap as number),
              marker: { color: imp.map((r) => col(String(r.feature))) }, hovertemplate: "%{y}: %{x:.4f}<extra></extra>" }]}
            layout={{ margin: { l: 170, r: 8, t: 8, b: 36 }, xaxis: { title: { text: "mean |SHAP|" } } }} />
          <figcaption><b>Feature importance.</b> Risk factors in their identity colours, marketing in amber, controls in grey.</figcaption>
        </figure>
        <div>
          <div className="table-wrap"><table>
            <caption>Share of total importance by feature group.</caption>
            <thead><tr><th>Group</th><th>Share</th></tr></thead>
            <tbody>{totals.map(([g, v]) => <tr key={g}><td>{GROUP_LABEL[g]}</td><td>{pct(v / sum, 0)}</td></tr>)}</tbody>
          </table></div>
        </div>
      </div>
      <div className="cols-2">
        {Object.entries(sh.dependence).map(([f, d]) => (
          <figure key={f}>
            <Plot label={`SHAP dependence for ${featureLabel(f)}`} height={220}
              data={[{ type: "scatter", mode: "lines+markers", x: d.x, y: d.shap, line: { color: factorOf(f) ? factorColor(factorOf(f)!) : teal, width: 2 }, marker: { size: 5 } }]}
              layout={{ margin: { l: 56, r: 8, t: 8, b: 36 }, xaxis: { title: { text: `${featureLabel(f)} (centred rank)` } }, yaxis: { title: { text: "SHAP" } } }} />
          </figure>))}
      </div>
    </Section>
  );
}
