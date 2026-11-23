import { useState } from "react";
import Plot from "../components/Plot";
import { Section } from "./Static";
import { FACTORS, cap, cssVar, factorColor, num, useData, useTheme } from "../lib";

type Explorer = {
  factors: string[]; years: number[]; risk_by_year: Record<string, number[]>;
  industries: string[]; risk_by_industry: Record<string, number[]>; words_by_year: number[];
  marketing_years: number[]; marketing_by_year: Record<string, number[]>; filings_by_year: number[];
  coverage: Record<string, number>[]; corr_constructs: Record<string, number | string>[];
};

const CONSTRUCT_LABEL: Record<string, string> = { mkt_orient: "Market orientation", mkt_capab: "Marketing capabilities", mkt_excel: "Marketing excellence" };

/** The eight factors as a heatmap: each row scaled to its own range, so changes over time are visible
 *  even for factors whose probabilities differ in level. Hover shows the raw median. */
export function Fingerprint({ d, by }: { d: Explorer; by: "year" | "industry" }) {
  useTheme();
  const cols = by === "year" ? d.years.map(String) : d.industries;
  const src = by === "year" ? d.risk_by_year : d.risk_by_industry;
  const z = FACTORS.map((f) => {
    const v = src[f]; const lo = Math.min(...v), hi = Math.max(...v);
    return v.map((x) => (hi > lo ? (x - lo) / (hi - lo) : 0.5));
  });
  const text = FACTORS.map((f) => src[f].map((x) => x.toFixed(3)));
  return (
    <Plot label={`Median risk-factor probability by ${by}, each factor scaled to its own range`} height={by === "year" ? 330 : 380}
      data={[{ type: "heatmap", z, x: cols, y: FACTORS.map(cap), text, hovertemplate: "%{y}, %{x}<br>median probability %{text}<extra></extra>",
        colorscale: [[0, cssVar("--paper-2")], [1, cssVar("--teal")]], showscale: false, xgap: 2, ygap: 2 }]}
      layout={{ margin: { l: 104, r: 8, t: 8, b: by === "year" ? 36 : 110 }, yaxis: { autorange: "reversed", gridcolor: "rgba(0,0,0,0)" },
        xaxis: { gridcolor: "rgba(0,0,0,0)", tickangle: by === "year" ? 0 : -40 } }} />
  );
}

export function Datasets() {
  const d = useData<Explorer>("explorer");
  const [by, setBy] = useState<"year" | "industry">("year");
  const [factor, setFactor] = useState("operations");
  useTheme();
  if (!d) return <p className="loading">Loading data…</p>;
  const total = d.filings_by_year.reduce((a, b) => a + b, 0);
  return (
    <Section id="data" title="The two datasets">
      <div className="cols-2">
        <div className="prose">
          <h3>Risk factors in 10-K filings</h3>
          <p>Astvansh &amp; Simpson (2026) extracted Item 1A from {num(total)} non-empty 10-Ks filed 2006–2024 and scored each
            on eight risk factors with fine-tuned transformers. The eight factors and their keyword dictionaries come from
            Schnatterly et al. (2021), who compared each firm's disclosed risks with the expertise of its board. Each score
            is a probability that the section discusses the factor.
            The dataset labels each filing by the year it was filed, which the timing rules here rely on.</p>
        </div>
        <div className="prose">
          <h3>Marketing emphasis in earnings calls</h3>
          <p>Damavandi, Mai &amp; Astvansh (2025) embedded every paragraph executives spoke on earnings calls with Sentence-BERT
            and measured its similarity to 19 marketing concepts, grouped into market orientation, marketing capabilities and
            marketing excellence. Scores are per calendar quarter of the call, 2003–2025.</p>
        </div>
      </div>
      <figure>
        <div className="controls">
          <span>Group filings by</span>
          <div className="seg" role="group" aria-label="Group by">
            {(["year", "industry"] as const).map((k) => <button key={k} aria-pressed={by === k} onClick={() => setBy(k)}>{k === "year" ? "Filing year" : "Industry (FF12)"}</button>)}
          </div>
        </div>
        <Fingerprint d={d} by={by} />
        <figcaption><b>Disclosed risk over time and across industries.</b> Median transformer probability per factor; each row is
          shaded relative to its own range. Item 1A grew from a median of {num(d.words_by_year[0])} words in 2006 to{" "}
          {num(d.words_by_year[d.words_by_year.length - 1])} in 2024, so scores are only compared within a month's cross-section.</figcaption>
      </figure>
      <div className="cols-2">
        <figure>
          <div className="controls">
            <label htmlFor="factor-sel">Factor</label>
            <select id="factor-sel" value={factor} onChange={(e) => setFactor(e.target.value)}>
              {FACTORS.map((f) => <option key={f} value={f}>{cap(f)}</option>)}
            </select>
          </div>
          <Plot label={`Median ${factor} risk probability by filing year`} height={260}
            data={[{ type: "scatter", mode: "lines+markers", x: d.years, y: d.risk_by_year[factor], line: { color: factorColor(factor), width: 2.5 },
              marker: { size: 5 }, hovertemplate: "%{x}: %{y:.3f}<extra></extra>" }]}
            layout={{ margin: { l: 48, r: 8, t: 8, b: 32 }, yaxis: { title: { text: "median probability" } } }} />
          <figcaption><b>One factor at a time.</b> The level of a single factor drifts with disclosure style, not only with risk.</figcaption>
        </figure>
        <figure>
          <div className="controls"><span>Marketing constructs, mean score by call year</span></div>
          <Plot label="Mean marketing-emphasis construct scores by year" height={260}
            data={Object.keys(d.marketing_by_year).map((c, i) => ({ type: "scatter", mode: "lines", name: CONSTRUCT_LABEL[c],
              x: d.marketing_years, y: d.marketing_by_year[c], line: { width: 2.2, color: [cssVar("--teal"), cssVar("--amber"), cssVar("--ink-2")][i] } }))}
            layout={{ margin: { l: 48, r: 8, t: 8, b: 32 }, legend: { y: -0.25 } }} />
          <figcaption><b>Three constructs, one direction.</b> They move together: pairwise correlations are{" "}
            {d.corr_constructs.flatMap((r) => Object.entries(r).filter(([k, v]) => k !== "index" && k > String(r.index) && typeof v === "number").map(([, v]) => (v as number).toFixed(2))).join(", ")}.
            The decorrelated (ZCA) components carry the distinct information.</figcaption>
        </figure>
      </div>
    </Section>
  );
}
