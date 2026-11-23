import Plot from "../components/Plot";
import { Section } from "./Static";
import { Row, cssVar, fx, pct, useData, useTheme } from "../lib";

type Studies = {
  survivorship: { coverage: Row[]; fate_lpm: Row[]; missing_decile: Row[]; breakeven: Row[]; summary: Row[] };
  event: { fm: Row[]; quintiles: Row[]; summary: Row[] };
  replication: { table: Row[]; summary: Row[] };
};

const SIG: Record<string, string> = {
  r_risk_all: "All eight factors (average rank)", r_nonop: "Non-operational (7 factors)", r_nonop_prob: "Non-operational (7 factors)",
  r_operations: "Operations", r_operations_prob: "Operations", r_d_risk_all: "Change, all factors", r_d_operations_prob: "Change, operations",
};
const sigName = (s: string) => SIG[s] ?? s.replace(/^r_/, "").replace(/_prob$/, "").replace(/^./, (c) => c.toUpperCase());
const VAR: Record<string, string> = { ln_at: "ln(total assets)", optional: "Item 1A optional", ln_words: "ln(Item 1A words)", rd_at: "R&D / assets",
  op: "Operational risk", nonop: "Non-operational risk", op_x_nonop: "Operational × non-operational" };
const tCls = (t: unknown) => (typeof t === "number" && Math.abs(t) >= 3 ? "sig" : typeof t === "number" && Math.abs(t) < 2 ? "weak" : "");

export function Studies() {
  const s = useData<Studies>("studies");
  useTheme();
  if (!s) return null;
  const sv = s.survivorship, ev = s.event, rp = s.replication;
  const cov = sv.coverage.filter((r) => (r.year as number) >= 2006 && (r.year as number) <= 2024);
  const c = (y: number) => cov.find((r) => r.year === y)!;
  const fate = Object.fromEntries(sv.summary.map((r) => [r.fate as string, r.firms as number]));
  const nFirms = Object.values(fate).reduce((a, b) => a + b, 0);
  const lpm = (o: string) => sv.fate_lpm.filter((r) => r.outcome === o);
  const bk = Object.fromEntries(lpm("bk_36m").map((r) => [r.signal as string, r]));
  const mg = Object.fromEntries(lpm("merged_36m").map((r) => [r.signal as string, r]));
  const dec = (sig: string, k: number) => sv.missing_decile.find((r) => r.signal === sig && r.decile === k)!;
  const be = sv.breakeven.find((r) => r.signal === "r_risk_all" && r.weight === "vw")!;
  const es = ev.summary[0];
  const efm = (x: string, y: string) => ev.fm.find((r) => r.signal === x && r.outcome === y && r.spec === "with_controls")!;
  const rs = rp.summary[0];
  const r4 = rp.table.filter((r) => r.model === "(4)");
  const r2op = rp.table.find((r) => r.model === "(2)" && r.var === "op")!;
  const op4 = r4.find((r) => r.var === "op")!, ix4 = r4.find((r) => r.var === "op_x_nonop")!;
  const sigs = ["r_risk_all", "r_operations_prob", "r_finance_prob", "r_legal_prob", "r_technology_prob", "r_international_prob", "r_d_risk_all"];
  const parts: [string, string, string][] = [["covered", "In our price data", "--teal"], ["missing_survivor", "Missing, still filing", "--ink-3"],
    ["missing_merged", "Missing, acquired", "--amber"], ["missing_bankrupt", "Missing, bankrupt", "--neg"], ["missing_exit_other", "Missing, other exit", "--rule"]];
  return (
    <Section id="studies" title="Further evidence">
      <div className="prose">
        <p>Three supplementary studies were added after the main design was frozen. They test the data rather than a
          trading rule, so they are not counted as strategy trials. Each answers a question the main results leave open:
          how much the survivorship bias matters, whether investors react to the risk language when the 10-K appears,
          and whether the risk scores behave as their authors report.</p>
      </div>

      <h3>Who is missing: a survivorship audit</h3>
      <div className="prose">
        <p>Free price data covers only firms that still trade, but the SEC keeps every filer's history. For each 10-K of a
          non-financial firm that ever had CRSP-listed stock, the audit checks whether we observe its price within a year of the
          filing, and how the firm ended. Some firms are still filing, some filed for bankruptcy (an 8-K under item 1.03), some
          were acquired (merger proxy or tender-offer forms), and some stopped filing for other reasons. Of {nFirms.toLocaleString("en-US")} such firms,
          {" "}{(fate.survivor ?? 0).toLocaleString("en-US")} still file, {(fate.merged ?? 0).toLocaleString("en-US")} were acquired, {(fate.bankrupt ?? 0).toLocaleString("en-US")} went
          bankrupt and {(fate.exit_other ?? 0).toLocaleString("en-US")} left for other reasons. Our prices cover {pct(c(2006).covered as number, 0)} of their 2006
          filings and {pct(c(2024).covered as number, 0)} of their 2024 filings. Among the missing, acquisitions far outnumber bankruptcies.</p>
      </div>
      <figure>
        <Plot label="Share of once-listed 10-K filers in our price data, by filing year and fate" height={300}
          data={parts.map(([k, name, col]) => ({ type: "bar", name, x: cov.map((r) => String(r.year)), y: cov.map((r) => r[k] as number),
            marker: { color: cssVar(col) }, hovertemplate: `%{x}: %{y:.0%}<extra>${name}</extra>` }))}
          layout={{ barmode: "stack", yaxis: { tickformat: ".0%", range: [0, 1] }, legend: { orientation: "h", y: -0.18 } }} />
        <figcaption><b>Coverage of once-listed filers.</b> Filings of non-financial, non-utility firms with a CRSP/Compustat link, by
          filing year. The missing share shrinks over time because recent filers have had less time to exit.</figcaption>
      </figure>
      <div className="prose">
        <p>Does the bias favour high-risk stocks? The opposite. Among development-years filings, {pct(dec("r_risk_all", 1).missing as number, 0)} of
          the lowest decile of aggregate risk language is missing, against {pct(dec("r_risk_all", 10).missing as number, 0)} of the highest.
          For the observed value-weighted high-minus-low risk spread ({pct(be.ls_mean_m as number, 2)} a month) to vanish once the missing firms
          are restored, they would have to outperform the observed firms in their decile by {pct(be.breakeven_gap_m as number, 1)} a month. That is
          implausible for a group that includes every bankruptcy. So survivorship cannot have created the (insignificant) risk premium,
          though it can still flatter the short leg of the machine-learning strategy, which this audit cannot score.</p>
      </div>
      <div className="table-wrap">
        <table>
          <caption>Does risk language predict how a firm ends? Linear probability models of bankruptcy or acquisition within 36 months of the
            10-K, on the within-year rank of each risk measure (lowest to highest = 1), with filing-year fixed effects and firm-clustered t.
            Filings 2006–2022, including firms absent from our price data. Base rates: bankruptcy {pct(bk.r_risk_all.base_rate as number, 1)},
            acquisition {pct(mg.r_risk_all.base_rate as number, 1)}.</caption>
          <thead><tr><th>Risk measure</th><th>Bankruptcy, low → high (pp)</th><th>t</th><th>Acquisition, low → high (pp)</th><th>t</th></tr></thead>
          <tbody>
            {["r_risk_all", "r_nonop", "r_finance", "r_operations", "r_legal", "r_accounting", "r_management", "r_marketing", "r_technology", "r_international"].map((k) => (
              <tr key={k}>
                <td>{sigName(k)}</td>
                <td>{fx((bk[k].coef as number) * 100, 1)}</td><td className={tCls(bk[k].t)}>{fx(bk[k].t as number)}</td>
                <td>{fx((mg[k].coef as number) * 100, 1)}</td><td className={tCls(mg[k].t)}>{fx(mg[k].t as number)}</td>
              </tr>))}
          </tbody>
        </table>
      </div>
      <div className="prose">
        <p>Finance-risk language is the clearest early warning: moving from the lowest to the highest rank adds {fx((bk.r_finance.coef as number) * 100, 1)} percentage points to
          the three-year bankruptcy probability (t = {fx(bk.r_finance.t as number, 1)}), against a base rate of {pct(bk.r_finance.base_rate as number, 1)}. Operations risk also predicts
          bankruptcy (t = {fx(bk.r_operations.t as number, 1)}). Technology, marketing and international risk language goes with acquisitions instead.</p>
      </div>

      <h3>How the market reacts on filing day</h3>
      <div className="prose">
        <p>For {(es.events as number).toLocaleString("en-US")} 10-Ks of {(es.firms as number).toLocaleString("en-US")} firms in the primary universe
          ({String(es.first).slice(0, 4)}–{String(es.last).slice(0, 4)}), the table relates the market-adjusted return over the filing day and the
          next, its absolute size, the drift over the following 60 trading days, and the change in daily return volatility to each
          risk measure. Signals are the panel's point-in-time ranks; controls are the ten characteristics measured before the filing.
          Legal-risk language comes with larger filing-day moves (t = {fx(efm("r_legal_prob", "abs_car01").t as number, 1)}), but no measure
          predicts a drift afterwards: if investors misread the risk sections, it does not show up in the following three months.</p>
      </div>
      <div className="table-wrap">
        <table>
          <caption>Filing-date event study. Fama-MacBeth slopes across filing years on a risk rank (lowest to highest = 1), with the ten
            controls. Returns are winsorised at 1%/99% within year.</caption>
          <thead><tr><th>Risk measure</th><th>CAR [0, +1]</th><th>t</th><th>|CAR [0, +1]|</th><th>t</th><th>CAR [+2, +60]</th><th>t</th><th>Δ log volatility</th><th>t</th></tr></thead>
          <tbody>
            {sigs.map((x) => (
              <tr key={x}>
                <td>{sigName(x)}</td>
                {["car01", "abs_car01", "car2_60", "dlogvol"].map((y) => {
                  const r = efm(x, y);
                  return [<td key={y}>{y === "dlogvol" ? fx(r.coef as number, 3) : pct(r.coef as number, 2)}</td>,
                    <td key={y + "t"} className={tCls(r.t)}>{fx(r.t as number)}</td>];
                })}
              </tr>))}
          </tbody>
        </table>
      </div>

      <h3>Replicating the source paper</h3>
      <div className="prose">
        <p>Astvansh and Simpson report that operational risk in a 10-K predicts higher cost of goods sold the next year, with firm
          and year fixed effects. Rebuilding their regression from SEC XBRL data ({(rs.n as number).toLocaleString("en-US")} firm-years,
          {" "}{(rs.firms as number).toLocaleString("en-US")} firms, fiscal years {rs.fyear_min}–{rs.fyear_max}, including firms that later disappeared)
          gives an operational-risk coefficient of {fx(op4.coef as number, 3)} (SE {fx(op4.se as number, 3)}) against their {fx(op4.paper_coef as number, 3)}
          {" "}({fx(op4.paper_se as number, 3)}); {fx(r2op.coef as number, 3)} without the non-operational terms against their {fx(r2op.paper_coef as number, 3)}.
          The main effect replicates in size, with a wider standard error on half their sample. Their interaction with non-operational
          risk does not (t = {fx(ix4.t as number, 1)} against their {fx((ix4.paper_coef as number) / (ix4.paper_se as number), 1)}).</p>
      </div>
      <div className="table-wrap">
        <table>
          <caption>Dependent variable: ln(COGS) in fiscal year y + 1. Column (4) of their Table 7 against the same specification here. Firm and
            fiscal-year fixed effects; standard errors clustered by firm. Their firm-size variable is ln(total assets), as in their Table 5.</caption>
          <thead><tr><th>Variable</th><th>This study</th><th>SE</th><th>t</th><th>Astvansh &amp; Simpson</th><th>SE</th></tr></thead>
          <tbody>
            {r4.map((r) => (
              <tr key={r.var as string}>
                <td>{VAR[r.var as string]}</td><td>{fx(r.coef as number, 3)}</td><td>{fx(r.se as number, 3)}</td>
                <td className={tCls(r.t)}>{fx(r.t as number)}</td><td>{fx(r.paper_coef as number, 3)}</td><td>{fx(r.paper_se as number, 3)}</td>
              </tr>))}
            <tr><td>Firm-years</td><td>{(rs.n as number).toLocaleString("en-US")}</td><td></td><td></td><td>{(rs.paper_n as number).toLocaleString("en-US")}</td><td></td></tr>
          </tbody>
        </table>
      </div>
    </Section>
  );
}
