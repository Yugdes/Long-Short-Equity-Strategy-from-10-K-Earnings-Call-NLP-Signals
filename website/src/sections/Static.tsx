import type { ReactNode } from "react";
import { Ledger } from "./Ledger";

export function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-h`}>
      <div className="sec-head"><span className="sec-no" aria-hidden="true" /><h2 id={`${id}-h`}>{title}</h2></div>
      {children}
    </section>
  );
}

export function Motivation() {
  return (
    <Section id="motivation" title="Why read the risk factors">
      <div className="prose">
        <p>Since December 2005 every U.S. public company has had to list, in Item 1A of its annual report, the
          risks that make its stock speculative. Most investors skim it as boilerplate. Two recent papers turned that
          text, and the language executives use on earnings calls, into numbers for nearly every listed firm.</p>
        <p>Neither paper asks whether those numbers say anything about future stock returns. That is the question here.
          If a firm's disclosed risk profile, or the emphasis its managers place on customers and markets, contains
          information the price does not yet reflect, a portfolio that is long the firms the signals favour and short the
          ones they disfavour should earn returns that standard risk factors cannot explain.</p>
        <p>The literature gives reasons to expect little. Risk disclosures load on risk the market already prices
          (Campbell et al. 2014), published anomalies decay (McLean &amp; Pontiff 2016), and with many signals to try,
          the bar for significance is a t-statistic near 3, not 2 (Harvey, Liu &amp; Zhu 2016). The study is built so that a
          null result would be believable too.</p>
      </div>
    </Section>
  );
}

export function Design() {
  return (
    <Section id="design" title="Research design and hypotheses">
      <div className="prose">
        <p>The hypotheses, the primary strategy and the significance hurdle were written down before any return was
          examined. The final 42 months (January 2022 to June 2025) were locked until the design was frozen.</p>
      </div>
      <Ledger />
    </Section>
  );
}

export function Pipeline() {
  const box = (x: number, y: number, w: number, t: string, s: string, key = false) => (
    <g className={key ? "key" : ""}>
      <rect x={x} y={y} width={w} height={58} rx={6} />
      <text x={x + 12} y={y + 24} fontSize={14} fontWeight={600}>{t}</text>
      <text x={x + 12} y={y + 44} fontSize={12} className="muted">{s}</text>
    </g>
  );
  return (
    <Section id="pipeline" title="From filings to portfolios">
      <div className="prose">
        <p>Every arrow below is code in the repository, rerun from raw files with one command. The one rule enforced at
          every step: a number enters month <em>t</em> only if it was public before month <em>t</em> began.</p>
      </div>
      <figure>
        <svg className="pipeline" viewBox="0 0 1000 300" role="img" aria-label="Pipeline diagram: four data sources feed a point-in-time panel, then features, walk-forward models and portfolios">
          {box(0, 10, 210, "10-K risk scores", "8 factors, 131,920 filings")}
          {box(0, 84, 210, "Earnings-call scores", "19 components, calendar quarters")}
          {box(0, 158, 210, "Prices (Yahoo)", "daily, splits, dividends")}
          {box(0, 232, 210, "SEC XBRL + French", "shares, book equity, factors")}
          {box(280, 121, 200, "Point-in-time panel", "release date + lag, caps", true)}
          {box(540, 121, 190, "Monthly ranks", "~90 features, 4 groups")}
          {box(790, 60, 210, "Walk-forward models", "refit yearly, tuned in-window", true)}
          {box(790, 182, 210, "Decile portfolios", "costs, factor alphas")}
          {[39, 113, 187, 261].map((y) => <path key={y} d={`M210 ${y} C245 ${y} 245 150 280 150`} />)}
          <path d="M480 150 L540 150" /><path d="M730 150 C760 150 760 89 790 89" /><path d="M895 118 L895 182" />
        </svg>
        <figcaption>Risk signals are usable from the first full month after filing date + 2 business days and expire after
          15 months. Call scores are usable from the second month after their calendar quarter ends and expire after 6.</figcaption>
      </figure>
    </Section>
  );
}

export function Validation() {
  return (
    <Section id="models" title="Models and validation">
      <div className="prose">
        <p>The model ladder climbs from transparent to flexible: single-signal decile sorts, Fama-MacBeth regressions,
          an elastic net, gradient-boosted trees (LightGBM and XGBoost), a two-layer neural network, and an equal-weight
          ensemble. Every model predicts the same thing, each stock's cross-sectional rank of next-month excess return.</p>
        <p>Validation is an expanding window. To forecast a year, a model trains on every earlier month, leaves a one-month
          embargo, tunes its hyperparameters on the last two years of that training window only, then refits and predicts
          the twelve months ahead. Scaling, imputation and tuning never see a future row. Feature sets are nested
          (controls; plus risk text; plus marketing; plus interactions) on identical rows, so differences come from the
          features, not the sample.</p>
      </div>
    </Section>
  );
}

export function Limitations() {
  return (
    <Section id="limitations" title="Limitations">
      <div className="prose">
        <p><strong>Survivorship.</strong> Prices come from Yahoo Finance, which only serves firms that still have a ticker.
          Firms that were delisted, merged or went bankrupt are missing, and so are their final, often worst, returns. Only
          12% to 33% of the risk-score filings in a year reach the primary universe. Both legs are affected, so the direction
          of the bias in a long-short spread is not known, but its size could be material.</p>
        <p><strong>Free fundamentals.</strong> Market capitalisation and book equity come from XBRL filings, which begin in
          2009 to 2011. Size and value controls and value weights are unavailable earlier, which is why the primary sample starts
          in 2010. Share counts needed sanity rules (decision log D-021), and a handful of misstated values can remain, which is
          why value weights are capped.</p>
        <p><strong>Current, not historical, identifiers.</strong> Industry codes and tickers are today's, applied backwards.</p>
        <p><strong>The scores themselves.</strong> The risk transformers were trained on 761 synthetic sentences labelled by
          the same dictionaries they were meant to improve on, with no human-labelled check on real filings. Their probabilities
          correlate only 0.14 with the dictionary counts for operations risk and fall as Item 1A gets longer. The three marketing
          constructs correlate 0.89 to 0.97 with one another, so they behave like one dimension.</p>
        <p><strong>Short horizon of the test.</strong> The locked window is 42 months, too short to separate a modest edge
          from luck on its own; it is a check on the development results, not a verdict by itself.</p>
        <p><strong>Not investment advice.</strong> Results are historical, gross of borrowing costs and market impact beyond
          the stated commissions, and for research and teaching only.</p>
        <h3>What would extend this</h3>
        <p>CRSP and Compustat data (delisting returns, historical identifiers, pre-2010 fundamentals), the authors'
          transformer models applied to 10-Q risk updates, and a human-labelled validation sample of real Item 1A sentences.</p>
      </div>
    </Section>
  );
}

const REFS: [string, string][] = [
  ["Astvansh, V., & Simpson, J. J. (2026). A firm's operational risk: Data set and empirical evidence. Manufacturing & Service Operations Management, 28(1), 326–341.", "https://doi.org/10.1287/msom.2025.0134"],
  ["Damavandi, H., Mai, F., & Astvansh, V. (2025). A new technique for measuring a firm's marketing emphasis. Marketing Letters, 36(4), 823–837.", "https://doi.org/10.1007/s11002-025-09781-7"],
  ["Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The probability of backtest overfitting. Journal of Computational Finance, 20(4).", "https://doi.org/10.21314/JCF.2016.322"],
  ["Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio. Journal of Portfolio Management, 40(5), 94–107.", "https://doi.org/10.3905/jpm.2014.40.5.094"],
  ["Campbell, J. L., Chen, H., Dhaliwal, D. S., Lu, H., & Steele, L. B. (2014). The information content of mandatory risk factor disclosures in corporate filings. Review of Accounting Studies, 19(1), 396–455.", "https://doi.org/10.1007/s11142-013-9258-3"],
  ["Carhart, M. M. (1997). On persistence in mutual fund performance. Journal of Finance, 52(1), 57–82.", "https://doi.org/10.1111/j.1540-6261.1997.tb03808.x"],
  ["Cohen, L., Malloy, C., & Nguyen, Q. (2020). Lazy prices. Journal of Finance, 75(3), 1371–1415.", "https://doi.org/10.1111/jofi.12885"],
  ["Edeling, A., & Fischer, M. (2016). Marketing's impact on firm value: Generalizations from a meta-analysis. Journal of Marketing Research, 53(4), 515–534.", "https://doi.org/10.1509/jmr.14.0046"],
  ["Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. Journal of Financial Economics, 116(1), 1–22.", "https://doi.org/10.1016/j.jfineco.2014.10.010"],
  ["Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. Journal of Political Economy, 81(3), 607–636.", "https://doi.org/10.1086/260061"],
  ["Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. Review of Financial Studies, 33(5), 2223–2273.", "https://doi.org/10.1093/rfs/hhaa009"],
  ["Harvey, C. R., Liu, Y., & Zhu, H. (2016). … and the cross-section of expected returns. Review of Financial Studies, 29(1), 5–68.", "https://doi.org/10.1093/rfs/hhv059"],
  ["Kravet, T., & Muslu, V. (2013). Textual risk disclosures and investors' risk perceptions. Review of Accounting Studies, 18(4), 1088–1122.", "https://doi.org/10.1007/s11142-013-9228-9"],
  ["Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. Journal of Finance, 66(1), 35–65.", "https://doi.org/10.1111/j.1540-6261.2010.01625.x"],
  ["McLean, R. D., & Pontiff, J. (2016). Does academic research destroy stock return predictability? Journal of Finance, 71(1), 5–32.", "https://doi.org/10.1111/jofi.12365"],
  ["Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica, 55(3), 703–708.", "https://doi.org/10.2307/1913610"],
  ["Price, S. M., Doran, J. S., Peterson, D. R., & Bliss, B. A. (2012). Earnings conference calls and stock returns: The incremental informativeness of textual tone. Journal of Banking & Finance, 36(4), 992–1011.", "https://doi.org/10.1016/j.jbankfin.2011.10.013"],
  ["Srinivasan, S., & Hanssens, D. M. (2009). Marketing and firm value: Metrics, methods, findings, and future directions. Journal of Marketing Research, 46(3), 293–312.", "https://doi.org/10.1509/jmkr.46.3.293"],
];

export function References() {
  return (
    <Section id="references" title="References">
      <ol className="refs">
        {REFS.map(([t, u]) => <li key={u}>{t} <a href={u}>{u.replace("https://", "")}</a></li>)}
      </ol>
      <p className="small prose">Data: Kenneth R. French Data Library (factors, industry definitions); SEC EDGAR (XBRL company
        facts, submissions, ticker file); Yahoo Finance via yfinance.</p>
    </Section>
  );
}

export function Team() {
  return (
    <footer>
      <div className="prose">
        <p>MS 499 independent research project. Authors: [Student name], [Student name]. Supervisor: [Faculty supervisor].</p>
        <p>Built from code in this repository; every figure on this page is read from the pipeline's exported results. Research and
          teaching use only; not investment advice.</p>
      </div>
    </footer>
  );
}
