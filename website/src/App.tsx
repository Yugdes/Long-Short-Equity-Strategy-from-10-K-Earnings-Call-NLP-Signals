import { useEffect, useState } from "react";
import { Datasets } from "./sections/Data";
import { Results } from "./sections/Results";
import { Explain, Robustness } from "./sections/Robust";
import { Design, Limitations, Motivation, Pipeline, References, Team, Validation } from "./sections/Static";
import { fx, num, pct, useData } from "./lib";

type Summary = {
  unlocked: boolean; test_window: [string, string]; n_filings: number; n_filings_scored: number; n_firms_scores: number; n_stock_months: number;
  n_firms_panel: number; median_names_per_month: number;
  primary: Record<string, number | string>; dsr: { dsr: number }; pbo: { pbo: number };
};

const TOC: [string, string][] = [["motivation", "Motivation"], ["data", "Datasets"], ["design", "Design"], ["pipeline", "Pipeline"],
  ["models", "Models"], ["results", "Results"], ["robustness", "Robustness"], ["explain", "Explainability"],
  ["limitations", "Limitations"], ["references", "References"]];

type ModelsLite = { vol_prediction_dev: Record<string, number | string>[] };

/** The verdict is assembled from exported numbers; the wording states what they support and nothing more. */
function verdict(s: Summary, m: ModelsLite | null) {
  const p = s.primary as Record<string, number>;
  const pos = (t: number | undefined) => t !== undefined && t >= 3;          // the hurdle, in the claimed (positive) direction
  const rows = (m?.vol_prediction_dev ?? []).filter((r) => r.spec === "with_controls");
  const fac = rows.filter((r) => /_prob$/.test(String(r.signal)) && !String(r.signal).includes("nonop"));
  const nVol = fac.filter((r) => Math.abs(r.t_nw as number) >= 3).length;
  const opsT = rows.find((r) => r.signal === "r_operations_prob")?.t_nw as number | undefined;
  const alpha = pos(p.dev_net_t_alpha) || pos(p.test_net_t_alpha);
  const head = alpha ? "Partly." : "Not as a trading strategy.";
  return `${head} The models rank next-month returns better than chance in both periods (rank IC ${fx(p.dev_ic_mean, 3)} in 2014–2021` +
    (p.test_ic_mean !== undefined ? ` and ${fx(p.test_ic_mean, 3)} in the locked 2022–2025 test` : "") +
    `), but the long-short portfolio built on them earns ${alpha ? "only partial" : "no significant"} alpha after the five Fama-French factors, momentum and 25 bps trading costs` +
    ` (t = ${fx(p.dev_net_t_alpha, 1)}` + (p.test_net_t_alpha !== undefined ? ` and ${fx(p.test_net_t_alpha, 1)}` : "") + `),` +
    ` and the text features add little beyond a controls-only model. Disclosed risk does carry information about future risk, though not as pre-registered:` +
    ` operations risk does not predict next-year volatility (t = ${fx(opsT, 1)}), while ${nVol} of the 8 individual factors do (|t| ≥ 3), some up and some down.`;
}

function Hero() {
  const s = useData<Summary>("summary");
  const m = useData<ModelsLite>("models");
  if (!s) return <header className="hero"><p className="loading">Loading…</p></header>;
  const p = s.primary;
  const hasTest = s.unlocked && p.test_net_sharpe !== undefined;
  return (
    <header className="hero">
      <h1>Do the risks firms write down predict their stock returns?</h1>
      <p className="lede">A point-in-time test of {num(s.n_filings_scored)} scored 10-K risk sections and quarterly earnings-call
        language, combined with machine learning into a long-short equity strategy across {num(s.n_firms_panel)} U.S. firms.</p>
      <p className="byline">Authors: Yug Mitulkumar Desai, Arin Mehta<br />Supervisor: Marcos Severo</p>
      <p className="verdict" id="verdict">{verdict(s, m)}</p>
      <div className="stats">
        <div className="stat"><div className="v num">{fx(p.dev_ic_mean as number, 3)}</div><div className="l">mean monthly rank IC, development window 2014–2021 (t = {fx(p.dev_ic_t as number, 1)})</div></div>
        <div className="stat"><div className="v num">{fx(p.dev_net_sharpe as number, 2)}</div><div className="l">Sharpe ratio, primary long-short net of 25 bps, development window 2014–2021</div></div>
        <div className="stat"><div className="v num">{pct(p.dev_net_alpha_ann as number)}</div><div className="l">FF5 + momentum alpha a year, net of 25 bps, 2014–2021 (t = {fx(p.dev_net_t_alpha as number, 1)})</div></div>
        {hasTest
          ? <div className="stat"><div className="v num">{fx(p.test_net_sharpe as number, 2)}</div><div className="l">net Sharpe in the locked test window, {s.test_window[0]} to {s.test_window[1]}</div></div>
          : <div className="stat"><div className="v num">{fx(s.dsr.dsr, 2)}</div><div className="l">deflated Sharpe probability after every specification tried</div></div>}
      </div>
    </header>
  );
}

export default function App() {
  const [active, setActive] = useState("");
  const [theme, setTheme] = useState<string | null>(() => { try { return localStorage.getItem("theme"); } catch { return null; } });
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme; else delete document.documentElement.dataset.theme;
    try { theme ? localStorage.setItem("theme", theme) : localStorage.removeItem("theme"); } catch { /* storage unavailable */ }
  }, [theme]);
  useEffect(() => {
    const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && setActive(e.target.id)), { rootMargin: "-40% 0px -55% 0px" });
    const t = setInterval(() => { document.querySelectorAll("main section[id]").forEach((el) => io.observe(el)); }, 800);
    return () => { clearInterval(t); io.disconnect(); };
  }, []);
  const dark = theme ? theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  return (
    <div className="page">
      <nav className="toc" aria-label="Sections">
        <ol>{TOC.map(([id, t]) => <li key={id}><a href={`#${id}`} aria-current={active === id}>{t}</a></li>)}</ol>
        <button className="theme-btn" onClick={() => setTheme(dark ? "light" : "dark")}>{dark ? "Light theme" : "Dark theme"}</button>
      </nav>
      <main>
        <Hero />
        <Motivation />
        <Datasets />
        <Design />
        <Pipeline />
        <Validation />
        <Results />
        <Robustness />
        <Explain />
        <Limitations />
        <References />
        <Team />
        <p className="mobile-only"><button className="theme-btn" onClick={() => setTheme(dark ? "light" : "dark")}>{dark ? "Switch to light theme" : "Switch to dark theme"}</button></p>
      </main>
    </div>
  );
}
