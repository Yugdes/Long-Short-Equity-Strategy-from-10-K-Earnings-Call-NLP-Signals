import { Row, fx, useData } from "../lib";

type Models = { ladder_dev: Row[]; single_sorts_dev: Row[]; fama_macbeth_dev: Row[]; vol_prediction_dev: Row[]; interaction_fm_dev: Row[] };

/** Verdict from the pre-registered rule: supported = deciding |t| >= 3 in the predicted direction;
 *  mixed = 2 <= |t| < 3 or support on only part of the hypothesis; otherwise not supported. */
function verdict(t: number, sign: 1 | -1 | 0): "yes" | "mixed" | "no" {
  const s = sign === 0 ? Math.abs(t) : sign * t;
  return s >= 3 ? "yes" : s >= 2 ? "mixed" : "no";
}
const TAG = { yes: "Supported", mixed: "Weak or partial", no: "Not supported" };

export function Ledger() {
  const m = useData<Models>("models");
  const sum = useData<{ ic_increment: Record<string, Row>; primary: Row }>("summary");
  if (!m || !sum) return null;
  const sorts = (h: string) => m.single_sorts_dev.filter((r) => r.hyp === h && r.weight === "vw");
  const fm = (h: string) => m.fama_macbeth_dev.filter((r) => r.hyp === h && r.spec === "with_controls");
  const maxBy = (rows: Row[], k: string, sign: 1 | -1) => rows.reduce((b, r) => (sign * (r[k] as number) > sign * (b[k] as number) ? r : b), rows[0]);
  const lad = (set: string, model: string) => m.ladder_dev.find((r) => r.set === set && r.model === model)!;

  // H1: level sorts should earn no alpha; risk levels should predict volatility
  const h1a = sorts("H1"); const h1Abs = Math.max(...h1a.map((r) => Math.abs(r.t_alpha as number)));
  const vol = m.vol_prediction_dev.filter((r) => r.spec === "with_controls" && /_(accounting|finance|international|legal|management|marketing|operations|technology)_prob$/.test(String(r.signal)));
  const volSig = vol.filter((r) => Math.abs(r.t_nw as number) >= 3).length;
  const volUp = vol.filter((r) => (r.t_nw as number) >= 3).length, volDown = vol.filter((r) => (r.t_nw as number) <= -3).length;
  // pre-registered sign was "+": significant effects in both directions make it only partial support
  const h1v: "yes" | "mixed" | "no" = volUp >= 4 && volDown === 0 && h1Abs < 3 ? "yes" : volSig >= 1 ? "mixed" : "no";
  // H2: increases predict lower returns (negative)
  const h2 = [...sorts("H2"), ...fm("H2").map((r) => ({ ...r, t_alpha: r.t_nw }))];
  const h2best = maxBy(h2, "t_alpha", -1);
  // H3: marketing predicts higher returns (positive)
  const h3 = [...sorts("H3"), ...fm("H3").map((r) => ({ ...r, t_alpha: r.t_nw }))];
  const h3best = maxBy(h3, "t_alpha", 1);
  // H4: Fama-MacBeth interaction slope (predicted positive); plus the IC the interaction features add to the model
  const h4best = maxBy(m.interaction_fm_dev, "t_nw", 1);
  const d4 = ["enet", "lgbm"].map((k) => (lad("ALL", k).ic_mean as number) - (lad("C+R+M", k).ic_mean as number));
  // H5 (pre-registered): paired monthly IC difference ALL vs controls (NW t) AND net alpha of the primary strategy
  const inc = sum.ic_increment.ens;
  const tInc = inc.diff_t_nw as number, tNet = sum.primary.dev_net_t_alpha as number;
  const h5v = Math.min(tInc, tNet) >= 3 ? "yes" : Math.max(tInc, tNet) >= 2 ? "mixed" : "no";

  const items = [
    { id: "H1", claim: "Disclosed risk levels are priced as risk, not mispricing: they predict volatility but earn no alpha.", v: h1v,
      out: `No level sort earns alpha (largest |t| = ${fx(h1Abs)}). ${volSig} of 8 risk factors predict next-year volatility with |t| ≥ 3 after controls, but in both directions: ${volUp} higher, ${volDown} lower, against the predicted uniformly higher.` },
    { id: "H2", claim: "Firms whose disclosed risk rises underperform afterwards (the Lazy Prices logic).", v: verdict(h2best.t_alpha as number, -1),
      out: `Most negative statistic across change and surprise signals: t = ${fx(h2best.t_alpha as number)} (${String(h2best.signal).replace("r_", "")}).` },
    { id: "H3", claim: "Marketing emphasis on earnings calls predicts higher returns.", v: verdict(h3best.t_alpha as number, 1),
      out: `Largest statistic across marketing levels and changes: t = ${fx(h3best.t_alpha as number)} (${String(h3best.signal).replace("r_", "")}).` },
    { id: "H4", claim: "Risk × marketing interactions carry information neither carries alone.", v: verdict(h4best.t_nw as number, 1),
      out: `Largest Fama-MacBeth interaction slope: t = ${fx(h4best.t_nw as number)} (${String(h4best.interaction).replace("x_", "")}); adding interaction features changes model IC by ${d4.map((x) => (x >= 0 ? "+" : "") + fx(x, 4)).join(" (elastic net) and ")} (LightGBM).` },
    { id: "H5", claim: "A machine-learning combination of text and controls beats controls alone and survives costs.", v: h5v as "yes" | "mixed" | "no",
      out: `Ensemble IC ${fx(inc.ic_all as number, 4)} with text vs ${fx(inc.ic_controls as number, 4)} controls only: difference t = ${fx(tInc)}; primary strategy net-of-cost FF5+Mom alpha t = ${fx(tNet)}.` },
  ];
  return (
    <ol className="ledger" aria-label="Hypotheses and outcomes">
      {items.map((h) => (
        <li key={h.id}>
          <span className="id">{h.id}</span>
          <span className="claim">{h.claim}</span>
          <span className="outcome"><span className={`tag ${h.v}`}>{TAG[h.v]}</span>{h.out}</span>
        </li>))}
    </ol>
  );
}
