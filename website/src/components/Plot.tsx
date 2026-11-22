import { useEffect, useRef } from "react";
import Plotly from "plotly.js-cartesian-dist-min";
import { cssVar, useTheme } from "../lib";

type Props = { data: object[]; layout?: object; height?: number; label: string };

/** Plotly figure that inherits the page's tokens and re-renders on theme change. */
export default function Plot({ data, layout = {}, height = 360, label }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const theme = useTheme();
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ink = cssVar("--ink"), ink3 = cssVar("--ink-3"), rule = cssVar("--rule");
    const base = {
      height, margin: { l: 56, r: 16, t: 12, b: 44 },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "Archivo, Arial, sans-serif", size: 12, color: ink },
      xaxis: { gridcolor: rule, linecolor: ink3, zerolinecolor: ink3, tickcolor: ink3 },
      yaxis: { gridcolor: rule, linecolor: ink3, zerolinecolor: ink3, tickcolor: ink3 },
      legend: { orientation: "h", y: -0.18, x: 0, font: { size: 12 } },
      hoverlabel: { font: { family: "Archivo, Arial, sans-serif" } },
    };
    const merged = deepMerge(base, layout);
    Plotly.react(el, data, merged, { displayModeBar: false, responsive: true });
    const ro = new ResizeObserver(() => Plotly.Plots.resize(el));
    ro.observe(el);
    return () => ro.disconnect();
  }, [data, layout, height, theme]);
  useEffect(() => () => { if (ref.current) Plotly.purge(ref.current); }, []);
  return <div className="plot" ref={ref} role="img" aria-label={label} style={{ height }} />;
}

function deepMerge(a: any, b: any): any {
  const out = { ...a };
  for (const k of Object.keys(b)) out[k] = b[k] && typeof b[k] === "object" && !Array.isArray(b[k]) && a[k] ? deepMerge(a[k], b[k]) : b[k];
  return out;
}
