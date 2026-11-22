import { useEffect, useState } from "react";

export type Series = { x: string[]; y: number[] };
export type Row = Record<string, number | string | null>;

const cache: Record<string, Promise<unknown>> = {};
export function load<T>(name: string): Promise<T> {
  cache[name] ??= fetch(`./data/${name}.json`).then((r) => {
    if (!r.ok) throw new Error(`data/${name}.json missing: run the pipeline export`);
    return r.json();
  });
  return cache[name] as Promise<T>;
}

export function useData<T>(name: string): T | null {
  const [d, setD] = useState<T | null>(null);
  useEffect(() => { load<T>(name).then(setD).catch((e) => console.error(e)); }, [name]);
  return d;
}

/** Reads a CSS custom property so charts follow the active theme. */
export const cssVar = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

export function useTheme(): string {
  const get = () => document.documentElement.dataset.theme
    ?? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const [t, setT] = useState(get);
  useEffect(() => {
    const mq = matchMedia("(prefers-color-scheme: dark)");
    const on = () => setT(get());
    mq.addEventListener("change", on);
    const mo = new MutationObserver(on);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => { mq.removeEventListener("change", on); mo.disconnect(); };
  }, []);
  return t;
}

export const FACTORS = ["accounting", "finance", "international", "legal", "management", "marketing", "operations", "technology"];
export const factorColor = (f: string) => cssVar(`--f-${f}`);
export const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

const minus = (s: string) => s.replace(/^-/, "−");
export const pct = (v: number | null | undefined, d = 1) => v == null || Number.isNaN(v) ? "–" : minus(`${(v * 100).toFixed(d)}%`);
export const fx = (v: number | null | undefined, d = 2) => v == null || Number.isNaN(v as number) ? "–" : minus((v as number).toFixed(d));
export const num = (v: number) => v.toLocaleString("en-US");
