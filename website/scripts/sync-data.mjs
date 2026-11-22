// Copy the pipeline's exported JSON (../results/json) into public/data so charts never hard-code numbers.
import { cpSync, existsSync, mkdirSync } from "node:fs";
const src = new URL("../../results/json/", import.meta.url);
const dst = new URL("../public/data/", import.meta.url);
if (!existsSync(src)) { console.error("results/json missing: run `python run_all.py` first"); process.exit(1); }
mkdirSync(dst, { recursive: true });
cpSync(src, dst, { recursive: true });
console.log("synced results/json -> public/data");
