// Parse every sample .board and report errors. Usage: npm run check
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parse, type BoardNode } from "../src/board/parse";

const dir = join(import.meta.dirname, "../../samples/synthetic");
let failed = false;
for (const f of readdirSync(dir).filter((f) => f.endsWith(".board")).sort()) {
  const b = parse(readFileSync(join(dir, f), "utf8"));
  const count = (ns: BoardNode[]): number => ns.reduce((s, n) => s + 1 + (n.type === "shape" ? count(n.children) : 0), 0);
  console.log(`${f}: ${count(b.items)} nodes, ${b.edges.length} edges, ${b.errors.length} errors`);
  for (const e of b.errors) console.log(`  line ${e.line}: ${e.message}`);
  if (b.errors.length) failed = true;
}
process.exit(failed ? 1 : 0);
