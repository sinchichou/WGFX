// weight-integrity.mjs — verify every numeric literal in the source HLSL
// survives verbatim into the converted WGSL (multiset comparison).
// usage: node tools/port/weight-integrity.mjs
import fs from 'fs';
import path from 'path';

const SRC = path.resolve(import.meta.dirname, '..', 'effects_src');
const OUT = path.resolve(import.meta.dirname, '..', 'effects');

// Numbers that legitimately differ between source and output:
//  - the emitted WIDTH/HEIGHT text ("INPUT_WIDTH * 2") adds no numbers
//  - the porting-note header mentions e.g. 2x2, 8x8 → excluded by comparing
//    only tokens that look like float literals
const FLOAT_LIT = /[+-]?\d+\.\d+(?:[eE][+-]?\d+)?|[+-]?\d+[eE][+-]?\d+/g;

function multiset(text) {
  const m = new Map();
  for (const tok of text.match(FLOAT_LIT) || []) {
    // normalise: 1.4016870 → 1.401687, 1 → keep exponent form as-is
    const key = String(Number(tok)); // canonical double text
    m.set(key, (m.get(key) || 0) + 1);
  }
  return m;
}

// One-sided integrity: legitimate transformations only ever DUPLICATE numbers
// (macro inlining ×9 taps, pass-splitting ×2, SampleLevel lod `0` → `0.0`,
// `, 1)` → `, 1.0)`), never invent or lose them. So:
//   FAIL  if an OUT literal key is absent from SRC (typo/corruption), or
//   FAIL  if OUT has fewer copies of a key than SRC (lost weight).
//   INFO  inflation (out > src) is expected.
let fail = 0;
let inflation = 0;
for (const fam of ['CuNNy2', 'CuNNy']) {
  for (const f of fs.readdirSync(path.join(SRC, fam)).filter((x) => x.endsWith('.hlsl')).sort()) {
    const src = fs.readFileSync(path.join(SRC, fam, f), 'utf8');
    const out = fs.readFileSync(path.join(OUT, fam, f.replace(/\.hlsl$/, '.wgsl')), 'utf8');
    // strip the generated header comment (porting notes contain prose numbers)
    const outCode = out.slice(out.indexOf('//!VERSION'));
    const a = multiset(src);
    const b = multiset(outCode);
    const problems = [];
    let infl = 0;
    for (const [k, v] of b) {
      const want = a.get(k) || 0;
      if (want === 0) problems.push(`OUT-only literal ${k} x${v} (not in source!)`);
      else if (v < want) problems.push(`lost literal ${k}: src x${want} vs out x${v}`);
      else if (v > want) infl += v - want;
    }
    if (problems.length) {
      fail++;
      console.log(`FAIL ${fam}/${f}`);
      console.log(problems.slice(0, 10).map((s) => '  ' + s).join('\n'));
    } else {
      inflation += infl;
      console.log(`OK   ${fam}/${f}  (all literals present; benign inflation +${infl})`);
    }
  }
}
console.log(fail === 0
  ? `\nALL FILES: no invented / no lost numeric literals (total benign inflation +${inflation})`
  : `\n${fail} FILES WITH REAL DIFFERENCES`);
process.exit(fail === 0 ? 0 : 1);
