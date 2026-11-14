/**
 * make_f16.mjs — O1 prototype: transform a CuNNy-*-NVL WGSL effect's conv math
 * into f16 (half precision) arithmetic.
 *
 * The generated Magpie/CuNNy pass bodies are extremely regular:
 *   fn f0(s0_0: vec4<f32>, …, s1_8: vec4<f32>) -> vec4<f32> {
 *       var r = vec4<f32>(b0, b1, b2, b3);
 *       r = ((mat4x4<f32>(W0…W15)) * (s0_0) + (r));   ×18
 *       return r;                                     (pass4: return tanh(r);)
 *   }
 * with pass 1 (in) using 9 f32 scalars and `s * vec4<f32>(w)` instead of mat4x4.
 *
 * Transform (passes 1–3 only; the out-shuffle pass 4 stays f32 — its tanh/YUV/
 * saturate epilogue mixes r.x…r.w with f32 luma and is a minority of GPU time):
 *   1. f0 signature: f32/vec4<f32> params and vec4<f32> return → f16/vec4<f16>.
 *   2. f0 body: vec4<f32>( / mat4x4<f32>( → vec4<f16>( / mat4x4<f16>(
 *      (decimal literals are AbstractFloat — legal f16 constructor args).
 *   3. Call site: wrap every argument at the f0( boundary — f16(s) for pass 1's
 *      scalars, vec4<f16>(s) for passes 2/3's vectors — leaving the tap sampling
 *      and ReLU math f32 (boundaries stay f32 per the audit's O1 design).
 *   4. textureStore: f0(...) → vec4<f32>(f0(...)) — storage rgba8snorm requires
 *      vec4<f32>.
 *
 * The module-level `enable f16;` is NOT injected here: the runtime gate
 * (preferF16 + shader-f16 feature + CAPABILITY FP16 metadata, commit 0e38703)
 * already emits it. Compiling this variant WITHOUT shader-f16 is expected to
 * fail — it is an f16-only artifact.
 *
 * Usage: node make_f16.mjs <in.wgsl> <out.wgsl> [passes=1,2,3]
 *   passes: which non-final passes to f16-ify. `1,2,3` = max speed; `2,3` =
 *   conv-only (verify_f16 shows pass 1's larger value range crosses snorm8
 *   buckets on ~3% of texels, while passes 2/3 are bit-identical in theory).
 */
import fs from 'node:fs';

const [, , inPath, outPath, passArg] = process.argv;
if (!inPath || !outPath) {
    console.error('usage: node make_f16.mjs <in.wgsl> <out.wgsl> [passes=1,2,3]');
    process.exit(2);
}
const WANTED = new Set((passArg || '1,2,3').split(',').map((s) => Number(s.trim())));

let src = fs.readFileSync(inPath, 'utf8');
const passes = src.split(/(?=\/\/!PASS \d+)/); // keep prologue as passes[0]

/** Transform the `fn f0(…) { … }` block of one pass into f16 math. */
function transformF0(block, { scalarArgs }) {
    // 1. signature
    let out = block.replace(
        /fn f0\(([^)]*)\) -> vec4<f32> \{/,
        (full, params) => {
            const ty = scalarArgs ? 'f16' : 'vec4<f16>';
            const np = params.replace(/: (?:vec4<f32>|f32)/g, `: ${ty}`);
            return `fn f0(${np}) -> vec4<f16> {`;
        });
    if (out === block) return null; // signature didn't match — leave pass untouched
    // 2. body types
    out = out.replace(/mat4x4<f32>\(/g, 'mat4x4<f16>(');
    out = out.replace(/vec4<f32>\(/g, 'vec4<f16>(');
    return out;
}

/** Wrap the f0(…) call at the store line: args → f16 boundary, result → f32. */
function wrapCallSite(block, { scalarArgs }) {
    const callRe = /f0\((s0_\d(?:, s1_\d|, s0_\d)*)\)/;
    const m = callRe.exec(block);
    if (!m) return null;
    const args = m[1].split(',').map((a) => a.trim());
    const cast = scalarArgs ? (a) => `f16(${a})` : (a) => `vec4<f16>(${a})`;
    const wrapped = `vec4<f32>(f0(${args.map(cast).join(', ')}))`;
    return block.replace(callRe, wrapped);
}

let transformed = 0;
for (let pi = 1; pi < passes.length; pi++) {
    const p = passes[pi];
    const desc = /\/\/!DESC (.*)/.exec(p);
    const passNo = /\/\/!PASS (\d+)/.exec(p);
    if (!passNo) continue;
    // out-shuffle (last pass) stays f32: tanh + YUV epilogue, minority of time.
    const isLast = pi === passes.length - 1;
    if (isLast) {
        console.log(`PASS ${passNo[1]} (${desc?.[1]}): left f32 (out-shuffle epilogue)`);
        continue;
    }
    if (!/fn f0\(/.test(p) || !/fn Pass\d+\(/.test(p)) continue;

    const scalarArgs = /fn f0\(s0_0: f32,/.test(p); // pass 1: 9 f32 scalars
    if (!WANTED.has(Number(passNo[1]))) {
        console.log(`PASS ${passNo[1]} (${desc?.[1]}): not in --passes list — left f32`);
        continue;
    }
    // f0 block = from `fn f0(` to the first line that is exactly `}`
    const f0Start = p.indexOf('fn f0(');
    const f0End = p.indexOf('\n}\n', f0Start);
    if (f0Start < 0 || f0End < 0) continue;

    const f0New = transformF0(p.slice(f0Start, f0End + 3), { scalarArgs });
    if (!f0New) { console.log(`PASS ${passNo[1]}: f0 signature unmatched — skipped`); continue; }

    let p2 = p.slice(0, f0Start) + f0New + p.slice(f0End + 3);
    const p3 = wrapCallSite(p2, { scalarArgs });
    if (!p3) { console.log(`PASS ${passNo[1]}: call site unmatched — skipped`); continue; }

    passes[pi] = p3;
    transformed++;
    console.log(`PASS ${passNo[1]} (${desc?.[1]}): f0 → ${scalarArgs ? 'f16 scalars' : 'vec4<f16> matmul'} ✓`);
}

fs.writeFileSync(outPath, passes.join(''));
console.log(`\ntransformed ${transformed} pass(es) → ${outPath}`);
console.log('note: this artifact requires shader-f16 (runtime gate emits `enable f16;`);');
console.log('      compiling it on an f16-less device is expected to fail.');
