/**
 * verify_f16.mjs — pure-JS numerical bound for the f16 conv transform (no GPU).
 *
 * Model: the GPU f16 path rounds every operand and every arithmetic result to
 * binary16 (RNE); we emulate exactly that in JS and compare against the f32
 * (double) reference chain for the same inputs. This is an *estimate* — real
 * hardware may contract mul+add into fma (fewer roundings, error only smaller).
 *
 * Decisive metric: the intermediates are stored as rgba8snorm (8-bit, step
 * 2/255 ≈ 0.00784, half-step ≈ 0.00392). If the f16 chain's max |Δ| per pass
 * stays under the half-step, every intermediate store rounds to the SAME 8-bit
 * value ⇒ passes 2–4 see bit-identical inputs ⇒ final output pixel-identical.
 *
 * Usage: node verify_f16.mjs [--conv-only] [trials=20000]
 *   --conv-only: bound the CuNNy-2x4C-NVL-f16conv.wgsl variant (passes 2–3 only).
 */
import { readFile } from 'node:fs/promises';

let TRIALS = 20000;
let VARIANT = 'CuNNy-2x4C-NVL-f16.wgsl';
for (const a of process.argv.slice(2)) {
    if (a === '--conv-only') VARIANT = 'CuNNy-2x4C-NVL-f16conv.wgsl';
    else if (/^\d+$/.test(a)) TRIALS = Number(a);
}

// ---- binary16 RNE -------------------------------------------------------
const f32buf = new Float32Array(1);
const u32buf = new Uint32Array(f32buf.buffer);
/** f32 → f16 bits (round-to-nearest-even, IEEE 754 binary16). */
function toF16Bits(v) {
    f32buf[0] = v;
    const x = u32buf[0];
    const sign = (x >>> 16) & 0x8000;
    const exp = (x >>> 23) & 0xff;
    let man = x & 0x7fffff;
    if (exp === 0xff) return sign | 0x7c00 | (man ? 0x200 : 0); // ±Inf / NaN
    let e = exp - 127 + 15;
    if (e >= 0x1f) return sign | 0x7c00; // overflow → Inf
    if (e <= 0) { // subnormal / zero
        if (e < -10) return sign;
        man |= 0x800000;
        const shift = 14 - e;
        let m = man >> shift;
        const rem = man & ((1 << shift) - 1);
        const half = 1 << (shift - 1);
        if (rem > half || (rem === half && (m & 1))) m++;
        return sign | m;
    }
    let m = man >> 13;
    const rem = man & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && (m & 1))) m++;
    if (m === 0x400) { m = 0; e++; } // mantissa overflow
    return sign | (e << 10) | m;
}
/** f32 → f16 → f32 (the observable value of an f16 operand/result). */
function r16(v) {
    const bits = toF16Bits(v);
    const s = (bits & 0x8000) ? -1 : 1;
    const e = (bits & 0x7c00) >> 10;
    const m = bits & 0x03ff;
    if (e === 0) return s * m * 2 ** -24; // subnormal
    if (e === 0x1f) return s * Infinity;
    return s * (1 + m / 1024) * 2 ** (e - 15);
}

// ---- parse f0 weights from an effect file -------------------------------
function parseF0s(src) {
    const passes = src.split(/(?=\/\/!PASS \d+)/);
    const out = [];
    for (let pi = 1; pi < passes.length; pi++) {
        const p = passes[pi];
        const f0 = /fn f0\(([^)]*)\) -> vec4<f(?:16|32)> \{([\s\S]*?)\n\}/.exec(p);
        if (!f0) continue;
        // scalar pass (in): params are `s0_0: f32|f16, …` (no vec4); conv passes take vec4<…>.
        const scalar = !/vec4</.test(f0[1]);
        // bias: `var r = vec4<f32|f16>(a, b, c, d);`
        const bias = [.../var r = vec4<f(?:16|32)>\(([^)]*)\);/.exec(f0[2])[1].split(','),].map(Number);
        // terms: pass1 `s0_k * (vec4<f32|f16>(…))` ; conv `mat4x4<f32|f16>(…) * (s)`
        const terms = [];
        if (scalar) {
            for (const m of f0[2].matchAll(/\(s0_(\d)\) \* \(vec4<f(?:16|32)>\(([^)]*)\)\)/g)) {
                terms.push({ arg: `s0_${m[1]}`, w: m[2].split(',').map(Number) });
            }
        } else {
            for (const m of f0[2].matchAll(/mat4x4<f(?:16|32)>\(([^)]*)\) \* \(s([01])_(\d)\)/g)) {
                terms.push({ arg: `${m[2]}_${m[3]}`, w: m[1].split(',').map(Number) });
            }
        }
        out.push({ pass: pi, scalar, f16: /<f16>/.test(f0[0]), bias, terms });
    }
    return out;
}

// ---- chains -------------------------------------------------------------
/** f32 (double) reference: r = M_k * s_k + r per term. */
function runF32(f, samples) {
    const r = [...f.bias];
    for (const t of f.terms) {
        const v = samples[t.arg]; // vec4 or scalar
        for (let j = 0; j < 4; j++) {
            if (f.scalar) r[j] += v * t.w[j];
            else r[j] += v[0] * t.w[4 * j + 0] + v[1] * t.w[4 * j + 1] + v[2] * t.w[4 * j + 2] + v[3] * t.w[4 * j + 3];
        }
    }
    return r;
}
/** f16 emulation: round every weight, every product, every accumulation. */
function runF16(f, samples) {
    const r = f.bias.map(r16);
    const w16 = (a) => a.map(r16);
    for (const t of f.terms) {
        const w = w16(t.w);
        const v = samples[t.arg];
        for (let j = 0; j < 4; j++) {
            if (f.scalar) { r[j] = r16(r[j] + r16(v * w[j])); continue; }
            let acc = r[j];
            for (let k = 0; k < 4; k++) acc = r16(acc + r16(v[k] * w[4 * j + k]));
            r[j] = acc;
        }
    }
    return r;
}

// ---- sampling models ----------------------------------------------------
// pass1 input s ∈ dot(lumaV, rgb∈[0,1]) + 1.794  → ≈ [-0.21, 1.794]; pad a bit.
const rand = (a, b) => a + Math.random() * (b - a);
function makeSamples(f) {
    const s = {};
    if (f.scalar) {
        for (const t of f.terms) s[t.arg] = rand(-0.25, 1.85); // arg is already `s0_k`
    } else {
        for (const t of f.terms) {
            if (s[t.arg]) continue;
            const raw = [rand(-1, 1), rand(-1, 1), rand(-1, 1), rand(-1, 1)];
            s[t.arg] = raw; // relu/neg-relu variants share the raw draws below
        }
        // conv passes feed relu(s) as s0_k and -relu(-s) as s1_k — but the f0
        // weights were trained on those *values*, and both span [-1, 1] either
        // way; uniform [-1,1] per component covers both argument families.
    }
    return s;
}

// ---- main ---------------------------------------------------------------
const orig = await readFile(new URL('../../../effects/CuNNy/CuNNy-2x4C-NVL.wgsl', import.meta.url), 'utf8');
const f16v = await readFile(new URL(`./${VARIANT}`, import.meta.url), 'utf8');
console.log(`variant: ${VARIANT}`);
const A = parseF0s(orig), B = parseF0s(f16v);
if (A.length !== B.length) { console.error('FATAL: f0 count mismatch (variant lost a pass?)'); process.exit(1); }

const HALF_STEP = 1 / 255; // snorm8: values map [-1,1]→[0,255]; half bucket = 1/255 in unit span
let allPass = true;
console.log(`=== f16 conv-chain numerical bound (${TRIALS} trials/pass, RNE emulation) ===`);
for (let i = 0; i < A.length; i++) {
    const f = A[i];
    if (i === A.length - 1) { // out-shuffle f0 stays f32 in both files by design
        console.log(`PASS ${f.pass} (out-shuffle): left f32 in the variant — nothing to bound.`);
        continue;
    }
    if (B[i].scalar !== f.scalar || B[i].terms.length !== f.terms.length) {
        console.error(`PASS ${f.pass}: STRUCTURE MISMATCH`); allPass = false; continue;
    }
    if (!B[i].f16) { // pass left f32 in this variant — nothing to bound
        console.log(`PASS ${f.pass}: left f32 in the variant — nothing to bound.`);
        continue;
    }
    let maxD = 0, sumD = 0, bucketMiss = 0;
    const maxPerComp = [0, 0, 0, 0];
    for (let n = 0; n < TRIALS; n++) {
        const s = makeSamples(f);
        const a = runF32(f, s), b = runF16(f, s);
        for (let j = 0; j < 4; j++) {
            const d = Math.abs(a[j] - b[j]);
            sumD += d; maxD = Math.max(maxD, d); maxPerComp[j] = Math.max(maxPerComp[j], d);
            const qa = Math.round((Math.max(-1, Math.min(1, a[j])) + 1) / 2 * 255);
            const qb = Math.round((Math.max(-1, Math.min(1, b[j])) + 1) / 2 * 255);
            if (qa !== qb) bucketMiss++;
        }
    }
    const mae = sumD / (TRIALS * 4);
    // bit-identical claim requires ZERO bucket crossings; max|Δ| < half-step is
    // necessary but NOT sufficient (a value near a bucket edge can still cross).
    const identical = bucketMiss === 0;
    if (!identical) allPass = false;
    console.log(`PASS ${f.pass} (${f.scalar ? 'in: 9 scalar taps' : 'conv: 18 vec4 matmuls'}):`
        + ` max|Δ|=${maxD.toExponential(2)}  MAE=${mae.toExponential(2)}`
        + `  max/comp=[${maxPerComp.map((v) => v.toExponential(1)).join(', ')}]`
        + `  snorm8 bucket mismatch=${bucketMiss}/${TRIALS * 4}`
        + `  → ${identical ? '✅ bit-identical stores expected' : `⚠ ±1 LSB on ${(100 * bucketMiss / (TRIALS * 4)).toFixed(1)}% of texels (drift may propagate)`}`);
}
console.log(allPass
    ? '\n✅ THEORY: zero snorm8 bucket crossings ⇒ intermediates bit-identical, final pixels expected identical.'
    : '\n⚠ THEORY: some stores cross snorm8 buckets (±1 LSB) ⇒ end-to-end drift needs the real-GPU MAE gate (browser bench).');
process.exit(allPass ? 0 : 1);
