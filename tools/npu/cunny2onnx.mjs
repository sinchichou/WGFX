/**
 * cunny2onnx.mjs — Export a CuNNy-*-NVL Magpie HLSL effect to an ONNX model.
 *
 * Reverses the HLSL weight layout into an equivalent ONNX graph so the same
 * network can run through ONNX Runtime — CPU or DirectML (GPU) today, and the
 * AMD XDNA NPU via Ryzen AI Software's VitisAI EP once installed.
 *
 * Mapping (verified against effects_src/CuNNy/CuNNy-2x4C-NVL.hlsl):
 *   PASS "in"   : l0 = dot(luma3, rgb) + bias → f0 = Σ_k mad(s_k, V4(w_k), r)
 *                 → ONE Conv 3x3 (3→4) with the colour projection folded into
 *                   the taps: W[o][i][ky][kx] = luma[i] * w_k[o],
 *                   bias[o] = f0bias[o] + lumaBias * Σ_k w_k[o].
 *   PASS "convN": l0 = sample(prev 4ch) → s0 = relu, s1 = -relu(-x) (per tap)
 *                 → f0 = bias + Σ_k MulAdd(s0_k, M4(w), r) + Σ_k MulAdd(s1_k, …)
 *                 → Concat([relu, neg]) 8ch → Conv 3x3 (8→4).
 *                 WGSL: acc + mat4x4<f32>(w) * v ⇒ out[o] = Σ_i w[o*4+i]·s_i
 *                 (verbatim, no transposition — porting-guide §2.1).
 *   PASS "out"  : same 8→4 conv (+ tanh head) → DepthToSpace(DCR, 2) reproduces
 *                 the 2x2 store order (r.x→(0,0), r.y→(1,0), r.z→(0,1), r.w→(1,1));
 *                 luma residual: Resize(input, linear, half_pixel) → rgb2yuv
 *                 → Clip(yuv_luma + r, 0, 1) → yuv2rgb → +alpha 1.
 *   Intermediates stored as R8G8B8A8_SNORM on the GPU → Quantize/DequantizeLinear
 *   (int8, scale 1/127) between feature passes so the ONNX numeric route matches.
 *
 * Usage:
 *   node tools/npu/cunny2onnx.mjs effects_src/CuNNy/CuNNy-2x4C-NVL.hlsl \
 *        --out dist/npu/CuNNy-2x4C-NVL.onnx [--size 256] [--no-quantize]
 *
 * Experimental: see docs/npu-experiment.md for status and limits.
 */

import fs from 'fs';
import path from 'path';
import { buildModel, node, initializer, int64Tensor, int8Tensor } from './onnx_writer.mjs';

// ---------- CLI ----------------------------------------------------------------

const argv = process.argv.slice(2);
const inPath = argv.find((a) => !a.startsWith('--'));
const outArg = (() => {
    const i = argv.indexOf('--out');
    return i >= 0 ? argv[i + 1] : null;
})();
const sizeArg = (() => {
    const i = argv.indexOf('--size');
    return i >= 0 ? Number(argv[i + 1]) : 256;
})();
const quantize = !argv.includes('--no-quantize');
const emitStage = (() => {
    const i = argv.indexOf('--emit-stage');
    return i >= 0 ? argv[i + 1] : null; // 'in' | 'conv1' | 'conv2' — debug: output an intermediate
})();

if (!inPath || !outArg) {
    console.error('usage: node tools/npu/cunny2onnx.mjs <cunny-nvl.hlsl> --out <model.onnx> [--size N] [--no-quantize]');
    process.exit(2);
}

// ---------- HLSL parsing -------------------------------------------------------

const numRe = /-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/g;
const floats = (s) => (s.match(numRe) ?? []).map(Number);

const src = fs.readFileSync(inPath, 'utf8');

/** Split into //!PASS sections (keeping the trailing //!COMMON block separately). */
const passSections = [];
{
    const re = /\/\/!PASS\s+(\d+)([\s\S]*?)(?=\/\/!PASS\s+\d+|$)/g;
    let m;
    while ((m = re.exec(src)) !== null) {
        passSections.push({ index: Number(m[1]), body: m[2] });
    }
}
if (passSections.length === 0) throw new Error('no //!PASS sections found');

/** Extract the f0 function body of a pass (from `V4 f0(` or `MF f0(` to its brace). */
function parseF0(body) {
    const start = body.search(/(?:V4|MF)\s+f0\s*\(/);
    if (start < 0) throw new Error('f0 not found');
    const brace = body.indexOf('{', start);
    let depth = 0;
    let end = -1;
    for (let i = brace; i < body.length; i++) {
        if (body[i] === '{') depth++;
        else if (body[i] === '}') {
            depth--;
            if (depth === 0) { end = i; break; }
        }
    }
    return body.slice(brace + 1, end);
}

/** Parse one pass: { kind, luma?, bias, taps: {k: {s0: number[], s1?: number[]}}, tanh } */
function parsePass(sec) {
    const body = sec.body;
    const l0m = body.match(/#define\s+l0\s*\([^)]*\)\s*(.+?)\r?\n/);
    const l0 = l0m ? l0m[1].trim() : '';
    const f0 = parseF0(body);

    const kind = /dot\s*\(\s*MF3/.test(l0) ? 'in' : (/void\s+Pass/.test(body) && body.includes('DepthToSpace') ? 'out' : 'conv');
    // The out pass is identified by the OUTPUT store + rgb2yuv, not just text.
    const isOut = body.includes('rgb2yuv');

    // bias init: `V4 r = { a, b, c, d };`
    const biasM = f0.match(/(?:V4|MF)\s+r\s*=\s*\{([^}]*)\}/);
    const bias = biasM ? floats(biasM[1]) : [0, 0, 0, 0];

    // taps: mad(s0_k, V4(w…), r) for the in pass; MulAdd(sX_k, M4(w…), r) for conv/out
    const taps = new Map();
    const lineRe = /r\s*=\s*(mad|MulAdd)\s*\(\s*(s0|s1)_(\d)\s*,\s*(?:V4|M4)\s*\(([^)]*)\)\s*,\s*r\s*\)/g;
    let lm;
    while ((lm = lineRe.exec(f0)) !== null) {
        const k = Number(lm[3]);
        if (!taps.has(k)) taps.set(k, {});
        taps.get(k)[lm[2]] = floats(lm[4]);
    }

    const tanh = /return\s+tanh\s*\(/.test(f0);

    return { index: sec.index, kind: isOut ? 'out' : kind, l0, bias, taps, tanh, body };
}

/** in-pass l0: dot(MF3(c0,c1,c2), O(INPUT,…).rgb) + MF(bias) — pull the two arg groups */
function parseLuma(l0) {
    const colorM = l0.match(/MF3\(([^)]*)\)/);
    const biasM = l0.match(/MF\(([^)]*)\)/);
    if (!colorM || !biasM) throw new Error(`unexpected l0 in-projection: ${l0}`);
    const color = floats(colorM[1]);
    const bias = floats(biasM[1]);
    if (color.length !== 3 || bias.length !== 1) throw new Error(`unexpected l0 in-projection: ${l0}`);
    return { color, bias: bias[0] };
}

const passes = passSections.map(parsePass);
const inPass = passes.find((p) => p.kind === 'in');
const outPass = passes.find((p) => p.kind === 'out');
const convPasses = passes.filter((p) => p.kind === 'conv');
if (!inPass || !outPass) throw new Error('expected one "in" and one "out" pass');

const luma = parseLuma(inPass.l0);

// out-pass colour matrices (static const MF3x3, row-major flat)
const rgb2yuv = floats(outPass.body.match(/rgb2yuv\s*=\s*\{([^}]*)\}/)[1]);
const yuv2rgb = floats(outPass.body.match(/yuv2rgb\s*=\s*\{([^}]*)\}/)[1]);

/** 3×3 flat row-major → transposed (WGSL mat3x3(w)*v ≡ W[o][i] = w[i*3+o]) */
function transpose3(w) {
    return [w[0], w[3], w[6], w[1], w[4], w[7], w[2], w[5], w[8]];
}

// ---------- graph assembly ------------------------------------------------------

const H = sizeArg;
const W = sizeArg;
const init = [];   // initializer tensors (encoded)
const nodes = [];  // graph nodes (encoded)
let tally = 0;
const fresh = (p) => `${p}_${tally++}`;

/** push a 3x3 conv over the relu-split input; returns output tensor name */
function convBlock(prev4, pass, suffix) {
    // relu split: s0 = relu(x), s1 = sub(relu(x), x)  (== -relu(-x), exact)
    const relu = fresh('relu');
    nodes.push(node('Relu', [prev4], [relu]));
    const neg = fresh('neg');
    nodes.push(node('Sub', [relu, prev4], [neg]));

    const cat = fresh('cat');
    nodes.push(node('Concat', [relu, neg], [cat], { axis: { i: 1 } }));

    // weights [4, 8, 3, 3]: taps scan s0_0..s0_8 then s1_0..s1_8,
    // tap k ↔ (ky, kx) = (k / 3, k % 3).
    // WGSL `mat4x4<f32>(w) * v` (column-major constructor, matrix·vector):
    // out[o] = Σ_i v_i · w[i*4 + o] — i.e. ONNX W[o][i] = w[i*4 + o]
    // (Magpie MulAdd(v, M4(w), acc) = acc + mul(v, M) in HLSL row-major; the
    // row-major/column-major conventions cancel in the flipped product — the
    // same sequence serves both, indexed [i*4 + o].)
    const w = new Array(4 * 8 * 3 * 3).fill(0);
    for (const [k, m] of pass.taps) {
        const ky = Math.floor(k / 3);
        const kx = k % 3;
        for (let o = 0; o < 4; o++) {
            for (let i = 0; i < 4; i++) {
                w[((o * 8 + i) * 3 + ky) * 3 + kx] = m.s0[i * 4 + o];
                w[((o * 8 + (4 + i)) * 3 + ky) * 3 + kx] = m.s1[i * 4 + o];
            }
        }
    }
    const wName = `W_${suffix}`;
    const bName = `B_${suffix}`;
    init.push(initializer(wName, [4, 8, 3, 3], w));
    init.push(initializer(bName, [4], pass.bias));
    // GPU texture sampling CLAMPS at edges; ONNX Conv zero-pads. Replicate the
    // clamp semantics with Pad(mode='edge') + a no-padding Conv.
    const padName = `pad_${suffix}`;
    const padded = fresh('pad');
    init.push(int64Tensor(padName, [8], [0, 0, 1, 1, 0, 0, 1, 1]));
    nodes.push(node('Pad', [cat, padName], [padded], { mode: { s: 'edge' } }));
    const out = fresh('conv');
    nodes.push(
        node('Conv', [padded, wName, bName], [out], {
            strides: { ints: [1, 1] },
            pads: { ints: [0, 0, 0, 0] },
            kernel_shape: { ints: [3, 3] },
            group: { i: 1 },
        }),
    );
    return out;
}

/** snorm-8 round trip (R8G8B8A8_SNORM intermediate quantization) */
function qdq(x, suffix) {
    if (!quantize) return x;
    const scale = `qscale_${suffix}`;
    init.push(initializer(scale, [], [1 / 127]));
    const zp = `qzp_${suffix}`;
    init.push(int8Tensor(zp, [0]));
    const q = fresh('q');
    const dq = fresh('dq');
    nodes.push(node('QuantizeLinear', [x, scale, zp], [q]));
    nodes.push(node('DequantizeLinear', [q, scale, zp], [dq]));
    return dq;
}

// ---- pass 1 (in): folded colour projection + 3x3 conv (3→4) -------------------

{
    const w = new Array(4 * 3 * 3 * 3).fill(0);
    for (const [k, m] of inPass.taps) {
        const ky = Math.floor(k / 3);
        const kx = k % 3;
        const v4 = m.s0; // V4(w0..w3) — 4 output channels
        for (let o = 0; o < 4; o++) {
            for (let i = 0; i < 3; i++) {
                w[((o * 3 + i) * 3 + ky) * 3 + kx] = luma.color[i] * v4[o];
            }
        }
    }
    const sumW = new Array(4).fill(0);
    for (const [, m] of inPass.taps) for (let o = 0; o < 4; o++) sumW[o] += m.s0[o];
    const b = inPass.bias.map((v, o) => v + luma.bias * sumW[o]);
    init.push(initializer('W_in', [4, 3, 3, 3], w));
    init.push(initializer('B_in', [4], b));
    init.push(int64Tensor('pad_in', [8], [0, 0, 1, 1, 0, 0, 1, 1]));
    nodes.push(node('Pad', ['input', 'pad_in'], ['input_padded'], { mode: { s: 'edge' } }));
    nodes.push(
        node('Conv', ['input_padded', 'W_in', 'B_in'], ['in_proj'], {
            strides: { ints: [1, 1] },
            pads: { ints: [0, 0, 0, 0] },
            kernel_shape: { ints: [3, 3] },
            group: { i: 1 },
        }),
    );
}

let cur = qdq('in_proj', 'in');
const stageOut = { in: cur };
for (const p of convPasses) {
    cur = convBlock(cur, p, `conv${p.index}`);
    cur = qdq(cur, `conv${p.index}`);
    stageOut[`conv${p.index}`] = cur;
}

// debug: emit an intermediate stage instead of the full graph
if (emitStage && stageOut[emitStage]) {
    const m2 = buildModel({
        name: `${path.basename(inPath, '.hlsl')}-${emitStage}`,
        nodes,
        initializers: init,
        inputs: [{ name: 'input', dims: [1, 3, H, W] }],
        outputs: [{ name: stageOut[emitStage], dims: [1, 4, H, W] }],
    });
    fs.mkdirSync(path.dirname(path.resolve(outArg)), { recursive: true });
    fs.writeFileSync(outArg, m2);
    console.log(`wrote ${outArg} (debug stage: ${emitStage}, output tensor '${stageOut[emitStage]}')`);
    process.exit(0);
}

// ---- out pass -------------------------------------------------------------------

const r4 = convBlock(cur, outPass, 'out');
const head = outPass.tanh ? (() => {
    nodes.push(node('Tanh', [r4], ['r_tanh']));
    return 'r_tanh';
})() : r4;

// DepthToSpace(DCR, 2): [1,4,H,W] → [1,1,2H,2W]; DCR d=1→(1,0), d=2→(0,1) matches
// the store order (r.x→(0,0), r.y→(1,0), r.z→(0,1), r.w→(1,1)).
nodes.push(node('DepthToSpace', [head], ['r_up'], { blocksize: { i: 2 }, mode: { s: 'DCR' } }));

// luma residual path: bilinear resize of the input to 2H×2W, then rgb2yuv
nodes.push(
    node('Resize', ['input', '', 'resize_scales'], ['rgb_up'], {
        mode: { s: 'linear' },
        coordinate_transformation_mode: { s: 'half_pixel' },
    }),
);
init.push(initializer('resize_scales', [4], [1, 1, 2, 2]));
// WGSL `mat3x3(w) * v` (column-major constructor): out[r] = Σ_c w[c*3+r]·v[c] —
// same transposed convention as the M4 convs above.
init.push(initializer('W_rgb2yuv', [3, 3, 1, 1], transpose3(rgb2yuv)));
nodes.push(node('Conv', ['rgb_up', 'W_rgb2yuv'], ['yuv'], { kernel_shape: { ints: [1, 1] } }));

// luma = yuv[:, 0:1] — Slice via int64 starts/ends/axes initializers
init.push(int64Tensor('sl_starts', [1], [0]));
init.push(int64Tensor('sl_ends', [1], [1]));
init.push(int64Tensor('sl_axes', [1], [1]));
nodes.push(node('Slice', ['yuv', 'sl_starts', 'sl_ends', 'sl_axes'], ['luma']));

nodes.push(node('Add', ['luma', 'r_up'], ['luma_r']));
nodes.push(node('Clip', ['luma_r', 'clip_lo', 'clip_hi'], ['luma_c']));
init.push(initializer('clip_lo', [], [0]));
init.push(initializer('clip_hi', [], [1]));

// yuv' = concat([luma', yuv[:,1:3]])
init.push(int64Tensor('ch_starts', [1], [1]));
init.push(int64Tensor('ch_ends', [1], [3]));
init.push(int64Tensor('ch_axes', [1], [1]));
nodes.push(node('Slice', ['yuv', 'ch_starts', 'ch_ends', 'ch_axes'], ['chroma']));
nodes.push(node('Concat', ['luma_c', 'chroma'], ['yuv2'], { axis: { i: 1 } }));

// back to RGB + opaque alpha (Concat on axis 1: [1,3,2H,2W] + [1,1,2H,2W])
init.push(initializer('W_yuv2rgb', [3, 3, 1, 1], transpose3(yuv2rgb)));
nodes.push(node('Conv', ['yuv2', 'W_yuv2rgb'], ['rgb_out'], { kernel_shape: { ints: [1, 1] } }));
{
    const ones = new Array(2 * H * 2 * W).fill(1);
    init.push(initializer('alpha', [1, 1, 2 * H, 2 * W], ones));
    nodes.push(node('Concat', ['rgb_out', 'alpha'], ['output'], { axis: { i: 1 } }));
}

// ---- write ---------------------------------------------------------------------

const model = buildModel({
    name: path.basename(inPath, '.hlsl'),
    nodes,
    initializers: init,
    inputs: [{ name: 'input', dims: [1, 3, H, W] }],
    outputs: [{ name: 'output', dims: [1, 4, 2 * H, 2 * W] }],
});

fs.mkdirSync(path.dirname(path.resolve(outArg)), { recursive: true });
fs.writeFileSync(outArg, model);
console.log(`wrote ${outArg} (${(model.length / 1024).toFixed(1)} KiB)`);
console.log(`  passes: in + ${convPasses.length} conv + out | size ${H}x${W} → ${2 * H}x${2 * W} | quantize=${quantize}${outPass.tanh ? ' | tanh head' : ''}`);
