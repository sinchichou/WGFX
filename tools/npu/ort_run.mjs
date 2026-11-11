/**
 * ort_run.mjs — Run a WGFX-exported ONNX model and compare its output against
 * the same effect executed through the WGFX WebGPU runtime.
 *
 * Usage:
 *   node tools/npu/ort_run.mjs effects/CuNNy/CuNNy-2x4C-NVL.wgsl \
 *        dist/npu/CuNNy-2x4C-NVL.onnx [--size 128] [--ep cpu|dml]
 *
 * Experimental: see docs/npu-experiment.md.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { create, globals } from 'webgpu';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..', '..');

const argv = process.argv.slice(2);
const positional = argv.filter((a) => !a.startsWith('--'));
const sizeArg = (() => { const i = argv.indexOf('--size'); return i >= 0 ? Number(argv[i + 1]) : 128; })();
const epArg = (() => { const i = argv.indexOf('--ep'); return i >= 0 ? argv[i + 1] : 'cpu'; })();

const [wgslPath, onnxPath] = positional;
if (!wgslPath || !onnxPath) {
    console.error('usage: node tools/npu/ort_run.mjs <effect.wgsl> <model.onnx> [--size N] [--ep cpu|dml]');
    process.exit(2);
}

const H = sizeArg;
const W = sizeArg;

// ---------- shared test pattern (matches test/verification_test.mjs) ----------
const patternRGBA = new Uint8Array(H * W * 4);
for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        const v = (x * 7 + y * 13 + ((x * y) % 97)) % 256;
        patternRGBA[i] = v;
        patternRGBA[i + 1] = 255 - v;
        patternRGBA[i + 2] = (v * 3) % 255;
        patternRGBA[i + 3] = 255;
    }
}

// ---------- WGFX GPU run ----------
const U = globals.GPUTextureUsage;
const BU = globals.GPUBufferUsage;
Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
const adapter = await nav.gpu.requestAdapter();
const device = await adapter.requestDevice({
    requiredLimits: {
        maxStorageTexturesPerShaderStage: 8,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    },
});

const { WGFXRuntime } = await import(`file:///${path.join(ROOT, 'dist', 'wgfx.esm.js').replace(/\\/g, '/')}`);
const code = fs.readFileSync(path.resolve(wgslPath), 'utf8');
const rt = new WGFXRuntime(device);
await rt.compile(code, {
    defines: { INPUT_WIDTH: H, INPUT_HEIGHT: W },
    textures: {
        INPUT: { size: [W, H], format: 'rgba8unorm', usage: U.TEXTURE_BINDING | U.COPY_DST },
        OUTPUT: { size: [2 * W, 2 * H], format: 'rgba16float', usage: U.STORAGE_BINDING | U.COPY_SRC },
    },
}, {});
rt.resourceManager.updateSceneBuffer(W, H, 2 * W, 2 * H);
device.queue.writeTexture(
    { texture: rt.resourceManager.getTexture('INPUT') },
    patternRGBA,
    { bytesPerRow: W * 4, rowsPerImage: H },
    [W, H, 1],
);

const f16ToF32 = (h) => {
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
};

device.pushErrorScope('validation');
{
    const enc = device.createCommandEncoder();
    for (const p of rt.shaderInfo.passes) rt.dispatchPass(`PASS_${p.index}`, enc);
    device.queue.submit([enc.finish()]);
}
const gpuErr = await device.popErrorScope();
if (gpuErr) throw new Error('GPU dispatch failed: ' + gpuErr.message);

const bytesPerRow = Math.ceil((2 * W * 8) / 256) * 256;
const staging = device.createBuffer({ size: bytesPerRow * 2 * H, usage: BU.COPY_DST | BU.MAP_READ });
{
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer(
        { texture: rt.resourceManager.getTexture('OUTPUT') },
        { buffer: staging, bytesPerRow, rowsPerImage: 2 * H },
        [2 * W, 2 * H, 1],
    );
    device.queue.submit([enc.finish()]);
}
await staging.mapAsync(GPUMapMode.READ);
const raw = new Uint16Array(staging.getMappedRange().slice(0));
staging.unmap();
staging.destroy();

const rowStride = bytesPerRow / 2;
const gpuOut = new Float32Array(2 * W * 2 * H * 4);
for (let y = 0; y < 2 * H; y++) {
    for (let x = 0; x < 2 * W; x++) {
        const src = y * rowStride + x * 4;
        const dst = (y * 2 * W + x) * 4;
        for (let c = 0; c < 4; c++) gpuOut[dst + c] = f16ToF32(raw[src + c]);
    }
}
rt.resourceManager.dispose();

// ---------- ONNX run ----------
const ort = (await import('onnxruntime-node')).default;
const epList = epArg === 'dml' ? ['dml', 'cpu'] : ['cpu'];
const session = await ort.InferenceSession.create(path.resolve(onnxPath), {
    executionProviders: epList,
});

const nchw = new Float32Array(3 * H * W);
for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
        const si = (y * W + x) * 4;
        const di = y * W + x;
        nchw[0 * H * W + di] = patternRGBA[si] / 255;
        nchw[1 * H * W + di] = patternRGBA[si + 1] / 255;
        nchw[2 * H * W + di] = patternRGBA[si + 2] / 255;
    }
}
const t0 = performance.now();
const out = await session.run({ input: new ort.Tensor('float32', nchw, [1, 3, H, W]) });
const onnxMs = performance.now() - t0;
const onnxOut = out.output.data;
const [_, C, OH, OW] = out.output.dims;

// ---------- compare ----------
// GPU readback is interleaved RGBA; the ONNX tensor is NCHW planes.
const plane = OH * OW;
let maxErr = 0, sumErr = 0, count = 0, worst = -1;
for (let i = 0; i < OH * OW; i++) {
    for (let c = 0; c < 4; c++) {
        const g = gpuOut[i * 4 + c];
        const o = onnxOut[c * plane + i];
        const e = Math.abs(g - o);
        sumErr += e;
        count++;
        if (e > maxErr) { maxErr = e; worst = i * 4 + c; }
    }
}
const mae = sumErr / count;
const chMae = [0, 0, 0, 0];
for (let i = 0; i < OH * OW; i++) {
    for (let c = 0; c < 4; c++) chMae[c] += Math.abs(gpuOut[i * 4 + c] - onnxOut[c * plane + i]);
}
console.log('per-channel MAE:', chMae.map((v) => (v / (OH * OW)).toExponential(2)).join('  '));
let mse = 0;
for (let i = 0; i < OH * OW * 4; i++) {
    const d = gpuOut[i] - onnxOut[(i % 4) * plane + Math.floor(i / 4)];
    mse += d * d;
}
mse /= OH * OW * 4;
const psnr = 10 * Math.log10(1 / (mse || 1e-12));

console.log(`WGFX GPU  : ${2 * W}x${2 * H} rgba16float | ONNX (${epArg}): ${C}x${OH}x${OW} | run ${onnxMs.toFixed(1)} ms`);
console.log(`max|Δ| = ${maxErr.toExponential(3)}  MAE = ${mae.toExponential(3)}  PSNR = ${psnr.toFixed(1)} dB`);
const wi = worst;
console.log(`worst px (${Math.floor((wi / 4) % OW)}, ${Math.floor(wi / 4 / OW)}) ch${wi % 4}: gpu=${gpuOut[wi].toFixed(4)} onnx=${onnxOut[(wi % 4) * plane + Math.floor(wi / 4)].toFixed(4)}`);
for (const [px, py] of [[10, 10], [220, 114], [64, 64], [5, 200]]) {
    const i = py * OW + px;
    const g = [0, 1, 2, 3].map((c) => gpuOut[i * 4 + c].toFixed(3));
    const o = [0, 1, 2, 3].map((c) => onnxOut[c * plane + i].toFixed(3));
    console.log(`px(${px},${py}) gpu=[${g}] onnx=[${o}]`);
}
// isolate r (the tanh conv result): r = R_out - yuv0 (unsaturated pixels),
// with the GPU's transposed yuv luma row: yuv0 = 0.299r - 0.169g + 0.5b (half-pixel upsampled)
const yuv0 = new Float32Array(OH * OW);
for (let Y = 0; Y < OH; Y++) {
    for (let X = 0; X < OW; X++) {
        const fx = (X + 0.5) / 2 - 0.5, fy = (Y + 0.5) / 2 - 0.5;
        const x0 = Math.max(0, Math.min(W - 1, Math.floor(fx))), x1 = Math.min(W - 1, x0 + 1);
        const y0 = Math.max(0, Math.min(H - 1, Math.floor(fy))), y1 = Math.min(H - 1, y0 + 1);
        const tx = Math.max(0, Math.min(1, fx - x0)), ty = Math.max(0, Math.min(1, fy - y0));
        const si = (yy, xx) => (yy * W + xx) * 4;
        const mix = (c) => {
            const v00 = patternRGBA[si(y0, x0) + c] / 255, v10 = patternRGBA[si(y0, x1) + c] / 255;
            const v01 = patternRGBA[si(y1, x0) + c] / 255, v11 = patternRGBA[si(y1, x1) + c] / 255;
            return (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
        };
        const r = mix(0), g = mix(1), b = mix(2);
        yuv0[Y * OW + X] = 0.299 * r - 0.169 * g + 0.5 * b;
    }
}
let rMax = 0, rSum = 0, rN = 0;
for (let Y = 1; Y < OH - 1; Y++) {
    for (let X = 1; X < OW - 1; X++) {
        const i = Y * OW + X;
        const gr = gpuOut[i * 4] - yuv0[i];
        const orr = onnxOut[0 * plane + i] - yuv0[i];
        if (Math.abs(gr) > 0.98 || Math.abs(orr) > 0.98) continue; // skip saturated
        const e = Math.abs(gr - orr);
        rSum += e; rN++;
        if (e > rMax) rMax = e;
    }
}
console.log(`isolated r: max=${rMax.toExponential(3)} MAE=${(rSum / rN).toExponential(3)} (n=${rN})`);
console.log(mae < 0.02 ? 'RESULT: MATCH (within experimental tolerance)' : 'RESULT: DIFFERS — investigate before trusting the export');
device.destroy();
