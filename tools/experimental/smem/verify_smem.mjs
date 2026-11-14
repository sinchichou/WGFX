/**
 * verify_smem.mjs — O2 prototype verification + bench.
 * 1. compile check: the transformed WGSL must create pipelines without errors.
 * 2. pixel-identity: smem variant output vs original, readback diff == 0.
 * 3. bench: per-frame GPU wall time (submit+onSubmittedWorkDone), both variants.
 */
import { create, globals } from 'webgpu';
import { readFile } from 'node:fs/promises';
import assert from 'node:assert';
Object.assign(globalThis, globals);

const { WGFXRuntime } = await import('../../../dist/wgfx.esm.js');
// NOTE: deliberately NOT importing dist/verification.esm.js — it top-level-imports
// the webgpu polyfill and re-assigns globals, which breaks pipeline creation for
// runtimes created on the original device instance (measured). Inline comparator.
const comparePixels = (a, b, w, h) => {
    let max = 0, sum = 0;
    for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); sum += d; if (d > max) max = d; }
    return { mae: sum / a.length, maxAbsDiff: max };
};

const nav = { gpu: create([]) };
const adapter = await nav.gpu.requestAdapter();
const device = await adapter.requestDevice();

const orig = await readFile(new URL('../../../effects/CuNNy/CuNNy-2x4C-NVL.wgsl', import.meta.url), 'utf8');
const smem = await readFile(new URL('./CuNNy-2x4C-NVL-smem.wgsl', import.meta.url), 'utf8');

const W = Number(process.argv[2] || 256), H = W, ROUNDS = 5, FRAMES = 30;
const pixels = new Uint8Array(W * H * 4);
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = (y * W + x) * 4;
    pixels[i] = (x * 255 / W) | 0;
    pixels[i + 1] = (y * 255 / H) | 0;
    pixels[i + 2] = ((x >> 3) ^ (y >> 3)) & 1 ? 200 : 40;
    pixels[i + 3] = 255;
}

async function makeRT(code, fmt = 'rgba16float') {
    const rt = new WGFXRuntime(device);
    await rt.compile(code, {
        defines: { INPUT_WIDTH: W, INPUT_HEIGHT: H },
        textures: {
            INPUT: { size: [W, H], format: 'rgba8unorm',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST },
            OUTPUT: { size: [W, H], format: fmt,
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC |
                    GPUTextureUsage.TEXTURE_BINDING },
        },
    }, { honorDeclaredOutput: false });
    return rt;
}

function frame(rt) {
    const e = device.createCommandEncoder();
    rt.dispatchAllPasses(e);
    device.queue.submit([e.finish()]);
    return device.queue.onSubmittedWorkDone();
}

async function readback(rt) {
    const out = rt.resourceManager.getTexture('OUTPUT');
    const bpr = Math.ceil(out.width * 8 / 256) * 256;
    const buf = device.createBuffer({ size: bpr * out.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = device.createCommandEncoder();
    e.copyTextureToBuffer({ texture: out }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: out.height },
        [out.width, out.height]);
    device.queue.submit([e.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const copy = new Uint8Array(new Uint8Array(buf.getMappedRange()));
    buf.unmap(); buf.destroy();
    return { bytes: copy, bpr, w: out.width, h: out.height };
}

function f16to8(raw, bpr, w, h) {
    const u16 = new Uint16Array(raw.buffer, raw.byteOffset, (bpr / 2) * h);
    const out = new Uint8Array(w * h * 4);
    const f16 = (v) => {
        const s = (v & 0x8000) >> 15, e = (v & 0x7c00) >> 10, m = v & 0x03ff;
        if (e === 0) return (m ? m * 2 ** -24 : 0) * (s ? -1 : 1);
        return (s ? -1 : 1) * (1 + m / 1024) * 2 ** (e - 15);
    };
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const s = (y * (bpr / 2) + x * 4), d = (y * w + x) * 4;
        for (let c = 0; c < 4; c++) out[d + c] = Math.max(0, Math.min(255, Math.round(f16(u16[s + c]) * 255)));
    }
    return out;
}

console.log(`=== O2 shared-memory prototype — CuNNy-2x4C-NVL @ ${W}x${H} ===`);
const rtA = await makeRT(orig), rtB = await makeRT(smem);
for (const rt of [rtA, rtB]) {
    device.queue.writeTexture({ texture: rt.resourceManager.getTexture('INPUT') }, pixels,
        { bytesPerRow: W * 4, rowsPerImage: H }, [W, H]);
    rt.resourceManager.updateSceneBuffer(W, H, W, H);
}

// pixel identity
console.log('rtA keys:', [...rtA.pipelineManager.pipelines.keys()], 'modules:', rtA.generatedModules?.map(m => m.passIndex));
console.log('rtB keys:', [...rtB.pipelineManager.pipelines.keys()], 'modules:', rtB.generatedModules?.map(m => m.passIndex));
try { await frame(rtA); console.log('rtA frame OK'); } catch (e) { console.log('rtA FAIL:', e.message); }
try { await frame(rtB); console.log('rtB frame OK'); } catch (e) { console.log('rtB FAIL:', e.message); }
const A = f16to8((await readback(rtA)).bytes, Math.ceil(W * 8 / 256) * 256, W, H);
const B = f16to8((await readback(rtB)).bytes, Math.ceil(W * 8 / 256) * 256, W, H);
const m = comparePixels(A, B, W, H);
console.log(`pixel identity: MAE=${m.mae}  max=${m.maxAbsDiff}`);
assert.ok(m.maxAbsDiff === 0, 'smem variant must be pixel-identical to the original');

// bench
async function bench(rt, label) {
    for (let i = 0; i < 10; i++) await frame(rt);
    const times = [];
    for (let r = 0; r < ROUNDS; r++) {
        const t0 = performance.now();
        for (let i = 0; i < FRAMES; i++) await frame(rt);
        times.push((performance.now() - t0) / FRAMES);
    }
    times.sort((a, b) => a - b);
    const med = times[Math.floor(times.length / 2)];
    console.log(`${label.padEnd(10)} med ${med.toFixed(3)} ms/frame  min ${times[0].toFixed(3)}  FPS(med) ${(1000 / med).toFixed(0)}`);
    return med;
}
const tOrig = await bench(rtA, 'original');
const tSmem = await bench(rtB, 'smem');
console.log(`\ndelta: ${(tOrig - tSmem).toFixed(3)} ms/frame  (${((tOrig / tSmem - 1) * 100).toFixed(1)}% ${tSmem < tOrig ? 'faster' : 'SLOWER'})`);


