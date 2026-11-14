/**
 * outputFormat test — WGFXOptions.outputFormat / WGFXRuntime compile option.
 *
 * Proves:
 *   1. Default (no outputFormat) → OUTPUT texture is rgba16float and the WGSL
 *      declares rgba16float (historical behavior unchanged).
 *   2. outputFormat:'rgba8unorm' → the GPUTexture is rgba8unorm AND the WGSL
 *      declaration matches (the trap this feature had to avoid: bind-group
 *      format mismatch between the host texture and the WGSL declaration).
 *   3. The rgba8unorm output differs from the rgba16float output by ≤1/255 per
 *      channel (pure quantization, no semantic drift) — full GPU execution.
 *   4. Public API path: WGFX.create({outputFormat}) creates an rgba8unorm OUTPUT.
 *
 * Uses the low-level WGFXRuntime + manual encoder (Node has no DOM video
 * elements; same harness approach as test/bench_cnn_caseE.mjs).
 *
 * Run: node test/output_format.test.mjs   (needs a GPU via the webgpu polyfill)
 */
import { create, globals } from 'webgpu';
import assert from 'node:assert';
import { readFile } from 'node:fs/promises';
Object.assign(globalThis, globals);

const { WGFX, WGFXRuntime } = await import('../dist/wgfx.esm.js');
const { comparePixels } = await import('../src/verification/diff-core.ts').catch(
    () => import('../dist/verification.esm.js').then(m => ({ comparePixels: m.comparePixels }))
);

const nav = { gpu: create([]) };
const adapter = await nav.gpu.requestAdapter();
const device = await adapter.requestDevice();

const EFFECT = await readFile(new URL('../effects/Scalers/Bilinear.wgsl', import.meta.url), 'utf8');
const W = 64, H = 64;

let pass = 0, fail = 0;
async function check(name, fn) {
    try { await fn(); console.log(`  ✅ ${name}`); pass++; }
    catch (e) { console.log(`  ❌ ${name}\n     ${String(e.message).slice(0, 300)}`); fail++; }
}

/** checker pattern (RGBA8) */
function checker(w, h) {
    const d = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const c = ((x >> 3) + (y >> 3)) % 2;
        d[i] = c ? 255 - x * 4 : x * 4;
        d[i + 1] = y * 4;
        d[i + 2] = c ? 128 : 0;
        d[i + 3] = 255;
    }
    return d;
}

/** Compile + run Bilinear once; return format / wgsl / decoded RGBA8 output. */
async function runOnce(compileOpts = {}) {
    const rt = new WGFXRuntime(device);
    await rt.compile(EFFECT, {
        defines: { INPUT_WIDTH: W, INPUT_HEIGHT: H },
        textures: {
            INPUT: {
                size: [W, H], format: 'rgba8unorm',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            },
            OUTPUT: {
                size: [W, H],
                format: compileOpts.outputFormat ?? 'rgba16float',
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC |
                    GPUTextureUsage.TEXTURE_BINDING,
            },
        },
    }, { honorDeclaredOutput: false, ...compileOpts });

    const outTex = rt.resourceManager.getTexture('OUTPUT');
    const format = outTex.format;
    const wgsl = rt.generatedModules[0].wgslCode;

    // upload pattern
    const input = rt.resourceManager.getTexture('INPUT');
    device.queue.writeTexture({ texture: input }, checker(W, H),
        { bytesPerRow: W * 4, rowsPerImage: H }, [W, H]);
    rt.resourceManager.updateSceneBuffer(W, H, W, H);

    // single encode+submit
    const e = device.createCommandEncoder();
    rt.dispatchAllPasses(e);
    device.queue.submit([e.finish()]);
    await device.queue.onSubmittedWorkDone();

    // readback
    const is16f = format.includes('16float');
    const bpp = is16f ? 8 : 4;
    const bpr = Math.ceil(W * bpp / 256) * 256;
    const buf = device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e2 = device.createCommandEncoder();
    e2.copyTextureToBuffer({ texture: outTex },
        { buffer: buf, bytesPerRow: bpr, rowsPerImage: H }, [W, H]);
    device.queue.submit([e2.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const raw = new Uint8Array(buf.getMappedRange());
    const bytes = new Uint8Array(raw); // copy before unmap
    buf.unmap(); buf.destroy();

    return { format, wgsl, bytes, bpr, is16f };
}

/** half-float (IEEE 754 binary16) → number */
function f16ToNumber(h) {
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, m = h & 0x03ff;
    if (e === 0) return (m ? m * 2 ** -24 : 0) * (s ? -1 : 1);
    if (e === 0x1f) return m ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * (1 + m / 1024) * 2 ** (e - 15);
}
function decode16(raw, bpr, w, h) {
    const u16 = new Uint16Array(raw.buffer, raw.byteOffset, (bpr / 2) * h);
    const out = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const s = (y * (bpr / 2) + x * 4), d = (y * w + x) * 4;
        for (let c = 0; c < 4; c++) {
            out[d + c] = Math.max(0, Math.min(255, Math.round(f16ToNumber(u16[s + c]) * 255)));
        }
    }
    return out;
}
function decode8(raw, bpr, w, h) {
    const out = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) out.set(raw.subarray(y * bpr, y * bpr + w * 4), y * w * 4);
    return out;
}

console.log('🧪 outputFormat tests (Bilinear 64x64, adapter:', adapter.info?.vendor || '?', ')');
console.log('');

await check('default → OUTPUT rgba16float, WGSL declares rgba16float', async () => {
    const r = await runOnce({});
    assert.equal(r.format, 'rgba16float', `texture format = ${r.format}`);
    assert.ok(r.wgsl.includes('texture_storage_2d<rgba16float, write>'),
        'WGSL must declare rgba16float');
});

await check("outputFormat:'rgba8unorm' → texture AND WGSL both rgba8unorm", async () => {
    const r = await runOnce({ outputFormat: 'rgba8unorm' });
    assert.equal(r.format, 'rgba8unorm', `texture format = ${r.format}`);
    assert.ok(r.wgsl.includes('texture_storage_2d<rgba8unorm, write>'),
        'WGSL must declare rgba8unorm (bind-group consistency)');
    assert.ok(!r.wgsl.includes('texture_storage_2d<rgba16float'),
        'no stale rgba16float declaration may remain');
});

await check('GPU outputs differ ≤1/255 per channel (quantization only)', async () => {
    const r16 = await runOnce({});
    const r8 = await runOnce({ outputFormat: 'rgba8unorm' });
    const a = decode16(r16.bytes, r16.bpr, W, H);
    const b = decode8(r8.bytes, r8.bpr, W, H);
    const m = comparePixels(a, b, W, H);
    assert.ok(m.maxAbsDiff <= 1,
        `max abs diff = ${m.maxAbsDiff} (expected ≤1), MAE = ${m.mae}`);
});

await check('WGFX.create({outputFormat}) public API creates rgba8unorm OUTPUT', async () => {
    const wgfx = await WGFX.create({ device, effectCode: EFFECT, width: W, height: H, outputFormat: 'rgba8unorm' });
    const t = wgfx.runtime.resourceManager.getTexture('OUTPUT');
    assert.equal(t.format, 'rgba8unorm');
    assert.ok(wgfx.runtime.generatedModules[0].wgslCode.includes('texture_storage_2d<rgba8unorm, write>'));
    wgfx.dispose();
});

await check('WGFX.create default keeps rgba16float (no regression)', async () => {
    const wgfx = await WGFX.create({ device, effectCode: EFFECT, width: W, height: H });
    const t = wgfx.runtime.resourceManager.getTexture('OUTPUT');
    assert.equal(t.format, 'rgba16float');
    wgfx.dispose();
});

console.log('');
console.log(`✅ ${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
