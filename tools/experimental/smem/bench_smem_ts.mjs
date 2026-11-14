/**
 * bench_smem_ts.mjs — O2 GPU-only timing via timestamp-query.
 *
 * Measures passes 2+3 (the transformed conv passes) of CuNNy-2x4C-NVL,
 * original vs shared-memory variant, K-repeated inside one command buffer
 * to beat D3D12's 65.536µs timestamp granularity (same technique as
 * test/bench_cnn_caseE.mjs, which also documents the BigInt epoch handling).
 *
 * Run: node --experimental-strip-types tools/experimental/smem/bench_smem_ts.mjs [size]
 */
import { create, globals } from 'webgpu';
import { readFile } from 'node:fs/promises';
Object.assign(globalThis, globals);

const { WGFXRuntime } = await import('../../../dist/wgfx.esm.js');

const nav = { gpu: create([]) };
const adapter = await nav.gpu.requestAdapter();
if (!adapter.features.has('timestamp-query')) {
    console.log('timestamp-query unavailable — cannot measure GPU time precisely.');
    process.exit(1);
}
const device = await adapter.requestDevice({
    requiredFeatures: ['timestamp-query'],
    requiredLimits: {
        maxStorageTexturesPerShaderStage: Math.min(8, adapter.limits.maxStorageTexturesPerShaderStage),
        maxComputeWorkgroupStorageSize: Math.min(32768, adapter.limits.maxComputeWorkgroupStorageSize),
    },
});

const W = Number(process.argv[2] || 256), H = W, K = 48, CHUNK = 8, ROUNDS = 5;
const orig = await readFile(new URL('../../../effects/CuNNy/CuNNy-2x4C-NVL.wgsl', import.meta.url), 'utf8');
const smem = await readFile(new URL('./CuNNy-2x4C-NVL-smem.wgsl', import.meta.url), 'utf8');
const pixels = new Uint8Array(W * H * 4);
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = (y * W + x) * 4;
    pixels[i] = (x * 255 / W) | 0; pixels[i + 1] = (y * 255 / H) | 0; pixels[i + 3] = 255;
}

function makeRT(code) {
    const rt = new WGFXRuntime(device);
    return rt.compile(code, {
        defines: { INPUT_WIDTH: W, INPUT_HEIGHT: H },
        textures: {
            INPUT: { size: [W, H], format: 'rgba8unorm',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST },
            OUTPUT: { size: [W, H], format: 'rgba16float',
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC |
                    GPUTextureUsage.TEXTURE_BINDING },
        },
    }, { honorDeclaredOutput: false }).then(() => rt);
}

const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
const resolveBuf = device.createBuffer({ size: 256, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
const readBuf = device.createBuffer({ size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

async function measure(rt) {
    const pm = rt.pipelineManager;
    const passIdx = [2, 3]; // the transformed conv passes
    for (const i of passIdx) {
        const p = pm.pipelines.get(i);
        if (!p) throw new Error(`pipeline ${i} missing`);
    }
    // Warm the cached bind groups: dispatchPass builds them lazily; the raw
    // encodePass below reads p.bindGroup directly (null until first dispatch).
    // 暖機：dispatchPass 會懶建 bind group，下面的 raw encodePass 直接讀取。
    {
        const e = device.createCommandEncoder();
        rt.dispatchAllPasses(e);
        device.queue.submit([e.finish()]);
        await device.queue.onSubmittedWorkDone();
    }
    async function timeSpanChunked() {
        // per-chunk measurement (case E pattern): each chunk is its own command
        // buffer with begin/end timestamps around the K' (pass2+pass3) pairs.
        let total = 0n;
        for (let c = 0; c < K; c += CHUNK) {
            const count = Math.min(CHUNK, K - c);
            const enc = device.createCommandEncoder();
            for (let k = 0; k < count; k++) {
                for (let j = 0; j < passIdx.length; j++) {
                    const p = pm.pipelines.get(passIdx[j]);
                    const first = (k === 0 && j === 0);
                    const last = (k === count - 1 && j === passIdx.length - 1);
                    const ts = first ? { querySet, beginningOfPassWriteIndex: 0 }
                        : last ? { querySet, endOfPassWriteIndex: 1 } : undefined;
                    const pe = enc.beginComputePass(ts);
                    pe.setPipeline(p.computePipeline);
                    pe.setBindGroup(0, p.bindGroup);
                    pe.dispatchWorkgroups(p.gridDx, p.gridDy, 1);
                    pe.end();
                }
            }
            device.queue.submit([enc.finish()]);
            await device.queue.onSubmittedWorkDone();
            const e2 = device.createCommandEncoder();
            e2.resolveQuerySet(querySet, 0, 2, resolveBuf, 0);
            e2.copyBufferToBuffer(resolveBuf, 0, readBuf, 0, 16);
            device.queue.submit([e2.finish()]);
            await readBuf.mapAsync(GPUMapMode.READ);
            const raw = new BigInt64Array(readBuf.getMappedRange().slice(0));
            readBuf.unmap();
            if (c === 0 && globalThis.__tsDebug !== true) {
                globalThis.__tsDebug = true;
                console.log(`[debug] chunk0 raw[0]=${raw[0]} raw[1]=${raw[1]} diff=${raw[1] - raw[0]}`);
            }
            total += raw[1] - raw[0];
        }
        return Number(total) / 1e6 / K; // ms per (pass2+pass3) execution
    }
    // warmup
    await timeSpanChunked();
    const out = [];
    for (let r = 0; r < ROUNDS; r++) out.push(await timeSpanChunked());
    out.sort((a, b) => a - b);
    return out[Math.floor(out.length / 2)];
}

console.log(`=== O2 timestamp bench — CuNNy-2x4C conv passes (2+3) @ ${W}x${H}, K=${K} ×${ROUNDS} rounds ===`);
const rtA = await makeRT(orig);
const rtB = await makeRT(smem);
device.queue.writeTexture({ texture: rtA.resourceManager.getTexture('INPUT') }, pixels,
    { bytesPerRow: W * 4, rowsPerImage: H }, [W, H]);
const tA = await measure(rtA);
const tB = await measure(rtB);
console.log(`original (textureSampleLevel)  : ${tA.toFixed(4)} ms  (${(1000 / tA).toFixed(0)} FPS-eq)`);
console.log(`smem (workgroup tile + load)   : ${tB.toFixed(4)} ms  (${(1000 / tB).toFixed(0)} FPS-eq)`);
console.log(`GPU-time delta                 : ${(tA - tB).toFixed(4)} ms  (${((tA / tB - 1) * 100).toFixed(1)}% ${tB < tA ? 'faster' : 'SLOWER'})`);
