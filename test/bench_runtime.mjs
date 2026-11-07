// 效能基準：編譯延遲 + 每幀編碼開銷（優化前後對照用）
// 執行：npm run build && npm run bench
// 量測項目：
//   1. compile() 牆鐘時間（3 次，新 runtime 實例）— 反映管線建立並行化成效
//   2. 每幀「編碼 + submit」CPU 端開銷 — 反映 bind group 快取、暫存重用成效
//      （每幀 = 建立 encoder → dispatch 全部 4 個 pass → submit；純 JS/CPU 端成本）
import { create, globals } from 'webgpu';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
const HERE = path.dirname(fileURLToPath(import.meta.url));
const { WGFXRuntime } = await import('../dist/wgfx.esm.js');

const adapter = await nav.gpu.requestAdapter();
const dev = await adapter.requestDevice({ requiredLimits: { maxStorageTexturesPerShaderStage: 8 } });
dev.onuncapturederror = (e) => console.error('[GPU 未捕捉錯誤]', String((e.error && e.error.message) || e));

const W = 256, H = 256;
const shaderText = fs.readFileSync(
    path.join(HERE, '..', 'examples', 'Anime4K_Upscale_Denoise_S.wgsl'), 'utf8'
);

const resources = {
    defines: { INPUT_WIDTH: W, INPUT_HEIGHT: H },
    textures: {
        INPUT: {
            size: [W, H], format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        }
    }
};

// ---- 1. 編譯延遲 ----
console.log(`--- compile() 延遲（${W}x${H}，4-pass CNN）---`);
for (let i = 0; i < 3; i++) {
    const r = new WGFXRuntime(dev);
    const t0 = performance.now();
    await r.compile(shaderText, resources);
    const t1 = performance.now();
    console.log(`compile #${i + 1}: ${(t1 - t0).toFixed(1)} ms`);
    r.resourceManager.dispose();
}

// ---- 2. 每幀編碼開銷 ----
const runtime = new WGFXRuntime(dev);
await runtime.compile(shaderText, resources);
const rm = runtime.resourceManager;

// 填入測試圖樣
const inPx = new Uint8Array(W * H * 4);
for (let i = 0; i < W * H; i++) {
    inPx[i * 4] = (i * 7) % 256; inPx[i * 4 + 1] = (i * 13) % 256;
    inPx[i * 4 + 2] = (i * 29) % 256; inPx[i * 4 + 3] = 255;
}
dev.queue.writeTexture({ texture: rm.getTexture('INPUT') }, inPx,
    { bytesPerRow: W * 4, rowsPerImage: H }, [W, H, 1]);
rm.updateSceneBuffer(W, H, W * 2, H * 2);

const passNames = runtime.shaderInfo.passes.map(p => `PASS_${p.index}`);

function encodeFrame() {
    const encoder = dev.createCommandEncoder();
    for (const n of passNames) runtime.dispatchPass(n, encoder);
    dev.queue.submit([encoder.finish()]);
}

// 暖機 + 等待 GPU 排空
for (let i = 0; i < 100; i++) encodeFrame();
await dev.queue.onSubmittedWorkDone();

const FRAMES = 2000;
const t0 = performance.now();
for (let i = 0; i < FRAMES; i++) encodeFrame();
const t1 = performance.now(); // 編碼/submit 為非同步，不等待 GPU
const perFrameUs = (t1 - t0) * 1000 / FRAMES;
await dev.queue.onSubmittedWorkDone(); // 確認全部 GPU 工作確實完成、無錯誤

console.log(`--- 每幀編碼開銷（4 passes，${FRAMES} 幀）---`);
console.log(`per-frame: ${perFrameUs.toFixed(1)} us  (${(1000 / (perFrameUs / 1000) / 1000).toFixed(0)}k fps 編碼上限)`);
runtime.resourceManager.dispose();
process.exit(0);
