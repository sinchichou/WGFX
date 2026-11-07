// 紅點重現探測（red-firefly reproduction probe）
// 場景：深色雜訊輸入（模擬 4:2:0 壓縮影片的暗部 chroma noise），逐特效跑完整管線後
// 讀回 OUTPUT，統計「R 遠大於 G/B」的紅色雜點（firefly）數量。
// 對照兩種模式：
//   (A) debugger 現況：WGFX.create 未傳 outWidth/outHeight → OUTPUT 建立為 1x
//   (B) 預期模式：OUTPUT 為效果宣告的 2x
// 執行：npm run build && node test/red_dot_probe.mjs（需真實 WebGPU 裝置）
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

const W = 256, H = 256;

// ---- deterministic PRNG（兩種模式用同一張輸入，公平對照）----
let seed = 987654321;
const rnd = () => {
    seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
};
const clamp255 = v => Math.max(0, Math.min(255, Math.round(v)));

function makeDarkNoisyFrame(w, h) {
    // 深色「樹葉」基底（luma 10~50/255，偏暗偏綠），每通道獨立 ±6/255 雜訊（模擬 4:2:0 chroma noise）
    const px = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const o = (y * w + x) * 4;
        const shade = rnd();
        const base = 10 + 40 * shade * shade; // 偏暗
        const noise = () => (rnd() - 0.5) * 12;
        let r = base * 0.85 + noise(), g = base * 1.05 + noise(), b = base * 0.75 + noise();
        // 0.4% 單像素 chroma 離群值（±25~40/255，僅作用在單一通道）——壓縮影片的典型 firefly 前身
        if (rnd() < 0.004) {
            const c = rnd();
            const spike = 25 + 15 * rnd();
            if (c < 0.45) r += spike; else if (c < 0.75) g += spike; else b += spike;
        }
        px[o] = clamp255(r); px[o + 1] = clamp255(g); px[o + 2] = clamp255(b); px[o + 3] = 255;
    }
    return px;
}

function f16(hbits) {
    const s = (hbits & 0x8000) >> 15, e = (hbits & 0x7c00) >> 10, m = hbits & 0x03ff;
    if (e === 0) return (m ? Math.pow(2, -14) * (m / 1024) : 0) * (s ? -1 : 1);
    if (e === 31) return s ? NaN : Infinity;
    return Math.pow(2, e - 15) * (1 + m / 1024) * (s ? -1 : 1);
}

async function readTexRGBA16(tex, w, h) {
    const bytesPerRow = Math.ceil(w * 8 / 256) * 256;
    const rb = dev.createBuffer({ size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const cmd = dev.createCommandEncoder();
    cmd.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow, rowsPerImage: h }, [w, h, 1]);
    dev.queue.submit([cmd.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const u16 = new Uint16Array(rb.getMappedRange());
    const out = new Float32Array(w * h * 4);
    const stride = bytesPerRow / 2;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 4; c++) {
        out[(y * w + x) * 4 + c] = f16(u16[y * stride + x * 4 + c]);
    }
    rb.unmap();
    rb.destroy();
    return out;
}

function countFireflies(data, w, h) {
    let red = 0, green = 0, blue = 0, nonFinite = 0, overshoot = 0;
    let worst = 0;
    for (let i = 0; i < w * h; i++) {
        const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
        if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b)) { nonFinite++; continue; }
        if (r > 1.0 || g > 1.0 || b > 1.0) overshoot++;
        const chromaExcess = Math.max(r - Math.max(g, b), g - Math.max(r, b), b - Math.max(r, g));
        if (chromaExcess > 0.12 && Math.max(r, g, b) > 0.15) {
            if (r - Math.max(g, b) === chromaExcess) red++;
            else if (g - Math.max(r, b) === chromaExcess) green++;
            else blue++;
        }
        worst = Math.max(worst, chromaExcess);
    }
    return { red, green, blue, nonFinite, overshoot, worstChroma: worst.toFixed(3) };
}

const EFFECTS = ['FSR_x2.wgsl', 'Anime4K_Upscale_S.wgsl', 'Anime4K_Restore_S.wgsl'];

// ---- 高階 API 驗證：WGFX.create 的輸出尺寸解析順序 ----
// 1) 未傳 outWidth/outHeight → 採納效果檔宣告（FSR/Upscale 為 2x）
// 2) 明確傳入 outWidth/outHeight → 宿主覆寫
// 3) 效果檔未宣告尺寸（test.wgsl）→ 退回輸入尺寸
{
    const { WGFX } = await import('../dist/wgfx.esm.js');
    const fsr = fs.readFileSync(path.join(HERE, '..', 'examples', 'FSR_x2.wgsl'), 'utf8');
    const t = fs.readFileSync(path.join(HERE, '..', 'examples', 'test.wgsl'), 'utf8');

    const a = await WGFX.create({ device: dev, effectCode: fsr, width: 64, height: 64 });
    const outA = a.runtime.resourceManager.getTexture('OUTPUT');
    console.log(`\n=== WGFX.create 輸出尺寸解析 ===`);
    console.log(`  FSR 未傳 outWidth → OUTPUT ${outA.width}x${outA.height}（預期 128x128）${outA.width === 128 && outA.height === 128 ? ' [PASS]' : ' [FAIL]'}`);

    const b = await WGFX.create({ device: dev, effectCode: fsr, width: 64, height: 64, outWidth: 100, outHeight: 100 });
    const outB = b.runtime.resourceManager.getTexture('OUTPUT');
    console.log(`  FSR 傳 outWidth=100 → OUTPUT ${outB.width}x${outB.height}（預期 100x100）${outB.width === 100 && outB.height === 100 ? ' [PASS]' : ' [FAIL]'}`);

    const c = await WGFX.create({ device: dev, effectCode: t, width: 64, height: 64 });
    const outC = c.runtime.resourceManager.getTexture('OUTPUT');
    console.log(`  test.wgsl（無宣告尺寸）→ OUTPUT ${outC.width}x${outC.height}（預期 64x64）${outC.width === 64 && outC.height === 64 ? ' [PASS]' : ' [FAIL]'}`);

    // dispatchAllPasses 熱路徑等價性：跑一幀並與逐 pass dispatch 對照像素
    const rmA = a.runtime.resourceManager;
    const inPx = new Uint8Array(64 * 64 * 4);
    for (let i = 0; i < 64 * 64; i++) { inPx[i * 4] = i % 256; inPx[i * 4 + 1] = (i * 3) % 256; inPx[i * 4 + 2] = (i * 7) % 256; inPx[i * 4 + 3] = 255; }
    dev.queue.writeTexture({ texture: rmA.getTexture('INPUT') }, inPx, { bytesPerRow: 256, rowsPerImage: 64 }, [64, 64, 1]);
    rmA.updateSceneBuffer(64, 64, outA.width, outA.height);
    let e1 = dev.createCommandEncoder(); a.runtime.dispatchAllPasses(e1); dev.queue.submit([e1.finish()]);
    await dev.queue.onSubmittedWorkDone();
    const pxAll = await readTexRGBA16(outA, outA.width, outA.height);
    let e2 = dev.createCommandEncoder();
    for (const p of a.runtime.shaderInfo.passes) a.runtime.dispatchPass(`PASS_${p.index}`, e2);
    dev.queue.submit([e2.finish()]);
    await dev.queue.onSubmittedWorkDone();
    const pxEach = await readTexRGBA16(outA, outA.width, outA.height);
    let same = true;
    for (let i = 0; i < pxAll.length; i++) if (Math.abs(pxAll[i] - pxEach[i]) > 1e-6) { same = false; break; }
    console.log(`  dispatchAllPasses ≡ 逐 pass dispatch：${same ? '[PASS]' : '[FAIL]'}`);
    a.dispose(); b.dispose(); c.dispose();
}

// ---- NaN 防護探測：AMD 近似倒數的存在理由 ----
// 影片暗部常被量化成完全平坦的 3x3 區塊（maxL-minL=0）或純黑（mx4=0）：
// 精確除法在這些 ring 上得 1/0=inf，再乘上 0 → NaN → rgba16float → 螢幕紅點；
// AMD 的位元技巧倒數對 0 回傳「有限巨大值」，0×rcp(0)=0，天然免疫。
{
    const code = fs.readFileSync(path.join(HERE, '..', 'examples', 'FSR_x2.wgsl'), 'utf8');
    for (const [name, fill] of [['平坦暗部 (8,8,8)', 8], ['純黑 (0,0,0)', 0]]) {
        const r = new WGFXRuntime(dev);
        await r.compile(code, {
            defines: { INPUT_WIDTH: W, INPUT_HEIGHT: H },
            textures: {
                INPUT: { size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST },
                OUTPUT: { size: [W, H], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING }
            }
        });
        const rm = r.resourceManager;
        dev.queue.writeTexture({ texture: rm.getTexture('INPUT') }, new Uint8Array(W * H * 4).fill(fill),
            { bytesPerRow: W * 4, rowsPerImage: H }, [W, H, 1]);
        rm.updateSceneBuffer(W, H, W, H);
        const cmd = dev.createCommandEncoder();
        for (const p of r.shaderInfo.passes) r.dispatchPass(`PASS_${p.index}`, cmd);
        dev.queue.submit([cmd.finish()]);
        await dev.queue.onSubmittedWorkDone();
        const data = await readTexRGBA16(rm.getTexture('OUTPUT'), W, H);
        const stats = countFireflies(data, W, H);
        console.log(`\n=== NaN 防護探測：FSR_x2 @ ${name} ===`);
        console.log(`  nonFinite(NaN/Inf) 像素數：${stats.nonFinite}（預期 0）${stats.nonFinite === 0 ? ' [PASS]' : ' [FAIL]'}`);
        r.resourceManager.dispose();
    }
}

for (const file of EFFECTS) {
    const code = fs.readFileSync(path.join(HERE, '..', 'examples', file), 'utf8');
    // 輸入本身的 firefly 基準
    seed = 987654321;
    const px = makeDarkNoisyFrame(W, H);
    let inRed = 0;
    for (let i = 0; i < W * H; i++) {
        const r = px[i * 4] / 255, g = px[i * 4 + 1] / 255, b = px[i * 4 + 2] / 255;
        if (r - Math.max(g, b) > 0.12 && r > 0.15) inRed++;
    }

    console.log(`\n=== ${file} ===`);
    console.log(`  input fireflies (R-maxGB>0.12): ${inRed}`);

    for (const [mode, outW, outH] of [['A: 1x (debugger 現況)', W, H], ['B: 2x (效果宣告尺寸)', W * 2, H * 2]]) {
        const r = new WGFXRuntime(dev);
        try {
            await r.compile(code, {
                defines: { INPUT_WIDTH: W, INPUT_HEIGHT: H },
                textures: {
                    INPUT: { size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST },
                    OUTPUT: { size: [outW, outH], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING }
                }
            });
            const rm = r.resourceManager;
            seed = 987654321;
            // 輸入已由 seed 重置生成，重寫同一張
            dev.queue.writeTexture({ texture: rm.getTexture('INPUT') }, px, { bytesPerRow: W * 4, rowsPerImage: H }, [W, H, 1]);
            rm.updateSceneBuffer(W, H, outW, outH);

            const cmd = dev.createCommandEncoder();
            for (const p of r.shaderInfo.passes) r.dispatchPass(`PASS_${p.index}`, cmd);
            dev.queue.submit([cmd.finish()]);
            await dev.queue.onSubmittedWorkDone();

            const outTex = rm.getTexture('OUTPUT');
            const data = await readTexRGBA16(outTex, outW, outH);
            const stats = countFireflies(data, outW, outH);
            console.log(`  ${mode}: OUTPUT ${outW}x${outH} → red=${stats.red} green=${stats.green} blue=${stats.blue} nonFinite=${stats.nonFinite} >1.0=${stats.overshoot} worstChroma=${stats.worstChroma}`);
        } catch (e) {
            console.log(`  ${mode}: FAILED — ${e.message}`);
        }
    }
}
dev.destroy?.();
