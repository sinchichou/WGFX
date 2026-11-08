// scalers_ref.mjs — Scalers 家族的 GPU vs CPU 參考比對
//
// 同時驗證 Magpie PS style 的 `pos` 語意（正規化 UV、texel 中心）：
//   * Nearest @1:1 必須與 INPUT **逐位元相同**（證明 pos 取 texel 中心而非角點）
//   * Bilinear @1:1 必須與 INPUT 逐通道相同（容差 1e-3）
//   * Nearest/Bilinear @2x 必須與 CPU 最近鄰／雙線性參考相符
//
// CPU 參考以「取樣點 = 輸出 texel 中心的 UV」對齊 GPU 的
// textureSampleLevel(POINT / LINEAR, CLAMP) 行為。
import { create, globals } from 'webgpu';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
const HERE = path.dirname(fileURLToPath(import.meta.url));
const { WGFXRuntime } = await import('../dist/wgfx.esm.js');

const W = 64, H = 64;
let fail = 0;
const check = (name, ok, detail = '') => {
    console.log(`  ${ok ? '[PASS]' : '[FAIL]'} ${name}${detail ? ' — ' + detail : ''}`);
    if (!ok) fail++;
};

const adapter = await nav.gpu.requestAdapter();
const dev = await adapter.requestDevice();
dev.onuncapturederror = (e) => console.error('  [GPU 未捕捉錯誤]', e.error && e.error.message);
process.on('uncaughtException', (e) => { console.error('UNCAUGHT', e && e.stack); process.exit(2); });
process.on('unhandledRejection', (e) => { console.error('UNHANDLED', e); process.exit(2); });

// ---------- CPU 參考 ----------
function f16(h) {
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

/** 產生測試圖樣（與 port_check 一致） */
function makeInput() {
    const d = new Uint8Array(W * H * 4);
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const i = (y * W + x) * 4;
            const v = (x * 7 + y * 13 + ((x * y) % 97)) % 256;
            d[i] = v; d[i + 1] = 255 - v; d[i + 2] = (v * 3) % 256; d[i + 3] = 255;
        }
    }
    return d;
}
const INPUT = makeInput();
const src = (x, y) => {
    const i = (y * W + x) * 4;
    return [INPUT[i] / 255, INPUT[i + 1] / 255, INPUT[i + 2] / 255, INPUT[i + 3] / 255];
};

/** CPU 最近鄰（POINT + CLAMP），uv = 輸出 texel 中心 */
function cpuNearest(ow, oh) {
    const out = [];
    for (let y = 0; y < oh; y++) {
        for (let x = 0; x < ow; x++) {
            const u = (x + 0.5) / ow, v = (y + 0.5) / oh;
            const sx = Math.min(W - 1, Math.max(0, Math.floor(u * W)));
            const sy = Math.min(H - 1, Math.max(0, Math.floor(v * H)));
            out.push(src(sx, sy));
        }
    }
    return out;
}

/** CPU 雙線性（LINEAR + CLAMP），uv = 輸出 texel 中心 */
function cpuBilinear(ow, oh) {
    const out = [];
    const cl = (a, n) => Math.min(n - 1, Math.max(0, a));
    for (let y = 0; y < oh; y++) {
        for (let x = 0; x < ow; x++) {
            const u = (x + 0.5) / ow, v = (y + 0.5) / oh;
            const fx = u * W - 0.5, fy = v * H - 0.5;
            const x0 = Math.floor(fx), y0 = Math.floor(fy);
            const wx = fx - x0, wy = fy - y0;
            const p = [];
            for (let c = 0; c < 4; c++) {
                const a = src(cl(x0, W), cl(y0, H))[c];
                const b = src(cl(x0 + 1, W), cl(y0, H))[c];
                const cc = src(cl(x0, W), cl(y0 + 1, H))[c];
                const d = src(cl(x0 + 1, W), cl(y0 + 1, H))[c];
                p.push((a * (1 - wx) + b * wx) * (1 - wy) + (cc * (1 - wx) + d * wx) * wy);
            }
            out.push(p);
        }
    }
    return out;
}

/** f32 → f16（**截斷**向零；Dawn 的實際行為，見 docs/porting-guide.md §6.1） */
function truncF16(v) {
    const f32 = new Float32Array(1);
    const u32 = new Uint32Array(f32.buffer);
    f32[0] = v;
    const bits = u32[0];
    const s = (bits >>> 16) & 0x8000;
    const e = (bits >>> 23) & 0xff;
    const m = bits & 0x7fffff;
    if (e === 0xff) return s | 0x7c00 | (m ? 0x200 : 0);
    const ex = e - 127 + 15;
    if (ex >= 31) return s | 0x7c00;
    if (ex <= 0) {
        const full = m | 0x800000;
        const sh = 14 - ex;
        if (sh > 24) return s;
        return s | (full >> sh);
    }
    return s | (ex << 10) | (m >> 13);
}

// ---------- GPU 執行 ----------
async function runEffect(file, outW, outH) {
    const code = fs.readFileSync(file, 'utf8');
    const runtime = new WGFXRuntime(dev);
    await runtime.compile(code, {
        defines: { INPUT_WIDTH: W, INPUT_HEIGHT: H },
        textures: {
            INPUT: { size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST },
            OUTPUT: { size: [outW, outH], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC },
        },
    });
    const rm = runtime.resourceManager;
    rm.updateSceneBuffer(W, H, outW, outH);
    dev.queue.writeTexture({ texture: rm.getTexture('INPUT') }, INPUT,
        { bytesPerRow: W * 4, rowsPerImage: H }, [W, H, 1]);
    const cmd = dev.createCommandEncoder();
    for (const p of runtime.shaderInfo.passes) runtime.dispatchPass(`PASS_${p.index}`, cmd);
    dev.queue.submit([cmd.finish()]);

    const bytesPerRow = Math.ceil((outW * 8) / 256) * 256;
    const staging = dev.createBuffer({ size: bytesPerRow * outH, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = dev.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: rm.getTexture('OUTPUT'), origin: [0, 0, 0] },
        { buffer: staging, bytesPerRow, rowsPerImage: outH }, [outW, outH, 1]);
    dev.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const raw = new Uint16Array(staging.getMappedRange().slice(0));
    staging.unmap(); staging.destroy();
    runtime.resourceManager.dispose();

    const rowStride = bytesPerRow / 2;
    const px = [];
    const bits = [];
    for (let y = 0; y < outH; y++) {
        for (let x = 0; x < outW; x++) {
            const i = y * rowStride + x * 4;
            px.push([f16(raw[i]), f16(raw[i + 1]), f16(raw[i + 2]), f16(raw[i + 3])]);
            bits.push([raw[i], raw[i + 1], raw[i + 2], raw[i + 3]]);
        }
    }
    return { px, bits };
}

/** 與參考比對：回傳 {bad, worst} */
function compare(gpu, ref, tol) {
    let bad = 0, worst = 0, worstAt = '';
    for (let i = 0; i < ref.length; i++) {
        for (let c = 0; c < 4; c++) {
            const d = Math.abs(gpu[i][c] - ref[i][c]);
            if (d > worst) { worst = d; worstAt = `p=${i} c=${c}`; }
            if (d > tol) bad++;
        }
    }
    return { bad, worst, worstAt };
}

// ---------- 測試 ----------
const SCALERS = path.join(HERE, '..', 'effects', 'Scalers');

console.log('=== Nearest @1:1（必須與 INPUT 位元相同）===');
{
    const { px, bits } = await runEffect(path.join(SCALERS, 'Nearest.wgsl'), W, H);
    const ref = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) ref.push(src(x, y));
    // 位元精確：GPU 的 f16 = trunc_f16(精確 f32 值)
    let badBits = 0, badVal = 0, worst = 0, worstAt = '';
    for (let i = 0; i < ref.length; i++) {
        for (let c = 0; c < 4; c++) {
            if (bits[i][c] !== truncF16(ref[i][c])) badBits++;
            const d = Math.abs(px[i][c] - ref[i][c]);
            if (d > worst) { worst = d; worstAt = `p=${i} c=${c}`; }
            if (d > 1e-3) badVal++;
        }
    }
    check('Nearest @1:1 ≡ INPUT（位元精確，模擬 f16 截斷）', badBits === 0,
        `位元不符=${badBits}/${ref.length * 4}`);
    check('Nearest @1:1 ≡ INPUT（值域，容差 1e-3）', badVal === 0,
        `Δ>1e-3 = ${badVal}, worstΔ=${worst.toExponential(2)} (${worstAt})`);
}

console.log('=== Bilinear @1:1（texel 中心取樣 → 應與 INPUT 相同）===');
{
    const { px } = await runEffect(path.join(SCALERS, 'Bilinear.wgsl'), W, H);
    const ref = [];
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) ref.push(src(x, y));
    const r = compare(px, ref, 1e-3);
    check('Bilinear @1:1 ≡ INPUT', r.bad === 0, `Δ>1e-3 = ${r.bad}, worstΔ=${r.worst.toExponential(2)} (${r.worstAt})`);
}

console.log('=== Nearest @2x（vs CPU 最近鄰參考）===');
{
    const { px } = await runEffect(path.join(SCALERS, 'Nearest.wgsl'), W * 2, H * 2);
    const ref = cpuNearest(W * 2, H * 2);
    const r = compare(px, ref, 1e-3);
    check('Nearest @2x ≡ CPU 最近鄰', r.bad === 0, `Δ>1e-3 = ${r.bad}, worstΔ=${r.worst.toExponential(2)} (${r.worstAt})`);
}

console.log('=== Bilinear @2x（vs CPU 雙線性參考）===');
{
    const { px } = await runEffect(path.join(SCALERS, 'Bilinear.wgsl'), W * 2, H * 2);
    const ref = cpuBilinear(W * 2, H * 2);
    const r = compare(px, ref, 1e-3);
    check('Bilinear @2x ≡ CPU 雙線性', r.bad === 0, `Δ>1e-3 = ${r.bad}, worstΔ=${r.worst.toExponential(2)} (${r.worstAt})`);
}

console.log(`\n=== 總結: ${fail === 0 ? 'ALL PASS' : fail + ' FAILED'} ===`);
dev.destroy?.();
process.exit(fail === 0 ? 0 : 1);
