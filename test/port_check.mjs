// port_check.mjs — 通用移植驗收（Magpie HLSL → WGFX WGSL）
//
// 用法：
//   node test/port_check.mjs <effect.wgsl> [更多檔案…] [--size N] [--quiet]
//
// 每個效果檢查：
//   1. 編譯無例外、無驗證錯誤。
//   2. 全部 pass dispatch 無驗證錯誤（單一 encoder、單一 submit）。
//   3. OUTPUT 非全黑（確實有資料流動）。
//   4. 報告輸出統計（min/max/mean）與對 INPUT 的平均絕對差，供移植者判斷合理性。
//
// 這是**通用**健全性檢查；數值正確性的強驗收請另寫專屬測試。
//
// 注意：node-webgpu（Dawn Node 綁定）在「多次 submit + 多次 mapAsync」交錯時
// 會不穩定地拋出無訊息的 Error，因此本工具每個效果只做一次 mapAsync。
import { create, globals } from 'webgpu';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
const HERE = path.dirname(fileURLToPath(import.meta.url));

const argv = process.argv.slice(2);
let SIZE = 256;
let quiet = false;
const files = [];
for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--size') { SIZE = Number(argv[++i]); continue; }
    if (a === '--quiet') { quiet = true; continue; }
    files.push(a);
}
if (files.length === 0) {
    console.error('usage: node test/port_check.mjs <effect.wgsl> [...] [--size N] [--quiet]');
    process.exit(2);
}

const { WGFXRuntime } = await import('../dist/wgfx.esm.js');

const adapter = await nav.gpu.requestAdapter();
const dev = await adapter.requestDevice();
const uncaptured = [];
dev.onuncapturederror = (e) => {
    uncaptured.push(String((e.error && e.error.message) || e));
};

let fail = 0;
const check = (name, ok, detail = '') => {
    if (!quiet) console.log(`  ${ok ? '[PASS]' : '[FAIL]'} ${name}${detail ? ' — ' + detail : ''}`);
    if (!ok) fail++;
};

// ---------- 紋理讀寫輔助 ----------
function f16(h) {
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function writePattern(device, texture, w, h) {
    const data = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const i = (y * w + x) * 4;
            const v = (x * 7 + y * 13 + ((x * y) % 97)) % 256;
            data[i] = v; data[i + 1] = 255 - v; data[i + 2] = (v * 3) % 256; data[i + 3] = 255;
        }
    }
    device.queue.writeTexture({ texture }, data, { bytesPerRow: w * 4, rowsPerImage: h }, [w, h, 1]);
}

async function readTexture(device, texture, w, h) {
    // bytesPerRow 必須是 256 的倍數（WebGPU 規範）
    const bytesPerRow = Math.ceil((w * 8) / 256) * 256;
    const staging = device.createBuffer({
        size: bytesPerRow * h,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture, origin: [0, 0, 0] },
        { buffer: staging, bytesPerRow, rowsPerImage: h }, [w, h, 1]);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Uint16Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
}

/** rgba16float 資料 → 每像素 [r,g,b,a] */
function decode(out, w, h) {
    const rowStride = Math.ceil((w * 8) / 256) * 256 / 2;
    const px = [];
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const i = y * rowStride + x * 4;
            px.push([f16(out[i]), f16(out[i + 1]), f16(out[i + 2]), f16(out[i + 3])]);
        }
    }
    return px;
}

// ---------- 主流程 ----------
for (const rel of files) {
    const file = path.resolve(rel);
    const name = path.relative(process.cwd(), file) || file;
    if (!quiet) console.log(`\n=== ${name} ===`);
    const code = fs.readFileSync(file, 'utf8');

    let runtime = null;
    try {
        runtime = new WGFXRuntime(dev);
        dev.pushErrorScope('validation');
        await runtime.compile(code, {
            defines: { INPUT_WIDTH: SIZE, INPUT_HEIGHT: SIZE },
            textures: {
                INPUT: {
                    size: [SIZE, SIZE], format: 'rgba8unorm',
                    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
                },
                OUTPUT: {
                    size: [SIZE, SIZE], format: 'rgba16float',
                    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
                },
            },
        });
        const compileErr = await dev.popErrorScope();
        if (compileErr) throw new Error('compile: ' + compileErr.message);

        const rm = runtime.resourceManager;
        const outTex = rm.getTexture('OUTPUT');
        const ow = outTex.width, oh = outTex.height;
        rm.updateSceneBuffer(SIZE, SIZE, ow, oh);
        writePattern(dev, rm.getTexture('INPUT'), SIZE, SIZE);

        // 單一 encoder 跑完所有 pass（貼近正式執行路徑）
        dev.pushErrorScope('validation');
        const cmd = dev.createCommandEncoder();
        for (const p of runtime.shaderInfo.passes) runtime.dispatchPass(`PASS_${p.index}`, cmd);
        dev.queue.submit([cmd.finish()]);
        const runErr = await dev.popErrorScope();
        if (runErr) throw new Error('dispatch: ' + runErr.message);

        const raw = await readTexture(dev, outTex, ow, oh);
        const px = decode(raw, ow, oh);

        let nonZero = 0;
        const mn = [Infinity, Infinity, Infinity, Infinity];
        const mx = [-Infinity, -Infinity, -Infinity, -Infinity];
        const sum = [0, 0, 0, 0];
        let nan = 0;
        for (const p of px) {
            for (let c = 0; c < 4; c++) {
                const v = p[c];
                if (Number.isNaN(v)) { nan++; continue; }
                if (v !== 0) nonZero++;
                if (v < mn[c]) mn[c] = v;
                if (v > mx[c]) mx[c] = v;
                sum[c] += v;
            }
        }
        const n = px.length;
        const mean = sum.map((s) => s / n);

        check(`${name} 編譯 + 全 pass 執行`, true,
            `passes=${runtime.shaderInfo.passes.length} out=${ow}x${oh}`);
        check(`${name} OUTPUT 非全黑`, nonZero > 0, `非零分量=${nonZero}/${n * 4}`);
        check(`${name} OUTPUT 無 NaN`, nan === 0, `NaN=${nan}`);
        if (!quiet) {
            console.log(`        min=[${mn.map((v) => v.toFixed(4)).join(', ')}] ` +
                `max=[${mx.map((v) => v.toFixed(4)).join(', ')}] ` +
                `mean=[${mean.map((v) => v.toFixed(4)).join(', ')}]`);
        }
    } catch (err) {
        try { await dev.popErrorScope(); } catch { /* scope 可能未 push */ }
        check(`${name} 編譯 + 全 pass 執行`, false, (err && err.message) || String(err));
    } finally {
        if (runtime) runtime.resourceManager.dispose();
    }
}

if (uncaptured.length) {
    console.error('\n[GPU 未捕捉錯誤]');
    for (const e of uncaptured) console.error('  ' + e);
    fail += uncaptured.length;
}

console.log(`\n=== 總結: ${fail === 0 ? 'ALL PASS' : fail + ' FAILED'} ===`);
dev.destroy?.();
process.exit(fail === 0 ? 0 : 1);
