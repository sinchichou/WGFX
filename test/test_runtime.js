// test_runtime.js — WGFX 執行時驗證套件（Runtime Verification Suite）
// 以真實 WebGPU 裝置執行四組驗證：A. 核心範例 test.wgsl 端到端（參數元數據 / 預設值 / 動態更新）
// B. sampled/storaged 成對別名恆等與全 pass 驗證範圍；C. 四個 examples 編譯、逐 pass 執行並讀回非零輸出；
// D. 中間紋理缺少 WIDTH/HEIGHT 時拋出含紋理名稱的明確錯誤。執行：npm run build && node test/test_runtime.js

// ===== 區段 0：WebGPU 全域初始化（必須在載入 dist 之前完成） =====
// 注意：Node 的 globalThis.navigator 只有 getter，無法整個覆寫；
// 因此只把 GPU* 類別指到 globalThis，gpu 實例改用區域變數 nav。
import 'webgpu';
import { create, globals } from 'webgpu';
Object.assign(globalThis, globals);
const nav = { gpu: create([]) };

import * as fs from 'fs/promises';
import * as path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.join(HERE, '..', 'dist', 'wgfx.esm.js');
const EXAMPLES_DIR = path.join(HERE, '..', 'examples');

// ===== 區段 1：小型測試框架（結果收集 / 斷言） =====
const results = [];
const uncapturedErrors = [];

function record(name, pass, detail = '') {
    results.push({ name, pass });
    console.log(`  [${pass ? 'PASS' : 'FAIL'}] ${name}${detail ? ' — ' + detail : ''}`);
}

async function runTest(name, fn) {
    try {
        await fn();
        record(name, true);
    } catch (err) {
        record(name, false, (err && err.message) ? err.message : String(err));
    }
}

function assert(cond, msg) {
    if (!cond) throw new Error(msg);
}

// 浮點近似比對：任一分量超出容差即失敗
function approx(actual, expected, tol, label) {
    const fmt = `(${actual.map((v) => v.toFixed(4)).join(', ')})`;
    for (let i = 0; i < expected.length; i++) {
        if (!(Math.abs(actual[i] - expected[i]) <= tol)) {
            throw new Error(`${label}: 分量 ${i} 期望 ${expected[i]}（容差 ${tol}），實際 ${fmt}`);
        }
    }
}

// ===== 區段 2：共用輔助函式（f16 解碼 / 紋理讀回 / 資源樣板） =====

// f16()：將 rgba16float 的 Uint16 半精度值解碼為 number
function f16(h) {
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : ((s ? -Infinity : Infinity));
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

// readTexture()：COPY_SRC → staging buffer（bytesPerRow 對齊 16）→ Uint16Array
async function readTexture(device, texture, w, h) {
    const bytesPerRow = Math.ceil((w * 8) / 16) * 16; // rgba16float = 每像素 8 bytes
    const staging = device.createBuffer({
        size: bytesPerRow * h,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        label: 'test-staging',
    });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer(
        { texture, origin: [0, 0, 0] },
        { buffer: staging, bytesPerRow, rowsPerImage: h },
        [w, h, 1]
    );
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Uint16Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
}

// 以單一灰階值填滿 INPUT（含 alpha）
function writeSolidTexture(device, texture, w, h, value) {
    const data = new Uint8Array(w * h * 4).fill(value);
    device.queue.writeTexture({ texture }, data, { bytesPerRow: w * 4, rowsPerImage: h }, [w, h, 1]);
}

// 以漸層圖樣填滿 INPUT（確保 CNN 類範例有足夠的非零輸入）
function writePatternTexture(device, texture, w, h) {
    const data = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const i = (y * w + x) * 4;
            const v = (x * 7 + y * 13 + ((x * y) % 97)) % 256;
            data[i] = v;
            data[i + 1] = 255 - v;
            data[i + 2] = (v * 3) % 256;
            data[i + 3] = 255;
        }
    }
    device.queue.writeTexture({ texture }, data, { bytesPerRow: w * 4, rowsPerImage: h }, [w, h, 1]);
}

// 編譯用外部資源樣板：INPUT(rgba8unorm) 與 OUTPUT(rgba16float)
function externalResources(w, h) {
    return {
        defines: { INPUT_WIDTH: w, INPUT_HEIGHT: h },
        textures: {
            INPUT: {
                size: [w, h],
                format: 'rgba8unorm',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            },
            OUTPUT: {
                size: [w, h],
                format: 'rgba16float',
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
            },
        },
    };
}

// 在單一 validation error scope 內 dispatch 全部 pass 並 submit
async function runEffectPasses(device, runtime) {
    device.pushErrorScope('validation');
    const encoder = device.createCommandEncoder({ label: 'WGFX test frame' });
    for (const pass of runtime.shaderInfo.passes) {
        runtime.dispatchPass(`PASS_${pass.index}`, encoder);
    }
    device.queue.submit([encoder.finish()]);
    return device.popErrorScope();
}

// 取出指定像素（預設為圖像中央）的 (r,g,b,a) 浮點值
function samplePixel(view, w, h, pixelIndex) {
    const idx = (pixelIndex !== undefined) ? pixelIndex : ((h >> 1) * w + (w >> 1)) * 4;
    return [f16(view[idx]), f16(view[idx + 1]), f16(view[idx + 2]), f16(view[idx + 3])];
}

function hasNonZero(view) {
    for (let i = 0; i < view.length; i++) {
        if (view[i] !== 0) return true;
    }
    return false;
}

const EXAMPLE_NAMES = [
    'Anime4K_Restore_Soft_UL.wgsl',
    'Anime4K_Denoise_Bilateral_Mode.wgsl',
    'Anime4K_Upscale_Denoise_S.wgsl',
    'test.wgsl',
];

// ===== 區段 3：測試 A — 核心範例 test.wgsl 端到端（WGFX 高階類別） =====
async function testA(device, bundle) {
    const { WGFX } = bundle;
    const code = await fs.readFile(path.join(EXAMPLES_DIR, 'test.wgsl'), 'utf-8');
    const wgfx = await WGFX.create({ device, effectCode: code, width: 64, height: 64 });

    // 步驟 1：initialize() 應回傳兩筆參數元數據（型別與預設值）
    const info = wgfx.initialize();
    assert(info.uniforms.length === 2, `參數應有 2 筆，實際 ${info.uniforms.length}`);
    const inv = info.uniforms.find((u) => u.name === 'InvertAmount');
    assert(!!inv && inv.type === 'float' && inv.default === 1,
        `InvertAmount 元數據不符: ${JSON.stringify(inv)}`);
    const thr = info.uniforms.find((u) => u.name === 'Threshold');
    assert(!!thr && thr.type === 'int' && thr.default === 128,
        `Threshold 元數據不符: ${JSON.stringify(thr)}`);

    // 步驟 2：以純色 200 填滿 INPUT（200/255 = 0.7843）
    writeSolidTexture(device, wgfx.runtime.resourceManager.getTexture('INPUT'), 64, 64, 200);
    const outTex = wgfx.runtime.resourceManager.getTexture('OUTPUT');

    // 步驟 3：預設參數 (InvertAmount=1, Threshold=128) → 反轉亮度 55 < 128 → (0,0,0,1)
    let err = await runEffectPasses(device, wgfx.runtime);
    assert(err === null, `預設參數執行觸發驗證錯誤: ${err && err.message}`);
    let px = samplePixel(await readTexture(device, outTex, 64, 64), 64, 64, 0);
    approx(px, [0, 0, 0, 1], 0.001, '預設參數');

    // 步驟 4：Threshold=0 → 反轉結果通過門檻 → (0.2158, 0.2158, 0.2158, 0.7843)
    wgfx.updateUniforms({ Threshold: 0 });
    err = await runEffectPasses(device, wgfx.runtime);
    assert(err === null, `Threshold=0 執行觸發驗證錯誤: ${err && err.message}`);
    px = samplePixel(await readTexture(device, outTex, 64, 64), 64, 64, 0);
    approx(px, [0.2158, 0.2158, 0.2158, 0.7843], 0.01, 'Threshold=0');

    // 步驟 5：InvertAmount=0.5 → mix(0.7843, 0.2157, 0.5) = 0.5，alpha 維持 0.7843
    wgfx.updateUniforms({ InvertAmount: 0.5 });
    err = await runEffectPasses(device, wgfx.runtime);
    assert(err === null, `InvertAmount=0.5 執行觸發驗證錯誤: ${err && err.message}`);
    px = samplePixel(await readTexture(device, outTex, 64, 64), 64, 64, 0);
    approx(px, [0.5, 0.5, 0.5, 0.7843], 0.01, 'InvertAmount=0.5');

    wgfx.dispose();
}

// ===== 區段 4：測試 B — 成對別名恆等（Anime4K_Restore_Soft_UL，64x64） =====
async function testB(device, bundle) {
    const { WGFXRuntime } = bundle;
    const code = await fs.readFile(path.join(EXAMPLES_DIR, 'Anime4K_Restore_Soft_UL.wgsl'), 'utf-8');
    const runtime = new WGFXRuntime(device);
    await runtime.compile(code, externalResources(64, 64));
    // 編譯後由宿主寫入 scene buffer（低階 API 契約，同 WGFX.create 內部行為），
    // 否則 scene.inputSize=(0,0) 會讓所有 pass 提前返回。
    runtime.resourceManager.updateSceneBuffer(64, 64, 64, 64);
    const rm = runtime.resourceManager;

    // 斷言 1：tex1_sampled 與 tex1_storaged 解析到同一個 GPUTexture 物件
    const tSampled = rm.getTexture('tex1_sampled');
    assert(!!tSampled, '找不到 tex1_sampled 紋理');
    assert(tSampled === rm.getTexture('tex1_storaged'), 'tex1_sampled 與 tex1_storaged 不是同一個 GPUTexture');

    // 斷言 2：textures Map 以邏輯名稱 'tex1' 儲存，不重複配置成對紋理
    assert(rm.textures.has('tex1'),
        `textures Map 應含邏輯名稱 tex1（實際 keys: ${[...rm.textures.keys()].join(', ')}）`);
    assert(!rm.textures.has('tex1_sampled') && !rm.textures.has('tex1_storaged'),
        'textures Map 不應出現成對別名鍵');

    // 斷言 3：單一 validation error scope 內 dispatch 全部 pass，popErrorScope 應為 null
    writePatternTexture(device, rm.getTexture('INPUT'), 64, 64);
    const err = await runEffectPasses(device, runtime);
    assert(err === null, `執行全部 pass 觸發驗證錯誤: ${err && err.message}`);

    // 斷言 4：OUTPUT 具有非零資料
    const out = await readTexture(device, rm.getTexture('OUTPUT'), 64, 64);
    assert(hasNonZero(out), 'OUTPUT 全為 0（無資料流動）');
    runtime.resourceManager.dispose();
}

// ===== 區段 5：測試 C — 四個範例全部編譯並流動資料（256x256） =====
async function testCExample(device, bundle, name) {
    const { WGFXRuntime } = bundle;
    const code = await fs.readFile(path.join(EXAMPLES_DIR, name), 'utf-8');
    const runtime = new WGFXRuntime(device);
    try {
        await runtime.compile(code, externalResources(256, 256));
        // 編譯後由宿主寫入 scene buffer（低階 API 契約，同 WGFX.create 內部行為）
        runtime.resourceManager.updateSceneBuffer(256, 256, 256, 256);
        const rm = runtime.resourceManager;
        writePatternTexture(device, rm.getTexture('INPUT'), 256, 256);

        const err = await runEffectPasses(device, runtime);
        assert(err === null, `error scope 回報驗證錯誤: ${err && err.message}`);

        const out = await readTexture(device, rm.getTexture('OUTPUT'), 256, 256);
        const px = samplePixel(out, 256, 256);
        const status = hasNonZero(out) ? 'DATA' : 'EMPTY';
        console.log(`  ${name}: passes=${runtime.shaderInfo.passes.length} ` +
            `pixel=(${px.map((v) => v.toFixed(3)).join(', ')}) status=${status}`);
        assert(status === 'DATA', `${name} 的 OUTPUT 為空（EMPTY）`);
    } finally {
        runtime.resourceManager.dispose();
    }
}

// ===== 區段 6：測試 D — 中間紋理缺少 WIDTH/HEIGHT 時的明確錯誤 =====
async function testD(device, bundle) {
    const { WGFXRuntime } = bundle;
    // tex1 成對宣告刻意省略 WIDTH/HEIGHT，編譯必須失敗且訊息提及紋理名稱與 WIDTH/HEIGHT
    const badEffect = [
        '//!TEXTURE INPUT\n//!WIDTH INPUT_WIDTH\n//!HEIGHT INPUT_HEIGHT\nTexture2D INPUT;',
        '//!TEXTURE OUTPUT\n//!WIDTH INPUT_WIDTH\n//!HEIGHT INPUT_HEIGHT\nTexture2D OUTPUT;',
        '//!TEXTURE\n//!FORMAT R16G16B16A16_FLOAT\nvar tex1_sampled: texture_2d<f32>;',
        'var tex1_storaged: texture_storage_2d<rgba16float, read_write>;',
        '//!PASS 1\n//!IN INPUT\n//!OUT tex1_storaged\n//!NUM_THREADS 8,8',
        'fn Pass1(@builtin(global_invocation_id) gid: vec3<u32>) {',
        '    textureStore(tex1_storaged, vec2<i32>(gid.xy), vec4<f32>(1.0, 0.5, 0.25, 1.0));',
        '}',
    ].join('\n');

    let thrown = null;
    try {
        const runtime = new WGFXRuntime(device);
        await runtime.compile(badEffect, { defines: { INPUT_WIDTH: 64, INPUT_HEIGHT: 64 } });
    } catch (e) {
        thrown = e;
    }
    assert(thrown !== null, '缺少 WIDTH/HEIGHT 時未拋出錯誤');
    assert(/tex1/.test(thrown.message), `錯誤訊息未提及紋理名稱: ${thrown.message}`);
    assert(/WIDTH/.test(thrown.message) && /HEIGHT/.test(thrown.message),
        `錯誤訊息未提及 WIDTH/HEIGHT: ${thrown.message}`);
    console.log(`  錯誤訊息: ${thrown.message.split('\n')[0]}`);
}

// ===== 區段 7：主流程（裝置請求 → 依序執行測試 → 總結） =====
async function main() {
    console.log('=== WGFX Runtime 驗證套件 ===');

    // dist 產物檢查：不存在時以清楚訊息失敗（需先 npm run build）
    try {
        await fs.access(DIST);
    } catch {
        throw new Error(`找不到 ${DIST}；請先在 repo 根目錄執行 npm run build 產生 dist/wgfx.esm.js`);
    }
    // Windows 路徑需轉 file:// URL 才能動態 import
    const bundle = await import('file:///' + DIST.replace(/\\/g, '/'));

    // 1. 取得真實 GPU 裝置
    const adapter = await nav.gpu.requestAdapter();
    assert(adapter, '無法取得 WebGPU adapter');
    const device = await adapter.requestDevice({
        requiredLimits: { maxStorageTexturesPerShaderStage: 8 },
    });
    device.onuncapturederror = (e) => {
        const msg = String((e.error && e.error.message) || e);
        uncapturedErrors.push(msg);
        console.error(`  [GPU 未捕捉錯誤] ${msg}`);
    };
    console.log('WebGPU 裝置就緒');

    // 2. 依序執行測試 A–D
    await runTest('Test A：核心範例 test.wgsl 端到端（元數據 + 三段參數更新）', () => testA(device, bundle));
    await runTest('Test B：成對別名恆等（Anime4K_Restore_Soft_UL）', () => testB(device, bundle));
    for (const name of EXAMPLE_NAMES) {
        await runTest(`Test C：${name} 編譯 + 全 pass 資料流`, () => testCExample(device, bundle, name));
    }
    await runTest('Test D：缺少 WIDTH/HEIGHT 的明確錯誤', () => testD(device, bundle));

    // 3. 範圍外（未捕捉）的 GPU 錯誤不得出現
    if (uncapturedErrors.length > 0) {
        record('GPU 未捕捉錯誤檢查', false, uncapturedErrors.join(' | '));
    } else {
        record('GPU 未捕捉錯誤檢查', true);
    }

    // 4. 總結：X/Y tests passed，任何失敗以 exit code 1 結束
    const passed = results.filter((r) => r.pass).length;
    console.log('');
    console.log(`=== 總結: ${passed}/${results.length} tests passed ===`);
    // webgpu（Dawn）會讓事件迴圈保持存活，明確結束程序以避免掛起
    process.exit(passed === results.length ? 0 : 1);
}

// 主流程包在 try/catch，確保不留未處理的 rejection
try {
    await main();
} catch (err) {
    console.error('測試套件執行失敗（未捕捉錯誤）:');
    console.error(err);
    process.exit(1);
}
