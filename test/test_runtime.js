// test_runtime.js — WGFX 執行時驗證套件（Runtime Verification Suite）
// 以真實 WebGPU 裝置執行四組驗證：A. 核心範例 test.wgsl 端到端（參數元數據 / 預設值 / 動態更新）
// B. sampled/storaged 成對別名恆等與全 pass 驗證範圍；C. 全部 examples（EXAMPLE_NAMES）編譯、逐 pass 執行並讀回非零輸出；
// D. 中間紋理缺少 WIDTH/HEIGHT 時拋出含紋理名稱的明確錯誤。
// 另含兩個回歸測試：F. codegen 過度包裹（test/fixtures/codegen_u32_wrap.wgsl）；
// G. //! DISPATCH 預編譯求值與 resize 快取失效。執行：npm run build && node test/test_runtime.js

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
// //! SOURCE 外部紋理的共用解析（與 test/port_check.mjs 同一份實作）
import { prepareExternalTextures } from '../tools/source_textures.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.join(HERE, '..', 'dist', 'wgfx.esm.js');
const EXAMPLES_DIR = path.join(HERE, '..', 'examples');
const EFFECTS_SRC_DIR = path.join(HERE, '..', 'effects_src');

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

// readTexture()：COPY_SRC → staging buffer（bytesPerRow 對齊 256）→ Uint16Array
async function readTexture(device, texture, w, h) {
    // bytesPerRow 必須是 256 的倍數（WebGPU 規範）
    const bytesPerRow = Math.ceil((w * 8) / 256) * 256; // rgba16float = 每像素 8 bytes
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
    // 第一波移植特效（Magpie src/Effects；真實 GPU 像素級驗證）
    'Anime4K_Upscale_S.wgsl',
    'Anime4K_Restore_S.wgsl',
    'FSR_x2.wgsl',
    'Anime4K_Denoise_Bilateral_Mean.wgsl',
    'Anime4K_Denoise_Bilateral_Median.wgsl',
    'test.wgsl',
];

// ===== 區段 2b：effects/ 樹的通用驗收清單 =====
//
// effects/ 下的移植效果一律納入 Test E；需要額外資源者（宿主必須餵 storage
// buffer 或 LUT 紋理）列於 test/effects-suite.json 的 skip 表，並附上原因。
// 那些效果由各自的專屬測試覆蓋（test/opennr_network.mjs、test/port_lut_check.mjs …）。
const EFFECTS_DIR = path.join(HERE, '..', 'effects');
const EFFECTS_SKIP_FILE = path.join(HERE, 'effects-suite.json');

async function loadEffectsSkipMap() {
    try {
        const raw = await fs.readFile(EFFECTS_SKIP_FILE, 'utf-8');
        const parsed = JSON.parse(raw);
        return parsed.skip || {};
    } catch (e) {
        if (e && e.code === 'ENOENT') return {};
        throw e;
    }
}

async function listWGSLFiles(dir) {
    const out = [];
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const ent of entries) {
        const full = path.join(dir, ent.name);
        if (ent.isDirectory()) {
            out.push(...await listWGSLFiles(full));
        } else if (ent.name.endsWith('.wgsl')) {
            out.push(full);
        }
    }
    return out.sort();
}

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

// ===== 區段 5：測試 C — EXAMPLE_NAMES 全部編譯並流動資料（256x256） =====
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

// ===== 區段 6b：測試 F — codegen「過度包裹」回歸（u32 場景尺寸 × vecN<f32> 建構子）=====
// 修正前：`vec4<f32>(scene.inputPt, vec2<f32>(scene.inputSize))` 的第二個引數
// 會被改寫成 `f32(vec2<f32>(scene.inputSize))` → Tint `no matching constructor for
// 'f32(vec2<f32>)'`（SMAA 移植時實際踩到）。SMAA 現行版本用逐分量繞法，不會覆蓋此 bug，
// 故以獨立最小 fixture（test/fixtures/codegen_u32_wrap.wgsl）納入本套件。
async function testF(device, bundle) {
    const { WGFXRuntime } = bundle;
    const code = await fs.readFile(path.join(HERE, 'fixtures', 'codegen_u32_wrap.wgsl'), 'utf-8');
    const runtime = new WGFXRuntime(device);
    try {
        // 1) 必須能編譯（bug 未修時 createComputePipelineAsync 會 reject）
        await runtime.compile(code, externalResources(64, 64));
        const wgsl = runtime.generatedModules[0].wgslCode;

        // 2) 產物字串：已是 f32 向量建構式的引數不得被再包一層
        assert(!/f32\s*\(\s*vec[234]\s*<\s*f32\s*>/.test(wgsl),
            '生成 WGSL 出現 f32(vecN<f32>(…)) 過度包裹');
        assert(wgsl.includes('vec2<f32>(scene.inputSize)'),
            '生成 WGSL 遺失合法引數 vec2<f32>(scene.inputSize)');

        // 3) 產物字串：真正的 u32 純量引數仍必須補 f32()（否則多引數向量建構子不合法）
        assert(wgsl.includes('f32(scene.inputSize.x)') && wgsl.includes('f32(scene.inputSize.y)'),
            '生成 WGSL 未對 u32 純量引數補 f32()');

        // 4) 實跑：無驗證錯誤、輸出有資料流動
        const rm = runtime.resourceManager;
        rm.updateSceneBuffer(64, 64, 64, 64);
        writePatternTexture(device, rm.getTexture('INPUT'), 64, 64);
        const err = await runEffectPasses(device, runtime);
        assert(err === null, `執行觸發驗證錯誤: ${err && err.message}`);
        assert(hasNonZero(await readTexture(device, rm.getTexture('OUTPUT'), 64, 64)),
            'OUTPUT 全為 0（轉換後的值未流入輸出）');
    } finally {
        runtime.resourceManager.dispose();
    }
}

// 檢查 rgba16float 輸出是否每個像素都被寫入（以 alpha 為覆蓋指標；未寫入區域為 0）
function assertAlphaCoverage(view, w, h, label) {
    let zeros = 0;
    for (let i = 3; i < w * h * 4; i += 4) if (view[i] === 0) zeros++;
    assert(zeros === 0, `${label}：${zeros}/${w * h} 個像素未被寫入（DISPATCH 網格不足或快取未失效）`);
}

// ===== 區段 6c：測試 G — //! DISPATCH 求值（P1 預編譯 + resize 快取失效）=====
// 兩個 buffer-style pass：Pass1 以 OUTPUT 尺寸推導網格，Pass2 以 defines(NW/NH) 推導。
// 驗證 (a) 預編譯路徑求值正確、(b) OUTPUT 紋理尺寸改變（資源版本遞增）後快取確實失效。
// 網格正確性以「輸出被寫入的範圍」量測：pass 寫入常數 alpha=1，未被 dispatch 到的
// 像素維持 0，因此任何網格不足（含快取未失效）都會留下可數的零像素。
async function testG(device, bundle) {
    const { WGFXRuntime } = bundle;
    const pass = (idx, dispatch) => [
        `//! PASS ${idx}`,
        '//! IN INPUT',
        '//! OUT OUTPUT',
        `//! DISPATCH ${dispatch}`,
        '//! NUM_THREADS 8,8,1',
        `fn Pass${idx}(@builtin(global_invocation_id) gid: vec3<u32>) {`,
        // 寫入常數（alpha=1）：覆蓋率即「DISPATCH 網格涵蓋的像素範圍」的直接量測。
        // 不可改為讀 INPUT——INPUT 只有 64x64，越界的 textureLoad 回傳 0，
        // 會讓 128x128 的覆蓋率檢查出現假性缺口（實測踩過）。
        '    textureStore(OUTPUT, vec2<i32>(gid.xy), vec4<f32>(1.0, 0.5, 0.25, 1.0));',
        '}',
    ].join('\n');
    const code = [
        '//! VERSION 4',
        '//! TEXTURE INPUT',
        '//! WIDTH INPUT_WIDTH',
        '//! HEIGHT INPUT_HEIGHT',
        'Texture2D INPUT;',
        '//! TEXTURE OUTPUT',
        '//! WIDTH INPUT_WIDTH',
        '//! HEIGHT INPUT_HEIGHT',
        'Texture2D OUTPUT;',
        pass(1, '(OUTPUT_WIDTH+7)/8, (OUTPUT_HEIGHT+7)/8'),
        pass(2, '(NW+7)/8, (NH+7)/8'),
    ].join('\n');

    const runtime = new WGFXRuntime(device);
    try {
        const res = externalResources(64, 64);
        res.defines = { ...res.defines, NW: 64, NH: 64 };
        await runtime.compile(code, res);
        const rm = runtime.resourceManager;
        rm.updateSceneBuffer(64, 64, 64, 64);
        writePatternTexture(device, rm.getTexture('INPUT'), 64, 64);

        // (a) DISPATCH 網格正確 → 每個像素都被寫入（alpha=1）
        let err = await runEffectPasses(device, runtime);
        assert(err === null, `64x64 執行觸發驗證錯誤: ${err && err.message}`);
        assertAlphaCoverage(await readTexture(device, rm.getTexture('OUTPUT'), 64, 64), 64, 64,
            '64x64 首次 DISPATCH');

        // (b) 把 OUTPUT 換成 128x128（createTexture → 資源版本遞增）：運算式必須重算為
        //     (128+7)/8 = 16 群組，否則右/下區域永遠不會被寫入（快取未失效）。
        rm.createTexture('OUTPUT', {
            size: [128, 128], format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
        });
        rm.updateSceneBuffer(64, 64, 128, 128);
        err = await runEffectPasses(device, runtime);
        assert(err === null, `128x128 執行觸發驗證錯誤: ${err && err.message}`);
        assertAlphaCoverage(await readTexture(device, rm.getTexture('OUTPUT'), 128, 128), 128, 128,
            '128x128 resize 後 DISPATCH');
    } finally {
        runtime.resourceManager.dispose();
    }
}

// ===== 區段 6：測試 E — effects/ 樹全部移植效果編譯並流動資料（256x256） =====
async function testEEffect(device, bundle, file, relPath) {
    const { WGFXRuntime } = bundle;
    const code = await fs.readFile(file, 'utf-8');
    const runtime = new WGFXRuntime(device);
    try {
        // 效果宣告 //! SOURCE 時自動備妥外部紋理（自 effects_src/ 依檔名解析），
        // 讓 LUT 系效果（RAVU / SMAA / NIS）也能由通用套件驗收，而不必列入 skip。
        const base = externalResources(256, 256);
        const externalTextures = await prepareExternalTextures(code, {
            textureDir: EFFECTS_SRC_DIR,
            quiet: true,
        });
        const resources = Object.keys(externalTextures).length > 0
            ? { ...base, textures: { ...base.textures, ...externalTextures } }
            : base;
        await runtime.compile(code, resources);
        const rm = runtime.resourceManager;
        const outTex = rm.getTexture('OUTPUT');
        const ow = outTex.width, oh = outTex.height;
        rm.updateSceneBuffer(256, 256, ow, oh);
        writePatternTexture(device, rm.getTexture('INPUT'), 256, 256);

        const err = await runEffectPasses(device, runtime);
        assert(err === null, `error scope 回報驗證錯誤: ${err && err.message}`);

        const out = await readTexture(device, outTex, ow, oh);
        assert(hasNonZero(out), 'OUTPUT 全為 0（無資料流動）');
        console.log(`  ${relPath}: passes=${runtime.shaderInfo.passes.length} out=${ow}x${oh} status=DATA`);
    } finally {
        runtime.resourceManager.dispose();
    }
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
    // 比照真實宿主索取 adapter 支援的上限：
    //   - maxStorageTexturesPerShaderStage:CNN 效果單一 pass 可寫多張 storage texture
    //   - maxComputeWorkgroupStorageSize:RAVU R3/R4 RGB 變體每 workgroup 需 19–23 KB
    //     （WebGPU 預設 16 KB；D3D 宿主本來就開 32 KB）→ 不索取會誤判為失敗
    const device = await adapter.requestDevice({
        requiredLimits: {
            maxStorageTexturesPerShaderStage: 8,
            maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        },
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
    await runTest('Test F：codegen 過度包裹回歸（u32 場景尺寸 × vecN<f32>）', () => testF(device, bundle));
    await runTest('Test G：//! DISPATCH 求值（預編譯 + resize 快取失效）', () => testG(device, bundle));

    // Test E：effects/ 樹的移植效果（需額外資源者依 test/effects-suite.json 排除）
    const skipMap = await loadEffectsSkipMap();
    const effectFiles = await listWGSLFiles(EFFECTS_DIR);
    const included = [];
    for (const file of effectFiles) {
        const rel = path.relative(EFFECTS_DIR, file).split(path.sep).join('/');
        if (skipMap[rel]) {
            console.log(`  [SKIP] effects/${rel} — ${skipMap[rel]}`);
            continue;
        }
        included.push({ file, rel });
    }
    for (const { file, rel } of included) {
        await runTest(`Test E：effects/${rel} 編譯 + 全 pass 資料流`,
            () => testEEffect(device, bundle, file, rel));
    }

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
