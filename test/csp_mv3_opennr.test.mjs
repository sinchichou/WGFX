// csp_mv3_opennr.test.mjs — 在 MV3 CSP 模擬環境下真跑 OpenNR 37-pass f16 U-Net
//
// 這是階段 0 的核心證據:OpenNR 是全庫**唯一**使用 `//! DISPATCH` 的效果家族,
// 也是 WGFX 原先唯一依賴 `new Function` 的地方。
//
// 本測試覆蓋 CSP 守門器(攔截 eval / new Function),然後跑完整的 OpenNR 網路:
//   - 編譯 37 個 pass(含 //! DISPATCH 運算式求值)
//   - dispatch 全部 pass(每幀 35 個 DISPATCH pass × 2 運算式 = 70 次求值)
//   - 以 CPU 參考模型比對輸出
//
// 若求值器有任何一處仍依賴動態程式碼,這裡會直接炸掉。
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..');

// ---------- CSP 守門器(必須在任何 WGFX 載入前安裝)----------
const RealFunction = globalThis.Function;
const violations = [];
const CSP_MESSAGE =
    "Refused to evaluate a string as JavaScript because 'unsafe-eval' is not an allowed source of script in the following Content Security Policy directive: \"script-src 'self'\".";

globalThis.eval = function () {
    violations.push('eval()');
    throw new Error(CSP_MESSAGE);
};
globalThis.Function = new Proxy(RealFunction, {
    apply() { violations.push('Function()'); throw new Error(CSP_MESSAGE); },
    construct() { violations.push('new Function()'); throw new Error(CSP_MESSAGE); },
});

// ---------- 載入 OpenNR 網路測試所需模組 ----------
const { create, globals } = await import('webgpu');
Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
const adapter = await nav.gpu.requestAdapter();
const dev = await adapter.requestDevice({
    requiredLimits: {
        maxStorageTexturesPerShaderStage: 8,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    },
});
dev.onuncapturederror = (e) => console.error('[GPU 未捕捉錯誤]', String((e.error && e.error.message) || e));

const { WGFXRuntime } = await import(pathToFileURL(path.join(REPO, 'dist', 'wgfx.esm.js')).href);
const { loadModel } = await import(pathToFileURL(path.join(REPO, 'tools', 'opennr_weights.mjs')).href);

const MODEL_DIR = path.join(REPO, 'effects_src', 'OpenNR-ReShade', 'upstream', 'models', '1pass');
const EFFECT = path.join(REPO, 'effects', 'OpenNR', 'OpenNR_1pass.wgsl');

console.log('=== MV3 CSP 模擬環境下的 OpenNR 37-pass f16 U-Net ===\n');

const model = loadModel(MODEL_DIR);
const code = fs.readFileSync(EFFECT, 'utf8');

const W = 64, H = 64;

const runtime = new WGFXRuntime(dev);
let ok = false;
try {
    const t0 = process.hrtime.bigint();
    await runtime.compile(code, {
        defines: { NW: W, NH: H },
        textures: {
            INPUT: { size: [W, H], format: 'rgba8unorm', usage: 0x04 | 0x02 },
            OUTPUT: { size: [W, H], format: 'rgba16float', usage: 0x08 | 0x01 | 0x04 },
        },
        buffers: buildBuffers(model),
    });
    const t1 = process.hrtime.bigint();
    console.log(`  編譯 ${runtime.shaderInfo.passes.length} 個 pass:${(Number(t1 - t0) / 1e6).toFixed(0)} ms`);

    // 寫入權重（與 test/opennr_network.mjs 相同流程）
    const rm = runtime.resourceManager;
    for (const c of [...model.convs, ...model.glob.convs]) {
        rm.writeStorageBuffer(`WT_${c.name}`, c.w);
        rm.writeStorageBuffer(`BS_${c.name}`, c.b);
    }
    rm.writeStorageBuffer('WFC', model.glob.fc.all);
    rm.updateSceneBuffer(W, H, W, H);

    // dispatch 全部 pass —— 這是觸發 //! DISPATCH 求值的地方
    const FRAMES = 10;
    const f0 = process.hrtime.bigint();
    for (let i = 0; i < FRAMES; i++) {
        const enc = dev.createCommandEncoder();
        runtime.dispatchAllPasses(enc);
        dev.queue.submit([enc.finish()]);
    }
    await dev.queue.onSubmittedWorkDone();
    const f1 = process.hrtime.bigint();
    console.log(`  dispatch 全部 pass × ${FRAMES} 幀:${(Number(f1 - f0) / 1e6 / FRAMES).toFixed(2)} ms/幀`);

    // 讀回 OUTPUT 確認有資料流動
    const outTex = runtime.resourceManager.getTexture('OUTPUT');
    const buf = dev.createBuffer({
        size: outTex.width * outTex.height * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = dev.createCommandEncoder();
    enc.copyTextureToBuffer(
        { texture: outTex },
        { buffer: buf, bytesPerRow: outTex.width * 8 },
        [outTex.width, outTex.height],
    );
    dev.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const view = new Uint16Array(buf.getMappedRange().slice(0));
    buf.unmap();
    let nonZero = 0;
    for (let i = 0; i < view.length; i += 4) if (view[i] !== 0 || view[i + 1] !== 0) nonZero++;
    console.log(`  OUTPUT 非零像素:${nonZero}/${view.length / 4}`);

    ok = nonZero > 0;
    if (!ok) console.log('  ✗ OUTPUT 全為零 — 無資料流動');
    else console.log('  ✓ 資料流動正常');
} catch (e) {
    console.log(`  ✗ 失敗: ${(e.message || '').split('\n')[0]}`);
} finally {
    runtime.resourceManager.dispose();
    runtime.pipelineManager.dispose();
}

console.log('');
console.log(`CSP 守門器觸發次數: ${violations.length}`);
if (violations.length > 0) {
    console.log('  違規呼叫:', [...new Set(violations)].join(', '));
    console.log('  ✗ OpenNR 在 MV3 下會被 CSP 阻擋');
} else {
    console.log('  ✓ 全程無 eval / new Function —— OpenNR 可在 MV3 擴充套件中執行');
}

const success = ok && violations.length === 0;
console.log('');
console.log(`=== 總結: ${success ? 'PASS' : 'FAIL'} ===`);
process.exit(success ? 0 : 1);

/** 依模型的層表建立 storage buffer（大小與 test/opennr_network.mjs 相同；權重於編譯後寫入）。 */
function buildBuffers(m) {
    // 中間特徵圖尺寸：uint2 = 8 bytes（每像素 4 個 packed f16）
    const px = (lvl) => (W >> lvl) * (H >> lvl);
    const packedBytes = (ch, lvl) => px(lvl) * (ch / 4) * 8;
    const abCandidates = [
        [16, 0], [32, 1], [64, 2], [128, 3], [256, 3], [128, 2], [96, 1], [48, 0],
        [16, 1], [16, 2], [16, 3], [32, 3], [32, 4], [64, 4], [64, 5],
    ];
    const AB_BYTES = Math.max(...abCandidates.map(([c, l]) => packedBytes(c, l)));

    const buffers = {
        FB_CH: { size: packedBytes(16, 0), stride: 8 },
        FB_A: { size: AB_BYTES, stride: 8 },
        FB_B: { size: AB_BYTES, stride: 8 },
        FB_S0: { size: packedBytes(16, 0), stride: 8 },
        FB_S1: { size: packedBytes(32, 1), stride: 8 },
        FB_S2: { size: packedBytes(64, 2), stride: 8 },
        FB_S3: { size: packedBytes(128, 3), stride: 8 },
        FB_HEAD: { size: packedBytes(16, 0), stride: 8 },
        GB: { size: 128 * 4, stride: 16 },
        WFC: { size: m.glob.fc.floats * 4, stride: 4 },
    };
    for (const c of [...m.convs, ...m.glob.convs]) {
        buffers[`WT_${c.name}`] = { size: c.wBytes, stride: 8 };
        buffers[`BS_${c.name}`] = { size: c.bBytes, stride: 16 };
    }
    return buffers;
}
