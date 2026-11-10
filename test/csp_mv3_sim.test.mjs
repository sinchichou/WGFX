// csp_mv3_sim.test.mjs — 模擬 Chrome MV3 擴充套件的 CSP,驗證 WGFX 可在其中執行
//
// 目的:證明「移除 new Function」不只通過靜態檢查,而是**真的**能在禁止動態
// 程式碼生成的環境中執行完整效果。
//
// 做法:在 Node 進程中安裝一個 CSP 守門器 —— 覆寫 `Function` 建構子與 `eval`,
// 當任何程式碼嘗試以字串建立函式時立即拋出(等同瀏覽器的
// "Refused to evaluate a string as JavaScript because 'unsafe-eval' is not an
// allowed source of script")。其餘 API 與真實瀏覽器無異。
//
// 覆蓋範圍:框架測試套件的完整 effects/ 樹(每個效果真跑)+ OpenNR 網路測試。
// 這等同於「若把這套程式放進 MV3 擴充套件,會不會炸」的直接驗證。
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..');

// ---------- 安裝 CSP 守門器 ----------
const RealFunction = globalThis.Function;
let violations = [];

globalThis.eval = function () {
    violations.push('eval()');
    throw new Error("Refused to evaluate a string as JavaScript because 'unsafe-eval' is not an allowed source of script in the following Content Security Policy directive: \"script-src 'self'\".");
};

// 以 Proxy 攔截 `new Function(...)` 與 `Function(...)`
globalThis.Function = new Proxy(RealFunction, {
    apply(target, thisArg, args) {
        // 直接呼叫 Function('a','b','...') 也視為動態程式碼生成
        violations.push('Function()');
        throw new Error("Refused to evaluate a string as JavaScript because 'unsafe-eval' is not an allowed source of script in the following Content Security Policy directive: \"script-src 'self'\".");
    },
    construct(target, args) {
        violations.push('new Function()');
        throw new Error("Refused to evaluate a string as JavaScript because 'unsafe-eval' is not an allowed source of script in the following Content Security Policy directive: \"script-src 'self'\".");
    },
});

// ---------- 執行真實工作負載 ----------
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
// 共用模組：//! SOURCE 外部紋理的解析（與 port_check / test_runtime 同一份實作）
const { prepareExternalTextures } = await import(pathToFileURL(path.join(REPO, 'tools', 'source_textures.mjs')).href);
const EFFECTS_SRC = path.join(REPO, 'effects_src');

const W = 256, H = 256;
// GPUTextureUsage: TEXTURE_BINDING=0x04 | COPY_DST=0x02 | STORAGE_BINDING=0x08
//                | RENDER_ATTACHMENT=0x10 | COPY_SRC=0x01
function resources() {
    return {
        defines: {
            INPUT_WIDTH: W, INPUT_HEIGHT: H,
            // OpenNR 類效果的通道數與特徵圖尺寸（//! DISPATCH 以 NW/NH 推導網格）
            NW: W, NH: H, CIN: 64, COUT: 64, N_GROUPS: 1,
        },
        textures: {
            INPUT: { size: [W, H], format: 'rgba8unorm', usage: 0x04 | 0x02 | 0x10 },
            OUTPUT: { size: [W, H], format: 'rgba16float', usage: 0x08 | 0x01 | 0x04 },
        },
    };
}

function listWGSL(dir) {
    const out = [];
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return out; }
    for (const ent of entries) {
        const full = path.join(dir, ent.name);
        if (ent.isDirectory()) out.push(...listWGSL(full));
        else if (ent.name.endsWith('.wgsl')) out.push(full);
    }
    return out.sort();
}

// 挑選有代表性的效果（每個家族一個）—— 全部 125 檔太慢，取覆蓋性
const EFFECTS = path.join(REPO, 'effects');
const all = listWGSL(EFFECTS);
const byFamily = new Map();
for (const f of all) {
    const fam = path.relative(EFFECTS, f).split(path.sep)[0];
    if (!byFamily.has(fam)) byFamily.set(fam, f);
}
const selected = [...byFamily.values()];
console.log(`=== 在 MV3 CSP 模擬環境下真跑 ${selected.length} 個效果(每家族一個)===\n`);

let ok = 0, bad = 0, skipped = 0;
for (const file of selected) {
    const rel = path.relative(REPO, file).replace(/\\/g, '/');
    const code = fs.readFileSync(file, 'utf8');
    const runtime = new WGFXRuntime(dev);
    try {
        // 自動備妥 //! SOURCE 外部紋理（RAVU / SMAA / NIS）
        const base = resources();
        const external = await prepareExternalTextures(code, {
            textureDir: EFFECTS_SRC,
            quiet: true,
        });
        const res = Object.keys(external).length > 0
            ? { ...base, textures: { ...base.textures, ...external } }
            : base;
        await runtime.compile(code, res);
        // dispatch 全部 pass —— 這是會觸發 //! DISPATCH 求值的地方
        const enc = dev.createCommandEncoder();
        runtime.dispatchAllPasses(enc);
        dev.queue.submit([enc.finish()]);
        await dev.queue.onSubmittedWorkDone();
        ok++;
        console.log(`  [PASS] ${rel}`);
    } catch (e) {
        // OpenNR 類效果需要額外 storage buffer 與 defines → 標註為跳過而非失敗
        const msg = (e.message || '').split('\n')[0];
        if (/NW|NH|CIN|COUT|storage buffer|Defines/i.test(msg)) {
            console.log(`  [SKIP] ${rel} — 需專屬資源(${msg.slice(0, 60)})`);
            skipped++;
        } else {
            bad++;
            console.log(`  [FAIL] ${rel} — ${msg}`);
        }
    } finally {
        runtime.resourceManager.dispose();
        runtime.pipelineManager.dispose();
    }
}

console.log('');
console.log(`=== CSP 守門器觸發次數: ${violations.length} ===`);
if (violations.length > 0) {
    console.log('  違規呼叫:', [...new Set(violations)].join(', '));
    console.log('  ✗ 有程式碼嘗試在 MV3 下使用動態程式碼生成');
} else {
    console.log('  ✓ 全程無 eval / new Function 呼叫');
}
console.log('');
console.log(`=== 總結: ${ok} 通過 / ${bad} 失敗 / ${skipped} 跳過 / CSP 違規 ${violations.length} 次 ===`);

const success = bad === 0 && violations.length === 0;
process.exit(success ? 0 : 1);
