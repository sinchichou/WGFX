// test/bench_hotpath.mjs — WGFX 每幀熱路徑微基準（P1~P4 優化前後同輪對照）
//
// 執行：node test/bench_hotpath.mjs [--rounds N] [--size N]
//
// 為何另開一支基準：test/bench_runtime.mjs 用的是 Anime4K_Upscale_Denoise_S（4 pass，
// 無 //! DISPATCH），量不到 P1（每幀 new Function 重新編譯 DISPATCH 運算式）的差異。
// 本檔以合成效果把各項成本拆開量測，並以「同輪多回合取樣 + min/median」對抗 GPU 噪音：
//
//   A. //! DISPATCH：16 個 buffer-style pass，每 pass 2 條運算式（每幀 32 次求值）→ P1
//   B. getOutput()：每次呼叫建立 TextureView → P2
//   C. updateSceneBuffer()：場景資訊每幀更新（Logger.debug 字串組建）→ P3
//   D. 4-pass CNN 每幀 dispatchAllPasses + submit → P4（整體熱路徑）
//
// 本檔只用公開 API，故可直接對「優化前」的 dist 取得 before 數字。
import { create, globals } from 'webgpu';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.join(HERE, '..');
const { WGFXRuntime } = await import('../dist/wgfx.esm.js');

const argv = process.argv.slice(2);
let ROUNDS = 5;
let W = 256, H = 256;
for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--rounds') ROUNDS = Math.max(1, Number(argv[++i]) | 0);
    else if (argv[i] === '--size') W = H = Math.max(16, Number(argv[++i]) | 0);
}

const adapter = await nav.gpu.requestAdapter();
const dev = await adapter.requestDevice({ requiredLimits: { maxStorageTexturesPerShaderStage: 8 } });
const uncaptured = [];
dev.onuncapturederror = (e) => uncaptured.push(String((e.error && e.error.message) || e));

// ---- 共用：多回合取樣，回報 min / median（GPU 與其他 agent 共用時以 min 為準） ----
function stats(perOp) {
    const s = [...perOp].sort((a, b) => a - b);
    const median = s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
    return { min: s[0], median, all: perOp };
}

async function bench(label, iters, fn, unit = 'us') {
    const scale = unit === 'us' ? 1e3 : 1e6; // ns → us / ns → ms
    // 暖機（含 JIT）
    fn(0);
    await dev.queue.onSubmittedWorkDone();
    const perOp = [];
    for (let r = 0; r < ROUNDS; r++) {
        const t0 = performance.now();
        fn(iters);
        const t1 = performance.now();
        perOp.push((t1 - t0) * scale / iters);
    }
    const { min, median } = stats(perOp);
    const fmt = (v) => v >= 100 ? v.toFixed(1) : v.toFixed(3);
    console.log(`  ${label.padEnd(46)} min=${fmt(min).padStart(9)} ${unit}  median=${fmt(median).padStart(9)} ${unit}`);
    console.log(`      rounds: [${perOp.map((v) => fmt(v)).join(', ')}]`);
    await dev.queue.onSubmittedWorkDone();
    return { label, min, median, unit };
}

function externalResources(w, h) {
    return {
        defines: { INPUT_WIDTH: w, INPUT_HEIGHT: h, NW: w, NH: h },
        textures: {
            INPUT: {
                size: [w, h], format: 'rgba8unorm',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            },
            OUTPUT: {
                size: [w, h], format: 'rgba16float',
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
            },
        },
    };
}

// ---- 合成 //! DISPATCH 效果（8 pass × 2 運算式）----
function dispatchEffect(dispatchPasses = 8) {
    const head = [
        '//! VERSION 4',
        '//! TEXTURE INPUT',
        '//! WIDTH INPUT_WIDTH',
        '//! HEIGHT INPUT_HEIGHT',
        'Texture2D INPUT;',
        '//! TEXTURE OUTPUT',
        '//! WIDTH INPUT_WIDTH',
        '//! HEIGHT INPUT_HEIGHT',
        'Texture2D OUTPUT;',
        '',
    ].join('\n');
    const passes = [];
    for (let i = 1; i <= dispatchPasses; i++) {
        const dx = i % 2 === 0 ? '(NW*NH+63)/64' : '(OUTPUT_WIDTH+7)/8';
        const dy = i % 2 === 0 ? '1' : '(OUTPUT_HEIGHT+7)/8';
        passes.push([
            `//! PASS ${i}`,
            '//! IN INPUT',
            '//! OUT OUTPUT',
            `//! DISPATCH ${dx}, ${dy}`,
            '//! NUM_THREADS 8,8,1',
            `fn Pass${i}(@builtin(global_invocation_id) gid: vec3<u32>) {`,
            '    let p = vec2<i32>(i32(gid.x), i32(gid.y));',
            '    textureStore(OUTPUT, p, textureLoad(INPUT, p, 0));',
            '}',
            '',
        ].join('\n'));
    }
    return head + passes.join('\n');
}

console.log(`=== WGFX 熱路徑微基準 (${W}x${H}, rounds=${ROUNDS}) ===`);
console.log(`(同輪取樣，GPU 可能與其他 agent 共用 → 以 min/median 判讀，勿看單一回合)`);

// ===== A. //! DISPATCH 每幀求值 =====
// 16 個 DISPATCH pass × 2 條運算式 = 每幀 32 次運算式求值；修正前每次都 new Function(...)。
{
    const rt = new WGFXRuntime(dev);
    await rt.compile(dispatchEffect(16), externalResources(W, H));
    await bench(`A. DISPATCH 16-pass encode+submit / frame`, 200, (n) => {
        for (let i = 0; i < n; i++) {
            const e = dev.createCommandEncoder();
            rt.dispatchAllPasses(e);
            dev.queue.submit([e.finish()]);
        }
    });
    rt.resourceManager.dispose();
}

// ===== B. getOutput() 視圖取得 =====
{
    const rt = new WGFXRuntime(dev);
    await rt.compile(dispatchEffect(1), externalResources(W, H));
    await bench('B. getOutput() 取得輸出視圖 / call', 200000, (n) => {
        for (let i = 0; i < n; i++) rt.getOutput();
    }, 'ns');
    rt.resourceManager.dispose();
}

// ===== C. updateSceneBuffer() 場景更新 =====
{
    const rt = new WGFXRuntime(dev);
    await rt.compile(dispatchEffect(1), externalResources(W, H));
    await bench('C. updateSceneBuffer() / call', 100000, (n) => {
        for (let i = 0; i < n; i++) rt.resourceManager.updateSceneBuffer(W, H, W, H);
    }, 'ns');
    rt.resourceManager.dispose();
}

// ===== D. 4-pass CNN 每幀 dispatchAllPasses（整體熱路徑）=====
{
    const shaderText = fs.readFileSync(
        path.join(REPO, 'examples', 'Anime4K_Upscale_Denoise_S.wgsl'), 'utf8'
    );
    const rt = new WGFXRuntime(dev);
    await rt.compile(shaderText, externalResources(W, H));
    rt.resourceManager.updateSceneBuffer(W, H, W, H);
    await bench('D. 4-pass CNN dispatchAllPasses+submit / frame', 300, (n) => {
        for (let i = 0; i < n; i++) {
            const e = dev.createCommandEncoder();
            rt.dispatchAllPasses(e);
            dev.queue.submit([e.finish()]);
        }
    });
    rt.resourceManager.dispose();
}

if (uncaptured.length > 0) {
    console.error(`[GPU 未捕捉錯誤] ${uncaptured.join(' | ')}`);
    process.exit(1);
}
console.log('=== done ===');
process.exit(0);
