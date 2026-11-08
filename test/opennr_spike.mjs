// OpenNR conv spike 驗收：storage-buffer conv（packed f16）vs CPU 參考（opennr_ref.mjs）
//
// 這是「WGFX storage buffer + f16 打包卷積」的端到端正確性證明，也是全網效果檔的地基。
// 對照 opennr_shaders.hpp 的 kConvHLSL：
//   - x/w 為 flat storage buffer（uint2 = 4 個 packed f16），b 為 StructuredBuffer<float4>（f32）
//   - f32 累加、TM=4 像素/執行緒、clamp 邊界（gZeroPad=0）
//   - 關鍵：kConvHLSL 的 gCout 傳入的是 **pad**（opennr_addon.cpp 的 conv 以 c.pad 當 gCout），
//     所以 shader 的 COUT define 必須是 pad，不是 cout。輸出 buffer 每像素寬 pad/4 個 uint2。
//
// 比對方式：不再要求 bit-exact（CPU 以 float64 累加、GPU 以 f32；且 f16 存在多個 NaN payload），
// 改以**逐通道 ≤ 1e-2** 的 f16 容差，與全網驗收同一標準；
// oracle 用 opennr_ref.mjs 的 conv3x3Fast（已由 opennr_conv_equiv.mjs 證明與可讀版位元相同）。
import { create, globals } from 'webgpu';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { FeatureMap, p4, h4, conv3x3Fast } from './opennr_ref.mjs';

Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
const HERE = path.dirname(fileURLToPath(import.meta.url));
const { WGFXRuntime } = await import('../dist/wgfx.esm.js');

const adapter = await nav.gpu.requestAdapter();
const dev = await adapter.requestDevice();

// ---- 決定性偽隨機（有限值；避免造出 NaN/Inf 的 f16 位元樣式）----
let _seed = 12345;
const rnd = () => {
    _seed = (Math.imul(_seed, 1103515245) + 12345) >>> 0;
    return ((_seed >>> 8) / 16777216) * 2 - 1; // [-1, 1)
};

/** 造一層隨機權重（packed f16）與 f32 bias，回傳 loadModel 相容的 layer 物件 */
function makeLayer(name, cin, cout, pad) {
    const w = new Uint32Array((9 * cin * pad / 4) * 2);
    for (let i = 0; i < w.length; i += 2) {
        const pk = p4([rnd() * 0.5, rnd() * 0.5, rnd() * 0.5, rnd() * 0.5]);
        w[i] = pk[0]; w[i + 1] = pk[1];
    }
    const b = new Float32Array(pad);
    for (let i = 0; i < b.length; i++) b[i] = ((i % 7) - 3) * 0.05;
    return { name, cin, cout, pad, w, b, act: true };
}

/** 造一個隨機特徵圖（packed f16），回傳 {fm, data} */
function makeInput(w, h, ch) {
    const fm = new FeatureMap(w, h, ch);
    for (let i = 0; i < fm.data.length; i += 2) {
        const pk = p4([rnd(), rnd(), rnd(), rnd()]);
        fm.data[i] = pk[0]; fm.data[i + 1] = pk[1];
    }
    return fm;
}

// ---- GPU 執行：單層 conv，讀回 o buffer ----
async function runCase({ name, W, H, cin, cout, pad, act }) {
    const npix = W * H, cin4 = cin / 4, cout4 = pad / 4;
    const code = fs.readFileSync(path.join(HERE, '..', 'effects', 'OpenNR', 'OpenNR_spike.wgsl'), 'utf8');

    const layer = makeLayer('spike', cin, cout, pad);
    layer.act = act;
    const x = makeInput(W, H, cin);

    const r = new WGFXRuntime(dev);
    await r.compile(code, {
        // COUT 必須是 pad（見檔頭）；groups = ceil((npix/TM) * (pad/4) / 64)
        defines: {
            NW: W, NH: H, CIN: cin, COUT: pad, ACT: act ? 1 : 0,
            N_GROUPS: Math.ceil((npix / 4) * cout4 / 64),
        },
        buffers: {
            x: { size: npix * cin4 * 8, stride: 8 },
            w: { size: 9 * cin * pad / 4 * 8, stride: 8 },
            b: { size: pad / 4 * 16, stride: 16 },
            o: { size: npix * cout4 * 8, stride: 8 },
        },
    });
    const rm = r.resourceManager;

    rm.writeStorageBuffer('x', x.data);
    rm.writeStorageBuffer('w', layer.w);
    rm.writeStorageBuffer('b', layer.b);

    const cmd = dev.createCommandEncoder();
    r.dispatchPass('PASS_1', cmd);
    dev.queue.submit([cmd.finish()]);
    await dev.queue.onSubmittedWorkDone();

    const gpu = new Uint32Array(await rm.readStorageBuffer('o'));

    // CPU oracle：conv3x3Fast（層參數與 GPU 完全相同）
    const ref = new FeatureMap(W, H, cout, pad);
    conv3x3Fast(x, ref, layer, act, false);

    // 逐通道比較（僅比較前 cout 個通道；pad 通道是緩衝墊，模型不使用）
    let bad = 0, worst = 0, worstAt = '';
    const chanCmp = Math.min(cout, pad);
    for (let p = 0; p < npix; p++) {
        const g4 = h4([gpu[p * cout4 * 2], gpu[p * cout4 * 2 + 1]]);
        const r4 = h4([ref.data[p * cout4 * 2], ref.data[p * cout4 * 2 + 1]]);
        for (let c = 0; c < Math.min(4, chanCmp); c++) {
            const d = Math.abs(g4[c] - r4[c]);
            if (d > worst) { worst = d; worstAt = `p=${p} c=${c}`; }
            if (d > 1e-2) bad++;
        }
    }
    // 若 pad 通道 > 4 也檢查（模型輸出層 cout=4 pad=16 的其餘通道仍由 shader 計算）
    for (let p = 0; p < npix; p++) {
        for (let c4 = 1; c4 < cout4; c4++) {
            const g4 = h4([gpu[p * cout4 * 2 + c4 * 2], gpu[p * cout4 * 2 + c4 * 2 + 1]]);
            const r4 = h4([ref.data[p * cout4 * 2 + c4 * 2], ref.data[p * cout4 * 2 + c4 * 2 + 1]]);
            for (let c = 0; c < 4; c++) {
                const d = Math.abs(g4[c] - r4[c]);
                if (d > worst) { worst = d; worstAt = `p=${p} c4=${c4} c=${c}`; }
                if (d > 1e-2) bad++;
            }
        }
    }

    const total = npix * cout4 * 4;
    const ok = bad === 0;
    console.log(`${name}: 逐通道 Δ>1e-2 = ${bad}/${total}  worstΔ=${worst.toExponential(2)} (${worstAt}) ${ok ? '[PASS]' : '[FAIL]'}`);
    r.resourceManager.dispose();
    return ok;
}

const cases = [
    { name: '(a) cin=16 pad=16 gelu  ', W: 16, H: 16, cin: 16, cout: 16, pad: 16, act: true },
    { name: '(b) cin=64 pad=128 gelu ', W: 8, H: 8, cin: 64, cout: 128, pad: 128, act: true },
    { name: '(c) cin=16 cout=4 pad=16 線性', W: 16, H: 16, cin: 16, cout: 4, pad: 16, act: false },
];

let all = true;
for (const c of cases) all = (await runCase(c)) && all;
console.log(`\nspike 總體: ${all ? '[PASS]' : '[FAIL]'}`);
dev.destroy?.();
process.exit(all ? 0 : 1);
