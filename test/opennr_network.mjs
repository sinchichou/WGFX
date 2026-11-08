// OpenNR 全網 GPU 驗收：effects/OpenNR/OpenNR_1pass.wgsl（37 pass）vs CPU oracle（opennr_ref.mjs）
//
// 驗收內容：
//   1. 全部 37 pass 編譯 + dispatch 無驗證錯誤。
//   2. 全網逐通道 ≤1e-2：19 層 conv（enc0a..out）的 GPU 輸出 vs CPU forward，逐通道比對。
//   3. skip 特徵圖 FB_S0..S3 逐通道 ≤1e-2。
//   4. 噪聲時域：同一 stage 在 frame 0 與 frame 1 產生不同 CH（>50% 位元不同），
//      且色彩通道不受 frame 影響。
//   5. composite：Strength=0 時 OUTPUT ≈ INPUT（f16 容差）。
//
// 讀回策略（重要）：
//   node-webgpu（Dawn Node 綁定）在「多次 submit + 多次 mapAsync」交錯時會不穩定地
//   拋出無訊息的 Error（空 message/stack），且崩潰點隨機。本測試因此採用
//   **單一 command encoder + 單一 submit + 單一 mapAsync**：
//   在各 conv pass 之後把該層輸出複製到一個快照緩衝區，全部 pass 跑完後再一次讀回。
//   這同時貼近正式執行路徑（WGFXRuntime.dispatchAllPasses：單 encoder 全 pass）。
import { create, globals } from 'webgpu';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadModel } from '../tools/opennr_weights.mjs';
import { loadModel as loadRefModel, forward, h4 } from './opennr_ref.mjs';

Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
const HERE = path.dirname(fileURLToPath(import.meta.url));
const { WGFXRuntime } = await import('../dist/wgfx.esm.js');

const MODEL_DIR = path.join(HERE, '..', 'effects_src', 'OpenNR-ReShade', 'upstream', 'models', '1pass');
const NW = 64, NH = 64;

let fail = 0;
const check = (name, ok, detail = '') => {
    console.log(`  ${ok ? '[PASS]' : '[FAIL]'} ${name}${detail ? ' — ' + detail : ''}`);
    if (!ok) fail++;
};

const adapter = await nav.gpu.requestAdapter();
const dev = await adapter.requestDevice();
dev.onuncapturederror = (e) => console.error('  [GPU 未捕捉錯誤]', e.error && e.error.message);
process.on('uncaughtException', (e) => {
    console.error('UNCAUGHT name=' + (e && e.name) + ' msg=' + (e && e.message) +
        ' str=' + String(e) + ' props=' + (e ? Object.getOwnPropertyNames(e).join(',') : '-'));
    console.error('stack=' + (e && e.stack));
    process.exit(2);
});
process.on('unhandledRejection', (e) => { console.error('UNHANDLED', e); process.exit(2); });

const model = loadModel(MODEL_DIR);      // 權重 + 尺寸（GPU 端）
const refModel = loadRefModel(MODEL_DIR); // CPU oracle（forward 需要其層表命名）

// ---- buffer 尺寸 ----
const px = (lvl) => (NW >> lvl) * (NH >> lvl);
const packedBytes = (ch, lvl) => px(lvl) * (ch / 4) * 8; // uint2 = 8 bytes

// ping-pong A/B 需容納所有中間特徵圖（含 upcat 輸出）
const abCandidates = [
    [16, 0], [32, 1], [64, 2], [128, 3], [256, 3], [128, 2], [96, 1], [48, 0], // main + upcat outs
    [16, 1], [16, 2], [16, 3], [32, 3], [32, 4], [64, 4], [64, 5],             // global branch maps
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
    WFC: { size: model.glob.fc.floats * 4, stride: 4 },
};
for (const c of [...model.convs, ...model.glob.convs]) {
    buffers[`WT_${c.name}`] = { size: c.wBytes, stride: 8 };
    buffers[`BS_${c.name}`] = { size: c.bBytes, stride: 16 };
}

// ---- 編譯 ----
const code = fs.readFileSync(path.join(HERE, '..', 'effects', 'OpenNR', 'OpenNR_1pass.wgsl'), 'utf8');
const INPUT_BYTES = new Uint8Array(NW * NH * 4);
for (let y = 0; y < NH; y++) {
    for (let x = 0; x < NW; x++) {
        const i = (y * NW + x) * 4;
        const v = (x * 7 + y * 13 + ((x * y) % 97)) % 256; // 與 CPU src 相同的圖樣
        INPUT_BYTES[i] = v; INPUT_BYTES[i + 1] = 255 - v; INPUT_BYTES[i + 2] = (v * 3) % 256; INPUT_BYTES[i + 3] = 255;
    }
}
// CPU src：byte/255（rgba8unorm 正規化）
const cpuSrc = (x, y) => {
    const i = (y * NW + x) * 4;
    return [INPUT_BYTES[i] / 255, INPUT_BYTES[i + 1] / 255, INPUT_BYTES[i + 2] / 255];
};

const runtime = new WGFXRuntime(dev);
dev.pushErrorScope('validation');
await runtime.compile(code, {
    defines: { NW, NH },
    textures: {
        INPUT: { size: [NW, NH], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST },
        OUTPUT: { size: [NW, NH], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC },
    },
    buffers,
});
const compileErr = await dev.popErrorScope();
check('37 pass 效果檔編譯', compileErr === null, compileErr ? compileErr.message : `${runtime.shaderInfo.passes.length} passes`);
console.log(`  解析到的 pass 數 = ${runtime.shaderInfo.passes.length}, buffer 數 = ${runtime.shaderInfo.buffers.length}`);

const rm = runtime.resourceManager;
// 寫入權重
for (const c of [...model.convs, ...model.glob.convs]) {
    rm.writeStorageBuffer(`WT_${c.name}`, c.w);
    rm.writeStorageBuffer(`BS_${c.name}`, c.b);
}
rm.writeStorageBuffer('WFC', model.glob.fc.all);
// 寫入 INPUT
dev.queue.writeTexture({ texture: rm.getTexture('INPUT') }, INPUT_BYTES,
    { bytesPerRow: NW * 4, rowsPerImage: NH }, [NW, NH, 1]);

// ---- CPU oracle（onLayer 收集每層輸出）----
const cpuLayers = new Map();
const trace = forward(refModel, NW, NH, {
    frame: 0, tone: 1.0, structure: 1.0, src: cpuSrc,
    onLayer: (name, fm) => cpuLayers.set(name, fm),
});

// conv 層名 → 該層輸出的 GPU buffer（與 pass 換手一致）
const layerMap = {
    enc0a: ['FB_B', 16, 0], enc0b: ['FB_S0', 16, 0],
    enc1a: ['FB_B', 32, 1], enc1b: ['FB_S1', 32, 1],
    enc2a: ['FB_B', 64, 2], enc2b: ['FB_S2', 64, 2],
    enc3a: ['FB_B', 128, 3], enc3b: ['FB_S3', 128, 3],
    mid_a: ['FB_A', 128, 3], mid_b: ['FB_B', 128, 3],
    dec3a: ['FB_B', 128, 3], dec3b: ['FB_A', 128, 3],
    dec2a: ['FB_A', 64, 2], dec2b: ['FB_B', 64, 2],
    dec1a: ['FB_B', 32, 1], dec1b: ['FB_A', 32, 1],
    dec0a: ['FB_A', 16, 0], dec0b: ['FB_B', 16, 0],
    out: ['FB_HEAD', 16, 0],
};
const passLayer = { 11: 'enc0a', 12: 'enc0b', 14: 'enc1a', 15: 'enc1b', 17: 'enc2a', 18: 'enc2b',
    20: 'enc3a', 21: 'enc3b', 22: 'mid_a', 23: 'mid_b', 25: 'dec3a', 26: 'dec3b',
    28: 'dec2a', 29: 'dec2b', 31: 'dec1a', 32: 'dec1b', 34: 'dec0a', 35: 'dec0b', 36: 'out' };

// ---- 1. 全網前向：單一 encoder 跑完 37 pass，逐層快照，最後一次讀回 ----
// 快照計畫：pass → { buffer, bytes }（bytes 取該層實際資料量：npix * ch4 * 8）
const snapPlan = [];
for (const [passStr, lname] of Object.entries(passLayer)) {
    const fm = cpuLayers.get(lname);
    if (!fm) throw new Error(`CPU oracle missing layer "${lname}"`);
    const [buf] = layerMap[lname];
    snapPlan.push({ pass: Number(passStr), lname, buf, bytes: fm.npix * fm.ch4 * 8 });
}
snapPlan.sort((a, b) => a.pass - b.pass);
const snapByPass = new Map(snapPlan.map((s) => [s.pass, s]));
const SNAP_TOTAL = snapPlan.reduce((s, x) => s + x.bytes, 0);

const snapBuf = dev.createBuffer({
    size: SNAP_TOTAL,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    label: 'opennr-snapshot',
});
const stageBuf = dev.createBuffer({
    size: SNAP_TOTAL,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label: 'opennr-staging',
});

runtime.setFrame(0);
rm.updateSceneBuffer(NW, NH, NW, NH);
dev.pushErrorScope('validation');
{
    const cmd = dev.createCommandEncoder();
    let off = 0;
    for (const p of runtime.shaderInfo.passes) {
        runtime.dispatchPass(`PASS_${p.index}`, cmd);
        const spec = snapByPass.get(p.index);
        if (spec) {
            const src = rm.getStorageBuffer(spec.buf);
            if (!src) throw new Error(`storage buffer "${spec.buf}" missing`);
            cmd.copyBufferToBuffer(src, 0, snapBuf, off, spec.bytes);
            spec.off = off;
            off += spec.bytes;
        }
    }
    if (off !== SNAP_TOTAL) throw new Error(`snapshot size mismatch: ${off} != ${SNAP_TOTAL}`);
    dev.queue.submit([cmd.finish()]);
}
const runErr = await dev.popErrorScope();
check('全 37 pass dispatch 無驗證錯誤', runErr === null, runErr ? runErr.message : 'clean');

// 一次 copy + 一次 mapAsync 讀回所有快照
{
    const cmd = dev.createCommandEncoder();
    cmd.copyBufferToBuffer(snapBuf, 0, stageBuf, 0, SNAP_TOTAL);
    dev.queue.submit([cmd.finish()]);
}
await stageBuf.mapAsync(GPUMapMode.READ);
const SNAP = new Uint8Array(stageBuf.getMappedRange().slice(0));
stageBuf.unmap();
stageBuf.destroy();
snapBuf.destroy();

/** 取出某層快照的 u32 視圖 */
const snapU32 = (spec) => new Uint32Array(SNAP.buffer, spec.off, spec.bytes / 4);

/** 逐通道比較：GPU packed f16 vs CPU oracle */
function compare(fm, gpu) {
    const ch4 = fm.ch4;
    let bad = 0, worst = 0, worstAt = '';
    for (let i = 0; i < fm.npix; i++) {
        for (let c4 = 0; c4 < ch4; c4++) {
            const g4 = h4([gpu[(i * ch4 + c4) * 2], gpu[(i * ch4 + c4) * 2 + 1]]);
            const r4 = fm.get4(i, c4);
            for (let c = 0; c < 4; c++) {
                const d = Math.abs(g4[c] - r4[c]);
                if (d > worst) { worst = d; worstAt = `p=${i} c4=${c4} c=${c}`; }
                if (d > 1e-2) bad++;
            }
        }
    }
    return { bad, worst, worstAt };
}

let firstBad = null;
for (const spec of snapPlan) {
    const fm = cpuLayers.get(spec.lname);
    const r = compare(fm, snapU32(spec));
    const ok = r.bad === 0;
    if (!ok && !firstBad) firstBad = spec.lname;
    check(`層 ${spec.lname} (ch=${fm.ch} ${fm.npix}px) 逐通道 ≤1e-2`, ok,
        `Δ>1e-2 = ${r.bad}  worstΔ=${r.worst.toExponential(2)} (${r.worstAt})`);
}
if (firstBad) console.log(`  首個失敗層: ${firstBad}`);

// skip 特徵圖 S0..S3（enc0b/enc1b/enc2b/enc3b 的輸出即為 skip 緩衝區）
for (let i = 0; i < 4; i++) {
    const spec = snapPlan.find((s) => s.lname === `enc${i}b`);
    const skipRef = trace.skip[i];
    const r = compare(skipRef, snapU32(spec));
    check(`skip S${i} 逐通道 ≤1e-2`, r.bad === 0,
        `ch=${skipRef.ch} Δ>1e-2 = ${r.bad}, worstΔ=${r.worst.toExponential(2)}`);
}

// ---- 2. 噪聲時域（frame 0 / frame 1 各快照一次，單次 mapAsync）----
{
    const CH_BYTES = packedBytes(16, 0);
    const nSnap = dev.createBuffer({ size: CH_BYTES * 2, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const nStage = dev.createBuffer({ size: CH_BYTES * 2, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    for (let f = 0; f < 2; f++) {
        runtime.setFrame(f);
        rm.updateSceneBuffer(NW, NH, NW, NH);
        const cmd = dev.createCommandEncoder();
        runtime.dispatchPass('PASS_1', cmd);
        cmd.copyBufferToBuffer(rm.getStorageBuffer('FB_CH'), 0, nSnap, f * CH_BYTES, CH_BYTES);
        dev.queue.submit([cmd.finish()]);
    }
    const cmd = dev.createCommandEncoder();
    cmd.copyBufferToBuffer(nSnap, 0, nStage, 0, CH_BYTES * 2);
    dev.queue.submit([cmd.finish()]);
    await nStage.mapAsync(GPUMapMode.READ);
    const raw = nStage.getMappedRange().slice(0);
    nStage.unmap(); nStage.destroy(); nSnap.destroy();

    const ch0 = new Uint32Array(raw, 0, CH_BYTES / 4);
    const ch1 = new Uint32Array(raw, CH_BYTES, CH_BYTES / 4);
    const px4 = 4; // 16ch = 4 uint2/pixel
    let diff = 0, total = 0, cdiff = 0, ctot = 0;
    for (let p = 0; p < NW * NH; p++) {
        // 通道 0..3 = noise（元素 0）
        for (const off of [0, 1]) {
            if (ch0[p * px4 * 2 + off] !== ch1[p * px4 * 2 + off]) diff++;
            total++;
        }
        // 通道 4..7 = colour（元素 1）
        for (const off of [0, 1]) {
            if (ch0[p * px4 * 2 + 2 + off] !== ch1[p * px4 * 2 + 2 + off]) cdiff++;
            ctot++;
        }
    }
    check('噪聲隨 frame 變化（>50% 位元不同）', diff / total > 0.5, `frac=${(diff / total).toFixed(4)}`);
    check('色彩通道不受 frame 影響', cdiff === 0, `diff=${cdiff}/${ctot}`);
}

// ---- 3. composite：Strength=0 → OUTPUT ≈ INPUT ----
{
    runtime.setFrame(0);
    rm.updateSceneBuffer(NW, NH, NW, NH);
    runtime.updateUniform('Strength', 0);
    const cmd = dev.createCommandEncoder();
    runtime.dispatchPass('PASS_37', cmd);
    dev.queue.submit([cmd.finish()]);

    // 讀回 OUTPUT
    const bytesPerRow = Math.ceil((NW * 8) / 16) * 16;
    const staging = dev.createBuffer({ size: bytesPerRow * NH, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = dev.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: rm.getTexture('OUTPUT'), origin: [0, 0, 0] },
        { buffer: staging, bytesPerRow, rowsPerImage: NH }, [NW, NH, 1]);
    dev.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Uint16Array(staging.getMappedRange().slice(0));
    staging.unmap(); staging.destroy();

    const f16 = (h) => {
        const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
        if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
        if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
        return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    };
    const rowStride = bytesPerRow / 2;
    let bad = 0, worst = 0;
    for (let y = 0; y < NH; y++) {
        for (let x = 0; x < NW; x++) {
            const i = (y * rowStride + x * 4);
            for (let c = 0; c < 3; c++) {
                const got = f16(out[i + c]);
                const exp = INPUT_BYTES[(y * NW + x) * 4 + c] / 255;
                const d = Math.abs(got - exp);
                if (d > worst) worst = d;
                if (d > 1e-2) bad++;
            }
        }
    }
    check('composite Strength=0 → OUTPUT ≈ INPUT', bad === 0, `Δ>1e-2 = ${bad}, worstΔ=${worst.toExponential(2)}`);
    runtime.updateUniform('Strength', 100);
}

rm.dispose();
console.log(`\n=== 總結: ${fail === 0 ? 'ALL PASS' : fail + ' FAILED'} ===`);
dev.destroy?.();
process.exit(fail === 0 ? 0 : 1);
