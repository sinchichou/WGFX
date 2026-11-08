// OpenNR 上游權重 → WGFX packed-f16 storage buffer 轉換器
//
// 讀 `effects_src/OpenNR-ReShade/upstream/models/<variant>/` 的 model.txt + *.bin，
// 輸出可直接 writeStorageBuffer 的型別化陣列與一份 JSON 清單。
//
// 佈局與 WGSL/HLSL 端索引完全一致（opennr_shaders.hpp:70 kConvHLSL）：
//   w : 元素 uint2（4 個 packed f16）= [tap][cin][cout4]，索引 (tap*cin + k4*4 + kk)*cout4 + n4
//       → 檔案為 f16 tap-major，直接「每 4 個 half 併成一個 u32 pair」即為最終 layout，
//         不需重排；本檔只做 little-endian 解讀。
//   b : 元素 float4（f32，**不是 f16**）→ StructuredBuffer<float4>，索引 [n4]
//   x : 元素 uint2 = [pixel][cin4]
//
// 依據：
//   - opennr_model.hpp:157  read(".w.bin", 9*cin*pad*2)      ← 用 pad，不是 cout
//   - opennr_model.hpp:158  read(".b.bin", pad*4)            ← f32
//   - opennr_model.hpp:164  buffer(9*cin*pad/4 個元素, stride 8)
//   - opennr_model.hpp:170  buffer(pad/4 個元素, stride 16)
//   - opennr_shaders.hpp:17-19 x/w 為 StructuredBuffer<uint2>，b 為 StructuredBuffer<float4>
//
// 用法：
//   node tools/opennr_weights.mjs               # 驗算 1pass + 2pass 並印表
//   node tools/opennr_weights.mjs --json        # 額外輸出 tools/out/*.json 清單
//   node tools/opennr_weights.mjs --bin         # 額外輸出 tools/out/*.w.packed.bin / *.b.f32.bin
//   node tools/opennr_weights.mjs --dump enc3a  # 印出某層前 8 個 f16 解碼值
//
// 程式化使用（task-3 消費端）：
//   import { loadModel, decodeF16, packWeights, DEFAULT_MODELS_DIR } from './tools/opennr_weights.mjs';
//   const m = loadModel('./effects_src/OpenNR-ReShade/upstream/models/1pass');

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = path.resolve(HERE, '..');
export const DEFAULT_MODELS_DIR = path.join(REPO_ROOT, 'effects_src', 'OpenNR-ReShade', 'upstream', 'models');
export const VARIANTS = ['1pass', '2pass'];

// ----------------------------------------------------------------- 基本工具

/** f16 位元 → f32（與 WGSL 端 f16tof32 同語義，含 subnormal 與 Inf/NaN） */
export function decodeF16(h) {
    const s = (h & 0x8000) ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    if (e === 0) return s * m * 5.9604644775390625e-8; // 2^-24
    if (e === 31) return m ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + m / 1024);
}

/** f32 → f16 位元（RNE，與 WGSL 端 f32tof16 同語義） */
export function encodeF16(v) {
    const b = new ArrayBuffer(4);
    new Float32Array(b)[0] = v;
    const u = new Uint32Array(b)[0];
    const sign = (u >>> 16) & 0x8000;
    const exp = (u >>> 23) & 0xff;
    const man = u & 0x7fffff;
    if (exp === 0xff) return (sign | 0x7c00 | (man !== 0 ? 0x200 : 0)) >>> 0;
    let e = exp - 112;
    if (e >= 31) return (sign | 0x7c00) >>> 0;
    if (e >= 1) {
        let m10 = (man + 0x0fff + ((man >> 13) & 1)) >> 13;
        if (m10 > 0x3ff) { m10 = 0; e += 1; if (e >= 31) return (sign | 0x7c00) >>> 0; }
        return (sign | (e << 10) | m10) >>> 0;
    }
    if (exp === 0) return sign >>> 0; // f32 denormal < f16 最小 subnormal → 0
    const full = man | 0x800000;
    const sh = 126 - exp;
    if (sh > 34) return sign >>> 0;
    const q = full >>> sh;
    const rem = full & ((1 << sh) - 1);
    const half = 1 << (sh - 1);
    let m10 = q + ((rem > half || (rem === half && (q & 1))) ? 1 : 0);
    if (m10 >= 0x400) return (sign | 0x400) >>> 0;
    return (sign | m10) >>> 0;
}

// ----------------------------------------------------------------- model.txt

/**
 * 解析 model.txt。
 * 格式（opennr_model.hpp:123-135）：
 *   <cin> <base> <levels> <cout>            第 1 行（4 欄，但 fscanf 讀 5 欄）
 *   <n>                                     第 2 行：層數
 *   <name> <cin> <cout> <pad> <level>       n 行（第 5 欄是 level！）
 *   global <gn> <g_in> <g_hidden> <g_out>   選用
 *   <name> <cin> <cout> <pad> <level>       gn 行
 * 注意：第 1 行只有 4 個數字，第 5 個（層數）落在第 2 行 → 用「整檔空白切分」的
 * 方式讀取，與 fscanf 的跨行行為一致，也讓第 1 行的 `cout` 不會被誤讀。
 */
export function parseModelTxt(text) {
    const t = text.split(/\s+/).filter(Boolean);
    let i = 0;
    const nextInt = () => { const v = parseInt(t[i++], 10); if (!Number.isFinite(v)) throw new Error('model.txt: 數字欄位解析失敗 @' + (i - 1)); return v; };

    const cin = nextInt(), base = nextInt(), levels = nextInt(), cout = nextInt();
    const n = nextInt();

    const readConv = (isLast) => {
        const name = t[i++];
        const c = { name, cin: nextInt(), cout: nextInt(), pad: nextInt(), level: nextInt() };
        // opennr_model.hpp:137-141 —— 只有最後一層允許 pad != cout
        if (!isLast && c.cout !== c.pad)
            throw new Error(`${name}: width ${c.cout} padded to ${c.pad} -- 此模型不受支援`);
        return c;
    };

    const convs = [];
    for (let k = 0; k < n; k++) convs.push(readConv(k + 1 === n));

    let glob = null;
    if (i < t.length && t[i] === 'global') {
        i++;
        glob = { n: nextInt(), gin: nextInt(), ghidden: nextInt(), gout: nextInt() };
        glob.convs = [];
        for (let k = 0; k < glob.n; k++) glob.convs.push(readConv(false));
    }

    return { cin, base, levels, cout, n, convs, glob };
}

// ----------------------------------------------------------------- 權重載入

/**
 * 讀一層的 .w.bin / .b.bin。
 * @returns {{w: Uint32Array, b: Float32Array, wBytes: number, bBytes: number, wHalves: number}}
 *   w —— 已打包的 uint2 陣列（長度 = 元素數 * 2 個 u32），可直接 writeStorageBuffer
 *   b —— f32 陣列（長度 = pad），對應 StructuredBuffer<float4> 的 [pad/4] 個元素
 */
export function loadConvWeights(dir, conv) {
    const wPath = path.join(dir, `${conv.name}.w.bin`);
    const bPath = path.join(dir, `${conv.name}.b.bin`);
    const wRaw = fs.readFileSync(wPath);
    const bRaw = fs.readFileSync(bPath);

    const wExpect = 9 * conv.cin * conv.pad * 2; // opennr_model.hpp:157
    const bExpect = conv.pad * 4;                // opennr_model.hpp:158
    if (wRaw.length !== wExpect)
        throw new Error(`${conv.name}.w.bin: ${wRaw.length} bytes != 9*cin*pad*2 = ${wExpect}`);
    if (bRaw.length !== bExpect)
        throw new Error(`${conv.name}.b.bin: ${bRaw.length} bytes != pad*4 = ${bExpect}`);

    // w：檔案本身就是 tap-major f16，每個 u32 = 2 個 half，故直接 little-endian 解讀即為
    //    WGSL 端 `w[(tap*cin + k4*4 + kk)*cout4 + n4]` 的 uint2 元素（index*2 + {0,1}）。
    const wHalves = new Uint16Array(wRaw.buffer, wRaw.byteOffset, wRaw.length >> 1);
    const w = new Uint32Array(wRaw.length >> 2);
    const wBytes = new Uint8Array(wRaw.buffer, wRaw.byteOffset, wRaw.length);
    for (let j = 0; j < w.length; j++) {
        const lo = wHalves[j * 2], hi = wHalves[j * 2 + 1];
        if (wBytes[j * 4] !== (lo & 0xff) || wBytes[j * 4 + 1] !== (lo >> 8) ||
            wBytes[j * 4 + 2] !== (hi & 0xff) || wBytes[j * 4 + 3] !== (hi >> 8))
            throw new Error(`${conv.name}.w.bin: little-endian u32 打包與原始位元組不一致`);
        w[j] = (lo | (hi << 16)) >>> 0;
    }
    if (wHalves.length !== 9 * conv.cin * conv.pad)
        throw new Error(`${conv.name}: halves ${wHalves.length} != 9*cin*pad ${9 * conv.cin * conv.pad}`);

    // b：f32，對齊 4 bytes
    if (bRaw.byteOffset % 4 !== 0)
        throw new Error(`${conv.name}.b.bin: 未對齊 4 bytes，無法直接以 Float32Array 解讀`);
    const b = new Float32Array(bRaw.buffer, bRaw.byteOffset, bRaw.length >> 2);

    return {
        w, b,
        wBytes: wRaw.length, bBytes: bRaw.length,
        wHalves: wHalves.length,
        wU32: w.length, wElements: w.length >> 1,
        bFloats: b.length, bElements: b.length >> 2,
        wRaw, bRaw
    };
}

/**
 * 讀 fc.bin，切分成 kGlobalHLSL 讀取的四段（opennr_shaders.hpp:138,165-176）。
 *   p = fc1 W[gH][gC] ++ b1[gH] ++ fc2 W[gO][gH] ++ b2[gO]   全部 f32
 * @returns {{all: Float32Array, fc1W: Float32Array, b1: Float32Array, fc2W: Float32Array, b2: Float32Array}}
 */
export function loadFc(dir, glob) {
    const raw = fs.readFileSync(path.join(dir, 'fc.bin'));
    const nf = glob.ghidden * glob.gin + glob.ghidden + glob.gout * glob.ghidden + glob.gout;
    if (raw.length !== nf * 4)
        throw new Error(`fc.bin: ${raw.length} bytes != ${nf * 4}（gH*gI + gH + gO*gH + gO 個 f32）`);
    const all = new Float32Array(raw.buffer, raw.byteOffset, nf);
    let o = 0;
    const fc1W = all.subarray(o, o += glob.ghidden * glob.gin);
    const b1 = all.subarray(o, o += glob.ghidden);
    const fc2W = all.subarray(o, o += glob.gout * glob.ghidden);
    const b2 = all.subarray(o, o += glob.gout);
    return { all, fc1W, b1, fc2W, b2, raw, floats: nf, bytes: raw.length };
}

// ----------------------------------------------------------------- 主 API

/**
 * 載入一個模型的全部權重，回傳可直接餵給 WGFX writeStorageBuffer 的結構。
 *
 * convs[i].w 是**已打包好的 uint2 陣列**（可直接 writeStorageBuffer，size = w.length*4 bytes）
 * convs[i].b 是 **f32 陣列**（StructuredBuffer<float4>，size = b.length*4 bytes）
 *
 * activation 判定（opennr_addon.cpp:2299/2317/2340 傳 true；:2346 輸出層傳 false）：
 *   除最後一層外全部 GELU。注意 filter 與 early-return 也要比對，以 out 為界最穩。
 *
 * @param {string} [dir] 模型目錄；預設 1pass
 */
export function loadModel(dir = path.join(DEFAULT_MODELS_DIR, '1pass')) {
    const txt = fs.readFileSync(path.join(dir, 'model.txt'), 'utf8');
    const meta = parseModelTxt(txt);
    const variant = path.basename(dir);

    const convs = meta.convs.map((c, i) => {
        const isOut = i === meta.convs.length - 1;
        const wl = loadConvWeights(dir, c);
        return {
            index: i,
            name: c.name,
            cin: c.cin,
            cout: c.cout,
            pad: c.pad,
            level: c.level,
            // 只有輸出層線性；其餘 18 層全部 GELU（見檔頭引用）
            act: !isOut,
            gelu: !isOut,
            w: wl.w,
            b: wl.b,
            wBytes: wl.wBytes,
            bBytes: wl.bBytes,
            wElements: wl.wElements,
            bElements: wl.bElements,
            wStride: 8,
            bStride: 16,
            wHalves: wl.wHalves,
            _wRaw: wl.wRaw,
            _bRaw: wl.bRaw
        };
    });

    let glob = null;
    if (meta.glob) {
        const g = meta.glob;
        const gconvs = g.convs.map((c) => {
            const wl = loadConvWeights(dir, c);
            return {
                index: g.convs.indexOf(c),
                name: c.name, cin: c.cin, cout: c.cout, pad: c.pad, level: c.level,
                act: true, gelu: true,   // addon.cpp:2281 傳 true
                w: wl.w, b: wl.b,
                wBytes: wl.wBytes, bBytes: wl.bBytes,
                wElements: wl.wElements, bElements: wl.bElements,
                wStride: 8, bStride: 16, wHalves: wl.wHalves,
                _wRaw: wl.wRaw, _bRaw: wl.bRaw
            };
        });
        const fc = loadFc(dir, g);
        // kGlobalHLSL 的第二個線性層之後「沒有」gelu（shaders.hpp:170-179 直接相加）
        glob = {
            n: g.n,
            gin: g.gin, ghidden: g.ghidden, gout: g.gout,
            convs: gconvs,
            fc: {
                floats: fc.floats, bytes: fc.bytes,
                fc1W: fc.fc1W, b1: fc.b1, fc2W: fc.fc2W, b2: fc.b2,
                all: fc.all,
                shapes: {
                    fc1W: [g.ghidden, g.gin],
                    b1: [g.ghidden],
                    fc2W: [g.gout, g.ghidden],
                    b2: [g.gout]
                }
            },
            _raw: fc.raw
        };
    }

    return {
        variant,
        dir,
        cin: meta.cin,
        base: meta.base,
        levels: meta.levels,
        cout: meta.cout,
        layerCount: meta.n,
        convs,
        glob,
        // 便捷視圖：main 分支（不含 global 的 g0/g1/g2）
        main: convs,
        source: {
            modelTxtBytes: Buffer.byteLength(txt),
            modelTxtRaw: txt
        }
    };
}

/** level → 解析度除數（level 0 = 全解析度） */
export const levelDivisor = (level) => 1 << level;

// ----------------------------------------------------------------- 驗算

/** 對一個模型做全層 byte 驗算，回傳逐層結果 */
export function verifyModel(model) {
    const rows = [];
    const all = [...model.convs, ...(model.glob ? model.glob.convs : [])];
    for (const c of all) {
        const expW = 9 * c.cin * c.pad * 2;
        const expB = c.pad * 4;
        const expWHalves = 9 * c.cin * c.pad;
        const expElems = 9 * c.cin * c.pad / 4;
        rows.push({
            name: c.name,
            cin: c.cin, cout: c.cout, pad: c.pad, level: c.level,
            act: c.act,
            wBytesExp: expW, wBytesAct: c.wBytes,
            wHalvesExp: expWHalves, wHalvesAct: c.wHalves,
            wElemsExp: expElems, wElemsAct: c.wElements,
            bBytesExp: expB, bBytesAct: c.bBytes,
            bElemsExp: c.pad / 4, bElemsAct: c.bElements,
            ok: c.wBytes === expW && c.bBytes === expB &&
                c.wHalves === expWHalves && c.wElements === expElems &&
                c.w.length * 4 === expW && c.b.length * 4 === expB
        });
    }
    return rows;
}

// ----------------------------------------------------------------- JSON 清單

/** 產生可序列化的 JSON 清單（不含二進位本體；以 base64 內嵌） */
export function buildManifest(model, { embed = 'none' } = {}) {
    const enc = (typed) => Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength).toString('base64');
    const layer = (c) => ({
        name: c.name,
        cin: c.cin, cout: c.cout, pad: c.pad, level: c.level,
        resolutionDivisor: levelDivisor(c.level),
        act: c.act, gelu: c.gelu,
        w: {
            elementType: 'uint2', componentType: 'f16', stride: 8,
            elements: c.wElements, u32: c.w.length, bytes: c.wBytes,
            indexFormula: '(tap*cin + k4*4 + kk)*cout4 + n4',
            taps: 9
        },
        b: {
            elementType: 'float4', componentType: 'f32', stride: 16,
            elements: c.bElements, floats: c.b.length, bytes: c.bBytes,
            indexFormula: '[cout/4]  ← n4',
            note: 'bias 是 f32 不是 f16（StructuredBuffer<float4>）'
        },
        ...(embed === 'base64' ? { wBase64: enc(c.w), bBase64: enc(c.b) } : {})
    });

    return {
        variant: model.variant,
        source: { dir: path.relative(REPO_ROOT, model.dir).replace(/\\/g, '/'), modelTxtBytes: model.source.modelTxtBytes },
        network: {
            cin: model.cin, base: model.base, levels: model.levels, cout: model.cout,
            layerCount: model.layerCount
        },
        layers: model.convs.map(layer),
        global: model.glob ? {
            n: model.glob.n, gin: model.glob.gin, ghidden: model.glob.ghidden, gout: model.glob.gout,
            convs: model.glob.convs.map(layer),
            fc: {
                elementType: 'float', componentType: 'f32', stride: 4,
                floats: model.glob.fc.floats, bytes: model.glob.fc.bytes,
                layout: 'fc1W[gH][gC] ++ b1[gH] ++ fc2W[gO][gH] ++ b2[gO]',
                shapes: model.glob.fc.shapes
            },
            ...(embed === 'base64' ? {
                fcBase64: Buffer.from(model.glob.fc.all.buffer, model.glob.fc.all.byteOffset,
                    model.glob.fc.all.byteLength).toString('base64')
            } : {})
        } : null,
        bufferPlan: model.glob ? {
            gbias: { elementType: 'float4', stride: 16, elements: model.glob.gout / 4, bytes: model.glob.gout * 4 }
        } : null
    };
}

// ----------------------------------------------------------------- 寫檔

/** 把整個模型寫成 packed 二進位 + JSON 清單，回傳輸出路徑 */
export function writeModelFiles(model, outDir) {
    fs.mkdirSync(outDir, { recursive: true });
    const files = [];
    const all = [
        ...model.convs.map((c) => ['main', c]),
        ...(model.glob ? model.glob.convs.map((c) => ['global', c]) : [])
    ];
    for (const [, c] of all) {
        const wOut = path.join(outDir, `${c.name}.w.packed.bin`);
        const bOut = path.join(outDir, `${c.name}.b.f32.bin`);
        fs.writeFileSync(wOut, Buffer.from(c.w.buffer, c.w.byteOffset, c.w.byteLength));
        fs.writeFileSync(bOut, Buffer.from(c.b.buffer, c.b.byteOffset, c.b.byteLength));
        files.push(wOut, bOut);
    }
    if (model.glob) {
        const fcOut = path.join(outDir, 'fc.f32.bin');
        const gb = model.glob.fc;
        fs.writeFileSync(fcOut, Buffer.from(gb.all.buffer, gb.all.byteOffset, gb.all.byteLength));
        files.push(fcOut);
    }
    const jOut = path.join(outDir, 'manifest.json');
    fs.writeFileSync(jOut, JSON.stringify(buildManifest(model, { embed: 'none' }), null, 2));
    files.push(jOut);
    return files;
}

// ----------------------------------------------------------------- CLI

function fmtBytes(n) { return String(n).padStart(8); }

function reportModel(model, { dump = null } = {}) {
    const rows = verifyModel(model);
    const bad = rows.filter((r) => !r.ok);
    console.log(`\n=== ${model.variant}  (${path.relative(REPO_ROOT, model.dir).replace(/\\/g, '/')}) ===`);
    console.log(`header: cin=${model.cin} base=${model.base} levels=${model.levels} cout=${model.cout}  layers=${model.layerCount}`);
    console.log('');
    console.log('  layer   cin cout  pad lvl act |  w.bin 期望   實測  halves 期望    實測 |  b.bin 期望  實測  f32# | 驗算');
    console.log('  ' + '-'.repeat(114));
    for (const r of rows) {
        console.log(
            '  ' + r.name.padEnd(7) +
            String(r.cin).padStart(3) + String(r.cout).padStart(5) + String(r.pad).padStart(5) +
            String(r.level).padStart(4) + String(r.act ? ' G ' : ' - ').padStart(4) + ' | ' +
            fmtBytes(r.wBytesExp) + fmtBytes(r.wBytesAct) +
            fmtBytes(r.wHalvesExp) + fmtBytes(r.wHalvesAct) + ' | ' +
            fmtBytes(r.bBytesExp) + fmtBytes(r.bBytesAct) + String(r.bElemsAct).padStart(6) + ' | ' +
            (r.ok ? '  ✓' : '  ✗')
        );
    }
    if (model.glob) {
        const g = model.glob;
        console.log(`\n  global: n=${g.n} gin=${g.gin} gH=${g.ghidden} gO=${g.gout}`);
        for (const c of g.convs)
            console.log(`    ${c.name.padEnd(5)} cin=${c.cin} cout=${c.cout} pad=${c.pad} level=${c.level} act=${c.act}`);
        const fc = g.fc;
        console.log(`    fc.bin: ${fc.bytes} bytes = ${fc.floats} f32  ` +
            `[fc1W ${fc.shapes.fc1W.join('x')}=${fc.fc1W.length}, b1 ${fc.b1.length}, ` +
            `fc2W ${fc.shapes.fc2W.join('x')}=${fc.fc2W.length}, b2 ${fc.b2.length}]  ` +
            `預期 ${g.ghidden * g.gin}+${g.ghidden}+${g.gout * g.ghidden}+${g.gout}=${fc.floats} ` +
            `${fc.floats === g.ghidden * g.gin + g.ghidden + g.gout * g.ghidden + g.gout ? '✓' : '✗'}`);
    }
    console.log(`\n  驗算: ${rows.length - bad.length}/${rows.length} 層通過` + (bad.length ? `  ✗ 失敗: ${bad.map((b) => b.name).join(', ')}` : '  ✓'));
    if (dump) {
        const c = [...model.convs, ...(model.glob ? model.glob.convs : [])].find((x) => x.name === dump);
        if (!c) { console.log(`  (找不到層 ${dump})`); return bad.length === 0; }
        const first8 = [];
        for (let j = 0; j < 8; j++) {
            const u = c.w[j >> 1];
            first8.push(decodeF16(((j & 1) ? (u >>> 16) : (u & 0xffff)) & 0xffff));
        }
        console.log(`\n  [dump] ${dump}.w.bin 前 8 個 f16 解碼（tap=0, k4=0, kk=0..3, n4=0..1）:`);
        console.log('    ' + first8.map((v) => v.toExponential(6)).join(', '));
        console.log(`  [dump] ${dump}.b.bin 前 8 個 f32: ` +
            Array.from(c.b.slice(0, 8)).map((v) => v.toExponential(6)).join(', '));
        const u0 = c.w[0];
        console.log(`  [dump] 原始位元組[0..7] = ${Array.from(c._wRaw.slice(0, 8)).map((x) => x.toString(16).padStart(2, '0')).join(' ')}`);
        console.log(`         解讀為 u32[0]=0x${u0.toString(16).padStart(8, '0')} ` +
            `= half[0]=0x${(u0 & 0xffff).toString(16).padStart(4, '0')}, half[1]=0x${(u0 >>> 16).toString(16).padStart(4, '0')} ✓ little-endian`);
    }
    return bad.length === 0;
}

function main() {
    const argv = process.argv.slice(2);
    const dump = argv.includes('--dump') ? argv[argv.indexOf('--dump') + 1] : null;
    const wantJson = argv.includes('--json');
    const wantBin = argv.includes('--bin');
    const modelsDir = argv.includes('--dir') ? argv[argv.indexOf('--dir') + 1] : DEFAULT_MODELS_DIR;

    let allOk = true;
    const models = {};
    for (const v of VARIANTS) {
        const dir = path.join(modelsDir, v);
        if (!fs.existsSync(dir)) { console.log(`(略過 ${v}：${dir} 不存在)`); continue; }
        const m = loadModel(dir);
        models[v] = m;
        allOk = reportModel(m, { dump }) && allOk;
    }

    // 變體差異
    if (models['1pass'] && models['2pass']) {
        const a = models['1pass'], b = models['2pass'];
        console.log('\n=== 1pass vs 2pass 差異 ===');
        const diffs = [];
        if (a.cout !== b.cout) diffs.push(`header cout: ${a.cout} → ${b.cout}`);
        if (a.cin !== b.cin) diffs.push(`header cin: ${a.cin} → ${b.cin}`);
        if (a.base !== b.base) diffs.push(`header base: ${a.base} → ${b.base}`);
        if (a.levels !== b.levels) diffs.push(`header levels: ${a.levels} → ${b.levels}`);
        for (let i = 0; i < Math.max(a.convs.length, b.convs.length); i++) {
            const x = a.convs[i], y = b.convs[i];
            if (!x || !y) { diffs.push(`層數不同 @${i}`); continue; }
            for (const k of ['name', 'cin', 'cout', 'pad', 'level'])
                if (x[k] !== y[k]) diffs.push(`層 ${x.name}.${k}: ${x[k]} → ${y[k]}`);
        }
        console.log(diffs.length ? diffs.map((d) => '  ' + d).join('\n') : '  （無差異）');
        console.log(`  → 輸出通道數 cout=${a.cout} vs ${b.cout}（2pass 為 ${b.cout / 4} 種風格 × 4 通道）`);
    }

    if (wantJson || wantBin) {
        const outRoot = path.join(HERE, 'out');
        for (const [v, m] of Object.entries(models)) {
            const outDir = path.join(outRoot, v);
            if (wantBin) {
                const files = writeModelFiles(m, outDir);
                console.log(`\n[寫檔] ${v}: ${files.length} 個檔案 → ${path.relative(REPO_ROOT, outDir).replace(/\\/g, '/')}`);
            } else {
                fs.mkdirSync(outDir, { recursive: true });
                const p = path.join(outDir, 'manifest.json');
                fs.writeFileSync(p, JSON.stringify(buildManifest(m), null, 2));
                console.log(`\n[寫檔] ${v}: ${path.relative(REPO_ROOT, p).replace(/\\/g, '/')}`);
            }
        }
    }

    console.log(`\n總體: ${allOk ? '[PASS]' : '[FAIL]'}`);
    process.exit(allOk ? 0 : 1);
}

if (import.meta.url === `file://${process.argv[1]?.replace(/\\/g, '/')}` ||
    process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    main();
}
