// OpenNR CPU 參考實作 —— 全 19 層 f16 U-Net 前向傳播
//
// 這是整個移植案的**數值驗收 oracle**：GPU（WGFX 效果檔）的每一層中間結果都與此對照。
// 語義來源（逐項對齊，不自行推導）：
//   - effects_src/OpenNR-ReShade/upstream/src/opennr_shaders.hpp
//       kConvHLSL      :14-84   3x3 conv + bias + gelu（TM=4 像素/執行緒，f32 累加，f16 打包）
//       kPoolHLSL      :87-111  2x2 average pooling
//       kUpCatHLSL     :114-132 nearest upsample + skip concat
//       kGlobalHLSL    :136-182 全幀分支（均值 → fc1 → gelu → fc2 → 加到 mid_a bias）
//       kStageHLSL     :186-233 16 輸入通道（種子噪聲 + 降採樣色彩 + 設定）
//   - effects_src/OpenNR-ReShade/upstream/src/opennr_addon.cpp:2247-2346  精確 pass 序列
//   - effects_src/OpenNR-ReShade/upstream/src/opennr_model.hpp            層表與 buffer 尺寸
//   - docs/opennr-pass-graph.md                                           本專案的規格正本
//
// 設計原則：**完全模擬 GPU 的資料表示**（f16 打包、f32 累加），
// 因為驗收要求是「逐通道 ≤1e-2（f16 容差）」而非 bit-exact。
// 所有特徵圖以 packed uint2（4 個 f16）為單位儲存，與 WGSL 端一致。

import fs from 'fs';
import path from 'path';

// ---------------------------------------------------------------- f16 轉換（與 WGSL bit-level 一致）

const _f32buf = new ArrayBuffer(4);
const _f32 = new Float32Array(_f32buf);
const _u32 = new Uint32Array(_f32buf);

/** f32 → f16 位元（round-to-nearest-even），與 WGSL f32tof16 同語義 */
export function f32tof16(v) {
    _f32[0] = v;
    const u = _u32[0];
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
    if (exp === 0) return sign >>> 0;
    const full = man | 0x800000;
    const s = 126 - exp;
    if (s > 34) return sign >>> 0;
    const q = full >>> s;
    const rem = full & ((1 << s) - 1);
    const half = 1 << (s - 1);
    let m10 = q + ((rem > half || (rem === half && (q & 1))) ? 1 : 0);
    if (m10 >= 0x400) return (sign | 0x400) >>> 0;
    return (sign | m10) >>> 0;
}

/** f16 位元 → f32，與 WGSL f16tof32 同語義 */
export function f16tof32(h) {
    const s = (h & 0x8000) ? -1 : 1, e = (h & 0x7c00) >> 10, m = h & 0x03ff;
    if (e === 0) return s * 2 ** -14 * (m / 1024);
    if (e === 31) return m ? NaN : s * Infinity;
    return s * 2 ** (e - 15) * (1 + m / 1024);
}

/** 4 個 f32 → uint2（packed f16），對應 HLSL p4() */
export function p4(v) {
    return [
        (f32tof16(v[0]) | (f32tof16(v[1]) << 16)) >>> 0,
        (f32tof16(v[2]) | (f32tof16(v[3]) << 16)) >>> 0,
    ];
}

/** uint2 → 4 個 f32，對應 HLSL h4() */
export function h4(u2) {
    return [
        f16tof32(u2[0] & 0xffff),
        f16tof32((u2[0] >>> 16) & 0xffff),
        f16tof32(u2[1] & 0xffff),
        f16tof32((u2[1] >>> 16) & 0xffff),
    ];
}

// ---------------------------------------------------------------- GELU

/** GELU tanh 形式，對應 HLSL gelu()（shaders.hpp:34-37）；clamp(±10) 防溢位 */
export function gelu(x) {
    const t = 0.7978845608028654 * (x + 0.044715 * x * x * x);
    const tc = Math.max(-10, Math.min(10, t));
    return 0.5 * x * (1 + Math.tanh(tc));
}

// ---------------------------------------------------------------- 特徵圖容器
//
// 一律以 packed uint2 儲存：[pixel][ch4]，與 WGSL 端 x/o buffer 佈局一致。
// stride 可以大於實際通道數（輸出層 pad=16 但 cout=4）。

export class FeatureMap {
    /** @param {number} w @param {number} h @param {number} ch 邏輯通道數 @param {number} [pad] 實體通道數（預設 = ch） */
    constructor(w, h, ch, pad) {
        this.w = w; this.h = h; this.ch = ch;
        this.pad = pad === undefined ? ch : pad;
        this.ch4 = this.pad / 4;
        this.npix = w * h;
        /** @type {Uint32Array} 每像素 ch4 個 uint2 = 2*ch4 個 u32 */
        this.data = new Uint32Array(this.npix * this.ch4 * 2);
    }
    /** 讀像素 p 的第 c4 組 4 通道 → [f32,f32,f32,f32] */
    get4(p, c4) {
        const o = (p * this.ch4 + c4) * 2;
        return h4([this.data[o], this.data[o + 1]]);
    }
    /** 寫像素 p 的第 c4 組 4 通道 */
    set4(p, c4, v) {
        const o = (p * this.ch4 + c4) * 2;
        const pk = p4(v);
        this.data[o] = pk[0]; this.data[o + 1] = pk[1];
    }
    /** 讀取單一通道（c 為邏輯通道索引，0..ch-1） */
    get(p, c) { return this.get4(p, c >> 2)[c & 3]; }
}

// ---------------------------------------------------------------- 核心算子

/**
 * 預先解碼權重為 f32 查表，供快速路徑使用。
 * `w` 為 packed uint2，每 4 個 f16 一組；解碼後以 Float32Array 保存，
 * 避免 conv 內層迴圈每次重複 f16→f32。
 */
export function decodeLayerW(layer) {
    if (layer._wf32) return layer._wf32;
    const n = layer.w.length * 2;              // 每個 u32 = 2 個 half
    const wf = new Float32Array(n);
    for (let i = 0; i < layer.w.length; i++) {
        wf[2 * i] = f16tof32(layer.w[i] & 0xffff);
        wf[2 * i + 1] = f16tof32((layer.w[i] >>> 16) & 0xffff);
    }
    Object.defineProperty(layer, '_wf32', { value: wf, enumerable: false });
    return wf;
}

/**
 * 快速版 3x3 conv（純 typed array 索引，語義與 conv3x3 完全相同）。
 * conv3x3 為可讀性參考實作；此版供大解析度全網驗收使用。
 * 兩者在 smoke 中以隨機資料交叉驗證相等。
 */
export function conv3x3Fast(x, out, layer, act, zeroPad = false) {
    const cin = layer.cin, pad = layer.pad;
    const cin4 = cin / 4, cout4 = pad / 4;
    const W = x.w, H = x.h, npix = W * H;
    const wf = decodeLayerW(layer);
    const xd = x.data, od = out.data;
    const xc4 = x.ch4, oc4 = out.ch4;

    for (let p = 0; p < npix; p++) {
        const px = p % W, py = (p / W) | 0;
        for (let n4 = 0; n4 < cout4; n4++) {
            let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (let tap = 0; tap < 9; tap++) {
                const dy = ((tap / 3) | 0) - 1, dx = (tap % 3) - 1;
                let qm, ok = true;
                if (zeroPad) {
                    const qx = px + dx, qy = py + dy;
                    ok = qx >= 0 && qy >= 0 && qx < W && qy < H;
                    qm = ok ? qy * W + qx : 0;
                } else {
                    qm = p + dy * W + dx;
                    if (qm < 0) qm = 0; else if (qm >= npix) qm = npix - 1;
                }
                const xo = qm * xc4 * 2;
                for (let k4 = 0; k4 < cin4; k4++) {
                    let xv0 = 0, xv1 = 0, xv2 = 0, xv3 = 0;
                    if (ok) {
                        const xi = xo + k4 * 2;
                        const u0 = xd[xi], u1 = xd[xi + 1];
                        xv0 = f16tof32(u0 & 0xffff);
                        xv1 = f16tof32((u0 >>> 16) & 0xffff);
                        xv2 = f16tof32(u1 & 0xffff);
                        xv3 = f16tof32((u1 >>> 16) & 0xffff);
                    }
                    for (let kk = 0; kk < 4; kk++) {
                        const xk = kk === 0 ? xv0 : kk === 1 ? xv1 : kk === 2 ? xv2 : xv3;
                        if (xk === 0) continue;
                        const wi = ((tap * cin + k4 * 4 + kk) * cout4 + n4) * 4;
                        a0 += xk * wf[wi];
                        a1 += xk * wf[wi + 1];
                        a2 += xk * wf[wi + 2];
                        a3 += xk * wf[wi + 3];
                    }
                }
            }
            const bi = n4 * 4;
            let v0 = a0 + layer.b[bi], v1 = a1 + layer.b[bi + 1];
            let v2 = a2 + layer.b[bi + 2], v3 = a3 + layer.b[bi + 3];
            if (act) { v0 = gelu(v0); v1 = gelu(v1); v2 = gelu(v2); v3 = gelu(v3); }
            const oi = (p * oc4 + n4) * 2;
            od[oi] = (f32tof16(v0) | (f32tof16(v1) << 16)) >>> 0;
            od[oi + 1] = (f32tof16(v2) | (f32tof16(v3) << 16)) >>> 0;
        }
    }
}

/**
 * 3x3 conv + bias + 可選 GELU，對應 kConvHLSL（shaders.hpp:39-83）。
 * @param {FeatureMap} x 輸入（cin 通道）
 * @param {FeatureMap} out 輸出（pad 通道，僅前 cout 有效）
 * @param {{w:Uint32Array,b:Float32Array}} layer 權重（已 packed）與 bias（f32）
 * @param {boolean} act gelu 與否
 * @param {boolean} zeroPad true = 影像外讀 0；false = clamp 攤平索引
 */
export function conv3x3(x, out, layer, act, zeroPad = false) {
    const cin = layer.cin, pad = layer.pad;
    const cin4 = cin / 4, cout4 = pad / 4;
    const W = x.w, H = x.h, npix = W * H;

    for (let p = 0; p < npix; p++) {
        const px = p % W, py = (p / W) | 0;
        for (let n4 = 0; n4 < cout4; n4++) {
            const acc = [0, 0, 0, 0];
            for (let tap = 0; tap < 9; tap++) {
                const dy = ((tap / 3) | 0) - 1, dx = (tap % 3) - 1;
                let qm, ok = true;
                if (zeroPad) {
                    const qx = px + dx, qy = py + dy;
                    ok = qx >= 0 && qy >= 0 && qx < W && qy < H;
                    qm = ok ? qy * W + qx : 0;
                } else {
                    qm = Math.max(0, Math.min(npix - 1, p + dy * W + dx));
                }
                for (let k4 = 0; k4 < cin4; k4++) {
                    const xv = ok ? x.get4(qm, k4) : [0, 0, 0, 0];
                    for (let kk = 0; kk < 4; kk++) {
                        // w 索引：(tap*cin + k4*4 + kk) * cout4 + n4 —— 與 HLSL 完全一致
                        const wi = (tap * cin + k4 * 4 + kk) * cout4 + n4;
                        const wv = h4([layer.w[wi * 2], layer.w[wi * 2 + 1]]);
                        const xk = xv[kk];
                        acc[0] += xk * wv[0];
                        acc[1] += xk * wv[1];
                        acc[2] += xk * wv[2];
                        acc[3] += xk * wv[3];
                    }
                }
            }
            const bv = [layer.b[n4 * 4], layer.b[n4 * 4 + 1], layer.b[n4 * 4 + 2], layer.b[n4 * 4 + 3]];
            const v = [acc[0] + bv[0], acc[1] + bv[1], acc[2] + bv[2], acc[3] + bv[3]];
            out.set4(p, n4, act ? v.map(gelu) : v);
        }
    }
}

/** 2x2 平均池化，對應 kPoolHLSL（shaders.hpp:100-110） */
export function pool2x2(x, out, ch) {
    const cin4 = ch / 4;
    const W = x.w, H = x.h;
    const ow = W / 2, oh = H / 2;
    for (let p = 0; p < ow * oh; p++) {
        const y = ((p / ow) | 0) * 2, xx = (p % ow) * 2;
        // 以像素索引實作（4 個來源像素各取一次），與 kPoolHLSL 的 a / a+cin4 /
        // a+gW*cin4 / a+gW*cin4+cin4 等價
        for (let c = 0; c < cin4; c++) {
            const p00 = (y * W + xx), p01 = p00 + 1, p10 = p00 + W, p11 = p10 + 1;
            const s = [0, 0, 0, 0];
            for (const pp of [p00, p01, p10, p11]) {
                const v = x.get4(pp, c);
                for (let k = 0; k < 4; k++) s[k] += v[k];
            }
            out.set4(p, c, s.map(v => 0.25 * v));
        }
    }
}

/**
 * nearest 上採樣 + skip concat，對應 kUpCatHLSL（shaders.hpp:121-131）。
 * 輸出前 clo4 組來自 lo（放大 scale 倍），其餘來自 skip。
 */
export function upcat(lo, skip, out, clo, cs, scale) {
    const clo4 = clo / 4, cs4 = cs / 4, cout4 = clo4 + cs4;
    const W = out.w, H = out.h;
    for (let p = 0; p < W * H; p++) {
        const y = (p / W) | 0, xx = p % W;
        for (let c = 0; c < cout4; c++) {
            if (c < clo4) {
                const lp = (((y / scale) | 0) * ((W / scale) | 0) + ((xx / scale) | 0));
                out.set4(p, c, lo.get4(lp, c));
            } else {
                out.set4(p, c, skip.get4(p, c - clo4));
            }
        }
    }
}

// ---------------------------------------------------------------- 模型載入

/** 讀檔為**精確大小**的 ArrayBuffer。
 *
 * 陷阱：`fs.readFileSync(...).buffer` 對小檔（< Buffer.poolSize/2）會回傳整個 8KB 的
 * Buffer 池，而非檔案本身；`new Uint16Array(buf.buffer)` 因此會多讀到池中殘留的其他資料
 * （含任意 f16 位元樣式 → NaN），污染權重。必須以 byteOffset/byteLength 精確切片。
 */
function readExact(p) {
    const buf = fs.readFileSync(p);
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length);
}

/** 從 model.txt 的目錄載入權重與層表（pad 由最後一層規則決定） */
export function loadModel(dir) {
    const txt = fs.readFileSync(path.join(dir, 'model.txt'), 'utf8').trim().split(/\r?\n/);
    const [cin, base, levels, cout] = txt[0].trim().split(/\s+/).map(Number);
    const n = Number(txt[1].trim());
    const convs = [];
    for (let i = 0; i < n; i++) {
        const [name, a, b2, lvl] = txt[2 + i].trim().split(/\s+/);
        const last = (i === n - 1);
        const cinL = Number(a), coutL = Number(b2), level = Number(lvl);
        const pad = last ? (fs.statSync(path.join(dir, `${name}.w.bin`)).size / 2) / (9 * cinL) : coutL;
        const wRaw = new Uint16Array(readExact(path.join(dir, `${name}.w.bin`)));
        const bRaw = new Float32Array(readExact(path.join(dir, `${name}.b.bin`)));
        // f16 位元 → packed uint2（相鄰兩個 half 併成一個 u32，與 GPU 佈局一致）
        const w = new Uint32Array(wRaw.length / 2);
        for (let k = 0; k < w.length; k++) w[k] = (wRaw[2 * k] | (wRaw[2 * k + 1] << 16)) >>> 0;
        convs.push({ name, cin: cinL, cout: coutL, pad, level, act: !last, w, b: bRaw });
    }
    // global 段
    let glob = null;
    const gi = txt.findIndex(l => l.trim().startsWith('global'));
    if (gi >= 0) {
        const [tag, gn, g_in, g_hidden, g_out] = txt[gi].trim().split(/\s+/);
        const gconvs = [];
        for (let i = 0; i < Number(gn); i++) {
            const [name, a, b2, lvl] = txt[gi + 1 + i].trim().split(/\s+/);
            const cinL = Number(a), coutL = Number(b2);
            const wRaw = new Uint16Array(readExact(path.join(dir, `${name}.w.bin`)));
            const bRaw = new Float32Array(readExact(path.join(dir, `${name}.b.bin`)));
            const w = new Uint32Array(wRaw.length / 2);
            for (let k = 0; k < w.length; k++) w[k] = (wRaw[2 * k] | (wRaw[2 * k + 1] << 16)) >>> 0;
            gconvs.push({ name, cin: cinL, cout: coutL, pad: coutL, level: Number(lvl), act: true, w, b: bRaw });
        }
        const nf = Number(g_hidden) * Number(g_in) + Number(g_hidden) + Number(g_out) * Number(g_hidden) + Number(g_out);
        const fc = new Float32Array(readExact(path.join(dir, 'fc.bin'))).subarray(0, nf);
        glob = { g_in: Number(g_in), g_hidden: Number(g_hidden), g_out: Number(g_out), gconvs, fc };
    }
    return { cin, base, levels, cout, convs, glob };
}

// ---------------------------------------------------------------- 全網路前向

/**
 * 完整 19 層前向傳播，回傳每層與關鍵中間結果。
 * @param {object} model loadModel() 的產物
 * @param {number} W 網路寬 @param {number} H 網路高
 * @param {object} [opts] { frame, tone, structure, onLayer }
 */
export function forward(model, W, H, opts = {}) {
    const { base, levels, cin } = model;
    const trace = {};
    const onLayer = opts.onLayer || null;
    // fast = 預設走 typed-array 路徑（大解析度）；opts.fast=false 用可讀版參考實作
    const conv = opts.fast === false ? conv3x3 : conv3x3Fast;

    // --- 輸入 16 通道（kStageHLSL 語義）---
    const chans = new FeatureMap(W, H, cin);
    buildStage(chans, W, H, opts);

    // ping-pong 緩衝：A / B，與 addon.cpp 的 a_srv/b_srv 對應
    let A = null, B = new FeatureMap(W, H, 128 * 2); // 足以容納最大 256 通道
    const mk = (w, h, ch, pad) => new FeatureMap(w, h, ch, pad);

    const skip = [];
    let cur = chans;

    // --- global 分支（addon.cpp:2272-2293）---
    let gbias = null;
    if (model.glob) {
        let gw = W, gh = H;
        for (let s = 0; s < 3; s++) {
            const nb = mk(gw / 2, gh / 2, cin);
            pool2x2(cur, nb, cin);
            cur = nb; gw /= 2; gh /= 2;
        }
        for (let s = 0; s < model.glob.gconvs.length; s++) {
            const c = model.glob.gconvs[s];
            const nb = mk(gw, gh, c.pad);
            conv(cur, nb, c, true, true);
            cur = nb;
            if (s + 1 < model.glob.gconvs.length) {
                const nb2 = mk(gw / 2, gh / 2, c.pad);
                pool2x2(cur, nb2, c.pad);
                cur = nb2; gw /= 2; gh /= 2;
            }
        }
        gbias = globalBranch(cur, model, gw * gh);
        cur = chans;
    }

    // --- encoder（addon.cpp:2294-2313）---
    let k = 0, ww = W, hh = H;
    for (let i = 0; i < levels; i++) {
        for (let j = 0; j < 2; j++) {
            const c = model.convs[k++];
            const last = (j === 1);
            const nb = mk(ww, hh, c.pad);
            conv(cur, nb, c, true, false);
            if (last) { skip[i] = nb; cur = nb; }
            else cur = nb;
            if (onLayer) onLayer(c.name, nb);
        }
        if (i + 1 < levels) {
            const nb = mk(ww / 2, hh / 2, base << i);
            pool2x2(skip[i], nb, base << i);
            cur = nb; ww /= 2; hh /= 2;
        }
    }

    // --- middle（addon.cpp:2314-2322）：第一層吃 gbias ---
    for (let j = 0; j < 2; j++) {
        const c = model.convs[k++];
        const nb = mk(ww, hh, c.pad);
        const layer = (j === 0 && gbias) ? { ...c, b: gbias } : c;
        conv(cur, nb, layer, true, false);
        cur = nb;
        if (onLayer) onLayer(c.name, nb);
    }

    // --- decoder（addon.cpp:2323-2345）---
    for (let i = levels - 1; i >= 0; i--) {
        const sw = W >> i, sh = H >> i;
        const c0 = model.convs[k];
        const cs = base << i;
        const clo = c0.cin - cs;
        const scale = (i + 1 < levels) ? 2 : 1;
        const nb = mk(sw, sh, c0.cin);
        upcat(cur, skip[i], nb, clo, cs, scale);
        cur = nb; ww = sw; hh = sh;
        for (let j = 0; j < 2; j++) {
            const c = model.convs[k++];
            const nbb = mk(ww, hh, c.pad);
            conv(cur, nbb, c, true, false);
            cur = nbb;
            if (onLayer) onLayer(c.name, nbb);
        }
    }

    // --- 輸出層（addon.cpp:2346）：act=false ---
    const outC = model.convs[k++];
    const head = mk(W, H, outC.cout, outC.pad);
    conv(cur, head, outC, false, false);
    if (onLayer) onLayer(outC.name, head);

    trace.head = head;
    trace.chans = chans;
    trace.gbias = gbias;
    trace.skip = skip;
    trace.layerCount = k;
    return trace;
}

/** kStageHLSL：16 輸入通道（噪聲 4 + 色彩 6 + 設定 4 的實際配置見 :228-231） */
function buildStage(chans, W, H, opts) {
    const frame = opts.frame === undefined ? 0 : opts.frame;
    const tone = opts.tone === undefined ? 1.0 : opts.tone;       // 已是 /100 後的值
    const structure = opts.structure === undefined ? 1.0 : opts.structure;
    const src = opts.src || null;  // (x,y)->[r,g,b]，null = 0.5 灰

    const mix32 = (v) => {
        v = (v ^ (v >>> 16)) >>> 0;
        v = Math.imul(v, 0x7feb352d) >>> 0;
        v = (v ^ (v >>> 15)) >>> 0;
        v = Math.imul(v, 0x846ca68b) >>> 0;
        v = (v ^ (v >>> 16)) >>> 0;
        return v;
    };
    const u01 = (v) => (v & 0x00ffffff) / 16777216 + 1 / 33554432;
    const TAU = 6.283185307179586;

    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const p = y * W + x;
            const s = mix32((Math.imul(x, 0x9e3779b9) ^ Math.imul(y, 0x85ebca6b) ^ Math.imul(frame, 0xc2b2ae35)) >>> 0);
            const u0 = u01(mix32((s + 1) >>> 0)), u1 = u01(mix32((s + 2) >>> 0));
            const u2 = u01(mix32((s + 3) >>> 0)), u3 = u01(mix32((s + 4) >>> 0));
            const r0 = Math.sqrt(-2 * Math.log(u0)), r1 = Math.sqrt(-2 * Math.log(u1));
            let c = src ? src(x, y) : [0.5, 0.5, 0.5];
            const v = [(c[0] - 0.5) * 2 * (1 / 16), (c[1] - 0.5) * 2 * (1 / 16), (c[2] - 0.5) * 2 * (1 / 16)];
            chans.set4(p, 0, [r1 * Math.cos(TAU * u3), r1 * Math.sin(TAU * u3), r0 * Math.cos(TAU * u2), 1.0]);
            chans.set4(p, 1, [v[0], v[1], v[2], v[0]]);
            chans.set4(p, 2, [v[1], v[2], 0.0, tone]);
            chans.set4(p, 3, [structure, -1.0, -1.0, 0.0]);
        }
    }
}

/** kGlobalHLSL：空間均值 → fc1 → gelu → fc2 → + mid_a bias */
function globalBranch(f, model, npix) {
    const { g_in: gC, g_hidden: gH, g_out: gO, fc } = model.glob;
    const c4 = gC / 4;
    // 空間均值
    const m = new Float64Array(gC);
    for (let t = 0; t < gC; t++) {
        let s = 0;
        for (let i = 0; i < npix; i++) s += f.get4(i, t >> 2)[t & 3];
        m[t] = s / npix;
    }
    // fc1 + gelu
    const hid = new Float64Array(gH);
    for (let t = 0; t < gH; t++) {
        let a = fc[gH * gC + t];
        for (let c = 0; c < gC; c++) a += fc[t * gC + c] * m[c];
        hid[t] = gelu(a);
    }
    // fc2 + mid_a bias
    const w2 = gH * gC + gH, b2 = w2 + gO * gH;
    const midA = model.convs[2 * model.levels];  // mid_a
    const r = new Float32Array(gO);
    for (let t = 0; t < gO / 4; t++) {
        for (let j = 0; j < 4; j++) {
            const oo = t * 4 + j;
            let a = fc[b2 + oo];
            for (let i = 0; i < gH; i++) a += fc[w2 + oo * gH + i] * hid[i];
            r[oo] = a + midA.b[oo];
        }
    }
    return r;
}

export function alignExtent(n) {
    const a = Math.ceil(n / 64) * 64;
    return a < 320 ? 320 : a;
}
