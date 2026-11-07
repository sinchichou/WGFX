// 脈衝響應驗證（impulse response probe）— 回答「runtime 產物有沒有真 bug」
//
// 方法：
//   1. 經 WGFXRuntime（Dawn）真跑 FSR_x2 完整管線（EASU→RCAS，2x）。
//   2. dispatch 前把 tex_easu/OUTPUT 預填 half-NaN bits → 讀回殘留 NaN = 覆蓋缺口
//      （直接抓 dispatch/bind/寫入座標錯誤）。
//   3. CPU 參考實作（fround 逐步 f32 模擬 + f16 量化模擬紋理儲存）：
//        refOfficial — AMD ffx_fsr1.h 原文數學（APrx* 位元技巧，逐位可重現）
//      GPU ≡ refOfficial（≤ f16 量化誤差）→ runtime 產物忠實；否則即為真 bug。
//   4. 案例：全平／全黑／全白／單紅 spike（內部、邊界、近邊界）／雙相鄰 spike／暗部雜訊場。
//   5. dump 生成後 WGSL → test/out_wgsl/。
//
// 執行：node test/impulse_response.mjs（需真實 WebGPU 裝置；先 npm run build）
//   - 每案例以「子行程」隔離執行（Dawn node polyfill 多案例累積會原生崩潰），
//     子行程輸出 @@JSON@@ 行由驅動器彙總。
import { create, globals } from 'webgpu';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawnSync } from 'child_process';
Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
const HERE = path.dirname(fileURLToPath(import.meta.url));
const SELF = fileURLToPath(import.meta.url);
const { WGFXRuntime } = await import('../dist/wgfx.esm.js');
let dev = null; // 子行程模式於驅動分支中建立

// ================= 共用：f32 / f16 bit helpers =================
const _ab = new ArrayBuffer(4);
const _f32 = new Float32Array(_ab);
const _u32 = new Uint32Array(_ab);
/** fround alias — every arithmetic op in the reference is rounded to f32. */
const f = Math.fround;
const f32bits = (x) => { _f32[0] = x; return _u32[0]; };
const f32from = (u) => { _u32[0] = u >>> 0; return _f32[0]; };
// AMD APrx* — integer ops, bit-exact reproducible in JS (u32 wrap = >>> 0)
const APrxLoRcpF1 = (a) => f32from((0x7ef07ebb - f32bits(a)) >>> 0);
const APrxLoRsqF1 = (a) => f32from((0x5f347d74 - (f32bits(a) >>> 1)) >>> 0);
const APrxMedRcpF1 = (a) => {
    const b = f32from((0x7ef19fff - f32bits(a)) >>> 0);
    return f(b * f(f(-b * a) + 2.0));
};

// f32 → f16 → f32 (round-to-nearest-even, subnormal + overflow handled)
function f16rt(v) {
    if (Number.isNaN(v)) return NaN;
    if (v === Infinity || v === -Infinity) return v;
    if (v === 0) return v; // keeps -0
    const x = f32bits(v);
    const sign = (x >>> 16) & 0x8000;
    const exp = (x >>> 23) & 0xFF;
    const man = x & 0x7FFFFF;
    if (exp === 0xFF) return sign ? -Infinity : Infinity;
    let e = exp - 112;
    let m10;
    if (e >= 31) return sign ? -Infinity : Infinity; // ≥ 2^16 → rounds to Inf
    if (e >= 1) {
        m10 = (man + 0x00000FFF + ((man >> 13) & 0x1)) >> 13;
        if (m10 > 0x3FF) { m10 = 0; e += 1; if (e >= 31) return sign ? -Infinity : Infinity; }
        return halfFromBits(sign, e, m10);
    }
    if (exp === 0) return sign ? -0 : 0;
    const full = man | 0x800000;
    const s = 126 - exp;
    if (s > 34) return sign ? -0 : 0;
    const q = full >>> s;
    const rem = full & ((1 << s) - 1);
    const halfBit = 1 << (s - 1);
    const m10s = q + (rem > halfBit || (rem === halfBit && (q & 1)) ? 1 : 0);
    if (m10s >= 0x400) return halfFromBits(sign, 1, 0);
    return (sign ? -1 : 1) * m10s * 2 ** -24;
}
function halfFromBits(sign, e, m10) {
    return sign ? -(2 ** (e - 15) * (1 + m10 / 1024)) : 2 ** (e - 15) * (1 + m10 / 1024);
}

const sat = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
const min2 = (a, b) => (a < b ? a : b);
const max2 = (a, b) => (a > b ? a : b);

// ================= CPU 參考：EASU（官方數學） =================
function refEasu(input, W, H, outW, outH, ipx, ipy) {
    const load = (x, y) => {
        const cx = Math.min(Math.max(x, 0), W - 1), cy = Math.min(Math.max(y, 0), H - 1);
        const o = (cy * W + cx) * 4;
        return [f(input[o] / 255), f(input[o + 1] / 255), f(input[o + 2] / 255)];
    };
    const con0x = f(f32from(f32bits(W)) / f32from(f32bits(outW)));
    const con0y = f(f32from(f32bits(H)) / f32from(f32bits(outH)));
    let ppX = f(f(ipx * con0x) + f(f(f(0.5) * con0x) - 0.5));
    let ppY = f(f(ipy * con0y) + f(f(f(0.5) * con0y) - 0.5));
    const fpX = Math.floor(ppX), fpY = Math.floor(ppY);
    ppX = f(ppX - fpX); ppY = f(ppY - fpY);

    const b = load(fpX, fpY - 1), c = load(fpX + 1, fpY - 1);
    const e = load(fpX - 1, fpY), ff = load(fpX, fpY), g = load(fpX + 1, fpY), h = load(fpX + 2, fpY);
    const i = load(fpX - 1, fpY + 1), j = load(fpX, fpY + 1), k = load(fpX + 1, fpY + 1), l = load(fpX + 2, fpY + 1);
    const n = load(fpX, fpY + 2), o = load(fpX + 1, fpY + 2);

    const luma = (p) => f(f(p[2] * 0.5) + f(f(p[0] * 0.5) + p[1]));
    const bL = luma(b), cL = luma(c), eL = luma(e), fL = luma(ff), gL = luma(g);
    const hL = luma(h), iL = luma(i), jL = luma(j), kL = luma(k), lL = luma(l), oL = luma(o), nL = luma(n);

    let dirX = 0, dirY = 0, len = 0;
    const setF = (biS, biT, biU, biV, lA, lB, lC, lD, lE) => {
        let w = 0;
        if (biS) w = f(f(1 - ppX) * f(1 - ppY));
        if (biT) w = f(ppX * f(1 - ppY));
        if (biU) w = f(f(1 - ppX) * ppY);
        if (biV) w = f(ppX * ppY);
        const dc = f(lD - lC), cb = f(lC - lB);
        let lenX = APrxLoRcpF1(max2(Math.abs(dc), Math.abs(cb)));
        const dirXc = f(lD - lB);
        dirX = f(dirX + f(dirXc * w));
        lenX = f(sat(f(Math.abs(dirXc) * lenX)));
        lenX = f(lenX * lenX);
        len = f(len + f(lenX * w));
        const ec = f(lE - lC), ca = f(lC - lA);
        let lenY = APrxLoRcpF1(max2(Math.abs(ec), Math.abs(ca)));
        const dirYc = f(lE - lA);
        dirY = f(dirY + f(dirYc * w));
        lenY = f(sat(f(Math.abs(dirYc) * lenY)));
        lenY = f(lenY * lenY);
        len = f(len + f(lenY * w));
    };
    setF(true, false, false, false, bL, eL, fL, gL, jL);
    setF(false, true, false, false, cL, fL, gL, hL, kL);
    setF(false, false, true, false, fL, iL, jL, kL, nL);
    setF(false, false, false, true, gL, jL, kL, lL, oL);

    const dir2x = f(dirX * dirX), dir2y = f(dirY * dirY);
    let dirR = f(dir2x + dir2y);
    const zro = dirR < f(1 / 32768);
    dirR = zro ? 1.0 : APrxLoRsqF1(dirR);
    dirX = zro ? 1.0 : f(dirX * dirR);
    dirY = f(dirY * dirR); // shader 僅 dir.x 由 zro 強制 1；dir.y 恆乘 dirR
    len = f(len * 0.5);    // len = len*0.5
    len = f(len * len);    // len *= len → (0.5·L)²
    const stretch = f(f(f(dirX * dirX) + f(dirY * dirY)) * APrxLoRcpF1(max2(Math.abs(dirX), Math.abs(dirY))));
    const len2x = f(1 + f(f(stretch - 1) * len));
    const len2y = f(1 + f(-0.5 * len));
    const lob = f(0.5 + f(f(f(1 / 4) - 0.04) - 0.5) * len);
    const clp = APrxLoRcpF1(lob);

    const min4 = [Math.min(ff[0], g[0], j[0], k[0]), Math.min(ff[1], g[1], j[1], k[1]), Math.min(ff[2], g[2], j[2], k[2])];
    const max4 = [Math.max(ff[0], g[0], j[0], k[0]), Math.max(ff[1], g[1], j[1], k[1]), Math.max(ff[2], g[2], j[2], k[2])];

    let aC = [0, 0, 0], aW = 0;
    const tap = (offX, offY, col) => {
        offX = f(offX - ppX); offY = f(offY - ppY); // shader: off = vec2(tap) - pp
        const vX = f(f(offX * dirX) + f(offY * dirY));
        const vY = f(f(offX * -dirY) + f(offY * dirX));
        const vx = f(vX * len2x), vy = f(vY * len2y);
        const d2 = min2(f(f(vx * vx) + f(vy * vy)), clp);
        let wB = f(f(2 / 5) * d2 - 1);
        let wA = f(lob * d2 - 1);
        wB = f(wB * wB); wA = f(wA * wA);
        wB = f(f(25 / 16) * wB - f(25 / 16 - 1));
        const w = f(wB * wA);
        aC = [f(aC[0] + f(col[0] * w)), f(aC[1] + f(col[1] * w)), f(aC[2] + f(col[2] * w))];
        aW = f(aW + w);
    };
    tap(0, -1, b); tap(1, -1, c); tap(-1, 1, i); tap(0, 1, j); tap(0, 0, ff); tap(-1, 0, e);
    tap(1, 1, k); tap(2, 1, l); tap(2, 0, h); tap(1, 0, g); tap(1, 2, o); tap(0, 2, n);

    const rW = f(1 / aW);
    return [
        min2(max4[0], max2(min4[0], f(aC[0] * rW))),
        min2(max4[1], max2(min4[1], f(aC[1] * rW))),
        min2(max4[2], max2(min4[2], f(aC[2] * rW))),
    ];
}

// ================= CPU 參考：RCAS（官方數學 + WGSL spec 語義） =================
const RCAS_LIMIT = 0.25 - 1 / 16;
function refRcas(easuF16, EW, EH, x, y, sharpness) {
    const load = (lx, ly) => {
        if (lx < 0 || ly < 0 || lx >= EW || ly >= EH) return [0, 0, 0]; // textureLoad OOB = 0
        const o = (ly * EW + lx) * 4;
        return [easuF16[o], easuF16[o + 1], easuF16[o + 2]];
    };
    const b = load(x, y - 1), d = load(x - 1, y), e = load(x, y), ff = load(x + 1, y), h = load(x, y + 1);
    const luma = (p) => f(f(p[2] * 0.5) + f(f(p[0] * 0.5) + p[1]));
    const bL = luma(b), dL = luma(d), eL = luma(e), fL = luma(ff), hL = luma(h);
    const nzRaw = f(f(f(0.25 * bL) + f(0.25 * dL)) + f(f(0.25 * fL) + f(0.25 * hL)) - eL);
    const range = f(max2(max2(max2(bL, dL), eL), max2(fL, hL)) - min2(min2(min2(bL, dL), eL), min2(fL, hL)));
    let nz = sat(f(Math.abs(nzRaw) * APrxMedRcpF1(range)));
    nz = f(-0.5 * nz + 1);
    const mn4 = [0, 1, 2].map(c => Math.min(b[c], d[c], ff[c], h[c]));
    const mx4 = [0, 1, 2].map(c => Math.max(b[c], d[c], ff[c], h[c]));
    const hitMin = [0, 1, 2].map(c => (mx4[c] === 0 ? 0 : f(min2(mn4[c], e[c]) * f(1 / f(4 * mx4[c])))));
    const hitMax = [0, 1, 2].map(c => (mn4[c] === 1 ? 0 : f(f(1 - max2(mx4[c], e[c])) * f(1 / f(f(4 * mn4[c]) - 4)))));
    const lobeC = [0, 1, 2].map(c => max2(-hitMin[c], hitMax[c]));
    let lobe = max2(-RCAS_LIMIT, min2(max2(max2(lobeC[0], lobeC[1]), lobeC[2]), 0));
    lobe = f(lobe * sharpness);
    lobe = f(lobe * nz);
    const rcpL = APrxMedRcpF1(f(f(4 * lobe) + 1));
    const res = [0, 1, 2].map(c =>
        f(f(f(f(f(lobe * b[c]) + f(lobe * d[c])) + f(lobe * h[c])) + f(lobe * ff[c])) + e[c]) * rcpL);
    return [0, 1, 2].map(c => min2(mx4[c], max2(mn4[c], res[c])));
}

// ================= 案例幀 =================
const baseLuma = 20;
function makeFrame(kind) {
    const W = 64, H = 64;
    const px = new Uint8Array(W * H * 4);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const o = (y * W + x) * 4;
        let rr = baseLuma, gg = baseLuma + 4, bb = baseLuma - 2;
        if (kind === 'black') { rr = gg = bb = 0; }
        if (kind === 'white') { rr = gg = bb = 255; }
        if (kind === 'noise') {
            const hsh = (x * 374761393 + y * 668265263) ^ 0x5bf03635;
            const nz = ((hsh >>> 8) & 0xff) / 255;
            rr = 8 + 40 * nz * nz; gg = 10 + 42 * nz * nz; bb = 7 + 30 * nz * nz;
        }
        if (kind === 'spike' && x === 32 && y === 32) { rr = 60; gg = baseLuma; bb = baseLuma; }
        if (kind === 'spike-strong' && x === 32 && y === 32) { rr = 90; gg = baseLuma; bb = baseLuma; }
        if (kind === 'spike-pair' && ((x === 32 && y === 32) || (x === 33 && y === 32))) { rr = 60; gg = baseLuma; bb = baseLuma; }
        if (kind === 'spike-border' && x === 0 && y === 0) { rr = 60; gg = baseLuma; bb = baseLuma; }
        if (kind === 'spike-near-border' && x === 1 && y === 1) { rr = 60; gg = baseLuma; bb = baseLuma; }
        px[o] = Math.max(0, Math.min(255, Math.round(rr)));
        px[o + 1] = Math.max(0, Math.min(255, Math.round(gg)));
        px[o + 2] = Math.max(0, Math.min(255, Math.round(bb)));
        px[o + 3] = 255;
    }
    return px;
}

// ================= 單案例執行（子行程模式） =================
async function runCase(kind) {
    const W = 64, H = 64, OW = 128, OH = 128;
    const code = fs.readFileSync(path.join(HERE, '..', 'examples', 'FSR_x2.wgsl'), 'utf8');
    const r = new WGFXRuntime(dev);
    const U = GPUTextureUsage;
    await r.compile(code, {
        defines: { INPUT_WIDTH: W, INPUT_HEIGHT: H },
        textures: {
            INPUT: { size: [W, H], format: 'rgba8unorm', usage: U.TEXTURE_BINDING | U.COPY_DST },
            OUTPUT: { size: [OW, OH], format: 'rgba16float', usage: U.STORAGE_BINDING | U.COPY_SRC | U.TEXTURE_BINDING | U.COPY_DST }
        }
    });
    const rm = r.resourceManager;
    rm.updateSceneBuffer(W, H, OW, OH);
    const sharpness = f(0.87);

    async function readTexF16(tex, w, h) {
        const bpr = Math.ceil(w * 8 / 256) * 256;
        const rb = dev.createBuffer({ size: bpr * h, usage: 0x8 | 0x1 });
        const cmd = dev.createCommandEncoder();
        cmd.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow: bpr, rowsPerImage: h }, [w, h, 1]);
        dev.queue.submit([cmd.finish()]);
        await rb.mapAsync(1);
        const u16 = new Uint16Array(rb.getMappedRange()).slice();
        rb.unmap(); rb.destroy();
        return u16;
    }
    const f16Decode = (hbits) => {
        const s = (hbits & 0x8000) ? -1 : 1, e = (hbits & 0x7c00) >> 10, m = hbits & 0x03ff;
        if (e === 0) return s * Math.pow(2, -14) * (m / 1024);
        if (e === 31) return m ? NaN : s * Infinity;
        return s * Math.pow(2, e - 15) * (1 + m / 1024);
    };
    const decodeF16Tex = (u16, w, h) => {
        const out = new Float32Array(w * h * 4);
        const stride = Math.ceil(w * 8 / 256) * 256 / 2;
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 4; c++)
            out[(y * w + x) * 4 + c] = f16Decode(u16[y * stride + x * 4 + c]);
        return out;
    };

    const px = makeFrame(kind);
    const nanFill = new Uint16Array(OW * OH * 4).fill(0x7e00);
    dev.queue.writeTexture({ texture: rm.getTexture('tex_easu') }, nanFill,
        { bytesPerRow: OW * 8, rowsPerImage: OH }, [OW, OH, 1]);
    dev.queue.writeTexture({ texture: rm.getTexture('OUTPUT') }, nanFill,
        { bytesPerRow: OW * 8, rowsPerImage: OH }, [OW, OH, 1]);
    dev.queue.writeTexture({ texture: rm.getTexture('INPUT') }, px,
        { bytesPerRow: W * 4, rowsPerImage: H }, [W, H, 1]);

    let cmd = dev.createCommandEncoder();
    r.dispatchPass('PASS_1', cmd);
    dev.queue.submit([cmd.finish()]);
    await dev.queue.onSubmittedWorkDone();
    const easuBits = await readTexF16(rm.getTexture('tex_easu'), OW, OH);
    const easu = decodeF16Tex(easuBits, OW, OH);

    cmd = dev.createCommandEncoder();
    r.dispatchPass('PASS_2', cmd);
    dev.queue.submit([cmd.finish()]);
    await dev.queue.onSubmittedWorkDone();
    const outBits = await readTexF16(rm.getTexture('OUTPUT'), OW, OH);
    const out = decodeF16Tex(outBits, OW, OH);

    let nanEasu = 0, nanOut = 0;
    for (let i = 0; i < OW * OH; i++) {
        if ((easuBits[i * 4] & 0x7c00) === 0x7c00 && (easuBits[i * 4] & 0x03ff) !== 0) nanEasu++;
        if ((outBits[i * 4] & 0x7c00) === 0x7c00 && (outBits[i * 4] & 0x03ff) !== 0) nanOut++;
    }

    const refE = new Float32Array(OW * OH * 4);
    for (let y = 0; y < OH; y++) for (let x = 0; x < OW; x++) {
        const v = refEasu(px, W, H, OW, OH, x, y);
        refE[(y * OW + x) * 4] = f16rt(v[0]); refE[(y * OW + x) * 4 + 1] = f16rt(v[1]); refE[(y * OW + x) * 4 + 2] = f16rt(v[2]); refE[(y * OW + x) * 4 + 3] = 1;
    }
    const refO = new Float32Array(OW * OH * 3);
    for (let y = 0; y < OH; y++) for (let x = 0; x < OW; x++) {
        const v = refRcas(refE, OW, OH, x, y, sharpness);
        refO[(y * OW + x) * 3] = f16rt(v[0]); refO[(y * OW + x) * 3 + 1] = f16rt(v[1]); refO[(y * OW + x) * 3 + 2] = f16rt(v[2]);
    }

    let dE = 0, dEpos = null, dO = 0, dOpos = null, nonFinite = 0;
    for (let y = 0; y < OH; y++) for (let x = 0; x < OW; x++) {
        for (let c = 0; c < 3; c++) {
            const gv = easu[(y * OW + x) * 4 + c], rv = refE[(y * OW + x) * 4 + c];
            if (!Number.isFinite(gv)) { nonFinite++; continue; }
            const dd = Math.abs(gv - rv);
            if (dd > dE) { dE = dd; dEpos = [x, y, c]; }
            const gv2 = out[(y * OW + x) * 4 + c], rv2 = refO[(y * OW + x) * 3 + c];
            if (!Number.isFinite(gv2)) { nonFinite++; continue; }
            const dd2 = Math.abs(gv2 - rv2);
            if (dd2 > dO) { dO = dd2; dOpos = [x, y, c]; }
        }
    }
    const result = { kind, nanEasu, nanOut, nonFinite, dE, dEpos, dO, dOpos };

    if (kind.startsWith('spike')) {
        const sx = kind === 'spike-border' ? 0 : kind === 'spike-near-border' ? 1 : 32;
        const sy = kind === 'spike-border' ? 0 : kind === 'spike-near-border' ? 1 : 32;
        const inChroma = (px[(sy * W + sx) * 4] - Math.max(px[(sy * W + sx) * 4 + 1], px[(sy * W + sx) * 4 + 2])) / 255;
        let worst = 0, worstP = null;
        for (let y = sy * 2; y < sy * 2 + 4 && y < OH; y++) for (let x = sx * 2; x < sx * 2 + 4 && x < OW; x++) {
            const rr = out[(y * OW + x) * 4], gg = out[(y * OW + x) * 4 + 1], bb = out[(y * OW + x) * 4 + 2];
            const ch = rr - Math.max(gg, bb);
            if (ch > worst) { worst = ch; worstP = [x, y]; }
        }
        result.blob = { inChroma, worst, worstP };
    }

    // dump 生成 WGSL（子行程也寫，冪等）
    fs.mkdirSync(path.join(HERE, 'out_wgsl'), { recursive: true });
    for (const m of r.generatedModules) {
        fs.writeFileSync(path.join(HERE, 'out_wgsl', `FSR_PASS_${m.passIndex}.wgsl`), m.wgslCode);
    }
    dev.destroy?.();
    return result;
}

// ================= 驅動器 =================
const isChild = process.argv.includes('--case');
const CASES = ['flat', 'black', 'white', 'spike', 'spike-strong', 'spike-pair', 'spike-border', 'spike-near-border', 'noise'];

if (isChild) {
    const kind = process.argv[process.argv.indexOf('--case') + 1];
    const adapter = await nav.gpu.requestAdapter();
    dev = await adapter.requestDevice({ requiredLimits: { maxStorageTexturesPerShaderStage: 8 } });
    const res = await runCase(kind);
    console.log('@@JSON@@' + JSON.stringify(res));
    process.exit(0);
} else {
    const results = [];
    for (const kind of CASES) {
        const p = spawnSync(process.execPath, [SELF, '--case', kind], { encoding: 'utf8', timeout: 180000 });
        const jsonLine = (p.stdout || '').split('\n').find(l => l.startsWith('@@JSON@@'));
        if (!jsonLine) {
            results.push({ kind, error: (p.stderr || p.stdout || 'no output').split('\n').slice(0, 6).join(' | ') });
            continue;
        }
        results.push(JSON.parse(jsonLine.slice(8)));
    }
    console.log('\n=== impulse_response 報告 ===');
    let allPass = true;
    for (const res of results) {
        if (res.error) { allPass = false; console.log(`${res.kind.padEnd(18)} 子行程失敗: ${res.error}`); continue; }
        const okE = res.dE <= 2e-3, okO = res.dO <= 3e-3;
        const ok = res.nanEasu === 0 && res.nanOut === 0 && res.nonFinite === 0 && okE && okO;
        allPass = allPass && ok;
        let line = `${res.kind.padEnd(18)} NaN(easu/out)=${res.nanEasu}/${res.nanOut} nonFinite=${res.nonFinite} |GPU-refEASU|=${res.dE.toExponential(2)}@${JSON.stringify(res.dEpos)} |GPU-refRCAS|=${res.dO.toExponential(2)}@${JSON.stringify(res.dOpos)} ${ok ? '[PASS]' : '[FAIL]'}`;
        if (res.blob) line += `\n    blob: inChroma=${res.blob.inChroma.toFixed(3)} → outChroma=${res.blob.worst.toFixed(3)} @${JSON.stringify(res.blob.worstP)}（bounded=${res.blob.worst <= res.blob.inChroma + 0.01}）`;
        console.log(line);
    }
    console.log(`\n覆蓋+忠實性總體: ${allPass ? '[PASS]' : '[FAIL]'}（生成 WGSL 已 dump 至 test/out_wgsl/）`);
}
