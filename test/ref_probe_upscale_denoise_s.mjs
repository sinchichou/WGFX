// 回歸探測（regression probe）：Anime4K_Upscale_Denoise_S 逐 pass 對照 JS 參考實作。
// 參考語義 = Magpie 原版 HLSL：MulAdd(v, float3x4/4x4) = Σ_i v_i·args[4i+k]（row-major 構建）。
// 執行：npm run build && npm run test:ref（需真實 WebGPU 裝置）
import { create, globals } from 'webgpu';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
const HERE = path.dirname(fileURLToPath(import.meta.url));
const { WGFXRuntime } = await import('../dist/wgfx.esm.js');

const adapter = await nav.gpu.requestAdapter();
const dev = await adapter.requestDevice({ requiredLimits: { maxStorageTexturesPerShaderStage: 8 } });

const W = 32, H = 32, OW = 64, OH = 64;

function f16(h) {
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, m = h & 0x03ff;
    if (e === 0) return (m ? Math.pow(2, -14) * (m / 1024) : 0) * (s ? -1 : 1);
    if (e === 31) return s ? NaN : Infinity;
    return Math.pow(2, e - 15) * (1 + m / 1024) * (s ? -1 : 1);
}

function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

async function readTexRGBA16(tex, w, h) {
    const bytesPerRow = Math.ceil(w * 8 / 256) * 256;
    const rb = dev.createBuffer({ size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const cmd = dev.createCommandEncoder();
    cmd.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow, rowsPerImage: h }, [w, h, 1]);
    dev.queue.submit([cmd.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const u16 = new Uint16Array(rb.getMappedRange());
    const out = new Float32Array(w * h * 4);
    const stride = bytesPerRow / 2;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 4; c++) {
        out[(y * w + x) * 4 + c] = f16(u16[y * stride + x * 4 + c]);
    }
    rb.unmap();
    return out;
}

// ---- JS 參考：解析 .wgsl 權重（與原版 HLSL 數字逐項一致） ----
function parseWeights(text, fnName, count) {
    const fnStart = text.indexOf(`fn ${fnName}(`);
    const nextFn = text.indexOf('\nfn ', fnStart + 1);
    const seg = text.slice(fnStart, nextFn === -1 ? fnStart + 40000 : nextFn);
    const lines = [...seg.matchAll(/MF[34]x4\(([^)]*)\)/g)];
    const taps = lines.slice(0, count).map(m => m[1].split(',').map(Number));
    if (taps.length !== count) throw new Error(`${fnName}: expected ${count} taps, got ${taps.length}`);
    return taps; // taps[t] = 12 或 16 個數字（原版 HLSL row-major）
}

function conv3(v3, args) { // mul(v, float3x4): result_k = Σ_r v_r args[4r+k]
    const out = [0, 0, 0, 0];
    for (let r = 0; r < 3; r++) for (let k = 0; k < 4; k++) out[k] += v3[r] * args[4 * r + k];
    return out;
}
function conv4(v4, args) { // mul(v, float4x4): result_k = Σ_r v_r args[4r+k]
    const out = [0, 0, 0, 0];
    for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) out[k] += v4[r] * args[4 * r + k];
    return out;
}

const shaderText = fs.readFileSync(path.join(HERE, '..', 'examples', 'Anime4K_Upscale_Denoise_S.wgsl'), 'utf8');
const W1 = parseWeights(shaderText, 'A4KS1', 9);
const W2 = parseWeights(shaderText, 'A4KS2', 18);
const W3 = parseWeights(shaderText, 'A4KS3', 18);
const B1 = [-0.004601904, -0.0030944077, 0.14569537, -0.016794242];
const B2 = [0.022899346, 0.033619333, 0.030674957, -0.017047008];
const B3 = [0.04921464, -0.0011432811, 0.062071066, -0.06594219];
const B4 = [-0.0031417734, -0.002754766, -0.004053268, -0.003937834];
const W4 = parseWeights(shaderText, 'A4KS4', 18);

// ---- 準備 runtime 與輸入 ----
const r = new WGFXRuntime(dev);
await r.compile(shaderText, {
    defines: { INPUT_WIDTH: W, INPUT_HEIGHT: H },
    textures: { INPUT: { size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } }
});
const rm = r.resourceManager;
const inPx = new Uint8Array(W * H * 4);
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const o = (y * W + x) * 4;
    inPx[o] = (x * 7 + y * 13) % 256; inPx[o + 1] = (x * 29 + y * 3) % 256; inPx[o + 2] = (x * 5 + y * 41) % 256; inPx[o + 3] = 255;
}
dev.queue.writeTexture({ texture: rm.getTexture('INPUT') }, inPx, { bytesPerRow: W * 4, rowsPerImage: H }, [W, H, 1]);
rm.updateSceneBuffer(W, H, OW, OH);

const inFloat = new Float32Array(W * H * 4);
for (let i = 0; i < W * H; i++) { inFloat[i * 4] = inPx[i * 4] / 255; inFloat[i * 4 + 1] = inPx[i * 4 + 1] / 255; inFloat[i * 4 + 2] = inPx[i * 4 + 2] / 255; inFloat[i * 4 + 3] = 1; }
function inTexel(tx, ty, c) {
    tx = clamp(tx, 0, W - 1); ty = clamp(ty, 0, H - 1);
    return inFloat[(ty * W + tx) * 4 + c];
}

function dispatchPasses(names) {
    const cmd = dev.createCommandEncoder();
    for (const n of names) r.dispatchPass(n, cmd);
    dev.queue.submit([cmd.finish()]);
}

// ---- PASS 1 對照 ----
dispatchPasses(['PASS_1']);
const tex1 = await readTexRGBA16(rm.getTexture('tex1_storaged'), W, H);
let maxDiff1 = 0;
for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) {
    // src[a][b] = texel(px+a-1, py+b-1)（gather 角點語義）
    const src = [];
    for (let a = 0; a < 4; a++) { src.push([]); for (let b = 0; b < 4; b++) {
        src[a].push([inTexel(px + a - 1, py + b - 1, 0), inTexel(px + a - 1, py + b - 1, 1), inTexel(px + a - 1, py + b - 1, 2)]);
    } }
    const subs = [[1, 1, 0, 0], [2, 1, 1, 0], [2, 2, 1, 1], [1, 2, 0, 1]]; // [i, j, dx, dy]
    for (const [i, j, dx, dy] of subs) {
        const out = [...B1];
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
            const c = conv3(src[i + a - 1][j + b - 1], W1[a * 3 + b]);
            for (let k = 0; k < 4; k++) out[k] += c[k];
        }
        const gx = px + dx, gy = py + dy;
        if (gx < W && gy < H) for (let k = 0; k < 3; k++) {
            maxDiff1 = Math.max(maxDiff1, Math.abs(out[k] - tex1[(gy * W + gx) * 4 + k]));
        }
    }
}
console.log('PASS1 maxDiff(GPU vs JS 參考):', maxDiff1.toFixed(5), maxDiff1 < 0.02 ? 'OK' : 'FAIL');

// ---- PASS 2 對照（以 GPU tex1 為輸入）----
dispatchPasses(['PASS_2']);
const tex2 = await readTexRGBA16(rm.getTexture('tex2_storaged'), W, H);
function t1(px, py, c) { return tex1[(clamp(py, 0, H - 1) * W + clamp(px, 0, W - 1)) * 4 + c]; }
let maxDiff2 = 0;
for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) {
    const src = [];
    for (let a = 0; a < 4; a++) { src.push([]); for (let b = 0; b < 4; b++) {
        src[a].push([t1(px + a - 1, py + b - 1, 0), t1(px + a - 1, py + b - 1, 1), t1(px + a - 1, py + b - 1, 2), t1(px + a - 1, py + b - 1, 3)]);
    } }
    const subs = [[1, 1, 0, 0], [2, 1, 1, 0], [2, 2, 1, 1], [1, 2, 0, 1]];
    for (const [i, j, dx, dy] of subs) {
        const out = [...B2];
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) for (let part = 0; part < 2; part++) {
            const v = src[i + a - 1][j + b - 1].map(c => (part === 0 ? 1 : -1) * c);
            const mv = v.map(c => Math.max(c, 0));
            const c4 = conv4(mv, W2[a * 3 + b + part * 9]);
            for (let k = 0; k < 4; k++) out[k] += c4[k];
        }
        const gx = px + dx, gy = py + dy;
        if (gx < W && gy < H) for (let k = 0; k < 3; k++) {
            maxDiff2 = Math.max(maxDiff2, Math.abs(out[k] - tex2[(gy * W + gx) * 4 + k]));
        }
    }
}
console.log('PASS2 maxDiff:', maxDiff2.toFixed(5), maxDiff2 < 0.02 ? 'OK' : 'FAIL');

// ---- PASS 3 對照 ----
dispatchPasses(['PASS_3']);
const tex3 = await readTexRGBA16(rm.getTexture('tex1_storaged'), W, H); // pass3 寫回 tex1
function t2(px, py, c) { return tex2[(clamp(py, 0, H - 1) * W + clamp(px, 0, W - 1)) * 4 + c]; }
let maxDiff3 = 0;
for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) {
    const src = [];
    for (let a = 0; a < 4; a++) { src.push([]); for (let b = 0; b < 4; b++) {
        src[a].push([t2(px + a - 1, py + b - 1, 0), t2(px + a - 1, py + b - 1, 1), t2(px + a - 1, py + b - 1, 2), t2(px + a - 1, py + b - 1, 3)]);
    } }
    const subs = [[1, 1, 0, 0], [2, 1, 1, 0], [2, 2, 1, 1], [1, 2, 0, 1]];
    for (const [i, j, dx, dy] of subs) {
        const out = [...B3];
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) for (let part = 0; part < 2; part++) {
            const v = src[i + a - 1][j + b - 1].map(c => (part === 0 ? 1 : -1) * c);
            const mv = v.map(c => Math.max(c, 0));
            const c4 = conv4(mv, W3[a * 3 + b + part * 9]);
            for (let k = 0; k < 4; k++) out[k] += c4[k];
        }
        const gx = px + dx, gy = py + dy;
        if (gx < W && gy < H) for (let k = 0; k < 3; k++) {
            maxDiff3 = Math.max(maxDiff3, Math.abs(out[k] - tex3[(gy * W + gx) * 4 + k]));
        }
    }
}
console.log('PASS3 maxDiff:', maxDiff3.toFixed(5), maxDiff3 < 0.02 ? 'OK' : 'FAIL');

// ---- PASS 4 對照（conv at texel centers + depth-to-space + bilinear residual）----
dispatchPasses(['PASS_4']);
const outTex = await readTexRGBA16(rm.getTexture('OUTPUT'), OW, OH);
function t3(px, py, c) { return tex3[(clamp(py, 0, H - 1) * W + clamp(px, 0, W - 1)) * 4 + c]; }
function bilinear(u, v, c) { // INPUT 的雙線性取樣（sam1 LINEAR, CLAMP）
    const fu = clamp(u, 0, 1) * W - 0.5, fv = clamp(v, 0, 1) * H - 0.5;
    const x0 = Math.floor(fu), y0 = Math.floor(fv);
    const fx = fu - x0, fy = fv - y0;
    const s = (xx, yy) => inTexel(xx, yy, c);
    return (s(x0, y0) * (1 - fx) + s(x0 + 1, y0) * fx) * (1 - fy)
         + (s(x0, y0 + 1) * (1 - fx) + s(x0 + 1, y0 + 1) * fx) * fy;
}
let maxDiff4 = 0;
for (let oy = 0; oy < OH; oy++) for (let ox = 0; ox < OW; ox++) {
    // gxy = (ox, oy)；thread 基點為偶數格，sub-pixel 依 (ox%2, oy%2) 對應 c.x/y/w/z
    const bx = ox - (ox % 2), by = oy - (oy % 2);
    const kx = bx >> 1, ky = by >> 1; // 輸入 texel 索引（整數）
    const px2 = kx + 0.5, py2 = ky + 0.5; // conv 中心（texel 座標系，供 residual 用）
    const taps = [];
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        taps.push([t3(px2 + dx, py2 + dy, 0), t3(px2 + dx, py2 + dy, 1), t3(px2 + dx, py2 + dy, 2), t3(px2 + dx, py2 + dy, 3)]);
    }
    const compIdx = (ox % 2 === 0 && oy % 2 === 0) ? 0 : (ox % 2 === 1 && oy % 2 === 0) ? 1 : (ox % 2 === 1 && oy % 2 === 1) ? 3 : 2; // c.x, c.y, c.w, c.z
    const out = [...B4];
    // 原版 tap 順序：a..i 以 dx 為外層（a=(-1,-1), b=(-1,0), c=(-1,+1), d=(0,-1), ...）
    let t = 0;
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
        const tap = [t3(kx + dx, ky + dy, 0), t3(kx + dx, ky + dy, 1), t3(kx + dx, ky + dy, 2), t3(kx + dx, ky + dy, 3)];
        for (let part = 0; part < 2; part++) {
            const v = tap.map(c => (part === 0 ? 1 : -1) * c);
            const mv = v.map(c => Math.max(c, 0));
            const c4 = conv4(mv, W4[t + part * 9]);
            for (let k = 0; k < 4; k++) out[k] += c4[k];
        }
        t++;
    }
    // residual: pos = conv中心*inputPt − 0.5*outputPt，再依子像素偏移
    const resU = (px2 / W) - 0.5 / OW + (ox % 2) / OW;
    const resV = (py2 / H) - 0.5 / OH + (oy % 2) / OH;
    const o = (oy * OW + ox) * 4;
    for (let k = 0; k < 3; k++) {
        const want = out[compIdx] + bilinear(resU, resV, k);
        if (ox < 2 && oy < 2 && k === 0) console.log("  JS out(" + ox + "," + oy + ") conv=" + out[0].toFixed(4) + " res=" + bilinear(resU, resV, 0).toFixed(4) + " want=" + want.toFixed(4));
        maxDiff4 = Math.max(maxDiff4, Math.abs(want - outTex[o + k]));
    }
}
console.log('PASS4 maxDiff:', maxDiff4.toFixed(5), maxDiff4 < 0.05 ? 'OK' : 'FAIL');
const allOk = maxDiff1 < 0.02 && maxDiff2 < 0.02 && maxDiff3 < 0.02 && maxDiff4 < 0.05;
console.log(allOk ? '=== REFERENCE PROBE PASSED ===' : '=== REFERENCE PROBE FAILED ===');
process.exit(allOk ? 0 : 1);
