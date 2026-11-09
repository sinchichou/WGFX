// test_external_textures.mjs — //!SOURCE 外部紋理支援驗證
//
// 涵蓋:
//   A. tools/dds.mjs DDS 讀取器:合成 DDS(f16 RGBA)解碼、L8(LUMINANCE)、
//      行距非 256 倍數的資料;以及真實 Coef_Scale.dds 的已知 texel 值。
//   B. ShaderParser://!SOURCE 附加到對應 TEXTURE(TextureInfo.source)。
//   C. ResourceManager:宿主按名稱以 { size, format, data, bytesPerRow } 餵外部
//      紋理 → 真跑 GPU,驗證 shader 讀到與 LUT 一致的 texel(含 256 對齊重排)。
//   D. 宣告 //!SOURCE 但宿主未提供 → 拋出含紋理名稱與 SOURCE 的明確錯誤。
//   E. port_check.mjs 整合:--texture NAME=path 顯式餵入(子程序實測 CLI 契約)。
//
// 執行:node test/test_external_textures.mjs(需先 npm run build)
import 'webgpu';
import { create, globals } from 'webgpu';
Object.assign(globalThis, globals);
const nav = { gpu: create([]) };

import fs from 'fs';
import path from 'path';
import os from 'os';
import { execFileSync } from 'child_process';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.join(HERE, '..');
const DIST = path.join(REPO, 'dist', 'wgfx.esm.js');

const results = [];
function record(name, pass, detail = '') {
    results.push({ name, pass });
    console.log(`  [${pass ? 'PASS' : 'FAIL'}] ${name}${detail ? ' — ' + detail : ''}`);
}
async function runTest(name, fn) {
    try { await fn(); record(name, true); }
    catch (err) { record(name, false, (err && err.message) || String(err)); }
}
function assert(cond, msg) { if (!cond) throw new Error(msg); }

// ---- f16 打包/解包(精確值測試用) ----
function f32tof16(v) {
    // 精確值(0.5、2.0、-1.25…)適用的最小實作
    const b = new DataView(new ArrayBuffer(4));
    b.setFloat32(0, v);
    const x = b.getUint32(0);
    const s = (x >>> 16) & 0x8000;
    let e = ((x >>> 23) & 0xff) - 127 + 15;
    let m = (x >>> 13) & 0x3ff;
    if (e <= 0) return s;
    if (e >= 0x1f) return s | 0x7c00;
    return s | (e << 10) | m;
}
function f16tof32(h) {
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x3ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

/** 建構最小 DDS 位元組(未壓縮 fourCC 標頭 + 像素) */
function makeDDS(width, height, fourCC, bpp, pixels /* Uint8Array */) {
    const header = Buffer.alloc(128);
    header.write('DDS ', 0, 'ascii');
    const dv = new DataView(header.buffer);
    dv.setUint32(4, 124, true);          // headerSize
    dv.setUint32(12, height, true);
    dv.setUint32(16, width, true);
    dv.setUint32(20, width * height * bpp, true); // linearSize
    dv.setUint32(28, 1, true);           // mipCount
    dv.setUint32(76, 32, true);          // pfSize
    dv.setUint32(80, 0x4, true);         // DDPF_FOURCC
    dv.setUint32(84, fourCC, true);      // fourCC(數值型)
    dv.setUint32(108, 0x1000, true);     // DDSCAPS_TEXTURE
    return Buffer.concat([header, pixels]);
}

// ---- 測試效果:直接把 LUT texel 拷到 OUTPUT ----
const LUT_W = 4, LUT_H = 2;
const LUT_VALUES = [
    [0.5, 0.25, 0.125, 1.0], [2.0, -1.25, 0.75, 1.0], [0.0, 1.0, 0.5, 1.0], [0.125, 0.875, 0.0625, 1.0],
    [-0.5, 0.375, 1.5, 1.0], [0.0625, 0.4375, 0.9375, 1.0], [1.0, 0.5, 0.25, 1.0], [0.25, 0.75, 0.5, 1.0],
];
function lutPixelBytes() {
    const out = new Uint8Array(LUT_W * LUT_H * 8);
    const dv = new DataView(out.buffer);
    LUT_VALUES.forEach((px, i) => {
        dv.setUint16(i * 8 + 0, f32tof16(px[0]), true);
        dv.setUint16(i * 8 + 2, f32tof16(px[1]), true);
        dv.setUint16(i * 8 + 4, f32tof16(px[2]), true);
        dv.setUint16(i * 8 + 6, f32tof16(px[3]), true);
    });
    return out;
}

const EFFECT_SOURCE = `
//! VERSION 4
//! SORT_NAME ext_texture_test

//! TEXTURE
//! WIDTH INPUT_WIDTH
//! HEIGHT INPUT_HEIGHT
Texture2D INPUT;

//! TEXTURE
//! WIDTH 4
//! HEIGHT 2
Texture2D OUTPUT;

//! TEXTURE
//! SOURCE fake_lut_4x2_f16.dds
//! FORMAT R16G16B16A16_FLOAT
Texture2D lut;

//! SAMPLER
//! FILTER LINEAR
SamplerState sam_lut;

//! PASS 1
//! DESC "LUT passthrough"
//! IN INPUT
//! OUT OUTPUT
//! NUM_THREADS 8,8,1
fn Pass1(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= scene.outputSize.x || gid.y >= scene.outputSize.y) {
        return;
    }
    let texel = textureLoad(lut, vec2<i32>(gid.xy), 0);
    textureStore(OUTPUT, vec2<i32>(gid.xy), texel);
}
//! END
`;

async function main() {
    console.log('=== !SOURCE 外部紋理驗證 ===');
    fs.accessSync(DIST);
    const bundle = await import('file:///' + DIST.replace(/\\/g, '/'));
    // parse 未由 dist 匯出 → 以原始 peggy 產物驗證 parser(避免只測框架上層)
    const { parse: parseEffect } = await import('file:///' + path.join(REPO, 'src', 'runtime', 'ShaderParser.js').replace(/\\/g, '/'));

    const adapter = await nav.gpu.requestAdapter();
    assert(adapter, '無法取得 WebGPU adapter');
    const device = await adapter.requestDevice();
    device.onuncapturederror = (e) => {
        throw new Error('GPU 未捕捉錯誤: ' + String((e.error && e.error.message) || e));
    };

    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'wgfx-ext-'));
    const ddsPath = path.join(tmp, 'fake_lut_4x2_f16.dds');
    const ddsBytes = makeDDS(LUT_W, LUT_H, 113, 8, lutPixelBytes());
    fs.writeFileSync(ddsPath, ddsBytes);

    // ===== A. DDS 讀取器 =====
    await runTest('A1: DDS 讀取器解碼合成 f16 RGBA LUT', async () => {
        const { readDDSFile } = await import('file:///' + path.join(REPO, 'tools', 'dds.mjs').replace(/\\/g, '/'));
        const tex = readDDSFile(ddsPath, ddsPath);
        assert(tex.width === LUT_W && tex.height === LUT_H, `尺寸不符: ${tex.width}x${tex.height}`);
        assert(tex.format === 'rgba16float', `格式不符: ${tex.format}`);
        assert(tex.bytesPerPixel === 8, `bpp 不符: ${tex.bytesPerPixel}`);
        const dv = new DataView(tex.data.buffer, tex.data.byteOffset);
        LUT_VALUES.forEach((expect, i) => {
            const got = [f16tof32(dv.getUint16(i * 8, true)), f16tof32(dv.getUint16(i * 8 + 2, true)),
                f16tof32(dv.getUint16(i * 8 + 4, true)), f16tof32(dv.getUint16(i * 8 + 6, true))];
            got.forEach((v, c) => assert(Math.abs(v - expect[c]) < 1e-6,
                `texel ${i} ch${c}: 期望 ${expect[c]},實際 ${v}`));
        });
    });

    await runTest('A2: DDS 讀取器支援 L8(LUMINANCE)與不可靠 pitch', async () => {
        const { readDDS } = await import('file:///' + path.join(REPO, 'tools', 'dds.mjs').replace(/\\/g, '/'));
        const header = Buffer.alloc(128);
        header.write('DDS ', 0, 'ascii');
        const dv = new DataView(header.buffer);
        dv.setUint32(4, 124, true);
        dv.setUint32(12, 16, true);   // h
        dv.setUint32(16, 64, true);   // w
        dv.setUint32(20, 1024, true); // 故意的錯誤 pitch(模擬 SMAA SearchTex)
        dv.setUint32(76, 32, true);
        dv.setUint32(80, 0x20000, true); // DDPF_LUMINANCE
        dv.setUint32(88, 8, true);       // bits
        dv.setUint32(92, 0xff, true);    // R mask
        const pixels = Buffer.alloc(64 * 16, 0x80);
        const tex = readDDS(Buffer.concat([header, pixels]), 'search');
        assert(tex.format === 'r8unorm', `格式: ${tex.format}`);
        assert(tex.bytesPerRow === 64, `rowBytes 應由 payload 回推為 64,實際 ${tex.bytesPerRow}`);
        assert(tex.data.length === 64 * 16, `資料長度: ${tex.data.length}`);
    });

    await runTest('A3: DDS 讀取器對真實 Coef_Scale.dds 的已知 texel', async () => {
        const { readDDSFile } = await import('file:///' + path.join(REPO, 'tools', 'dds.mjs').replace(/\\/g, '/'));
        const tex = readDDSFile(path.join(REPO, 'effects_src', 'NIS', 'Coef_Scale.dds'));
        assert(tex.width === 2 && tex.height === 64 && tex.format === 'rgba16float',
            `${tex.width}x${tex.height} ${tex.format}`);
        const dv = new DataView(tex.data.buffer, tex.data.byteOffset);
        // tools/_dds_probe.mjs 實測值:row0 tex0 = [0, 0, 1, 0]
        const got = [f16tof32(dv.getUint16(0, true)), f16tof32(dv.getUint16(2, true)),
            f16tof32(dv.getUint16(4, true)), f16tof32(dv.getUint16(6, true))];
        assert(got[2] === 1.0 && got[0] === 0.0 && got[1] === 0.0 && got[3] === 0.0, `row0 tex0 = ${got}`);
    });

    // ===== B. Parser //!SOURCE =====
    await runTest('B: ShaderParser 將 //!SOURCE 附加到 TEXTURE(source 欄位)', () => {
        const info = parseEffect(EFFECT_SOURCE);
        const lut = info.textures.find((t) => t.name === 'lut');
        assert(!!lut, '找不到 lut 紋理');
        assert(lut.source === 'fake_lut_4x2_f16.dds', `source 欄位: ${lut.source}`);
        assert(lut.format === 'rgba16float', `format: ${lut.format}`);
        const input = info.textures.find((t) => t.name === 'INPUT');
        assert(!input.source, 'INPUT 不應有 source');
    });

    // ===== C. ResourceManager 外部紋理真跑 =====
    await runTest('C: 宿主按名稱餵 {size,format,data,bytesPerRow} → shader 讀到 LUT texel', async () => {
        const { WGFXRuntime } = bundle;
        const runtime = new WGFXRuntime(device);
        try {
            await runtime.compile(EFFECT_SOURCE, {
                defines: { INPUT_WIDTH: LUT_W, INPUT_HEIGHT: LUT_H },
                textures: {
                    OUTPUT: {
                        size: [LUT_W, LUT_H], format: 'rgba16float',
                        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
                    },
                    lut: {
                        size: [LUT_W, LUT_H], format: 'rgba16float',
                        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
                        data: lutPixelBytes(),
                        bytesPerRow: LUT_W * 8, // 32 bytes → 非 256 倍數,驗證框架自動重排
                    },
                },
            });
            runtime.resourceManager.updateSceneBuffer(LUT_W, LUT_H, LUT_W, LUT_H);
            // 單 encoder 執行
            device.pushErrorScope('validation');
            const enc = device.createCommandEncoder();
            for (const p of runtime.shaderInfo.passes) runtime.dispatchPass(`PASS_${p.index}`, enc);
            device.queue.submit([enc.finish()]);
            const err = await device.popErrorScope();
            assert(!err, `dispatch 驗證錯誤: ${err && err.message}`);
            // 讀回 OUTPUT 並比對 LUT
            const bytesPerRow = Math.ceil((LUT_W * 8) / 256) * 256;
            const staging = device.createBuffer({
                size: bytesPerRow * LUT_H,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            const enc2 = device.createCommandEncoder();
            enc2.copyTextureToBuffer({ texture: runtime.resourceManager.getTexture('OUTPUT') },
                { buffer: staging, bytesPerRow, rowsPerImage: LUT_H }, [LUT_W, LUT_H, 1]);
            device.queue.submit([enc2.finish()]);
            await staging.mapAsync(GPUMapMode.READ);
            const out = new Uint16Array(staging.getMappedRange().slice(0));
            staging.unmap();
            staging.destroy();
            const rowStride = bytesPerRow / 2; // 每列的 u16 數(行距 256 對齊)
            LUT_VALUES.forEach((expect, i) => {
                const x = i % LUT_W, y = (i / LUT_W) | 0;
                const base = y * rowStride + x * 4;
                for (let c = 0; c < 4; c++) {
                    const got = f16tof32(out[base + c]);
                    assert(Math.abs(got - expect[c]) < 1e-3,
                        `texel ${i} ch${c}: 期望 ${expect[c]},實際 ${got}`);
                }
            });
        } finally {
            runtime.resourceManager.dispose();
        }
    });

    // ===== D. 缺少外部紋理時的明確錯誤 =====
    await runTest('D: //!SOURCE 未提供紋理 → 錯誤訊息含紋理名與 SOURCE 檔名', async () => {
        const { WGFXRuntime } = bundle;
        const runtime = new WGFXRuntime(device);
        let thrown = null;
        try {
            await runtime.compile(EFFECT_SOURCE, {
                defines: { INPUT_WIDTH: LUT_W, INPUT_HEIGHT: LUT_H },
                textures: {
                    OUTPUT: {
                        size: [LUT_W, LUT_H], format: 'rgba16float',
                        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
                    },
                },
            });
        } catch (e) {
            thrown = e;
        } finally {
            try { runtime.resourceManager.dispose(); } catch { /* 未初始化完成 */ }
        }
        assert(thrown !== null, '未拋出錯誤');
        assert(/lut/.test(thrown.message), `訊息未含紋理名: ${thrown.message}`);
        assert(/fake_lut_4x2_f16\.dds/.test(thrown.message), `訊息未含 SOURCE 檔名: ${thrown.message}`);
    });

    // ===== E. port_check CLI 契約(--texture 顯式餵入) =====
    await runTest('E: port_check.mjs --texture NAME=path 顯式外部紋理(子程序)', () => {
        const effectPath = path.join(tmp, 'ext_effect.wgsl');
        // port_check 需要可推導的 INPUT/OUTPUT 尺寸(它以 --size 提供 INPUT;
        // OUTPUT 用效果宣告,此處與 LUT 同尺寸以便逐 texel 檢查非全黑即可)
        fs.writeFileSync(effectPath, EFFECT_SOURCE);
        const out = execFileSync(process.execPath,
            ['test/port_check.mjs', effectPath, '--size', '4', '--quiet',
                '--texture', `LUT=${ddsPath}`],
            { cwd: REPO, encoding: 'utf8' });
        assert(/ALL PASS/.test(out), `port_check 未 ALL PASS:\n${out}`);
    });

    // 清理暫存
    fs.rmSync(tmp, { recursive: true, force: true });

    const passed = results.filter((r) => r.pass).length;
    console.log(`\n=== 總結: ${passed}/${results.length} tests passed ===`);
    device.destroy?.();
    process.exit(passed === results.length ? 0 : 1);
}

try {
    await main();
} catch (err) {
    console.error('測試套件執行失敗:', err);
    process.exit(1);
}
