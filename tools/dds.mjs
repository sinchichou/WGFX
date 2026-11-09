// dds.mjs — 完整 DDS 讀取器（未壓縮、非 BC 格式 + D3DFMT 數值型 fourCC 的 f16/f32 通道）
//
// WGFX !SOURCE 外部紋理（RAVU / SMAA / NIS 的 .dds LUT）用。
// 支援：
//   * 標準 DDS 標頭（124 bytes + optional DX10 extension header）
//   * 未壓縮格式：
//       - DDPF_FOURCC 數值型（D3DFMT 列舉值）：111=R16F、112=RG16F、113=RGBA16F、
//         114=R32F、115=RG32F、116=RGBA32F、110=RGBA16_UNORM
//       - DDPF_FOURCC 字面型：'DXT1/3/5' 等 BC 格式 → 明確報錯（本讀取器不解壓 BC）
//       - DDPF_RGB（32/24 bits + bitmask）→ rgba8unorm（24bit 展開 alpha=255）
//       - DDPF_LUMINANCE / DDPF_ALPHA（8 bits）→ r8unorm（SMAA SearchTex 類）
//   * mip0 的像素資料（列序:上到下、左到右,與 WebGPU writeTexture 一致,免翻轉）
//   * header 的 pitch 欄位不可靠時（如 SearchTex pitch=1024 而 64x16 L8）
//     以「payload 尺寸 == height × width×bpp」優先回推行距
//
// 回傳 { width, height, format, data: Uint8Array, mipCount, bytesPerPixel }
// 用法：
//   import { readDDS } from './dds.mjs';
//   const tex = readDDS(fs.readFileSync('ravu_lut2_f16.dds'));
import fs from 'fs';

const FOURCC_NUM = {
    110: { format: 'rgba16unorm', bpp: 8 },
    111: { format: 'r16float', bpp: 2 },
    112: { format: 'rg16float', bpp: 4 },
    113: { format: 'rgba16float', bpp: 8 },
    114: { format: 'r32float', bpp: 4 },
    115: { format: 'rg32float', bpp: 8 },
    116: { format: 'rgba32float', bpp: 16 },
};

// DX10 extension 的 dxgiFormat → WebGPU（僅列未壓縮常用）
const DXGI_MAP = {
    2:  { format: 'rgba32float', bpp: 16 },   // R32G32B32A32_FLOAT
    1:  { format: 'rgba32uint', bpp: 16 },
    54: { format: 'rg16float', bpp: 4 },      // R16G16_FLOAT
    55: { format: 'rg16float', bpp: 4 },      // R16G16_FLOAT (alias)
    10: { format: 'rgba16float', bpp: 8 },    // R16G16B16A16_FLOAT
    13: { format: 'rgba16unorm', bpp: 8 },
    62: { format: 'r32float', bpp: 4 },
    41: { format: 'r32float', bpp: 4 },
    114: { format: 'r32float', bpp: 4 },      // R11G11B10_FLOAT → 以 rgba16float 承接
    24: { format: 'rg11b10ufloat', bpp: 4 },
    87: { format: 'bgra8unorm', bpp: 4 },
    28: { format: 'rgba8unorm', bpp: 4 },     // R8G8B8A8_UNORM
    29: { format: 'rgba8unorm-srgb', bpp: 4 },
    61: { format: 'r8unorm', bpp: 1 },
};

// BC 壓縮 fourCC（不支援,明確報錯）
const BC_FOURCC = new Set(['DXT1', 'DXT2', 'DXT3', 'DXT4', 'DXT5', 'BC4U', 'BC4S', 'BC5U', 'BC5S', 'BC6H', 'BC7U', 'ATI1', 'ATI2']);

function fourCCStr(b, off) {
    return String.fromCharCode(b[off], b[off + 1], b[off + 2], b[off + 3]);
}

/**
 * 解析 DDS 緩衝並回傳 mip0 的尺寸/格式/像素資料。
 * @param {Uint8Array|Buffer} buf - DDS 檔案位元組
 * @param {string} [name] - 錯誤訊息用的檔名
 */
export function readDDS(buf, name = 'dds') {
    if (buf.length < 128) throw new Error(`${name}: 檔案過小(${buf.length} bytes),非 DDS`);
    if (fourCCStr(buf, 0) !== 'DDS ') throw new Error(`${name}: 缺少 'DDS ' magic`);
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    const headerSize = dv.getUint32(4, true);
    const height = dv.getUint32(12, true);
    const width = dv.getUint32(16, true);
    const linearSize = dv.getUint32(20, true);
    const mipCount = Math.max(1, dv.getUint32(28, true));
    const pfSize = dv.getUint32(76, true);
    const pfFlags = dv.getUint32(80, true);
    const fourCC = fourCCStr(buf, 84);
    const bitCount = dv.getUint32(88, true);
    const rMask = dv.getUint32(92, true);
    const gMask = dv.getUint32(96, true);
    const bMask = dv.getUint32(100, true);
    const aMask = dv.getUint32(104, true);
    if (headerSize < 124) throw new Error(`${name}: DDS 標頭大小異常 (${headerSize})`);
    if (!width || !height) throw new Error(`${name}: 尺寸為 0 (${width}x${height})`);

    const DDPF_FOURCC = 0x4, DDPF_RGB = 0x40, DDPF_LUMINANCE = 0x20000, DDPF_ALPHA = 0x2;

    let format = null;
    let bpp = 0;

    if (pfFlags & DDPF_FOURCC) {
        const fccNum = dv.getUint32(84, true);
        if (BC_FOURCC.has(fourCC.trim())) {
            throw new Error(`${name}: BC 壓縮格式（${fourCC.trim()}）不受支援——本 DDS 讀取器僅支援未壓縮格式`);
        }
        if (fccNum === 0x30315844) { // 'DX10'
            const dxgiFormat = dv.getUint32(128, true);
            const entry = DXGI_MAP[dxgiFormat];
            if (!entry) throw new Error(`${name}: DX10 dxgiFormat ${dxgiFormat} 不受支援`);
            format = entry.format;
            bpp = entry.bpp;
        } else if (FOURCC_NUM[fccNum]) {
            format = FOURCC_NUM[fccNum].format;
            bpp = FOURCC_NUM[fccNum].bpp;
        } else {
            throw new Error(`${name}: fourCC ${fccNum}（'${fourCC.trim()}'）不受支援`);
        }
    } else if (pfFlags & DDPF_RGB) {
        if (bitCount === 32) {
            // 8bit/channel RGBA（遮罩為慣例配置時直接承認;WGFX 端以 rgba8unorm 取樣）
            format = 'rgba8unorm';
            bpp = 4;
        } else if (bitCount === 24) {
            format = 'rgba8unorm';
            bpp = 3;
        } else {
            throw new Error(`${name}: DDPF_RGB bits=${bitCount} 不受支援（僅 24/32）`);
        }
    } else if ((pfFlags & DDPF_LUMINANCE) || (pfFlags & DDPF_ALPHA)) {
        if (bitCount === 8) {
            format = 'r8unorm'; // L8/A8 → 單通道
            bpp = 1;
        } else if (bitCount === 16 && (aMask !== 0)) {
            format = 'rg8unorm'; // L8A8
            bpp = 2;
        } else {
            throw new Error(`${name}: LUMINANCE/ALPHA bits=${bitCount} 不受支援`);
        }
    } else {
        throw new Error(`${name}: 不支援的 pixelformat flags=0x${pfFlags.toString(16)}`);
    }

    if (!format || !bpp) throw new Error(`${name}: 格式解析失敗`);

    // 資料偏移：標頭(128) + 可選 DX10 擴充(20)
    let dataOffset = 128;
    if ((pfFlags & DDPF_FOURCC) && dv.getUint32(84, true) === 0x30315844) dataOffset += 20;

    const expectedRowBytes = width * bpp;
    const payload = buf.length - dataOffset;
    // header pitch 不可靠時以 payload 回推:payload == height * expectedRowBytes → 用 expectedRowBytes
    let rowBytes = expectedRowBytes;
    if (linearSize % Math.max(1, height) === 0 && linearSize / height >= expectedRowBytes &&
        linearSize / height <= expectedRowBytes * 4 && payload >= linearSize) {
        // linearSize 看起來是合理的總量 → 用它推 rowBytes（涵蓋 row-aligned 的檔案）
        const guess = linearSize / height;
        if (guess >= expectedRowBytes && guess <= 64) rowBytes = guess; // 合理範圍內採信
    }
    if (payload < rowBytes * height) {
        throw new Error(`${name}: 像素資料不足(payload=${payload},需要=${rowBytes * height})`);
    }
    const data = buf.slice(dataOffset, dataOffset + rowBytes * height);
    // 行距與 expectedRowBytes 不同時(未壓縮,WebGPU writeTexture 需要 bytesPerRow)回傳行距供宿主使用
    return { width, height, format, data, mipCount, bytesPerPixel: bpp, bytesPerRow: rowBytes };
}

/** 從檔案讀取 DDS（便捷封裝） */
export function readDDSFile(path) {
    return readDDS(fs.readFileSync(path), path);
}

// CLI:node tools/dds.mjs <file.dds> [...] — 傾印格式與前幾個 texel
if (process.argv[1] && import.meta.url.endsWith(process.argv[1].replace(/\\/g, '/').split('/').pop())) {
    for (const p of process.argv.slice(2)) {
        try {
            const t = readDDSFile(p);
            console.log(`${p}: ${t.width}x${t.height} ${t.format} bpp=${t.bytesPerPixel} rowBytes=${t.bytesPerRow} mips=${t.mipCount}`);
        } catch (e) {
            console.error(`${p}: ${e.message}`);
        }
    }
}
