/// <reference lib="dom" />

/**
 * DDS 讀取器 — WGFX `//! SOURCE` 外部紋理用。
 * ---
 * 支援 WGFX 的外部 LUT 紋理（RAVU / SMAA / NIS 的 `.dds`）。
 *
 * 這份實作是 `tools/dds.mjs` 的 runtime 版本 —— **同一套格式判斷**，
 * 使建置期（打包器）與執行期（擴充套件）對同一個 DDS 檔得到相同結果。
 *
 * 支援的格式
 * ---
 * - 標準 DDS 標頭（124 bytes + 選用的 DX10 擴充標頭）
 * - 未壓縮格式：
 *   - `DDPF_FOURCC` **數值型**（D3DFMT 列舉值）：110=RGBA16_UNORM、111=R16F、
 *     112=RG16F、113=RGBA16F、114=R32F、115=RG32F、116=RGBA32F
 *   - `DDPF_FOURCC` 字面型：'DXT1/3/5' 等 BC 格式 → 明確報錯（不解壓 BC）
 *   - `DDPF_RGB`（32/24 bits）→ rgba8unorm
 *   - `DDPF_LUMINANCE` / `DDPF_ALPHA`（8 bits）→ r8unorm（SMAA SearchTex 類）
 * - mip0 的像素資料（列序：上到下、左到右，與 WebGPU `writeTexture` 一致，免翻轉）
 * - header 的 pitch 欄位不可靠時（如 SearchTex pitch=1024 而實際 64×16 L8），
 *   以「payload 尺寸 == height × width×bpp」優先回推行距
 *
 * @group Core
 * @category Resources
 */

/** DDS 解析結果。 */
export interface DDSImage {
    /** 寬度 */
    width: number;
    /** 高度 */
    height: number;
    /** WebGPU 紋理格式 */
    format: string;
    /** mip0 的像素資料 */
    data: Uint8Array;
    /** mip 層數 */
    mipCount: number;
    /** 每像素位元組數 */
    bytesPerPixel: number;
    /** 每列位元組數（WebGPU `writeTexture` 的 bytesPerRow） */
    bytesPerRow: number;
}

/** DDPF_FOURCC 數值型（D3DFMT 列舉值）→ WebGPU 格式 */
const FOURCC_NUM: Record<number, { format: string; bpp: number }> = {
    110: { format: 'rgba16unorm', bpp: 8 },
    111: { format: 'r16float', bpp: 2 },
    112: { format: 'rg16float', bpp: 4 },
    113: { format: 'rgba16float', bpp: 8 },
    114: { format: 'r32float', bpp: 4 },
    115: { format: 'rg32float', bpp: 8 },
    116: { format: 'rgba32float', bpp: 16 },
};

/** DX10 extension 的 dxgiFormat → WebGPU（僅列未壓縮常用） */
const DXGI_MAP: Record<number, { format: string; bpp: number }> = {
    1: { format: 'rgba32uint', bpp: 16 },
    2: { format: 'rgba32float', bpp: 16 },
    10: { format: 'rgba16float', bpp: 8 },
    13: { format: 'rgba16unorm', bpp: 8 },
    24: { format: 'rg11b10ufloat', bpp: 4 },
    28: { format: 'rgba8unorm', bpp: 4 },
    29: { format: 'rgba8unorm-srgb', bpp: 4 },
    41: { format: 'r32float', bpp: 4 },
    54: { format: 'rg16float', bpp: 4 },
    55: { format: 'rg16float', bpp: 4 },
    61: { format: 'r8unorm', bpp: 1 },
    62: { format: 'r32float', bpp: 4 },
    87: { format: 'bgra8unorm', bpp: 4 },
    114: { format: 'r32float', bpp: 4 },
};

/** BC 壓縮 fourCC（不支援，明確報錯） */
const BC_FOURCC = new Set([
    'DXT1', 'DXT2', 'DXT3', 'DXT4', 'DXT5',
    'BC4U', 'BC4S', 'BC5U', 'BC5S', 'BC6H', 'BC7U', 'ATI1', 'ATI2',
]);

function fourCCStr(b: Uint8Array, off: number): string {
    return String.fromCharCode(b[off], b[off + 1], b[off + 2], b[off + 3]);
}

/**
 * 解析 DDS 緩衝並回傳 mip0 的尺寸／格式／像素資料。
 * @param buf DDS 檔案位元組
 * @param name 錯誤訊息用的檔名
 */
export function readDDS(buf: Uint8Array, name = 'dds'): DDSImage {
    if (buf.length < 128) {
        throw new Error(`${name}: 檔案過小(${buf.length} bytes)，非 DDS`);
    }
    if (fourCCStr(buf, 0) !== 'DDS ') {
        throw new Error(`${name}: 缺少 'DDS ' magic`);
    }
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    const headerSize = dv.getUint32(4, true);
    const height = dv.getUint32(12, true);
    const width = dv.getUint32(16, true);
    const linearSize = dv.getUint32(20, true);
    const mipCount = Math.max(1, dv.getUint32(28, true));
    const pfFlags = dv.getUint32(80, true);
    const fourCC = fourCCStr(buf, 84);
    const bitCount = dv.getUint32(88, true);
    const aMask = dv.getUint32(104, true);

    if (headerSize < 124) throw new Error(`${name}: DDS 標頭大小異常 (${headerSize})`);
    if (!width || !height) throw new Error(`${name}: 尺寸為 0 (${width}x${height})`);

    const DDPF_FOURCC = 0x4;
    const DDPF_RGB = 0x40;
    const DDPF_LUMINANCE = 0x20000;
    const DDPF_ALPHA = 0x2;

    let format: string | null = null;
    let bpp = 0;

    if (pfFlags & DDPF_FOURCC) {
        const fccNum = dv.getUint32(84, true);
        if (BC_FOURCC.has(fourCC.trim())) {
            throw new Error(
                `${name}: BC 壓縮格式（${fourCC.trim()}）不受支援——本讀取器僅支援未壓縮格式`);
        }
        if (fccNum === 0x30315844) {   // 'DX10'
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
            format = 'r8unorm';
            bpp = 1;
        } else if (bitCount === 16 && aMask !== 0) {
            format = 'rg8unorm';
            bpp = 2;
        } else {
            throw new Error(`${name}: LUMINANCE/ALPHA bits=${bitCount} 不受支援`);
        }
    } else {
        throw new Error(`${name}: 不支援的 pixelformat flags=0x${pfFlags.toString(16)}`);
    }

    if (!format || !bpp) throw new Error(`${name}: 格式解析失敗`);

    // 資料偏移：標頭(128) + 選用的 DX10 擴充(20)
    let dataOffset = 128;
    if ((pfFlags & DDPF_FOURCC) && dv.getUint32(84, true) === 0x30315844) dataOffset += 20;

    const expectedRowBytes = width * bpp;
    const payload = buf.length - dataOffset;

    // header 的 pitch 不可靠時以 payload 回推
    let rowBytes = expectedRowBytes;
    if (
        linearSize % Math.max(1, height) === 0
        && linearSize / height >= expectedRowBytes
        && linearSize / height <= expectedRowBytes * 4
        && payload >= linearSize
    ) {
        const guess = linearSize / height;
        if (guess >= expectedRowBytes && guess <= 64) rowBytes = guess;
    }
    if (payload < rowBytes * height) {
        throw new Error(`${name}: 像素資料不足(payload=${payload}，需要=${rowBytes * height})`);
    }

    const data = buf.slice(dataOffset, dataOffset + rowBytes * height);
    return { width, height, format, data, mipCount, bytesPerPixel: bpp, bytesPerRow: rowBytes };
}
