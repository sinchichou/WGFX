// _dds_probe.mjs — DDS 探測器（完整讀取器在 tools/dds.mjs,本檔為其除錯/檢視介面）
//
// 原版行為保留：傾印 effects_src/NIS 的 Coef_Scale / Coef_USM 前幾個 texel 的
// f16 解碼值,供人工確認 LUT 內容。擴充後:
//   * 以 tools/dds.mjs 的完整標頭解析取代硬編碼 offset（格式/尺寸自動辨識）
//   * node tools/_dds_probe.mjs                → NIS 兩檔的前 6 列 texel
//   * node tools/_dds_probe.mjs <file.dds> ... → 任意 DDS 的格式資訊 + 前 6 列 texel
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { readDDSFile } from './dds.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));

/** f16 半精度 → number */
function h2f(h) {
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x3ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function dump(file, rows = 6) {
    const t = readDDSFile(file);
    console.log(`=== ${path.basename(file)}: ${t.width}x${t.height} ${t.format} bpp=${t.bytesPerPixel} rowBytes=${t.bytesPerRow} mips=${t.mipCount}`);
    const dv = new DataView(t.data.buffer, t.data.byteOffset, t.data.byteLength);
    const showRows = Math.min(rows, t.height);
    for (let y = 0; y < showRows; y++) {
        const vals = [];
        for (let x = 0; x < Math.min(2, t.width); x++) {
            const off = y * t.bytesPerRow + x * t.bytesPerPixel;
            let q;
            if (t.format === 'rgba16float') {
                q = [h2f(dv.getUint16(off, true)), h2f(dv.getUint16(off + 2, true)), h2f(dv.getUint16(off + 4, true)), h2f(dv.getUint16(off + 6, true))];
            } else if (t.format === 'rgba8unorm') {
                q = [t.data[off] / 255, t.data[off + 1] / 255, t.data[off + 2] / 255, t.data[off + 3] / 255];
            } else if (t.format === 'r8unorm') {
                q = [t.data[off] / 255];
            } else {
                q = ['…'];
            }
            vals.push('[' + q.map((v) => (typeof v === 'number' ? v.toFixed(4) : v)).join(', ') + ']');
        }
        console.log(` row${y}  ${vals.join('  ')}`);
    }
}

const args = process.argv.slice(2);
if (args.length === 0) {
    // 原版行為：NIS 的兩個係數表
    for (const n of ['Coef_Scale', 'Coef_USM']) {
        dump(path.join(HERE, '..', 'effects_src', 'NIS', n + '.dds'));
    }
} else {
    for (const p of args) dump(p);
}
