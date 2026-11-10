// dds_consistency.test.mjs — 驗證兩份 DDS 實作的一致性
//
// WGFX 有兩份 DDS 讀取實作：
//   · tools/dds.mjs         —— 建置期（打包器）使用（Node，純 JS）
//   · src/runtime/DDSReader.ts —— 執行期（.wgx 套件包在瀏覽器中載入外部紋理）使用
//
// 兩者必須對**每一個**真實 DDS 檔給出完全相同的結果，否則同一個 LUT 在
// 打包期與執行期會被解讀成不同的紋理。
//
// 本測試對 effects_src/ 下全部 DDS 檔逐一比對：
//   尺寸、格式、bytesPerRow、mipCount、像素資料（逐位元組）
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..');

let pass = 0, fail = 0;
function check(name, ok, detail = '') {
    if (ok) { pass++; console.log(`  [PASS] ${name}`); }
    else { fail++; console.log(`  [FAIL] ${name}${detail ? ' — ' + detail : ''}`); }
}

// 建置期實作
const { readDDS: readDDSBuild } = await import(pathToFileURL(path.join(REPO, 'tools', 'dds.mjs')).href);
// 執行期實作（取自建置產物）
const { readDDS: readDDSTime } = await import(pathToFileURL(path.join(REPO, 'dist', 'wgfx.esm.js')).href);

function listDDS(dir, out = []) {
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return out; }
    for (const ent of entries) {
        const full = path.join(dir, ent.name);
        if (ent.isDirectory()) listDDS(full, out);
        else if (ent.name.toLowerCase().endsWith('.dds')) out.push(full);
    }
    return out;
}

const ddsFiles = listDDS(path.join(REPO, 'effects_src'));
console.log(`=== 1. 一致性:effects_src/ 下 ${ddsFiles.length} 個 DDS 檔 ===\n`);

for (const file of ddsFiles) {
    const rel = path.relative(REPO, file).replace(/\\/g, '/');
    const buf = fs.readFileSync(file);
    let a, b, errA = null, errB = null;
    try { a = readDDSBuild(buf, rel); } catch (e) { errA = e.message; }
    try { b = readDDSTime(new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength), rel); } catch (e) { errB = e.message; }

    if (errA || errB) {
        check(`${rel} — 兩者錯誤行為一致`, !!errA === !!errB,
            `build=${errA?.slice(0, 40)} runtime=${errB?.slice(0, 40)}`);
        continue;
    }
    const same = a.width === b.width
        && a.height === b.height
        && a.format === b.format
        && a.bytesPerRow === b.bytesPerRow
        && a.mipCount === b.mipCount
        && a.data.length === b.data.length
        && Buffer.compare(Buffer.from(a.data), Buffer.from(b.data)) === 0;
    check(`${rel} — ${a.width}×${a.height} ${a.format} 兩者完全一致`, same,
        same ? '' : `build(${a.width}×${a.height} ${a.format} ${a.data.length}B) vs runtime(${b.width}×${b.height} ${b.format} ${b.data.length}B)`);
}

// ---------- 2. 兩者都必須能處理真實 RAVU / SMAA / NIS 的 LUT ----------
console.log('\n=== 2. 關鍵 LUT 的格式辨識 ===');
const keyFiles = [
    // 檔名雖含 f16，實際 fourCC 為 D3DFMT 數值 113 = RGBA16F → rgba16float
    ['effects_src/RAVU/ravu_3x_lut2_f16.dds', 'rgba16float'],
    ['effects_src/SMAA/AreaTex.dds', 'rgba8unorm'],
    ['effects_src/SMAA/SearchTex.dds', 'r8unorm'],
    ['effects_src/NIS/Coef_Scale.dds', 'rgba16float'],
];
for (const [rel, expectFormat] of keyFiles) {
    const p = path.join(REPO, rel);
    if (!fs.existsSync(p)) {
        console.log(`  [SKIP] ${rel} — 不存在`);
        continue;
    }
    const buf = fs.readFileSync(p);
    const a = readDDSBuild(buf, rel);
    const b = readDDSTime(new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength), rel);
    const ok = a.format === b.format && (!expectFormat || a.format === expectFormat);
    check(`${rel} → ${a.format}${expectFormat ? `（預期 ${expectFormat}）` : ''}`, ok,
        `runtime=${b.format}`);
}

console.log('');
console.log(`=== 總結: ${pass}/${pass + fail} tests passed ===`);
process.exit(fail === 0 ? 0 : 1);
