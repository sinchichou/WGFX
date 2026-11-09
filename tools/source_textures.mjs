// source_textures.mjs — //! SOURCE 外部紋理的掃描、解析與載入（共用模組）
//
// 由 test/port_check.mjs 與 test/test_runtime.js 共用，確保兩邊行為一致：
//   - 掃描效果碼中的 //! SOURCE 宣告
//   - 依「顯式指定 → effects_src 遞迴檔名索引」順序解析來源檔
//   - 以 tools/dds.mjs 載入為 ResourceManager 可接受的紋理描述符
//
// 載入後的描述符欄位：size / format / usage / data / bytesPerRow
// （ResourceManager 會直接以 queue.writeTexture 上傳）。
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));

/**
 * 掃描效果碼中的 //! SOURCE 宣告。
 *
 * Magpie 的宣告順序是「//!TEXTURE … //!SOURCE <file> … Texture2D <name>;」，
 * 因此以 TEXTURE 區塊為單位配對 SOURCE 與隨後的 Texture2D 名稱。
 *
 * @param {string} code 效果原始碼
 * @returns {Map<string,string>} 紋理名稱 → SOURCE 檔名（如 ravu_3x_lut2 → ravu_3x_lut2_f16.dds）
 */
export function scanSourceDecls(code) {
    const map = new Map();
    let cur = null; // { name, source }
    const BLOCK_STARTS = ['TEXTURE', 'SAMPLER', 'PARAMETER', 'BUFFER', 'COMMON', 'PASS'];
    const META_DIRS = ['VERSION', 'SORT_NAME', 'USE', 'CAPABILITY', 'STYLE', 'DESC'];
    for (const raw of code.split(/\r?\n/)) {
        const line = raw.trim();
        const dirM = line.match(/^\/\/!\s*([A-Za-z_]+)\s*(.*)$/);
        if (dirM) {
            const dir = dirM[1].toUpperCase();
            const val = dirM[2].trim();
            if (dir === 'TEXTURE') {
                if (cur && cur.name && cur.source) map.set(cur.name, cur.source);
                cur = { name: null, source: null };
                continue;
            }
            if (cur) {
                if (dir === 'SOURCE') cur.source = val.replace(/^"|"$/g, '');
                else if (BLOCK_STARTS.includes(dir) || META_DIRS.includes(dir)) {
                    if (cur.name && cur.source) map.set(cur.name, cur.source);
                    cur = null;
                }
            }
            continue;
        }
        if (cur) {
            const declM = line.match(/^(?:Texture2D|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*[;:]/);
            if (declM) {
                // _sampled / _storaged 是同一邏輯紋理的成對別名
                cur.name = declM[1].replace(/_(sampled|storaged)$/, '');
            }
        }
    }
    if (cur && cur.name && cur.source) map.set(cur.name, cur.source);
    return map;
}

/**
 * 遞迴收集 dir 下所有檔案，建立「小寫檔名 → 完整路徑」索引（先到先得）。
 * @param {string} dir
 * @param {Map<string,string>} [index]
 * @returns {Map<string,string>}
 */
export function indexFilesByBasename(dir, index = new Map()) {
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return index; }
    for (const ent of entries) {
        const full = path.join(dir, ent.name);
        if (ent.isDirectory()) indexFilesByBasename(full, index);
        else {
            const key = ent.name.toLowerCase();
            if (!index.has(key)) index.set(key, full);
        }
    }
    return index;
}

/**
 * 載入一個 DDS 為外部紋理描述符。
 * @param {string} file DDS 檔路徑
 * @returns {Promise<{size:number[],format:string,usage:number,data:Uint8Array,bytesPerRow:number,height:number}>}
 */
export async function loadDDSTexture(file) {
    const { readDDS } = await import(new URL('./dds.mjs', import.meta.url).href);
    const tex = readDDS(fs.readFileSync(file), file);
    return {
        size: [tex.width, tex.height],
        format: tex.format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        data: tex.data,
        bytesPerRow: tex.bytesPerRow,
    };
}

/**
 * 為單一效果準備外部紋理資源。
 *
 * 解析順序：explicitTextures（顯式指定）優先，其次 textureDir 底下的檔名索引。
 * 效果未宣告任何 //! SOURCE 時回傳空物件（零成本）。
 *
 * @param {string} code 效果原始碼
 * @param {object} opts
 * @param {string} opts.textureDir 自動解析根目錄（通常為 effects_src/）
 * @param {Map<string,string>} [opts.explicitTextures] 紋理名（小寫）→ 檔案路徑
 * @param {boolean} [opts.quiet] 不輸出資訊列
 * @param {(msg:string)=>void} [opts.onInfo] 自訂資訊輸出（預設 console.log）
 * @param {string} [opts.label] 錯誤訊息中的效果名稱
 * @returns {Promise<Record<string, object>>} 紋理名 → 描述符
 */
export async function prepareExternalTextures(code, opts = {}) {
    const { textureDir, explicitTextures, quiet = false, onInfo, label = '' } = opts;
    const decls = scanSourceDecls(code);
    if (decls.size === 0) return {};

    const index = textureDir ? indexFilesByBasename(textureDir) : new Map();
    const resolved = {};
    for (const [texName, sourceFile] of decls) {
        const key = texName.toLowerCase();
        let file = explicitTextures ? explicitTextures.get(key) : undefined;
        let via = '--texture';
        if (!file) {
            file = index.get(String(sourceFile).toLowerCase());
            via = 'effects_src 自動解析';
        }
        if (!file) {
            throw new Error(
                `效果宣告 //! SOURCE ${sourceFile}（紋理 ${texName}）但找不到檔案:` +
                `請以 --texture ${texName}=<路徑> 提供或確認 effects_src/ 下存在該檔` +
                (label ? `（${label}）` : '')
            );
        }
        if (!fs.existsSync(file)) throw new Error(`外部紋理檔不存在: ${file}`);
        resolved[texName] = await loadDDSTexture(file);
        if (!quiet) {
            const msg = `  [TEXTURE] ${texName} ← ${path.relative(process.cwd(), file)} (${via})`;
            if (onInfo) onInfo(msg); else console.log(msg);
        }
    }
    return resolved;
}
