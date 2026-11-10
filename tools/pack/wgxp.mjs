// wgxp.mjs — `.wgx` 套件包打包器（建置期工具）
//
// 把一個 WGFX 效果（`effects/**/*.wgsl`）打包成「預編」套件包：
//
//   1. 解析效果 DSL → 取得 pass / resource / parameter metadata
//   2. 以 WGSLCodeGenerator 預生成各 pass 的 WGSL（省掉執行期的解析與 codegen）
//   3. 收集 pass 實際用到的資源（紋理、sampler、storage buffer、uniform）
//   4. 寫入 manifest.json + 各 pass 的 WGSL，壓縮成 zip（.wgx）
//
// 用法
// ---
//   node tools/pack/wgxp.mjs <效果.wgsl> [-o out.wgx]
//   node tools/pack/wgxp.mjs --batch effects/Scalers --out-dir dist/wgx
//   node tools/pack/wgxp.mjs --batch effects --out-dir dist/wgx [--only CuNNy,NNEDI3]
//
// 設計要點
// ---
// · **不碰 WGSL 語意**：預生成的 WGSL 與執行期 `WGSLCodeGenerator.generate()`
//   的輸出完全相同（同一支 codegen），故語意零風險。
// · **defines 於執行期注入**：輸入尺寸在執行期才知道，故生成的 WGSL 保留
//   `NW`/`NH` 等識別字，由宿主在 runtime 以 storage buffer / uniform 帶入
//   （見 `runtime/WGFXPackage.ts`）。此設計讓同一套件包能處理任意尺寸。
// · **壓縮**：zip entry 以 deflate-raw 壓縮（WGSL 文字壓縮率約 22%）。
//
// 對應計畫：docs/wgxp-format.md（格式規格）

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createManifest, validateManifest, MANIFEST_VERSION } from './manifest.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');

const DIST = path.join(REPO, 'dist', 'wgfx.esm.js');
const PARSER = path.join(REPO, 'src', 'runtime', 'ShaderParser.js');

/** 最小 zip writer（只支援 deflate-raw store/method 8 + CRC32）。 */
class ZipWriter {
    constructor() {
        this.entries = [];
    }

    /**
     * 加入一個檔案。
     * @param {string} name 套件包內的路徑（以 / 分隔）
     * @param {Buffer} data 內容
     * @param {boolean} compress 是否用 deflate-raw 壓縮
     */
    add(name, data, compress = true) {
        const crc = crc32(data);
        let method = 0;
        let payload = data;
        if (compress && data.length > 0) {
            const packed = zlib.deflateRawSync(data, { level: 9 });
            // 壓縮後反而更大時（例如已是壓縮格式）就不壓
            if (packed.length < data.length) {
                method = 8;
                payload = packed;
            }
        }
        this.entries.push({ name, crc, method, payload, uncompressedSize: data.length });
    }

    /** 組出 zip bytes。 */
    toBuffer() {
        const chunks = [];
        const central = [];
        let offset = 0;

        for (const e of this.entries) {
            const nameBuf = Buffer.from(e.name, 'utf8');
            const local = Buffer.alloc(30);
            local.writeUInt32LE(0x04034b50, 0);   // local file header signature
            local.writeUInt16LE(20, 4);           // version needed
            local.writeUInt16LE(0, 6);            // flags
            local.writeUInt16LE(e.method, 8);     // compression method
            local.writeUInt16LE(0, 10);           // mod time
            local.writeUInt16LE(0, 12);           // mod date
            local.writeUInt32LE(e.crc, 14);
            local.writeUInt32LE(e.payload.length, 18);
            local.writeUInt32LE(e.uncompressedSize, 22);
            local.writeUInt16LE(nameBuf.length, 26);
            local.writeUInt16LE(0, 28);           // extra length
            chunks.push(local, nameBuf, e.payload);

            const cd = Buffer.alloc(46);
            cd.writeUInt32LE(0x02014b50, 0);      // central directory signature
            cd.writeUInt16LE(20, 4);              // version made by
            cd.writeUInt16LE(20, 6);              // version needed
            cd.writeUInt16LE(0, 8);               // flags
            cd.writeUInt16LE(e.method, 10);
            cd.writeUInt16LE(0, 12);              // mod time
            cd.writeUInt16LE(0, 14);              // mod date
            cd.writeUInt32LE(e.crc, 16);
            cd.writeUInt32LE(e.payload.length, 20);
            cd.writeUInt32LE(e.uncompressedSize, 24);
            cd.writeUInt16LE(nameBuf.length, 28);
            cd.writeUInt16LE(0, 30);              // extra length
            cd.writeUInt16LE(0, 32);              // comment length
            cd.writeUInt16LE(0, 34);              // disk number start
            cd.writeUInt16LE(0, 36);              // internal attrs
            cd.writeUInt32LE(0, 38);              // external attrs
            cd.writeUInt32LE(offset, 42);         // relative offset of local header
            central.push(cd, nameBuf);

            offset += local.length + nameBuf.length + e.payload.length;
        }

        const centralBuf = Buffer.concat(central);
        const eocd = Buffer.alloc(22);
        eocd.writeUInt32LE(0x06054b50, 0);       // EOCD signature
        eocd.writeUInt16LE(0, 4);
        eocd.writeUInt16LE(0, 6);
        eocd.writeUInt16LE(this.entries.length, 8);
        eocd.writeUInt16LE(this.entries.length, 10);
        eocd.writeUInt32LE(centralBuf.length, 12);
        eocd.writeUInt32LE(offset, 16);
        eocd.writeUInt16LE(0, 20);

        return Buffer.concat([...chunks, centralBuf, eocd]);
    }
}

let CRC_TABLE = null;
function crc32(buf) {
    if (!CRC_TABLE) {
        CRC_TABLE = new Int32Array(256);
        for (let n = 0; n < 256; n++) {
            let c = n;
            for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
            CRC_TABLE[n] = c;
        }
    }
    let crc = -1;
    for (let i = 0; i < buf.length; i++) {
        crc = (crc >>> 8) ^ CRC_TABLE[(crc ^ buf[i]) & 0xff];
    }
    return (crc ^ -1) >>> 0;
}

/** 從檔名推導效果 id（去除副檔名、將路徑分隔符換成 '-'）。 */
function deriveId(relPath) {
    return relPath
        .replace(/\\/g, '/')
        .replace(/\.wgsl$/, '')
        .replace(/[^A-Za-z0-9._-]+/g, '-')
        .replace(/-+/g, '-')
        .replace(/^-|-$/g, '');
}

/**
 * 在 `effects_src/` 底下依檔名尋找外部來源檔（`//! SOURCE` 指向的 DDS 等）。
 * 大小寫不敏感；先到先得。
 */
const SOURCE_INDEX = (() => {
    let index = null;
    return () => {
        if (index) return index;
        index = new Map();
        const root = path.join(REPO, 'effects_src');
        const walk = (d) => {
            let entries;
            try { entries = fs.readdirSync(d, { withFileTypes: true }); } catch { return; }
            for (const ent of entries) {
                const full = path.join(d, ent.name);
                if (ent.isDirectory()) walk(full);
                else {
                    const key = ent.name.toLowerCase();
                    if (!index.has(key)) index.set(key, full);
                }
            }
        };
        walk(root);
        return index;
    };
})();

function findSourceFile(fileName) {
    return SOURCE_INDEX().get(String(fileName).toLowerCase());
}

/** 載入 WGFX 的 parser 與 codegen（同一支實作，確保語意一致）。 */
async function loadToolchain() {
    const { parse } = await import(pathToFileURL(PARSER).href);
    const { WGSLCodeGenerator } = await import(pathToFileURL(DIST).href);
    return { parse, WGSLCodeGenerator };
}

/**
 * 打包單一效果。
 * @param {object} opts
 * @param {string} opts.effectPath 效果 .wgsl 的絕對路徑
 * @param {string} opts.outPath  輸出 .wgx 的絕對路徑
 * @param {object} [opts.toolchain] 已載入的 parse/codegen
 * @param {object} [opts.defines] 產生期的 defines（尺寸相關；預設留空以維持執行期彈性）
 */
export async function packEffect(opts) {
    const { effectPath, outPath } = opts;
    const toolchain = opts.toolchain || await loadToolchain();
    const { parse, WGSLCodeGenerator } = toolchain;

    const code = fs.readFileSync(effectPath, 'utf8');
    const shaderInfo = parse(code);

    // 產生各 pass 的 WGSL。defines 留空 → 生成的 WGSL 保留 NW/NH 等識別字，
    // 由執行期以 uniform/storage buffer 帶入（同一套件包可處理任意尺寸）。
    const generator = new WGSLCodeGenerator();
    const modules = generator.generate(shaderInfo, opts.defines || {});

    const relFromEffects = path.relative(path.join(REPO, 'effects'), effectPath)
        .replace(/\\/g, '/');
    const id = deriveId(relFromEffects);
    const family = relFromEffects.includes('/') ? relFromEffects.split('/')[0] : '';

    const zip = new ZipWriter();
    const passes = [];

    // ---- 收集外部紋理（//! SOURCE）並收進套件包 ----
    // 套件包必須自足：外部 LUT 紋理（DDS）要一併打包，執行期才不必
    // 回頭找 effects_src/。
    const externalTextures = new Map();
    for (const tex of shaderInfo.textures || []) {
        if (!tex.source) continue;
        const fileName = tex.source;
        const found = findSourceFile(fileName);
        if (!found) {
            throw new Error(
                `找不到 //! SOURCE 指定的檔案: ${fileName}（效果 ${relFromEffects}）`);
        }
        const entryPath = `textures/${path.basename(fileName)}`;
        const data = fs.readFileSync(found);
        zip.add(entryPath, data, true);
        externalTextures.set(tex.name, {
            kind: 'texture',
            name: tex.name,
            file: entryPath,
            source: fileName,
            bytes: data.length,
        });
    }

    for (const m of modules) {
        const passInfo = shaderInfo.passes.find((p) => p.index === m.passIndex);
        const file = `shaders/pass${m.passIndex}.wgsl`;
        zip.add(file, Buffer.from(m.wgslCode, 'utf8'), true);

        const resources = [];
        for (const tex of m.resources.textures || []) {
            resources.push({
                kind: 'texture',
                name: tex.name,
                binding: tex.binding,
                format: tex.format,
                isStorage: !!tex.isStorage,
            });
        }
        for (const samp of m.resources.samplers || []) {
            resources.push({ kind: 'sampler', name: samp.name, binding: samp.binding });
        }
        for (const buf of m.resources.storageBuffers || []) {
            resources.push({
                kind: 'storageBuffer',
                name: buf.name,
                binding: buf.binding,
                mode: buf.mode,
                // 權重與特徵圖的大小取決於效果架構與輸入尺寸；打包期無法得知
                // 特徵圖的最終大小（隨解析度變動），故標記為執行期決定。
                // 權重則會在階段 2 轉為二進位檔（帶 file 欄位）。
                runtimeSized: !buf.file,
            });
        }

        passes.push({
            index: m.passIndex,
            file,
            dispatch: passInfo?.dispatch || undefined,
            workgroupSize: passInfo?.blockSize || passInfo?.numThreads || undefined,
            inputs: passInfo?.in || [],
            outputs: passInfo?.out || [],
            resources,
        });
    }

    // 參數：沿用 WGFX 的 ParameterInfo 形狀
    const parameters = (shaderInfo.parameters || []).map((p) => ({
        name: p.name,
        type: p.type || 'float',
        default: p.default ?? 0,
        min: p.min ?? 0,
        max: p.max ?? 1,
        step: p.step ?? 0.01,
        label: p.label,
    }));

    // 裝置上限需求：偵測此效果是否需要高於預設的上限
    const maxStorageTextures = Math.max(
        0,
        ...modules.map((m) => (m.resources.textures || []).filter((t) => t.isStorage).length),
    );
    const limits = {};
    if (maxStorageTextures > 4) {
        limits.maxStorageTexturesPerShaderStage = maxStorageTextures;
    }

    const manifest = createManifest({
        id,
        name: shaderInfo.metadata?.sortName || path.basename(effectPath, '.wgsl'),
        family,
        description: shaderInfo.metadata?.desc || undefined,
        source: relFromEffects,
        tags: [family].filter(Boolean),
        metadata: {
            sortName: shaderInfo.metadata?.sortName,
            version: shaderInfo.metadata?.version,
            use: shaderInfo.metadata?.use || {},
            capability: shaderInfo.metadata?.capability || {},
        },
        parameters,
        passes,
        resources: [...externalTextures.values()],
        limits: Object.keys(limits).length ? limits : undefined,
        buildTool: `wgxp/1.0.0 (manifest v${MANIFEST_VERSION})`,
    });

    // 驗證 manifest（交叉檢查套件包內實際存在的檔案）
    const fileSet = new Set([
        ...passes.map((p) => p.file),
        ...[...externalTextures.values()].map((r) => r.file),
    ]);
    const issues = validateManifest(manifest, { files: fileSet });
    if (issues.length) {
        throw new Error(`manifest 驗證失敗:\n  - ${issues.join('\n  - ')}`);
    }

    zip.add('manifest.json', Buffer.from(JSON.stringify(manifest, null, 2), 'utf8'), true);

    const out = zip.toBuffer();
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.writeFileSync(outPath, out);

    return {
        id,
        outPath,
        bytes: out.length,
        rawBytes: Buffer.byteLength(code, 'utf8'),
        passes: passes.length,
        parameters: parameters.length,
        manifest,
    };
}

/** 遞迴列出目錄下的 .wgsl。 */
function listEffects(dir) {
    const out = [];
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return out; }
    for (const ent of entries) {
        const full = path.join(dir, ent.name);
        if (ent.isDirectory()) out.push(...listEffects(full));
        else if (ent.name.endsWith('.wgsl')) out.push(full);
    }
    return out.sort();
}

/** CLI */
async function main() {
    const argv = process.argv.slice(2);
    if (argv.length === 0 || argv.includes('--help')) {
        console.log(`用法:
  node tools/pack/wgxp.mjs <效果.wgsl> [-o out.wgx]
  node tools/pack/wgxp.mjs --batch <目錄> --out-dir <輸出目錄> [--only 家族,家族]

選項:
  --batch <dir>     打包目錄下所有 .wgsl（遞迴）
  --out-dir <dir>   批次輸出目錄（預設 dist/wgx）
  --out <file>      單檔輸出路徑
  --only <list>     批次時只打包指定家族（逗號分隔）
  --quiet           只輸出摘要`);
        return;
    }

    const quiet = argv.includes('--quiet');
    const log = (...a) => { if (!quiet) console.log(...a); };

    const batchIdx = argv.indexOf('--batch');
    const onlyIdx = argv.indexOf('--only');
    const outDirIdx = argv.indexOf('--out-dir');
    const outIdx = argv.indexOf('--out');

    const toolchain = await loadToolchain();
    const results = [];

    if (batchIdx >= 0) {
        const dir = path.resolve(argv[batchIdx + 1]);
        const outDir = path.resolve(outDirIdx >= 0 ? argv[outDirIdx + 1] : path.join(REPO, 'dist', 'wgx'));
        const only = onlyIdx >= 0 ? argv[onlyIdx + 1].split(',').map((s) => s.trim()) : null;

        let files = listEffects(dir);
        if (only) {
            files = files.filter((f) => only.some((o) => f.replace(/\\/g, '/').includes(`/${o}/`)));
        }
        log(`打包 ${files.length} 個效果 → ${outDir}\n`);
        for (const f of files) {
            const rel = path.relative(dir, f).replace(/\\/g, '/');
            const outPath = path.join(outDir, rel.replace(/\.wgsl$/, '.wgx'));
            try {
                const r = await packEffect({ effectPath: f, outPath, toolchain });
                results.push(r);
                log(`  ✓ ${rel.padEnd(46)} ${String(r.passes).padStart(3)} pass  ` +
                    `${(r.rawBytes / 1024).toFixed(1).padStart(8)} KiB → ${(r.bytes / 1024).toFixed(1).padStart(7)} KiB`);
            } catch (e) {
                log(`  ✗ ${rel}: ${(e.message || '').split('\n')[0]}`);
                process.exitCode = 1;
            }
        }
    } else {
        const effectPath = path.resolve(argv[0]);
        const outPath = path.resolve(outIdx >= 0 ? argv[outIdx + 1] : effectPath.replace(/\.wgsl$/, '.wgx'));
        const r = await packEffect({ effectPath, outPath, toolchain });
        results.push(r);
        log(`✓ ${path.basename(outPath)}: ${r.passes} pass, ` +
            `${(r.rawBytes / 1024).toFixed(1)} KiB → ${(r.bytes / 1024).toFixed(1)} KiB`);
    }

    if (results.length) {
        const raw = results.reduce((s, r) => s + r.rawBytes, 0);
        const packed = results.reduce((s, r) => s + r.bytes, 0);
        console.log('');
        console.log(`=== 總計 ${results.length} 個套件包:`);
        console.log(`    原始 ${(raw / 1024 / 1024).toFixed(2)} MiB → 打包 ${(packed / 1024 / 1024).toFixed(2)} MiB` +
            ` (${((packed / raw) * 100).toFixed(1)}%)`);
    }
}

// 僅在「以本檔為主程式」時執行 CLI（以 process.argv[1] 判定；被 import 時不跑）。
const invokedDirectly = process.argv[1]
    && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (invokedDirectly) {
    main().catch((e) => {
        console.error('[wgxp] 失敗:', e);
        process.exit(1);
    });
}
