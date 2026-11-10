// split_passes.mjs — WGFX 效果檔的 pass 自動拆分器(storage texture 數量門檻)
//
// 背景:Dawn 預設 maxStorageTexturesPerShaderStage = 4;CuNNy2 的 NVL 變體每個
// conv pass 寫 6–8 張 storage texture,必須拆成連續子 pass(每個子 pass 寫不同
// 組 textureStore 目標、重讀同一組取樣)。每個輸出 vec4 是「同一組取樣」的獨立
// 函式,因此「整段計算原樣保留、只刪除非本組的 textureStore 陳述式」在數值上
// 與單一 pass 完全等價(tools/cunny2wgsl.mjs 的內建拆分即此語意)。
//
// 用法:
//   node tools/pack/split_passes.mjs <in.wgsl> <out.wgsl> [--max-storage-textures N]
//   node tools/pack/split_passes.mjs --merge <in.wgsl> <out.wgsl>   # 把 (part k/n) 子 pass 合回單一 pass
//
// 函式 API(供測試):
//   parseEffect(text)        → { prefixLines, passes }
//   storageTargetsOf(pass)   → string[](//!OUT 中以 _storaged 結尾者,保持順序)
//   storeStatementsOf(pass)  → { line, index, target }[]
//   splitEffect(text, opts)  → { text, report }   opts: { maxStorageTextures = 4 }
//   mergeSplitPasses(text)   → { text, report }   # 反向操作(等價性驗證用)
//
// 白名單契約:任何無法證明「拆分不改變語意」的形狀一律 throw(絕不產出錯誤產物):
//   1. //!OUT 混有非 _storaged 項(如 OUTPUT)→ 無法只按 storage 分組 → throw
//   2. textureStore 目標集合 ≠ //!OUT 集合(漏 store 或多 store)→ throw
//   3. textureStore 不是「單行完整陳述式」(跨行/複合)→ 位置刪除不安全 → throw
//   4. pass 內讀取自己要寫的紋理(出現 `${T}_sampled`)→ 讀寫競爭,拆分會改變
//      取樣時序 → throw
//   5. //!DISPATCH pass(網格語意與紋理寫入交織)→ throw
//   6. 拆分後某子 pass 仍超過門檻(數學上不可能,防禦性檢查)→ throw
import fs from 'fs';
import path from 'path';

const DEFAULT_MAX_STORAGE_TEXTURES = 4;

// ----------------------------------------------------------------------------
// 效果檔解析(輕量、只辨識 //! 指令;程式碼行原樣保留)
// ----------------------------------------------------------------------------

const PASS_SUB_DIRECTIVES = new Set(['DESC', 'BLOCK_SIZE', 'NUM_THREADS', 'IN', 'OUT', 'DISPATCH']);
const BLOCK_STARTS = new Set(['PARAMETER', 'TEXTURE', 'SAMPLER', 'BUFFER', 'COMMON', 'PASS']);
const GLOBAL_DIRECTIVES = new Set(['VERSION', 'SORT_NAME', 'USE', 'CAPABILITY', 'MAGPIE']);

/**
 * 解析效果檔 → { prefixLines, passes }
 * prefixLines:第一個 //!PASS 之前的所有行(原樣保留,含 TEXTURE/SAMPLER 區塊)
 * passes:每個 { no, desc, blockSize, numThreads, in, out, dispatch, codeLines, directiveLines }
 *   directiveLines = 該 pass 的 //! 指令行原樣(重組時以結構化欄位重建,保留未知指令)
 *   codeLines = pass 程式碼行(不含 //! 指令)
 */
export function parseEffect(text) {
    const lines = text.split(/\r?\n/);
    const prefixLines = [];
    const passes = [];
    let cur = null;
    let sawPass = false;

    for (const line of lines) {
        const dm = line.match(/^\/\/!\s*([A-Za-z_]+)\s*(.*)$/);
        if (dm) {
            const dir = dm[1].toUpperCase();
            const val = (dm[2] || '').trim();
            if (dir === 'PASS') {
                // 新 pass 開始(前一個 pass 未明確關閉時,以新 //!PASS 為界)
                cur = {
                    no: parseInt(val, 10),
                    desc: '',
                    blockSize: null,
                    numThreads: null,
                    in: null,
                    out: null,
                    dispatch: null,
                    codeLines: [],
                    directiveLines: [],
                    unknownDirectives: [],
                };
                passes.push(cur);
                sawPass = true;
                continue;
            }
            if (cur && PASS_SUB_DIRECTIVES.has(dir)) {
                cur.directiveLines.push(line);
                if (dir === 'DESC') cur.desc = val.replace(/^"|"$/g, '');
                else if (dir === 'BLOCK_SIZE') cur.blockSize = val;
                else if (dir === 'NUM_THREADS') cur.numThreads = val;
                else if (dir === 'IN') cur.in = val;
                else if (dir === 'OUT') cur.out = val;
                else if (dir === 'DISPATCH') cur.dispatch = val;
                continue;
            }
            // 其他區塊起始(如 //!TEXTURE)出現在 pass 之間:關閉當前 pass
            if (cur && BLOCK_STARTS.has(dir) && dir !== 'PASS') {
                cur = null;
                // 掉入 prefix(延續原樣輸出語意)
            }
            if (cur && !BLOCK_STARTS.has(dir) && !GLOBAL_DIRECTIVES.has(dir) && dir !== 'END') {
                // pass 區塊內的不明子指令:保留原樣(重組時輸出)
                cur.unknownDirectives.push(line);
                continue;
            }
            if (dir === 'END' && cur) {
                cur.hasEnd = true;
                cur = null;
                continue;
            }
        }
        if (cur) cur.codeLines.push(line);
        else prefixLines.push(line);
    }
    if (!sawPass) {
        throw new Error('split_passes: 檔案中沒有任何 //!PASS 區塊,不是可拆分的效果檔');
    }
    return { prefixLines, passes };
}

/** pass 的 //!OUT 中以 _storaged 結尾的目標(保持宣告順序) */
export function storageTargetsOf(pass) {
    if (pass.out == null) return [];
    return pass.out.split(',').map((s) => s.trim()).filter((s) => s.endsWith('_storaged'));
}

/** pass 程式碼中的單行 textureStore 陳述式(含目標與行索引) */
export function storeStatementsOf(pass) {
    const out = [];
    pass.codeLines.forEach((line, index) => {
        const m = line.match(/^(\s*)textureStore\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,(.*)$/);
        if (!m) return;
        // 白名單:必須是單行完整陳述式(以 ; 結尾且括號平衡)
        const body = m[3];
        if (!/;\s*$/.test(line)) {
            throw new Error(
                `split_passes: PASS ${pass.no} 的 textureStore 不是單行完整陳述式(跨行或複合),` +
                `無法安全刪除:\n    ${line.trim().slice(0, 120)}`);
        }
        // 整行(去掉結尾 ;)的括號必須平衡——呼叫自身的右括號使總計歸零
        const withoutSemi = line.replace(/;\s*$/, '');
        let depth = 0;
        for (const c of withoutSemi) {
            if (c === '(') depth++;
            else if (c === ')') depth--;
        }
        if (depth !== 0 || !/\)\s*$/.test(withoutSemi)) {
            throw new Error(
                `split_passes: PASS ${pass.no} 的 textureStore 括號不平衡,無法安全刪除:\n    ${line.trim().slice(0, 120)}`);
        }
        out.push({ line, index, target: m[2] });
    });
    return out;
}

/** 重組效果檔(prefix 原樣 + 依序輸出 passes,pass 編號重排) */
function serializeEffect(prefixLines, passes) {
    const out = [...prefixLines];
    passes.forEach((p, i) => {
        const no = i + 1;
        out.push(`//!PASS ${no}`);
        if (p.desc !== undefined && p.desc !== null) {
            if (p.desc.length) out.push(`//!DESC ${p.desc}`);
        }
        if (p.blockSize) out.push(`//!BLOCK_SIZE ${p.blockSize}`);
        if (p.numThreads) out.push(`//!NUM_THREADS ${p.numThreads}`);
        if (p.in != null) out.push(`//!IN ${p.in}`);
        if (p.out != null) out.push(`//!OUT ${p.out}`);
        if (p.dispatch != null) out.push(`//!DISPATCH ${p.dispatch}`);
        for (const d of p.unknownDirectives || []) out.push(d);
        out.push('');
        // 進入點改名為目前 pass 編號(WGFX 以 fn Pass<N> 錨定 @compute)
        out.push(...(p.codeLines || []).map((l) => l.replace(/^fn Pass\d+\(/, `fn Pass${no}(`)));
        if (p.hasEnd) out.push('//! END');
        out.push('');
    });
    return out.join('\n').replace(/\n{3,}/g, '\n\n');
}

/**
 * 自動拆分:每個 storage 寫入數超過 maxStorageTextures 的 pass,
 * 拆成 ceil(n/limit) 個連續子 pass(同一組取樣、不同 textureStore 目標集合)。
 * 白名單外形狀一律 throw(見檔頭契約)。
 */
export function splitEffect(text, { maxStorageTextures = DEFAULT_MAX_STORAGE_TEXTURES } = {}) {
    if (!Number.isInteger(maxStorageTextures) || maxStorageTextures < 1) {
        throw new Error(`split_passes: --max-storage-textures 必須是正整數,收到 ${maxStorageTextures}`);
    }
    const { prefixLines, passes } = parseEffect(text);
    const report = [];

    const finalPasses = [];
    for (const pass of passes) {
        const storTargets = storageTargetsOf(pass);
        const stores = storeStatementsOf(pass); // 可能 throw(白名單)

        if (pass.dispatch != null) {
            throw new Error(`split_passes: PASS ${pass.no} 使用 //!DISPATCH,拆分器不支援(網格語意與寫入交織)`);
        }
        if (pass.out == null) {
            finalPasses.push(pass);
            continue;
        }
        const allOut = pass.out.split(',').map((s) => s.trim()).filter(Boolean);
        const nonStoraged = allOut.filter((s) => !s.endsWith('_storaged'));
        if (storTargets.length <= maxStorageTextures) {
            // 未超限:原樣保留(OUT 可含 OUTPUT 等非 storage 項,拆分器不介入)
            finalPasses.push(pass);
            report.push({ passNo: pass.no, action: 'keep', storageTextures: storTargets.length });
            continue;
        }
        if (nonStoraged.length) {
            throw new Error(
                `split_passes: PASS ${pass.no} 超過門檻且 //!OUT 混有非 _storaged 項(${nonStoraged.join(', ')}),` +
                `無法只按 storage texture 分組`);
        }

        // 一致性:store 目標集合 == OUT 集合
        const outSet = new Set(storTargets);
        const storeSet = new Set(stores.map((s) => s.target));
        for (const t of outSet) {
            if (!storeSet.has(t)) {
                throw new Error(`split_passes: PASS ${pass.no} 的 //!OUT ${t} 在程式碼中沒有對應的 textureStore`);
            }
        }
        for (const t of storeSet) {
            if (!outSet.has(t)) {
                throw new Error(`split_passes: PASS ${pass.no} 的 textureStore(${t}, …) 不在 //!OUT 清單中`);
            }
        }
        // 讀寫競爭:pass 內不得取樣自己要寫的紋理
        for (const t of storTargets) {
            const readRe = new RegExp(`\\b${t.replace(/_storaged$/, '')}_sampled\\b`);
            if (readRe.test(pass.codeLines.join('\n'))) {
                throw new Error(
                    `split_passes: PASS ${pass.no} 讀取 ${t.replace(/_storaged$/, '')}_sampled(自己要寫的紋理),` +
                    `拆分會改變取樣時序,拒絕處理`);
            }
        }

        if (storTargets.length <= maxStorageTextures) {
            finalPasses.push(pass);
            report.push({ passNo: pass.no, action: 'keep', storageTextures: storTargets.length });
            continue;
        }

        // 連續分塊(與 tools/cunny2wgsl.mjs 的分塊規則一致:每塊 max 個,最後一塊餘數)
        const chunks = [];
        for (let i = 0; i < storTargets.length; i += maxStorageTextures) {
            chunks.push(storTargets.slice(i, i + maxStorageTextures));
        }
        const descBase = (pass.desc || '').replace(/\s*\(part \d+\/\d+\)\s*$/, '');
        chunks.forEach((chunk, ci) => {
            const keep = new Set(chunk);
            const sub = {
                ...pass,
                desc: `${descBase} (part ${ci + 1}/${chunks.length})`,
                out: chunk.join(', '),
                codeLines: pass.codeLines.filter((line) => {
                    const m = line.match(/^\s*textureStore\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,/);
                    return !m || keep.has(m[1]);
                }),
            };
            finalPasses.push(sub);
        });
        report.push({
            passNo: pass.no,
            action: 'split',
            storageTextures: storTargets.length,
            parts: chunks.length,
            desc: descBase,
        });
    }

    // 防禦性檢查:每個輸出 pass 的 storage 數不得超過門檻
    for (let i = 0; i < finalPasses.length; i++) {
        const n = storageTargetsOf(finalPasses[i]).length;
        if (n > maxStorageTextures) {
            throw new Error(
                `split_passes: 內部錯誤——輸出 pass ${i + 1} 仍有 ${n} 個 storage texture(> ${maxStorageTextures})`);
        }
    }

    return { text: serializeEffect(prefixLines, finalPasses), report, passes: finalPasses };
}

/**
 * 反向操作:把連續的 (part k/n) 子 pass 合回單一 pass(等價性驗證用)。
 * 條件:相鄰子 pass 的 DESC base 相同、(IN, BLOCK_SIZE, NUM_THREADS) 相同、
 * part 編號完整(1..n)且以 1 開頭;合併方式為「以 part 1 的行為基底,
 * 依 part 2..n 的非 store 行對齊,插入其 textureStore 行」。
 */
export function mergeSplitPasses(text) {
    const { prefixLines, passes } = parseEffect(text);

    // 分組:連續且 DESC 帶 (part k/n)
    const groups = [];
    for (const pass of passes) {
        const m = (pass.desc || '').match(/^(.*)\s*\(part (\d+)\/(\d+)\)$/);
        if (m) {
            const last = groups[groups.length - 1];
            if (last && last.base === m[1].trim() && last.total === parseInt(m[3], 10) &&
                last.passes.length === parseInt(m[2], 10) - 1 &&
                sameHeader(last.passes[0], pass)) {
                last.passes.push(pass);
                continue;
            }
            groups.push({ base: m[1].trim(), total: parseInt(m[3], 10), passes: [pass] });
        } else {
            groups.push({ base: null, total: 1, passes: [pass] });
        }
    }

    const mergedPasses = [];
    const report = [];
    for (const g of groups) {
        if (g.base === null || g.passes.length !== g.total || g.passes.length === 1) {
            mergedPasses.push(...g.passes);
            continue;
        }
        // 完整性:part 1..n
        for (let i = 0; i < g.passes.length; i++) {
            const dm = g.passes[i].desc.match(/\(part (\d+)\/(\d+)\)$/);
            if (!dm || parseInt(dm[1], 10) !== i + 1) {
                throw new Error(`split_passes: 「${g.base}」的 part 序列不完整(缺 part ${i + 1}),無法合併`);
            }
        }
        const parts = g.passes;
        const merged = {
            ...parts[0],
            desc: g.base,
            out: parts.map((p) => p.out).filter((o) => o != null).join(', '),
            codeLines: mergeCodeLines(parts, g.base),
        };
        mergedPasses.push(merged);
        report.push({ desc: g.base, parts: parts.length, storageTextures: storageTargetsOf(merged).length });
    }
    return { text: serializeEffect(prefixLines, mergedPasses), report, passes: mergedPasses };
}

function sameHeader(a, b) {
    return a.in === b.in && a.blockSize === b.blockSize && a.numThreads === b.numThreads;
}

const isStoreLine = (l) => /^\s*textureStore\s*\(/.test(l);
const isEntryLine = (l) => /^fn\s+Pass\d+\(/.test(l);

/**
 * 合併 part k/n 的程式碼行(兩步、可證偽):
 * 1. 驗證:每個 part 的「非 store、非進入點」行序列必須與 part 1 **逐行相同**
 *   （拆分的定義就是只刪 store 行,故此性質必然成立;不一致即白名單拒絕）。
 * 2. 重建:輸出 part 1 全部行;part k 的每個 store 行記錄其「前方最近非 store 行」
 *   為錨點;第二次走訪時,在每個錨點行之後按序插入屬於它的 store 行。
 *   錨點衝突(同一錨點同時有 base 的 store 與 part k 的 store)在「每個累加器的
 *   activation+store 相鄰」的拆分形狀下不可能發生;發生時 throw(無法唯一定序)。
 */
function mergeCodeLines(parts, baseDesc) {
    const base = parts[0].codeLines;
    const bodyOf = (lines) => lines.filter((l) => !isEntryLine(l));
    const baseBody = bodyOf(base);

    // 1. 一致性驗證 + 錨點蒐集
    const insertAfter = new Map(); // 錨點行(字串) → [store 行]
    let startAnchored = [];        // 無前導非 store 行的 store(罕見)→ 插在最前
    for (let pi = 1; pi < parts.length; pi++) {
        const body = bodyOf(parts[pi].codeLines);
        const nonStore = body.filter((l) => !isStoreLine(l) && l.trim() !== '');
        const baseNonStore = baseBody.filter((l) => !isStoreLine(l) && l.trim() !== '');
        for (let i = 0; i < Math.max(nonStore.length, baseNonStore.length); i++) {
            if (nonStore[i] !== baseNonStore[i]) {
                throw new Error(
                    `split_passes: 合併「${baseDesc}」失敗——part ${pi + 1} 與 part 1 的非 store 行不一致:\n` +
                    `    part1: ${JSON.stringify((baseNonStore[i] || '').trim().slice(0, 90))}\n` +
                    `    part${pi + 1}: ${JSON.stringify((nonStore[i] || '').trim().slice(0, 90))}`);
            }
        }
        // 錨點:part k 每個 store 行的前一個非 store 行(空白行不當錨點,沿用更早的)
        let anchor = null;
        for (const l of body) {
            if (isStoreLine(l)) {
                if (anchor === null) startAnchored.push(l);
                else {
                    if (!insertAfter.has(anchor)) insertAfter.set(anchor, []);
                    insertAfter.get(anchor).push(l);
                }
            } else if (l.trim() !== '') {
                anchor = l;
            }
        }
    }

    // 2. 重建:part 1 全行 + 錨點插入
    const out = [];
    for (const l of startAnchored) out.push(l);
    for (const l of base) {
        out.push(l);
        if (insertAfter.has(l)) {
            out.push(...insertAfter.get(l));
            insertAfter.delete(l);
        }
    }
    if (insertAfter.size) {
        throw new Error(
            `split_passes: 合併「${baseDesc}」失敗——有 store 行找不到錨點(拆分形狀非預期):` +
            [...insertAfter.values()].flat().map((l) => `\n    ${l.trim().slice(0, 90)}`).join(''));
    }
    return out;
}

// ----------------------------------------------------------------------------
// CLI
// ----------------------------------------------------------------------------

function main() {
    const args = process.argv.slice(2);
    let maxStorageTextures = DEFAULT_MAX_STORAGE_TEXTURES;
    const positional = [];
    let merge = false;
    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--max-storage-textures') maxStorageTextures = parseInt(args[++i], 10);
        else if (args[i] === '--merge') merge = true;
        else positional.push(args[i]);
    }
    const [inPath, outPath] = positional;
    if (!inPath || !outPath) {
        console.error('usage: node tools/pack/split_passes.mjs <in.wgsl> <out.wgsl> [--max-storage-textures N] [--merge]');
        process.exit(2);
    }
    const text = fs.readFileSync(path.resolve(inPath), 'utf8');
    try {
        const result = merge ? mergeSplitPasses(text) : splitEffect(text, { maxStorageTextures });
        fs.writeFileSync(path.resolve(outPath), result.text, 'utf8');
        const splits = (result.report || []).filter((r) => r.action === 'split');
        if (merge) {
            console.log(`OK  ${path.basename(outPath)}  (merged ${result.report.length} groups)`);
        } else if (splits.length) {
            for (const r of splits) {
                console.log(`  split PASS ${r.passNo}「${r.desc}」: ${r.storageTextures} 張 storage → ${r.parts} 個子 pass`);
            }
            console.log(`OK  ${path.basename(outPath)}  (${result.passes.length} passes)`);
        } else {
            console.log(`OK  ${path.basename(outPath)}  (無需拆分,${result.passes.length} passes)`);
        }
    } catch (e) {
        console.error(`FAIL ${inPath}: ${e.message}`);
        process.exit(1);
    }
}

if (process.argv[1]) {
    // 僅在直接執行本檔時跑 CLI(比照 tools/magpie_prescaler2wgsl.mjs 的判定方式)
    const here = new URL(import.meta.url).pathname.replace(/^\/(\w):/, '$1:');
    const invoked = process.argv[1].replace(/\\/g, '/');
    if (invoked.endsWith('pack/split_passes.mjs') || here === process.argv[1]) {
        main();
    }
}
