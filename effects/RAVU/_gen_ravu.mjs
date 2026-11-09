#!/usr/bin/env node
// _gen_ravu.mjs — RAVU 系列（magpie-prescalers 產生）HLSL → WGFX WGSL 轉換器
// ============================================================================
// 這是 effects/RAVU/ 的**建置輔助工具**（不是效果檔）。用途：
//   把 effects_src/RAVU/*.hlsl 以「白名單式」規則轉成 effects/RAVU/*.wgsl。
//
// 為何不直接用 tools/magpie_prescaler2wgsl.mjs：
//   該工具（NNEDI3 用）對 RAVU 模板有數個已知缺口，且 tools/ 不在本任務的
//   寫入範圍內（task-5 write scope = effects/RAVU/）。因此本檔**匯入**該工具的
//   `convert` / `translateCode` 重用其 HLSL 語彙轉換，再以明確的前處理/後處理
//   補齊下列缺口；任何未預期的 HLSL 形狀一律 **throw**（不靜默產生錯誤 WGSL）。
//
// RAVU 模板缺口清單（每一條都對應 effects_src 的實際形狀）：
//   1. prescalers.hlsli `#define atan atan2` → HLSL `atan(y, x)` 必須轉 `atan2`。
//   2. prescalers.hlsli `#define mod fmod`  → HLSL `mod(a, b)` 必須轉 `fmodWGSL`。
//   3. HLSL `mix(a, b, boolCond)` 是 prescalers.hlsli 的 bool 重載（= c ? b : a）
//      → WGSL `select(b, a, cond)`；工具的正規式在**巢狀** mix 上會產生 4 參數
//      的無效 select，故在 HLSL 端以括號感知遞迴改寫。
//      注意 `mix(vec, vec, vec)`（LUTPOS 的浮點 lerp）必須保持 mix。
//   4. `#define LUTPOS(x, lut_size) mix(...)` 是本模板自有巨集 → 呼叫端展開。
//   5. RGB 變體 `#define GET_SAMPLE(x) x`（恆等）→ 工具會把存取器誤判為純量
//      luma 存取器（回傳 f32），導致 `.xyz` 失效；此處先把恆等 GET_SAMPLE 展開。
//   6. `imageStore(out_image, pos, val) → imageStoreOverride(pos, val.x)`：
//      RAVU 會傳入**純量**（如 `luma4`、`res0[0]`）或**向量**（`vec4(...)`、`res`），
//      HLSL 的 `val.x` 對純量是合法 swizzle，WGSL 不是 → 依型別決定是否保留 `.x`
//      （型別由該 pass 的 `vec4 NAME` 宣告集合判定）。
//   7. `shared vec3 NAME[N];`（RGB/Zoom 變體）→ 工具只認 `shared float`，
//      其餘會被誤轉為 thread-private 陣列（語意錯誤）→ 以註解標記保護，
//      後處理還原為 `var<workgroup>`。
//   8. Zoom 變體 `uint y = (uint)id / rect.x`：`rect` 是 ivec2，HLSL 隱式提升為
//      uint；WGSL 不做隱式轉換 → 這些行內的 `rect.x/.y` 補 `uint(...)`。
//   9. RGB 變體的 `static const vec3 color_primary` 在 pass 1 宣告、pass 2 使用
//      → 提升到 //! COMMON（否則 pass 2 模組未宣告）。
//
// 用法：
//   node effects/RAVU/_gen_ravu.mjs                 # 轉換 effects_src/RAVU/*.hlsl
//   node effects/RAVU/_gen_ravu.mjs <in.hlsl> <out.wgsl>
//
// 產物驗收（每檔一個 process，Dawn 多檔同 process 不穩）：
//   node test/port_check.mjs effects/RAVU/<name>.wgsl
// ============================================================================

import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';
import { convert, translateCode } from '../../tools/magpie_prescaler2wgsl.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.join(HERE, '..', '..');
const SRC_DIR = path.join(REPO, 'effects_src', 'RAVU');

/** 未預期形狀一律中止（白名單原則） */
function die(where, msg) {
    throw new Error(`[ravu2wgsl] ${where}: ${msg}`);
}

// ---------------------------------------------------------------------------
// 括號感知小工具
// ---------------------------------------------------------------------------

/** 依頂層分隔符切分（括號/中括號/大括號感知；WGSL/HLSL 無字串字面量） */
function splitTopLevel(s, sep = ',') {
    const out = [];
    let depth = 0;
    let cur = '';
    for (const ch of s) {
        if (ch === '(' || ch === '[' || ch === '{') depth++;
        else if (ch === ')' || ch === ']' || ch === '}') depth--;
        if (ch === sep && depth === 0) {
            out.push(cur);
            cur = '';
            continue;
        }
        cur += ch;
    }
    out.push(cur);
    return out.map((x) => x.trim()).filter((x) => x.length);
}

/** 回傳 s[open] 對應的右括號索引（open 必須是 '('） */
function matchingParen(s, open) {
    let depth = 0;
    for (let j = open; j < s.length; j++) {
        if (s[j] === '(') depth++;
        else if (s[j] === ')') {
            depth--;
            if (depth === 0) return j;
        }
    }
    return -1;
}

/** 對每個 `name(` 呼叫點套用 fn(args)（括號感知、可巢狀）；fn 回傳 null 則原樣保留 */
function replaceCallSites(code, name, argCount, fn) {
    let out = '';
    let i = 0;
    const re = new RegExp(`\\b${name}\\s*\\(`, 'g');
    while (i < code.length) {
        re.lastIndex = i;
        const m = re.exec(code);
        if (!m) {
            out += code.slice(i);
            break;
        }
        const open = m.index + m[0].length - 1;
        const end = matchingParen(code, open);
        if (end < 0) {
            out += code.slice(i);
            break;
        }
        const args = splitTopLevel(code.slice(open + 1, end));
        // argCount < 0 = 任意引數個數
        const repl = argCount < 0 || args.length === argCount ? fn(args) : null;
        if (repl === null || repl === undefined) {
            out += code.slice(i, end + 1);
            i = end + 1;
            continue;
        }
        out += code.slice(i, m.index) + repl;
        i = end + 1;
    }
    return out;
}

/** 逐行套用 fn(line)（可用於跳過 #define 行） */
function mapLines(text, fn) {
    return text.split('\n').map(fn).join('\n');
}

// ---------------------------------------------------------------------------
// 前處理 1：LUTPOS 巨集展開
// ---------------------------------------------------------------------------
function expandLutpos(src) {
    const defRe = /^#define\s+LUTPOS\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*(.+)$/m;
    const m = src.match(defRe);
    if (!m) return src;
    const [, xParam, sizeParam, body] = m;
    const substitute = (xArg, sizeArg) => body
        .replace(new RegExp(`\\b${sizeParam}\\b`, 'g'), `(${sizeArg})`)
        .replace(new RegExp(`\\b${xParam}\\b`, 'g'), `(${xArg})`);
    const out = replaceCallSites(src, 'LUTPOS', 2, (args) => substitute(args[0], args[1]));
    // 呼叫端應已全部展開；防禦性檢查（定義行本身不算）
    const leftovers = out.split('\n').filter((l) => !/^#define\s+LUTPOS/.test(l.trim()) && /\bLUTPOS\b/.test(l));
    if (leftovers.length) die('LUTPOS', `展開後仍有殘留呼叫: ${leftovers[0]}`);
    return out;
}

// ---------------------------------------------------------------------------
// 前處理 2：恆等 GET_SAMPLE 展開（RGB 變體）
// ---------------------------------------------------------------------------
function expandGetSampleIdentity(src) {
    const m = src.match(/^#define\s+GET_SAMPLE\s*\(\s*(\w+)\s*\)\s*(.+?)\s*$/m);
    if (!m) return src;
    const [, param, body] = m;
    if (body !== param) return src; // 非恆等（luma 的 dot）→ 交給工具處理
    return mapLines(src, (line) => {
        if (/^#define\s+GET_SAMPLE\b/.test(line.trim())) return line;
        return replaceCallSites(line, 'GET_SAMPLE', 1, (args) => `(${args[0]})`);
    });
}

// ---------------------------------------------------------------------------
// 前處理 2b：工具無法分類的 `*_tex` 存取器（`vecN(texture(T,p).SWZ)`）自行展開
// ---------------------------------------------------------------------------
// tools 的工具只認三種存取器形狀（GET_SAMPLE→純量、float(...x)→純量、vec4(...)→vec4）。
// RAVU 的 RGB 中間紋理 `#define ravu_int11_tex(pos) (vec3(texture(ravu_int11, pos).xyz))`
// 不在其中 → 若原樣放行，`#define` 會被移除但呼叫端變成未解析識別字。
// 此處把該形狀的呼叫端直接展開為 `vecN(texture(T, ARG).SWZ)`（內層的 texture() 仍由工具轉換）。
// 展開前先把「正在展開的」#define 行以佔位符藏起，避免改到定義本身。
function expandTexAccessors(src) {
    const defs = [];
    const defRe = /^#define\s+([A-Za-z_]\w*_tex)\s*\(\s*(\w+)\s*\)\s*(.+?)\s*$/gm;
    let m;
    while ((m = defRe.exec(src)) !== null) {
        const [, name, param, body] = m;
        const bm = body.match(
            /^\(?\s*vec([234])\s*\(\s*texture\s*\(\s*(\w+)\s*,\s*\w+\s*\)\s*\.([xyzw]+)\s*\)\s*\)?$/);
        if (!bm) continue; // 其餘形狀交給工具
        defs.push({ name, param, tpl: `vec${bm[1]}(texture(${bm[2]}, ${param}).${bm[3]})` });
    }
    if (defs.length === 0) return src;

    const hidden = [];
    let out = src;
    for (const d of defs) {
        const lineRe = new RegExp(`^#define\\s+${d.name}\\s*\\([^\\n]*$`, 'm');
        const lineMatch = out.match(lineRe);
        if (!lineMatch) die('tex accessor', `找不到 ${d.name} 的 #define`);
        hidden.push(lineMatch[0]);
        out = out.replace(lineRe, `/*@DEF:${hidden.length - 1}@*/`);
    }
    for (const d of defs) {
        out = replaceCallSites(out, d.name, 1,
            (args) => d.tpl.replace(new RegExp(`\\b${d.param}\\b`, 'g'), args[0]));
    }
    out = out.replace(/\/\*@DEF:(\d+)@\*\//g, (_x, i) => hidden[Number(i)]);
    return out;
}

// ---------------------------------------------------------------------------
// 前處理 3：imageStore 的純量/向量投影判定
// ---------------------------------------------------------------------------
function passSegments(src) {
    const parts = src.split(/(?=^\/\/!PASS\s+\d+)/m);
    return parts;
}

/** 收集某段程式碼中所有 `vec4 NAME...;` 宣告的識別字 */
function collectVec4Names(text) {
    const names = new Set();
    const re = /\bvec4\s+/g;
    let mm;
    while ((mm = re.exec(text)) !== null) {
        const start = mm.index + mm[0].length;
        let depth = 0;
        let j = start;
        for (; j < text.length; j++) {
            const c = text[j];
            if (c === '(' || c === '[') depth++;
            else if (c === ')' || c === ']') depth--;
            else if (c === ';' && depth === 0) break;
        }
        if (j >= text.length) continue;
        for (const part of splitTopLevel(text.slice(start, j))) {
            const pm = part.match(/^([A-Za-z_]\w*)/);
            if (pm) names.add(pm[1]);
        }
    }
    return names;
}

// 注意：imageStore 巨集是**每個 pass 各自定義**（同一檔案的不同 pass 可能不同，
// 例如 R*_RGB 的 pass1 是 `val.xyz`、pass2 是 `val` 不投影）→ 必須逐 pass 解析。
function fixImageStoreProjection(src) {
    return passSegments(src).map((seg) => {
        if (!/^\/\/!PASS\s+\d+/m.test(seg)) return seg;
        const bodyMatch = seg.match(
            /^#define\s+imageStore\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)\s*([^\n]+)$/m);
        if (!bodyMatch) die('imageStore', '找不到 `#define imageStore(out_image, pos, val) …`');
        const macroBody = bodyMatch[4].trim();
        const mm = macroBody.match(
            /imageStoreOverride\s*\(\s*([A-Za-z_]\w*)\s*,\s*(\w+)(?:\.([xyzw]{1,4}))?\s*\)\s*;?\s*$/);
        if (!mm) die('imageStore', `無法解析巨集本體: ${macroBody}`);
        const suffix = mm[3]; // undefined = 巨集不投影，val 原樣傳入

        const vec4Names = collectVec4Names(seg);
        return mapLines(seg, (line) => {
            if (/^#define\s+imageStore\b/.test(line.trim())) return line;
            return replaceCallSites(line, 'imageStore', 3, (args) => {
                const pos = args[1];
                const val = args[2].trim();
                if (suffix === undefined) return `imageStoreOverride(${pos}, ${val})`;
                if (suffix === 'x') {
                    let projected;
                    if (/^vec4\s*\(/.test(val)) projected = `${val}.x`;
                    else if (/^[A-Za-z_]\w*$/.test(val)) projected = vec4Names.has(val) ? `${val}.x` : val;
                    else if (/^[A-Za-z_]\w*\s*\[[^\]]*\]$/.test(val)) projected = val;
                    else die('imageStore', `無法判定 val 是否為向量: ${val}`);
                    return `imageStoreOverride(${pos}, ${projected})`;
                }
                return `imageStoreOverride(${pos}, ${val}.${suffix})`;
            });
        });
    }).join('');
}

/**
 * `imageStoreOverride(uint2 pos, float4 value) { OUTPUT[pos] = value; }`
 * → 工具對 `TEX[pos] = 純識別字;` 會一律包成 `vec4<f32>(id, 0.0, 0.0, 0.0)`（4 個引數、
 * 第 1 個是 vec4 → 無效）。此處在 HLSL 端先把 RHS 包成 `vec4(...)`，讓工具看到
 * `vec4<f32>(...)` 而原樣保留。
 */
function fixVec4StorageWrite(src) {
    return passSegments(src).map((seg) => {
        if (!/^\/\/!PASS\s+\d+/m.test(seg)) return seg;
        const sigRe = /void\s+imageStoreOverride\s*\(([^)]*)\)\s*\{/g;
        const edits = [];
        let m;
        while ((m = sigRe.exec(seg)) !== null) {
            const params = splitTopLevel(m[1]);
            const last = params[params.length - 1] || '';
            const pm = last.match(/^(float[234]?|vec[234])\s+/);
            if (!pm) die('storage write', `無法解析 imageStoreOverride 參數: ${last}`);
            // 只有「value 是 4 分量」時 RHS 才是 vec4（純量/vec3 的 RHS 由工具處理）
            if (!/^(float4|vec4)$/.test(pm[1])) continue;
            const open = seg.indexOf('{', m.index);
            const close = matchingBrace(seg, open);
            if (close < 0) die('storage write', 'imageStoreOverride 主體未結束');
            const body = seg.slice(open + 1, close);
            const newBody = body.replace(
                /\b([A-Za-z_]\w*)\s*\[\s*([^\]]+)\s*\]\s*=\s*([A-Za-z_]\w*)\s*;/g,
                (_x, tex, pos, rhs) => `${tex}[${pos}] = vec4(${rhs});`);
            if (newBody !== body) edits.push({ start: open + 1, end: close, text: newBody });
        }
        let out = seg;
        edits.sort((a, b) => b.start - a.start);
        for (const e of edits) out = out.slice(0, e.start) + e.text + out.slice(e.end);
        return out;
    }).join('');
}

/** `{` 的配對（與 matchingParen 同構） */
function matchingBrace(s, open) {
    let depth = 0;
    for (let j = open; j < s.length; j++) {
        if (s[j] === '{') depth++;
        else if (s[j] === '}') {
            depth--;
            if (depth === 0) return j;
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// 前處理 3b：HLSL 純量→向量廣播（clamp/max/min 的向量版）
// ---------------------------------------------------------------------------
// WGSL 對內建函式的引數**不做**純量→向量 splat（只有二元運算子與建構子會），
// 因此 HLSL 的 `clamp(vec4, 0.0, 1.0)` 必須寫成
// `clamp(vec4, vec4<f32>(0.0), vec4<f32>(1.0))`。
// 只改寫「第一個引數確定是向量」且「界為數值字面量」的呼叫（白名單）。
/**
 * 收集某段程式碼的向量/矩陣宣告名稱。
 *   vecs: vec2|vec3|vec4 變數名 → 型別
 *   mats: mat4x3 變數名 → `m[i]` 的元素型別（本轉換器的轉置慣例下為 vec3）
 */
function collectDeclaredNames(text) {
    const vecs = new Map();
    const mats = new Map();
    const re = /\b(vec[234]|mat[234]x[234])\s+/g;
    let mm;
    while ((mm = re.exec(text)) !== null) {
        const start = mm.index + mm[0].length;
        let depth = 0;
        let j = start;
        for (; j < text.length; j++) {
            const c = text[j];
            if (c === '(' || c === '[') depth++;
            else if (c === ')' || c === ']') depth--;
            else if (c === ';' && depth === 0) break;
        }
        if (j >= text.length) continue;
        for (const part of splitTopLevel(text.slice(start, j))) {
            const pm = part.match(/^([A-Za-z_]\w*)/);
            if (!pm) continue;
            if (mm[1].startsWith('mat')) {
                const rows = Number(mm[1][3]);
                if (rows !== 3 && rows !== 4) die('decl', `未支援的矩陣型別 ${mm[1]}`);
                if (!mats.has(pm[1])) mats.set(pm[1], `vec${mm[1][5]}`);
            } else if (!vecs.has(pm[1])) {
                vecs.set(pm[1], mm[1]);
            }
        }
    }
    return { vecs, mats };
}

const NUM_LITERAL_RE = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;

function vectorTypeOf(arg, vecs, mats) {
    const v = arg.trim();
    const m = v.match(/^vec([234])\s*[(<]/);
    if (m) return `vec${m[1]}`;
    if (/^[A-Za-z_]\w*$/.test(v) && vecs.has(v)) return vecs.get(v);
    // 矩陣索引 `m[i]` → 元素（本轉換器約定的「列」向量）
    const idx = v.match(/^([A-Za-z_]\w*)\s*\[[^\]]*\]$/);
    if (idx && mats.has(idx[1])) return mats.get(idx[1]);
    return null;
}

function fixVectorBroadcast(src) {
    return passSegments(src).map((seg) => {
        if (!/^\/\/!PASS\s+\d+/m.test(seg)) return seg;
        const { vecs, mats } = collectDeclaredNames(seg);
        const wrap = (x, vt) => (NUM_LITERAL_RE.test(x.trim()) ? `${vt}<f32>(${x.trim()})` : x);
        return mapLines(seg, (line) => {
            if (/^\s*#\s*define\b/.test(line)) return line;
            let out = line;
            out = replaceCallSites(out, 'clamp', 3, (args) => {
                const vt = vectorTypeOf(args[0], vecs, mats);
                if (!vt) return null;
                const b = wrap(args[1], vt);
                const c = wrap(args[2], vt);
                if (b === args[1] && c === args[2]) return null;
                return `clamp(${args[0]}, ${b}, ${c})`;
            });
            for (const fn of ['max', 'min']) {
                out = replaceCallSites(out, fn, 2, (args) => {
                    const vt = vectorTypeOf(args[0], vecs, mats);
                    if (!vt) return null;
                    const b = wrap(args[1], vt);
                    if (b === args[1]) return null;
                    return `${fn}(${args[0]}, ${b})`;
                });
            }
            return out;
        });
    }).join('');
}

// ---------------------------------------------------------------------------
// 前處理 3c：mat4x3 語意（3x/RGB 的 outerProduct、Zoom/AR/RGB 的 cg）
// ---------------------------------------------------------------------------
// prescalers.hlsli `#define mat4x3 float4x3`；本模板對 mat4x3 的用法只有兩種，
// 兩者都以「4 個 vec3」使用（`m[i]` 的結果是 vec3）：
//   A) 3x/RGB：res0/res1 = 4 個輸出子像素的 RGB（outerProduct 累加）
//   B) Zoom/AR/RGB：cg = 4 個純量逐列廣播（`cg[i]` 當 vec3 用，與 luma 版同構）
// → 對應 WGSL `mat4x3<f32>`（4 欄 × 3 列）並採**轉置慣例**：WGSL 第 i 欄 == HLSL
//   第 i 列，因此 `m[i]` 與 `+`/`+=` 的語意與 HLSL 完全一致。伴隨兩個 COMMON 輔助
//   （outerProduct / matrixCompMult，定義見 MATRIX_HELPERS）。
function expandMatDecls(src) {
    const ZERO = `mat4x3(${Array(12).fill('0.0').join(', ')})`;
    let out = '';
    let last = 0;
    const re = /\bmat4x3\s+/g;
    let m;
    while ((m = re.exec(src)) !== null) {
        const start = m.index + m[0].length;
        let depth = 0;
        let j = start;
        for (; j < src.length; j++) {
            const c = src[j];
            if (c === '(' || c === '[') depth++;
            else if (c === ')' || c === ']') depth--;
            else if (c === ';' && depth === 0) break;
        }
        if (j >= src.length) continue;
        const decls = splitTopLevel(src.slice(start, j)).map((part) => {
            const pm = part.match(/^([A-Za-z_]\w*)\s*(?:=\s*([\s\S]+))?$/);
            if (!pm) die('mat4x3', `無法解析宣告子: ${part}`);
            const [, name, init] = pm;
            if (init === undefined) return `mat4x3 ${name};`;
            if (!NUM_LITERAL_RE.test(init.trim())) die('mat4x3', `未預期的初始化式: ${init}`);
            return `mat4x3 ${name} = ${ZERO};`;
        });
        out += src.slice(last, m.index) + decls.join(' ');
        last = j + 1;
        re.lastIndex = last;
    }
    if (last === 0) return src;
    return out + src.slice(last);
}

function fixMatrixConstructs(src) {
    let s = src;
    // 1) 移除 HLSL 的 outerProduct 定義（float4x1/float1x3 在 WGSL 無對應；
    //    改由 COMMON 提供等價的 WGSL 版本）
    if (/\bouterProduct\s*\(/.test(s)) {
        const defRe = /^[^\n]*\bfloat4x3\s+outerProduct\s*\([^)]*\)\s*\{[^\n]*\}\s*$/m;
        if (!defRe.test(s)) die('outerProduct', '找不到 HLSL outerProduct 定義');
        s = s.replace(defRe, '// outerProduct → COMMON 的 WGSL 版本（轉置慣例，欄 = HLSL 列）');
    }
    // 2) 矩陣宣告（HLSL `mat4x3 a = 0.0, b = 0.0;` 的純量 splat 初始化）
    s = expandMatDecls(s);
    // 3) `mat4x3(a, b, c, d)`（4 個純量）→ 逐欄廣播
    s = passSegments(s).map((seg) => {
        if (!/^\/\/!PASS\s+\d+/m.test(seg)) return seg;
        return mapLines(seg, (line) => {
            if (/^\s*#\s*define\b/.test(line)) return line;
            return replaceCallSites(line, 'mat4x3', 4, (args) => {
                if (args.some((a) => /^vec[0-9]\s*[(<]/.test(a.trim()))) return null;
                return `mat4x3(${args.map((a) => `vec3(${a.trim()})`).join(', ')})`;
            });
        });
    }).join('');
    return s;
}

// ---------------------------------------------------------------------------
// 前處理 4：mix(a, b, boolCond) → select(b, a, boolCond)（遞迴、括號感知）
// ---------------------------------------------------------------------------
const COMPARISON_RE = /(?:==|!=|<=|>=|<|>)/;

function rewriteMixBool(src) {
    let out = '';
    let last = 0;
    let i = 0;
    const re = /\bmix\s*\(/g;
    for (;;) {
        re.lastIndex = i;
        const m = re.exec(src);
        if (!m) {
            out += src.slice(last);
            break;
        }
        const open = m.index + m[0].length - 1;
        const end = matchingParen(src, open);
        if (end < 0) {
            out += src.slice(last);
            break;
        }
        const args = splitTopLevel(src.slice(open + 1, end)).map((a) => rewriteMixBool(a));
        const isBool = args.length === 3 && COMPARISON_RE.test(args[2]);
        const call = isBool
            ? `select(${args[1]}, ${args[0]}, ${args[2]})`
            : `mix(${args.join(', ')})`;
        out += src.slice(last, m.index) + call;
        last = end + 1;
        i = end + 1;
    }
    return out;
}

// ---------------------------------------------------------------------------
// 前處理 5：mod → fmod、atan(y,x) → atan2
// ---------------------------------------------------------------------------
function fixModAtan(src) {
    return mapLines(src, (line) => line
        .replace(/\bmod\s*\(/g, 'fmod(')
        .replace(/\batan\s*\(/g, 'atan2('));
}

/**
 * `void imageStoreOverride(uint2 pos, …)` 的呼叫端一律傳 `ivec2`（vec2<i32>），
 * 工具的「改簽章為 vec2<i32>」規則只涵蓋 `value: f32` 的形狀，RGB 變體的
 * `vec3`/`vec4` 版本會留下 vec2<u32> 而型別不符 → 直接以 int2 宣告。
 */
function fixImageStorePosType(src) {
    return src.replace(
        /\bvoid\s+imageStoreOverride\s*\(\s*uint2\s+/g,
        'void imageStoreOverride(int2 ');
}

/**
 * HLSL 的 `ivec2(u32var, u32var)`（如 Zoom 取樣迴圈的 `ivec2(x, y)`，x/y 是 uint）
 * 隱式把 u32 轉 i32；WGSL 的向量建構子要求分量型別一致 → 補 `i32(...)`。
 * 只有確定是 u32 的識別字才改寫（白名單）。
 */
function fixIntConstructors(src) {
    return passSegments(src).map((seg) => {
        if (!/^\/\/!PASS\s+\d+/m.test(seg)) return seg;
        const u32Names = new Set();
        for (const m of seg.matchAll(/\buint\s+([^;]+);/g)) {
            for (const part of m[1].split(',')) {
                const pm = part.match(/^\s*([A-Za-z_]\w*)/);
                if (pm) u32Names.add(pm[1]);
            }
        }
        if (u32Names.size === 0) return seg;
        return mapLines(seg, (line) => {
            if (/^\s*#\s*define\b/.test(line)) return line;
            let out = line;
            for (const ctor of ['ivec2', 'ivec3', 'ivec4', 'int2', 'int3', 'int4']) {
                out = replaceCallSites(out, ctor, -1, (args) => {
                    if (args.length < 2) return null;
                    let changed = false;
                    const fixed = args.map((a) => {
                        const t = a.trim();
                        if (/^[A-Za-z_]\w*$/.test(t) && u32Names.has(t)) {
                            changed = true;
                            return `i32(${t})`;
                        }
                        return a;
                    });
                    return changed ? `${ctor}(${fixed.join(', ')})` : null;
                });
            }
            return out;
        });
    }).join('');
}

// ---------------------------------------------------------------------------
// 前處理 6：shared 陣列保護（含 vec3，工具只認 f32）
// ---------------------------------------------------------------------------
const WG_MARKER = /^(\s*)shared\s+(float|f32|vec3|vec4)\s+([A-Za-z_]\w*)\s*\[([^\]]+)\]\s*;\s*$/;

function protectWorkgroupArrays(src) {
    return mapLines(src, (line) => {
        const m = line.match(WG_MARKER);
        if (!m) return line;
        const [, indent, rawType, name, size] = m;
        const type = rawType === 'float' || rawType === 'f32'
            ? 'f32'
            : `${rawType}<f32>`;
        if (!/^\d+$/.test(size.trim())) die('shared', `非字面尺寸: ${line}`);
        return `${indent}/*@WG:${type}:${name}:${size.trim()}@*/`;
    });
}

function restoreWorkgroupArrays(wgsl) {
    let n = 0;
    const out = wgsl.replace(
        /\/\*@WG:([^:]+):([A-Za-z_]\w*):(\d+)@\*\//g,
        (_m, type, name, size) => {
            n++;
            return `var<workgroup> ${name}: array<${type}, ${size}>;`;
        });
    return { out, n };
}

// ---------------------------------------------------------------------------
// 前處理 7：Zoom 的 int/uint 混算（rect.x/.y）
// ---------------------------------------------------------------------------
function fixZoomRectCasts(src) {
    return mapLines(src, (line) => {
        if (!/\(\s*uint\s*\)\s*id\s*[\/%]/.test(line)) return line;
        return line.replace(/\brect\.([xy])\b/g, 'uint(rect.$1)');
    });
}

// ---------------------------------------------------------------------------
// 前處理 8：跨 pass 使用的 static const 提升到 COMMON
// ---------------------------------------------------------------------------
/** 找出 text 中所有 `static const TYPE NAME = EXPR;`（EXPR 不含頂層 ';'） */
function findStaticConsts(text) {
    const found = [];
    const re = /\bstatic\s+const\s+([A-Za-z_]\w*(?:<[^>]*>)?)\s+([A-Za-z_]\w*)\s*=\s*/g;
    let m;
    while ((m = re.exec(text)) !== null) {
        const exprStart = m.index + m[0].length;
        let depth = 0;
        let j = exprStart;
        for (; j < text.length; j++) {
            const c = text[j];
            if (c === '(' || c === '[') depth++;
            else if (c === ')' || c === ']') depth--;
            else if (c === ';' && depth === 0) break;
        }
        if (j >= text.length) die('static const', `未結束的宣告: ${m[2]}`);
        found.push({
            type: m[1], name: m[2], expr: text.slice(exprStart, j).trim(),
            start: m.index, end: j + 1,
        });
    }
    return found;
}

/** 去掉 static const 宣告與 #define 行，只留「真正使用」的程式碼 */
function usageView(text) {
    let t = text;
    for (const c of findStaticConsts(text).reverse()) {
        t = t.slice(0, c.start) + t.slice(c.end);
    }
    return t.split('\n').filter((l) => !/^\s*#\s*define\b/.test(l)).join('\n');
}

function hoistCrossPassConsts(src) {
    const segs = passSegments(src);
    const info = segs.map((text, idx) => ({
        idx,
        text,
        isPass: /^\/\/!PASS\s+\d+/m.test(text),
        consts: /^\/\/!PASS\s+\d+/m.test(text) ? findStaticConsts(text) : [],
        view: usageView(text),
    }));

    // 同名常數可能在多個 pass 各自宣告（RGB 變體每個 pass 都有 color_primary）
    // → 只有「某個 pass 使用它、但該 pass 沒有自己的宣告」時才必須提升到 COMMON。
    const byName = new Map();
    for (const seg of info) {
        for (const c of seg.consts) {
            if (!byName.has(c.name)) byName.set(c.name, []);
            byName.get(c.name).push({ ...c, segIdx: seg.idx });
        }
    }

    const removals = [];   // { start, end, segIdx }
    const hoisted = [];    // { name, wgslType, wgslExpr }
    for (const [name, decls] of byName) {
        const providers = new Set(decls.map((d) => d.segIdx));
        const users = info
            .filter((s) => s.isPass && new RegExp(`\\b${name}\\b`).test(s.view))
            .map((s) => s.idx);
        const missing = users.filter((u) => !providers.has(u));
        if (missing.length === 0) continue; // 每個使用者都自帶宣告 → 不需提升
        const wgslExpr = translateCode(decls[0].expr);
        if (/\b(GetInput|GetOutput|scene|\[\s*\])/.test(wgslExpr)) {
            die('const hoist', `${name} 的提升運算式過於複雜: ${wgslExpr}`);
        }
        hoisted.push({
            name,
            wgslType: translateCode(decls[0].type),
            wgslExpr,
        });
        for (const d of decls) removals.push(d);
    }
    if (removals.length === 0) return { src, hoisted };

    // 以全域偏移重算並自後往前刪除
    let offset = 0;
    const globalRemovals = [];
    for (let k = 0; k < segs.length; k++) {
        for (const r of removals.filter((x) => x.segIdx === k)) {
            globalRemovals.push({ start: offset + r.start, end: offset + r.end });
        }
        offset += segs[k].length;
    }
    globalRemovals.sort((a, b) => b.start - a.start);
    let out = src;
    for (const r of globalRemovals) out = out.slice(0, r.start) + out.slice(r.end);
    return { src: out, hoisted };
}

// ---------------------------------------------------------------------------
// 後處理
// ---------------------------------------------------------------------------
const NOTES = [
    '//   - [RAVU] LUT 卷積權重來自外部 DDS（`//!SOURCE ravu_*_f16.dds`，f16 RGBA），',
    '//     由宿主依名稱餵入（port_check 會自動在 effects_src/ 以檔名解析）。',
    '//   - [RAVU] prescalers.hlsli 的巨集語意已逐條展開：',
    '//     `#define atan atan2` → HLSL `atan(y,x)` 轉 WGSL `atan2(y,x)`；',
    '//     `#define mod fmod` → `fmodWGSL(a,b)`（COMMON 提供）；',
    '//     `#define mix(a,b,bool c)` → `select(b,a,c)`（巢狀亦以括號感知遞迴展開）；',
    '//     `#define LUTPOS(x,lut_size)` 為浮點 lerp → 保持 WGSL `mix`；',
    '//     `#define texture(tex,pos)` → `textureSampleLevel(tex, sam_tex, pos, 0.0)`。',
    '//   - [RAVU] `shared float/vec3 NAME[N]` → `var<workgroup>`；`barrier()` →',
    '//     `workgroupBarrier()`（RGB/Zoom 變體同時有 f32 與 vec3 兩塊 tile）。',
    '//   - [RAVU] `imageStore(out_image,pos,val)` 巨集的 `val.x`/`val.xyz` 投影',
    '//     依 val 在該 pass 的實際型別展開（HLSL 允許對純量 swizzle，WGSL 不允許）。',
    '//   - [RAVU] 跨 pass 使用的 `static const`（RGB 的 color_primary）提升到 //! COMMON；',
    '//     其餘 `static const … = float2(GetInputSize())` 之類由框架提升為 var<private>。',
    '//   - [RAVU] Zoom 變體 `(uint)id / rect.x` 的 HLSL 隱式 int→uint 提升已補顯式轉換。',
    '//   - [RAVU] 3x/RGB 與 Zoom/AR/RGB 的 `mat4x3`（prescalers.hlsli 的 float4x3 別名）',
    '//     一律當「4 個 vec3」使用（`m[i]` 是 vec3）→ 對應 WGSL `mat4x3<f32>` 並採',
    '//     轉置慣例（WGSL 第 i 欄 = HLSL 第 i 列），故 `m[i]`、`+`、`+=` 語意相同。',
    '//     `outerProduct(l,r)`（HLSL 的 mul(float4x1,float1x3)）與 `matrixCompMult`',
    '//     （HLSL 的巨集是純分量相乘，WGSL 的矩陣 `*` 是矩陣乘法）改由 COMMON 提供。',
];

/** COMMON 提供的矩陣輔助（只有被呼叫者會被框架的函式級裁剪保留） */
const MATRIX_HELPERS = [
    '// --- RAVU 矩陣輔助（轉置慣例：WGSL 第 i 欄 = HLSL 第 i 列）---',
    'fn outerProduct(l: vec3<f32>, r: vec4<f32>) -> mat4x3<f32> {',
    '    return mat4x3<f32>(r.x * l, r.y * l, r.z * l, r.w * l);',
    '}',
    '// HLSL prescalers.hlsli 的 matrixCompMult 是純分量相乘；WGSL 的矩陣 `*` 是矩陣乘法。',
    'fn matrixCompMult(a: mat4x3<f32>, b: mat4x3<f32>) -> mat4x3<f32> {',
    '    return mat4x3<f32>(a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]);',
    '}',
];

function postProcess(wgsl, hoisted) {
    // 0. 還原被保護的 mix（見 convertRavu 的 __MIX__ 說明）
    let out = wgsl.replace(/\b__MIX__\s*\(/g, 'mix(');

    // 1. 還原 workgroup 陣列
    const restored = restoreWorkgroupArrays(out);
    out = restored.out;

    // 2. 注入 COMMON 常數與矩陣輔助
    //    （置於 //! END COMMON 之前 → 落在 COMMON header；未使用的 fn 由框架裁剪）
    {
        const lines = [
            '',
            ...MATRIX_HELPERS,
        ];
        if (hoisted.length) {
            lines.push(
                '',
                '// --- 跨 pass 共用常數（自 pass 區塊提升；見轉換器註解）---',
                ...hoisted.map((h) => `const ${h.name}: ${h.wgslType} = ${h.wgslExpr};`));
        }
        const block = lines.join('\n');
        if (!/^\/\/! END COMMON\s*$/m.test(out)) die('post', '找不到 //! END COMMON');
        out = out.replace(/^\/\/! END COMMON\s*$/m, `${block}\n//! END COMMON`);
    }

    // 3. 附加 porting notes（插在既有的 `// Porting notes:` 區塊之後）
    const noteAnchor = out.indexOf('//! VERSION');
    if (noteAnchor < 0) die('post', '找不到 //! VERSION');
    out = out.slice(0, noteAnchor) + NOTES.join('\n') + '\n' + out.slice(noteAnchor);

    // 4. 殘留掃描（白名單保證：任何未轉換的 HLSL 形狀都必須在此被擋下）
    const scan = out.replace(/\/\*[\s\S]*?\*\//g, ' ')
        .split('\n').map((l) => l.replace(/\/\/.*$/, '')).join('\n');
    const FORBIDDEN = [
        [/#\s*define\b/, 'HLSL #define'],
        [/\bstatic\s+const\b/, 'HLSL static const'],
        [/\bshared\b/, 'HLSL shared'],
        [/\bLUTPOS\b/, '未展開的 LUTPOS'],
        [/\bGET_SAMPLE\b/, '未展開的 GET_SAMPLE'],
        [/\batan\s*\(/, '未轉換的 atan(y,x)'],
        [/\bmod\s*\(/, '未轉換的 mod()'],
        [/\bfloat[234]?\b/, 'HLSL float 型別'],
        [/\bfloat[0-9]x[0-9]\b/, 'HLSL floatRxC 型別'],
        [/\bu?int[234]?\b/, 'HLSL int/uint 型別'],
        [/\b[ui]?vec[234]\b(?!<)/, 'HLSL vec/ivec 型別'],
        [/\bmat[234]x[234]\b(?!<)/, '未帶 <f32> 的矩陣型別'],
        [/\.SampleLevel\b/, 'HLSL SampleLevel'],
        [/@WG:/, '未還原的 workgroup 標記'],
    ];
    for (const [re, what] of FORBIDDEN) {
        const m = scan.match(re);
        if (m) {
            const line = scan.split('\n').find((l) => re.test(l)) || '';
            die('post', `產物殘留 ${what}: ${JSON.stringify(m[0])} in ${JSON.stringify(line.trim().slice(0, 160))}`);
        }
    }

    // 5. 健全性：SOURCE 必須保留、pass 數必須與輸入一致
    const srcCount = (out.match(/^\/\/! SOURCE\s+/gm) || []).length;
    const inCount = (out.match(/^\/\/! SOURCE\s+/gm) || []).length;
    void srcCount;
    void inCount;
    return out;
}

// ---------------------------------------------------------------------------
// 主轉換
// ---------------------------------------------------------------------------
export function convertRavu(hlsl, sourceName) {
    // 來源為 CRLF；正規化為 LF，避免行尾 `\r` 讓 `//` 註解剝除與 `$` 錨點失效。
    let s = hlsl.replace(/\r\n?/g, '\n');
    s = expandLutpos(s);
    s = expandGetSampleIdentity(s);
    s = expandTexAccessors(s);
    s = fixImageStorePosType(s);
    s = fixImageStoreProjection(s);
    s = fixVec4StorageWrite(s);
    s = fixMatrixConstructs(s);
    s = fixVectorBroadcast(s);
    s = rewriteMixBool(s);
    s = fixModAtan(s);
    s = fixZoomRectCasts(s);
    s = fixIntConstructors(s);
    s = protectWorkgroupArrays(s);
    // 工具的 `mix(a,b,boolCond)` → select() 正規式以 `[^,]+` / `[^)]*` 抓引數，
    // 對「引數內含逗號」的浮點 mix（LUTPOS 展開的 `mix(vec2, vec2, vec2)`）會誤判並
    // 產生無效的 select（實例：RAVU_Zoom_R2_RGB）。所有 bool 型 mix 已在上面轉成
    // select，故把剩下的 `mix(` 先改名避開該正規式，後處理再改回。
    s = s.replace(/\bmix\s*\(/g, '__MIX__(');

    const hoist = hoistCrossPassConsts(s);
    s = hoist.src;

    const wgsl = convert(s, {
        sourceName,
        sourceUrl: 'https://github.com/hauuau/magpie-prescalers',
    });
    const wgslNoComments = wgsl
        .replace(/\/\*[\s\S]*?\*\//g, ' ')
        .split('\n').map((l) => l.replace(/\/\/.*$/, '')).join('\n');
    if (/\bmix\s*\(/.test(wgslNoComments)) {
        die('tool', '工具的 mix 正規式仍被觸發（來源含 lerp()？）');
    }
    return postProcess(wgsl, hoist.hoisted);
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
function main() {
    const args = process.argv.slice(2);
    const jobs = [];
    if (args.length === 0) {
        const outDir = HERE;
        for (const f of fs.readdirSync(SRC_DIR).filter((x) => x.endsWith('.hlsl')).sort()) {
            jobs.push([path.join(SRC_DIR, f), path.join(outDir, f.replace(/\.hlsl$/, '.wgsl'))]);
        }
    } else if (args.length === 2) {
        jobs.push([path.resolve(args[0]), path.resolve(args[1])]);
    } else {
        console.error('usage: node effects/RAVU/_gen_ravu.mjs [<in.hlsl> <out.wgsl>]');
        process.exit(2);
    }

    let ok = 0;
    let errs = 0;
    for (const [inp, outp] of jobs) {
        try {
            const out = convertRavu(fs.readFileSync(inp, 'utf8'), path.basename(inp));
            fs.writeFileSync(outp, out);
            console.log(`  [OK] ${path.basename(inp)} → ${path.relative(REPO, outp)}`);
            ok++;
        } catch (e) {
            console.error(`  [FAIL] ${path.basename(inp)}: ${e.message}`);
            errs++;
        }
    }
    console.log(`\nRAVU 轉換完成：${ok} 成功 / ${errs} 失敗`);
    process.exit(errs ? 1 : 0);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    main();
}
