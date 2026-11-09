#!/usr/bin/env node
// magpie_prescaler2wgsl.mjs — magpie-prescalers 產生的 HLSL → WGFX WGSL 轉換器
//
// 適用於 https://github.com/hauuau/magpie-prescalers 產生的檔案（NNEDI3 系列、
// RAVU 系列）。這些檔案是**機器生成**的，所有變體共用同一份模板，差異只在：
//   * 權重函式的常數（nnedi3 / cnn 的 W()、WS() 巨集展開）
//   * pass 的 BLOCK_SIZE / NUM_THREADS / 視窗尺寸
//   * 紋理尺寸宣告
//
// 因此以「模板式轉換」處理：先做 HLSL 前處理（展開 W/WS/T 巨集、評估
// #if CURRENT_PASS == LAST_PASS），再做 HLSL→WGSL 的語彙替換，最後重組為
// WGFX 的效果檔結構（//! TEXTURE / //! SAMPLER / //! COMMON / //! PASS）。
//
// 用法：
//   node tools/magpie_prescaler2wgsl.mjs <in.hlsl> <out.wgsl>
//   node tools/magpie_prescaler2wgsl.mjs --batch effects_src/NNEDI3 effects/NNEDI3
//
// 轉換後必須以 node test/port_check.mjs <out.wgsl> 驗證。

import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.join(HERE, '..');

// ---------------------------------------------------------------------------
// 1. HLSL 前處理：展開模板巨集、評估 #if
// ---------------------------------------------------------------------------

/** 把整數常數轉成 WGSL 的 i32 字面量（bitcast 需要具體型別） */
function i32lit(n) {
    return `${n}i`;
}

/**
 * 展開 magpie-prescalers 模板的巨集。
 *   W(i, w0, w1, w2, w3) → dot(samples[i], vec4<f32>(bitcast<f32>(...), ...))
 *   WS(w0, w1)          → 四行陳述式（展開為區塊）
 *   T(x)                → bitcast<f32>(x)
 */
function expandTemplateMacros(src) {
    let s = src;

    // W(i, a, b, c, d)：dot(samples[i], vec4(...))
    s = s.replace(/\bW\(\s*(\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)/g,
        (_m, i, a, b, c, d) =>
            `dot(samples[${i}], vec4<f32>(bitcast<f32>(${i32lit(a)}), bitcast<f32>(${i32lit(b)}), ` +
            `bitcast<f32>(${i32lit(c)}), bitcast<f32>(${i32lit(d)})))`);

    // WS(w0, w1)：四行陳述式（HLSL 巨集是陳述式序列）
    s = s.replace(/\bWS\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*;/g,
        (_m, a, b) =>
            `sum1 = exp(sum1 * mstd2 + bitcast<f32>(${i32lit(a)}));\n` +
            `sum2 = sum2 * mstd2 + bitcast<f32>(${i32lit(b)});\n` +
            `wsum += sum1;\n` +
            `vsum += sum1 * (sum2 / (1.0 + abs(sum2)));`);

    // T(x)：intBitsToFloat
    s = s.replace(/\bT\(\s*(-?\d+)\s*\)/g, (_m, a) => `bitcast<f32>(${i32lit(a)})`);

    return s;
}

/**
 * 評估模板中的條件編譯區塊。
 * 目前只需要處理 `#if CURRENT_PASS == LAST_PASS` / `#else` / `#endif`。
 */
function evaluateConditionals(src, currentPass, lastPass) {
    const lines = src.split('\n');
    const out = [];
    // stack: { taken: bool, active: bool }（active = 目前是否輸出）
    const stack = [];
    const active = () => stack.every((f) => f.active);

    for (const line of lines) {
        const t = line.trim();
        if (t.startsWith('#if ')) {
            const cond = t.slice(4).trim();
            const parentActive = active();
            let val = false;
            if (cond === 'CURRENT_PASS == LAST_PASS') {
                val = currentPass === lastPass;
            } else if (cond === 'CURRENT_PASS != LAST_PASS') {
                val = currentPass !== lastPass;
            } else {
                throw new Error(`不支援的條件編譯: ${cond}`);
            }
            stack.push({ active: parentActive && val, taken: val });
            continue;
        }
        if (t === '#else') {
            const top = stack[stack.length - 1];
            if (!top) throw new Error('#else 沒有對應的 #if');
            const parentActive = stack.slice(0, -1).every((f) => f.active);
            top.active = parentActive && !top.taken;
            top.taken = true;
            continue;
        }
        if (t === '#endif') {
            if (!stack.length) throw new Error('#endif 沒有對應的 #if');
            stack.pop();
            continue;
        }
        if (active()) out.push(line);
    }
    return out.join('\n');
}

// ---------------------------------------------------------------------------
// 2. HLSL → WGSL 語彙替換
// ---------------------------------------------------------------------------

/** 依括號深度切分字串 */
function splitTopLevel(s, sep = ',') {
    const out = [];
    let depth = 0, cur = '';
    for (const ch of s) {
        if (ch === '(' || ch === '[') depth++;
        else if (ch === ')' || ch === ']') depth--;
        if (ch === sep && depth === 0) { out.push(cur); cur = ''; continue; }
        cur += ch;
    }
    if (cur.trim().length) out.push(cur);
    return out.map((x) => x.trim()).filter((x) => x.length);
}

/** WGSL 型別正規化（vecN<T> 保持原樣，裸 vecN → vecN<f32>） */
function normalizeWgslType(t) {
    const s = t.trim();
    if (/^vec[234]$/.test(s)) return `${s}<f32>`;
    if (/^mat[234]x[234]$/.test(s)) return `${s}<f32>`;
    return s;
}

/** 可作為區域變數宣告的型別樣式 */
const WGSL_TYPE_RE = '(?:f32|u32|i32|bool|vec[234]<[fiu]32>|mat[234]x[234]<f32>)';

/**
 * 函式參數列表：HLSL `uint2 pos, float value` → WGSL `pos: vec2<u32>, value: f32`
 * 陣列參數 `vec4<f32> samples[8]` → `samples: array<vec4<f32>, 8>`
 */
function convertParams(params) {
    return splitTopLevel(params).map((p) => {
        const m = p.match(/^(.*?[^\s])\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[([^\]]+)\]$/);
        if (m) return `${m[2]}: array<${normalizeWgslType(m[1])}, ${m[3].trim()}>`;
        const m2 = p.match(/^(.*?[^\s])\s+([A-Za-z_][A-Za-z0-9_]*)$/);
        if (m2) return `${m2[2]}: ${normalizeWgslType(m2[1])}`;
        return p;
    }).join(', ');
}

/**
 * 剝除前處理指示（#define / #pragma / #include）。
 * 函式巨集可能跨行（行尾以 `\` 接續），必須整組移除，否則續行會變成孤兒程式碼。
 */
function stripPreprocessor(text) {
    const lines = text.split('\n');
    const out = [];
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const t = line.trim();
        if (/^#\s*(pragma|include)\b/.test(t)) continue;
        if (/^#\s*define\b/.test(t)) {
            // 函式巨集：持續吃到沒有行尾反斜線的那一行為止
            while (i + 1 < lines.length && /\\\s*$/.test(lines[i])) i++;
            continue;
        }
        out.push(line);
    }
    return out.join('\n');
}

/**
 * 區域變數宣告：HLSL `f32 a = 0.0, b = 0.0, c;` → WGSL 逐個 `var a: f32 = 0.0;` 等。
 * HLSL 的區域變數預設可變，故一律用 `var`。
 * 值的擷取需括號感知（值內可以有逗號，例如 select(a, b, c)）。
 */
function convertLocalDecls(text) {
    // 括號感知的「值」樣式：允許一层巢狀括號與其中的逗號
    const VAL = '((?:[^,;()]|\\((?:[^()]|\\([^()]*\\))*\\))+)';
    const NAME = '[A-Za-z_][A-Za-z0-9_]*';
    const declRe = new RegExp(
        `\\b(${WGSL_TYPE_RE})\\s+` +
        `(${NAME}(?:\\s*=\\s*${VAL})?(?:\\s*,\\s*${NAME}(?:\\s*=\\s*${VAL})?)*)` +
        `\\s*;`, 'g');

    // 陣列宣告：`vec4<f32> samples[8];` → `var samples: array<vec4<f32>, 8>;`
    let s = text.replace(
        new RegExp(`\\b(${WGSL_TYPE_RE})\\s+(${NAME})\\s*\\[([^\\]]+)\\]\\s*;`, 'g'),
        (_m, type, name, n) => `var ${name}: array<${type}, ${n.trim()}>;`);

    s = s.replace(declRe, (_m, type, decls) => splitTopLevel(decls).map((d) => {
        const mm = d.match(new RegExp(`^(${NAME})\\s*(=\\s*([\\s\\S]+))?$`));
        return mm && mm[2] ? `var ${mm[1]}: ${type} ${mm[2].trim()};` : `var ${d}: ${type};`;
    }).join(' '));
    return s;
}

/**
 * 型別 / 內建 / 語句替換（依序套用，順序有意義）。
 * @param {string} src - HLSL 片段
 * @param {Set<string>} [textureNames] - 可作為 storage texture 寫入的紋理名稱
 */
export function translateCode(src, textureNames = null) {
    let s = src;

    // --- 移除 HLSL 專用指示（巨集可能跨行，須整組移除）---
    s = stripPreprocessor(s);

    // --- 移除 HLSL 屬性 ---
    s = s.replace(/\[\s*(unroll|loop|branch|flatten|allow_uav_condition)\s*\]\s*/g, '');

    // --- 矩陣型別（必須先於純向量型別）---
    s = s.replace(/\bfloat2x3\b/g, 'mat3x2<f32>');
    s = s.replace(/\bfloat3x3\b/g, 'mat3x3<f32>');
    s = s.replace(/\bfloat4x3\b/g, 'mat3x4<f32>');
    s = s.replace(/\bfloat3x4\b/g, 'mat4x3<f32>');
    s = s.replace(/\bfloat4x4\b/g, 'mat4x4<f32>');

    // --- prescalers.hlsli 的別名（vec2/vec3/vec4 → float2/3/4）---
    s = s.replace(/\bvec([234])\b(?!\s*<)/g, 'vec$1<f32>');
    s = s.replace(/\bmat([234]x[234])\b(?!\s*<)/g, 'mat$1<f32>');

    // --- 向量 / 純量型別 ---
    s = s.replace(/\bfloat4\b/g, 'vec4<f32>');
    s = s.replace(/\bfloat3\b/g, 'vec3<f32>');
    s = s.replace(/\bfloat2\b/g, 'vec2<f32>');
    s = s.replace(/\bivec2\b/g, 'vec2<i32>');
    s = s.replace(/\bivec3\b/g, 'vec3<i32>');
    s = s.replace(/\buint3\b/g, 'vec3<u32>');
    s = s.replace(/\buint2\b/g, 'vec2<u32>');
    s = s.replace(/\bint2\b/g, 'vec2<i32>');
    s = s.replace(/\bint3\b/g, 'vec3<i32>');
    s = s.replace(/\bfloat\b/g, 'f32');
    s = s.replace(/\buint\b/g, 'u32');
    s = s.replace(/\bint\b/g, 'i32');

    // --- Magpie 的場景資訊存取子 → WGFX 的 scene 欄位 ---
    s = s.replace(/\bGetInputSize\s*\(\s*\)/g, 'scene.inputSize');
    s = s.replace(/\bGetInputPt\s*\(\s*\)/g, 'scene.inputPt');
    s = s.replace(/\bGetOutputSize\s*\(\s*\)/g, 'scene.outputSize');
    s = s.replace(/\bGetOutputPt\s*\(\s*\)/g, 'scene.outputPt');

    // --- 內建函式 ---
    s = s.replace(/\binversesqrt\s*\(/g, 'inverseSqrt(');
    s = s.replace(/\brsqrt\s*\(/g, 'inverseSqrt(');
    s = s.replace(/\bintBitsToFloat\s*\(/g, 'bitcast<f32>(');
    s = s.replace(/\bfloatBitsToInt\s*\(/g, 'bitcast<i32>(');
    s = s.replace(/\bfrac\s*\(/g, 'fract(');
    s = s.replace(/\bfmod\s*\(/g, 'fmodWGSL(');
    s = s.replace(/\blerp\s*\(/g, 'mix(');

    // --- mix(a, b, bool) → select(b, a, bool) ---
    s = s.replace(/\bmix\s*\(([^,]+),\s*([^,]+),\s*([^)]*(?:>=|<=|==|!=|>|<)[^)]*)\)/g,
        (_m, a, b, c) => `select(${b.trim()}, ${a.trim()}, ${c.trim()})`);

    // --- C 風格轉型：(uint)id → u32(id) ---
    s = s.replace(/\(\s*u32\s*\)\s*([A-Za-z_][A-Za-z0-9_]*)/g, 'u32($1)');
    s = s.replace(/\(\s*i32\s*\)\s*([A-Za-z_][A-Za-z0-9_]*)/g, 'i32($1)');

    // --- ivec2(常量) 用於與 u32 相乘 → vec2<u32>（HLSL 會隱式提升）---
    s = s.replace(/\bvec2<i32>\(\s*(\d+)\s*,\s*(\d+)\s*\)/g, 'vec2<u32>($1u, $2u)');

    // --- 取樣 ---
    s = s.replace(/\b([A-Za-z_][A-Za-z0-9_]*)\.SampleLevel\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([^,]+),\s*([0-9.]+)\s*\)/g,
        (_m, tex, smp, uv, lvl) => `textureSampleLevel(${tex}, ${smp}, ${uv.trim()}, ${lvl})`);
    s = s.replace(/\btexture\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([^)]+)\)/g,
        (_m, tex, pos) => `textureSampleLevel(${tex}, sam_${tex}, ${pos.trim()}, 0.0)`);
    s = s.replace(/\b([A-Za-z_][A-Za-z0-9_]*)\.Gather(Red|Green|Blue|Alpha)\s*\(/g,
        (_m, tex, ch) => `textureGather(${({ Red: 0, Green: 1, Blue: 2, Alpha: 3 })[ch]}, ${tex}, `);

    // --- static const X name = expr; → const name: X = expr; ---
    s = s.replace(
        new RegExp(`\\bstatic\\s+const\\s+(${WGSL_TYPE_RE})\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*([^;]+);`, 'g'),
        (_m, type, name, val) => `const ${name}: ${normalizeWgslType(type)} = ${val.trim()};`);

    // --- 群組共用記憶體與同步 ---
    s = s.replace(/\bshared\s+f32\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[([^\]]+)\]\s*;/g,
        (_m, name, n) => `var<workgroup> ${name}: array<f32, ${n.trim()}>;`);
    s = s.replace(/\bbarrier\s*\(\s*\)\s*;/g, 'workgroupBarrier();');

    // --- 儲存紋理：tex[pos] = value; → textureStore(tex, vec2<i32>(pos), value); ---
    // 只作用於確實是紋理的名稱（避免把區域陣列 ret[0] 當成紋理）。
    if (textureNames && textureNames.size) {
        const alt = [...textureNames].map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
        s = s.replace(new RegExp(`\\b(${alt})\\s*\\[\\s*([A-Za-z0-9_]+)\\s*\\]\\s*=\\s*([^;\\n]+);`, 'g'),
            (_m, tex, pos, val) => {
                const v = val.trim();
                // WGSL 的 textureStore 一律接受 vec4<f32>（單通道紋理亦然）
                const vv = /^vec4</.test(v) ? v : `vec4<f32>(${v}, 0.0, 0.0, 0.0)`;
                return `textureStore(${tex}, vec2<i32>(${pos}), ${vv});`;
            });
    }

    // --- for 迴圈標頭（先處理，避免被區域變數宣告規則誤傷）---
    s = s.replace(/\bfor\s*\(\s*(i32|u32|f32|bool)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=/g,
        'for (var $2: $1 =');

    // --- 函式定義：f32 name(params) { → fn name(params) -> f32 { ---
    s = s.replace(/\bf32\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*\{/g,
        (_m, name, params) => `fn ${name}(${convertParams(params)}) -> f32 {`);
    // --- void name(params) { → fn name(params) {（進入點 Pass<N> 保留，交給 rewriteEntry）---
    s = s.replace(/\bvoid\s+(?!Pass\d+\s*\()([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*\{/g,
        (_m, name, params) => `fn ${name}(${convertParams(params)}) {`);

    // --- 區域變數宣告 ---
    s = convertLocalDecls(s);

    // --- imageStore(out_image, pos, val) → imageStoreOverride(pos, val.x) ---
    s = s.replace(/\bimageStore\s*\(\s*out_image\s*,\s*([^,]+),\s*([^)]+)\)/g,
        (_m, pos, val) => `imageStoreOverride(${pos.trim()}, ${val.trim()})`);

    // --- imageStoreOverride 的 pos 改為 vec2<i32>（呼叫端以 ivec2 傳入）---
    s = s.replace(/\bfn\s+imageStoreOverride\s*\(\s*pos\s*:\s*vec2<u32>\s*,\s*value\s*:\s*f32\s*\)/g,
        'fn imageStoreOverride(pos: vec2<i32>, value: f32)');

    return s;
}

// ---------------------------------------------------------------------------
// 3. 效果檔組裝
// ---------------------------------------------------------------------------

/** Magpie/WGFX 的區塊起始指令 */
const BLOCK_STARTS = ['PARAMETER', 'TEXTURE', 'SAMPLER', 'BUFFER', 'COMMON', 'PASS'];
/** 全域指令（不屬於任何區塊） */
const GLOBAL_DIRECTIVES = ['VERSION', 'SORT_NAME', 'USE', 'CAPABILITY', 'MAGPIE'];
/** 區塊內的子指令（必須保留在所屬區塊） */
const SUB_DIRECTIVES = [
    'DESC', 'IN', 'OUT', 'BLOCK_SIZE', 'NUM_THREADS', 'DISPATCH',
    'WIDTH', 'HEIGHT', 'FORMAT', 'FILTER', 'ADDRESS', 'LABEL',
    'MIN', 'MAX', 'DEFAULT', 'STEP', 'TYPE', 'MODE',
];

/** 把 HLSL 效果檔拆成區塊 */
function splitEffect(src) {
    const lines = src.split('\n');
    const blocks = { meta: [], resources: [], common: null, passes: [] };
    let cur = null;       // { kind, index?, directives: [], code: [] }

    const flush = () => {
        if (!cur) return;
        if (cur.kind === 'PASS') blocks.passes.push(cur);
        else if (cur.kind === 'COMMON') blocks.common = cur;
        else blocks.resources.push(cur);
        cur = null;
    };

    for (const line of lines) {
        const t = line.trim();
        if (t.startsWith('//!')) {
            const m = t.match(/^\/\/!\s*([A-Za-z_]+)\s*(.*)$/);
            if (m) {
                const dir = m[1].toUpperCase();
                const val = m[2].trim();

                // 子指令 → 附在目前的區塊上
                if (SUB_DIRECTIVES.includes(dir) && cur) {
                    cur.directives.push(`//! ${dir}${val ? ' ' + val : ''}`);
                    continue;
                }
                // 全域指令
                if (GLOBAL_DIRECTIVES.includes(dir)) {
                    flush();
                    if (dir === 'MAGPIE') continue;   // Magpie 專用標記，WGFX 不支援
                    blocks.meta.push(`//! ${dir}${val ? ' ' + val : ''}`);
                    continue;
                }
                // 區塊起始指令
                if (BLOCK_STARTS.includes(dir)) {
                    flush();
                    if (dir === 'COMMON') {
                        cur = { kind: 'COMMON', directives: [], code: [] };
                    } else if (dir === 'PASS') {
                        cur = { kind: 'PASS', index: Number(val), directives: [], code: [] };
                    } else {
                        cur = { kind: dir, directives: [`//! ${dir}${val ? ' ' + val : ''}`], code: [] };
                    }
                    continue;
                }
                // 未知指令：當成註解保留（不影響轉換）
                flush();
                blocks.meta.push(`// (略過未知指令) ${t}`);
                continue;
            }
        }
        if (!cur) { blocks.meta.push(line); continue; }
        cur.code.push(line);
    }
    flush();
    return blocks;
}

/** COMMON 區塊中的 prescalers 巨集 → WGFL 輔助函式 */
const WGSL_COMMON_PRELUDE = `
// --- magpie-prescalers 模板輔助（由 tools/magpie_prescaler2wgsl.mjs 產生）---
const rgb2y: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);
// HLSL float2x3（列優先）→ WGSL mat3x2（欄優先）；mul(M, v) 的結果完全一致。
const rgb2uv: mat3x2<f32> = mat3x2<f32>(
    vec2<f32>(-0.2126 / 1.8556, 0.7874 / 1.5748),
    vec2<f32>(-0.7152 / 1.8556, -0.7152 / 1.5748),
    vec2<f32>(0.9278 / 1.8556, -0.0722 / 1.5748)
);
// HLSL float3x3（列優先）→ WGSL mat3x3（欄優先）；mul(M, v) 的結果完全一致。
const yuv2rgb: mat3x3<f32> = mat3x3<f32>(
    vec3<f32>(1.0, 1.0, 1.0),
    vec3<f32>(0.0, -0.187324, 1.8556),
    vec3<f32>(1.5748, -0.468124, 0.0)
);
// HLSL fmod（對負數亦成立）；WGSL 沒有內建 fmod。
fn fmodWGSL(x: f32, y: f32) -> f32 {
    return x - y * trunc(x / y);
}
`;

/** 產生 pass 的 WGFX 進入點前導（blockStart / threadId / gl_* 對應） */
function passPrelude(blockSize, numThreads) {
    const [bx, by] = blockSize;
    const [tx, ty] = numThreads;
    return `    let blockStart: vec2<u32> = wgid.xy * vec2<u32>(${bx}u, ${by}u);
    let threadId: vec3<u32> = lid;
    let gl_WorkGroupID: vec2<u32> = wgid.xy;
    let gl_WorkGroupSize: vec2<u32> = vec2<u32>(${tx}u, ${ty}u);
    let gl_LocalInvocationID: vec3<u32> = lid;
    let gl_LocalInvocationIndex: u32 = lid.y * ${tx}u + lid.x;
    let gl_GlobalInvocationID: vec2<u32> = wgid.xy * gl_WorkGroupSize + lid.xy;
`;
}

/** 把 pass 的 HLSL 函式簽章換成 WGFX compute 進入點 */
function rewriteEntry(code, index, prelude) {
    // void PassN(uint2 blockStart, uint3 threadId) {  →  fn PassN(@builtin(...) ...)
    // （型別可能已被 translateCode 轉換，故兩種寫法都接受）
    const re = new RegExp(
        `void\\s+Pass${index}\\s*\\(\\s*(?:uint2|vec2<u32>)\\s+blockStart\\s*,\\s*` +
        `(?:uint3|vec3<u32>)\\s+threadId\\s*\\)\\s*\\{`);
    if (!re.test(code)) {
        if (process.env.PRESCALER_DEBUG) {
            console.error(`[debug] index=${index} regex=${re}`);
            console.error('[debug] code head:\n' + code.slice(0, 600));
        }
        throw new Error(`找不到 Pass${index} 的 HLSL 進入點簽章`);
    }
    const replacement =
        `fn Pass${index}(@builtin(workgroup_id) wgid: vec3<u32>, ` +
        `@builtin(local_invocation_id) lid: vec3<u32>) {\n${prelude}`;
    return code.replace(re, replacement);
}

/** 把 `float name(params) {` / `void name(params) {` 的 HLSL 定義改為 WGFL */
function rewriteFunctionDefs(code) {
    // 函式回傳型別：float → f32（已由 translateCode 處理），簽章需補上 WGSL 型別
    return code.replace(/\bf32\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[([^\]]+)\]\s*\)/g,
        (_m, name, arg, n) => `fn ${name}(${arg}: array<vec4<f32>, ${n.trim()}>) -> f32`);
}

/** 從 pass 區塊抽出頂層函式（以大括號配對找函式邊界） */
function extractTopLevelFunctions(code) {
    const funcs = [];
    const re = /^(?:f32|void)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*\{/gm;
    let m;
    while ((m = re.exec(code)) !== null) {
        const start = m.index;
        // 往前吃掉行首空白
        let s = start;
        while (s > 0 && (code[s - 1] === ' ' || code[s - 1] === '\t')) s--;
        // 找對應的右大括號
        let depth = 0, i = m.index + m[0].length - 1;
        for (; i < code.length; i++) {
            if (code[i] === '{') depth++;
            else if (code[i] === '}') { depth--; if (depth === 0) break; }
        }
        const end = i + 1;
        funcs.push({ name: m[1], text: code.slice(s, end), start: s, end });
    }
    return funcs;
}

// ---------------------------------------------------------------------------
// 4. 主流程
// ---------------------------------------------------------------------------

export function convert(src, { sourceName = 'input.hlsl', sourceUrl = '' } = {}) {
    const blocks = splitEffect(src);
    const lastPass = blocks.passes.length;

    // 4a. 先在每個 pass 內做前處理（展開巨集 + 條件編譯）
    const prepared = blocks.passes.map((p) => {
        const raw = p.code.join('\n');
        const expanded = expandTemplateMacros(raw);
        const evaluated = evaluateConditionals(expanded, p.index, lastPass);
        return { ...p, code: evaluated };
    });

    // 4b. 找出跨 pass 共用��頂層函式 → 移到 COMMON
    //     （若後續 pass 自己有同名定義，則保留為各 pass 自己的版本）
    const shared = new Map();
    for (let i = 0; i < prepared.length; i++) {
        const funcs = extractTopLevelFunctions(prepared[i].code);
        for (const f of funcs) {
            if (/^Pass\d+$/.test(f.name)) continue;
            const later = prepared.slice(i + 1);
            const usedLater = later.some((p) => p.code.includes(f.name + '('));
            const redefinedLater = later.some((p) =>
                extractTopLevelFunctions(p.code).some((g) => g.name === f.name));
            if (usedLater && !redefinedLater && !shared.has(f.name)) shared.set(f.name, f.text);
        }
    }

    // 4b2. 紋理名稱（供 storage texture 寫入規則辨識）
    const textureNames = new Set();
    for (const r of blocks.resources) {
        if (r.kind !== 'TEXTURE') continue;
        for (const line of r.code) {
            const m = line.match(/^\s*(?:Texture2D|var)\s+([A-Za-z_][A-Za-z0-9_]*)/);
            if (m) textureNames.add(m[1]);
        }
    }

    // 4c. 組出 WGFX 文字
    const out = [];
    out.push(`// ${path.basename(sourceName, '.hlsl')} — Magpie v4 → WGFX WGSL port`);
    out.push(`// 由 tools/magpie_prescaler2wgsl.mjs 自動轉換（magpie-prescalers 模板）。`);
    if (sourceUrl) out.push(`// 來源: ${sourceUrl}`);
    out.push('//');
    out.push('// Porting notes:');
    out.push('//   - Magpie 的 `//!STYLE CS` pass（blockStart / threadId）改為 WGFX compute：');
    out.push('//     `@builtin(workgroup_id)` / `@builtin(local_invocation_id)`，並依模板的');
    out.push('//     prescalers.hlsli 定義重建 blockStart 與 gl_* 別名（數值完全一致）。');
    out.push('//   - 權重以 `bitcast<f32>(整數位元)` 表示，逐字保留 HLSL 的 int 位元常數。');
    out.push('//   - HLSL `mix(a, b, bool)` → WGSL `select(b, a, bool)`（HLSL 為 c ? b : a）。');
    out.push('//   - HLSL `float2x3` / `float3x3` 為列優先，WGSL 為欄優先；此處以「欄 = HLSL 的欄」');
    out.push('//     重建，`M * v` 的結果與 HLSL `mul(M, v)` 完全相同。');
    out.push('//   - `shared` 陣列 → `var<workgroup>`；`barrier()` → `workgroupBarrier()`。');
    out.push('//   - 跨 pass 共用的權重函式移入 //! COMMON（每個 pass 模組各自一份）。');
    out.push('');
    out.push(...blocks.meta.filter((l) => l.trim().length > 0));

    // 資源區塊（原樣保留：WGFX 解析器支援 HLSL 風格的宣告）
    for (const r of blocks.resources) {
        out.push('');
        out.push(...r.directives);
        out.push(...r.code.filter((l) => l.trim().length > 0));
    }

    // COMMON
    out.push('');
    out.push('//! COMMON');
    out.push(WGSL_COMMON_PRELUDE.trimEnd());
    for (const [name, text] of shared) {
        out.push('');
        out.push(translateCode(rewriteFunctionDefs(text), textureNames));
    }
    out.push('//! END COMMON');

    // PASS 區塊
    for (const p of prepared) {
        const blockSize = getDirective(p.directives, 'BLOCK_SIZE') || [8, 8];
        const numThreads = getDirective(p.directives, 'NUM_THREADS') || [64, 1, 1];
        const bs = Array.isArray(blockSize) ? blockSize : [blockSize, blockSize];
        const nt = Array.isArray(numThreads) ? numThreads : [numThreads, 1, 1];

        let code = p.code;
        // 移除已被移入 COMMON 的函式
        for (const name of shared.keys()) {
            const funcs = extractTopLevelFunctions(code);
            const f = funcs.find((x) => x.name === name);
            if (f) code = code.slice(0, f.start) + code.slice(f.end);
        }
        code = translateCode(code, textureNames);
        code = rewriteFunctionDefs(code);
        code = rewriteEntry(code, p.index, passPrelude(bs, nt));

        out.push('');
        out.push(`//! PASS ${p.index}`);
        for (const d of p.directives) out.push(d);
        // NUM_THREADS 在 WGFX 需要 3 個分量
        const idx = out.length - 1;
        for (let i = out.length - 1; i >= 0; i--) {
            if (/^\/\/!\s*NUM_THREADS/i.test(out[i])) {
                out[i] = `//! NUM_THREADS ${nt[0]},${nt[1]},${nt[2] ?? 1}`;
                break;
            }
        }
        out.push(code.trimEnd());
        out.push('//! END');
        void idx;
    }

    return out.join('\n') + '\n';
}

/** 取得 pass 的子指令值 */
function getDirective(directives, name) {
    for (const d of directives) {
        const m = d.match(new RegExp(`^//!\\s*${name}\\s+(.*)$`, 'i'));
        if (m) {
            return m[1].split(',').map((s) => Number(s.trim())).filter((n) => !Number.isNaN(n));
        }
    }
    return null;
}

// ---------------------------------------------------------------------------
// 5. CLI
// ---------------------------------------------------------------------------

function main() {
    const args = process.argv.slice(2);
    if (args.length === 0) {
        console.error('usage: node tools/magpie_prescaler2wgsl.mjs <in.hlsl> <out.wgsl>');
        console.error('       node tools/magpie_prescaler2wgsl.mjs --batch <inDir> <outDir>');
        process.exit(2);
    }
    if (args[0] === '--batch') {
        const inDir = path.resolve(args[1]);
        const outDir = path.resolve(args[2]);
        fs.mkdirSync(outDir, { recursive: true });
        let n = 0, errs = 0;
        for (const f of fs.readdirSync(inDir).filter((x) => x.endsWith('.hlsl'))) {
            try {
                const src = fs.readFileSync(path.join(inDir, f), 'utf8');
                const out = convert(src, {
                    sourceName: f,
                    sourceUrl: `https://github.com/hauuau/magpie-prescalers`,
                });
                fs.writeFileSync(path.join(outDir, f.replace(/\.hlsl$/, '.wgsl')), out);
                console.log(`  [OK] ${f} → ${path.relative(REPO, path.join(outDir, f.replace(/\.hlsl$/, '.wgsl')))}`);
                n++;
            } catch (e) {
                console.error(`  [FAIL] ${f}: ${e.message}`);
                errs++;
            }
        }
        console.log(`\n轉換完成：${n} 成功 / ${errs} 失敗`);
        process.exit(errs ? 1 : 0);
    }
    const src = fs.readFileSync(path.resolve(args[0]), 'utf8');
    const out = convert(src, { sourceName: path.basename(args[0]) });
    fs.writeFileSync(path.resolve(args[1]), out);
    console.log(`已寫出 ${args[1]}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    main();
}
