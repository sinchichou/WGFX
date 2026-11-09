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
 * 值的擷取以**括號平衡掃描**（任意巢狀深度）——舊版的固定兩層括號 regex 會在
 * `vec2<f32> UV = mul(rgb2uv, textureSampleLevel(INPUT, sam, HOOKED_map(pos), 0).rgb);`
 * 這種深巢狀初始化式上失配，導致 HLSL typed declaration 原樣殘留。
 * 陣列宣告：`vec4<f32> samples[8];` → `var samples: array<vec4<f32>, 8>;`
 */
function convertLocalDecls(text) {
    const NAME = '[A-Za-z_][A-Za-z0-9_]*';
    const typeRe = new RegExp(`\\b(${WGSL_TYPE_RE})\\s+`, 'g');

    // 陣列宣告：`vec4<f32> samples[8];` → `var samples: array<vec4<f32>, 8>;`
    let s = text.replace(
        new RegExp(`\\b(${WGSL_TYPE_RE})\\s+(${NAME})\\s*\\[([^\\]]+)\\]\\s*;`, 'g'),
        (_m, type, name, n) => `var ${name}: array<${type}, ${n.trim()}>;`);

    // 純量/向量/矩陣宣告：掃描每個 `TYPE` 出現位置，向後括號平衡掃描到頂層 `;`
    let out = '';
    let last = 0;
    for (const m of s.matchAll(typeRe)) {
        const i = m.index + m[0].length;
        // 讀宣告子序列至頂層 `;`（頂層 `,` 分隔多宣告子）
        let depth = 0;
        let j = i;
        for (; j < s.length; j++) {
            const ch = s[j];
            if (ch === '(' || ch === '[') depth++;
            else if (ch === ')' || ch === ']') depth--;
            else if (ch === ';' && depth === 0) break;
        }
        if (j >= s.length || depth !== 0) continue;
        const body = s.slice(i, j);
        // 宣告子必須是 `NAME` 或 `NAME = VALUE`（逗號分隔,值可含任意巢狀括號）
        const parts = splitTopLevel(body);
        let ok = true;
        const decls = [];
        for (const p of parts) {
            const pm = p.match(new RegExp(`^(${NAME})\\s*(?:=\\s*([\\s\\S]+))?$`));
            if (!pm) { ok = false; break; }
            decls.push(pm[2] !== undefined
                ? `var ${pm[1]}: ${m[1]} = ${pm[2].trim()};`
                : `var ${pm[1]}: ${m[1]};`);
        }
        if (!ok) continue;
        // 誤傷防護：這不是宣告（如 for 迴圈頭 `f32 i = 0;` 由 for 規則先處理）——
        // TYPE 前面緊接 `(` 的情況（函式參數等）不在 matchAll 範圍;此處僅在
        // 宣告子形狀完全吻合時改寫。
        out += s.slice(last, m.index) + decls.join(' ');
        last = j + 1;
    }
    if (last === 0) return s;
    return out + s.slice(last);
}

/**
 * 型別 / 內建 / 語句替換（依序套用，順序有意義）。
 * @param {string} src - HLSL 片段
 * @param {Set<string>} [textureNames] - 可作為 storage texture 寫入的紋理名稱
 */
export function translateCode(src, textureNames = null) {
    let s = src;

    // --- group_base 以 u32 語意計算 ---
    // HLSL: `ivec2 group_base = ivec2(gl_WorkGroupID) * ivec2(gl_WorkGroupSize);`
    // 之後與 `uint x/y` 混算（group_base.x + x）——HLSL 隱式 int→uint 提升使整條
    // 座標鏈為 uint;WGSL 不做隱式提升,故直接以 u32 計算（模板的 gl_* 別名本來
    // 就是 vec2<u32>,數值語意與 HLSL 的 uint 運算完全一致）。
    s = s.replace(
        /\bivec2\s+group_base\s*=\s*ivec2\s*\(\s*gl_WorkGroupID\s*\)\s*\*\s*ivec2\s*\(\s*gl_WorkGroupSize\s*\)\s*;/g,
        'uint2 group_base = gl_WorkGroupID * gl_WorkGroupSize;');

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

    // --- ivec2 常數保持 i32（WGSL 不做 int↔uint 隱式提升）---
    // 舊版把 `ivec2(1, 2)` 一律轉 `vec2<u32>(1u, 2u)`,會把全 i32 的座標鏈
    //（`ivec2(gl_GlobalInvocationID) * ivec2(1, 2) + ivec2(0, 1)`）打混成
    // i32*u32 的無效運算。u32 語境（`threadId.xy * ivec2(1, 2)`）由下方
    // 專屬規則顯式升為 u32,其餘保持 i32（`vec2<i32>(1, 2)` 合法）。
    s = s.replace(
        /\b(threadId\.xy|wgid\.xy|gl_GlobalInvocationID|gl_WorkGroupID|gl_WorkGroupSize|blockStart)\s*\*\s*vec2<i32>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)/g,
        '$1 * vec2<u32>($2u, $3u)');
    s = s.replace(
        /\bvec2<i32>\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*\*\s*(threadId\.xy|wgid\.xy|gl_GlobalInvocationID|gl_WorkGroupID|gl_WorkGroupSize|blockStart)\b/g,
        'vec2<u32>($1u, $2u) * $3');

    // --- 取樣 ---
    s = s.replace(/\b([A-Za-z_][A-Za-z0-9_]*)\.SampleLevel\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([^,]+),\s*([0-9.]+)\s*\)/g,
        (_m, tex, smp, uv, lvl) => `textureSampleLevel(${tex}, ${smp}, ${uv.trim()}, ${lvl})`);
    // prescalers.hlsli 的 `#define texture(tex, pos) tex.SampleLevel(sam_##tex, pos, 0.0)`
    // — pos 以括號平衡掃描（`texture(lut, vec2(0.05, coord_y))` 的巢狀括號會讓
    // `[^)]+` 在第一個 `)` 截斷,產出不平衡的 WGSL）
    s = replaceCallSites(s, 'texture', 2, (args) => {
        const tex = args[0].trim();
        if (!/^[A-Za-z_]\w*$/.test(tex)) return null; // 非識別字：原樣保留
        return `textureSampleLevel(${tex}, sam_${tex}, ${args[1].trim()}, 0.0)`;
    });
    s = s.replace(/\b([A-Za-z_][A-Za-z0-9_]*)\.Gather(Red|Green|Blue|Alpha)\s*\(/g,
        (_m, tex, ch) => `textureGather(${({ Red: 0, Green: 1, Blue: 2, Alpha: 3 })[ch]}, ${tex}, `);

    // --- HLSL mul(a, b) → ((a) * (b))（括號平衡,任意巢狀）---
    s = expandHlslMul(s);

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
    'MIN', 'MAX', 'DEFAULT', 'STEP', 'TYPE', 'MODE', 'SOURCE',
];

// ---------------------------------------------------------------------------
// 2b. 格式對照表（Magpie/DXGI 名稱 → WebGPU 格式；storage 能力映射）
// ---------------------------------------------------------------------------

/**
 * Magpie/DXGI 格式名稱 → WebGPU 格式名稱。
 * 與 src/runtime/ShaderParser.pegjs 的 toWebGPUFormat 相同語意（此處供轉換器
 * 在產物端就寫出正確格式，並對「會被當 storage 寫入的紋理」預先映射到
 * storage-capable 格式——Dawn 拒絕 r16float 等的 storage 存取，
 * 詳見 tools/_storage_probe.mjs 的實測矩陣與 WGSLCodeGenerator.storageCapableFormat）。
 */
const FORMAT_TABLE = {
    'R8_UNORM': 'r8unorm', 'R8_SNORM': 'r8snorm', 'R8_UINT': 'r8uint', 'R8_SINT': 'r8sint',
    'R8G8_UNORM': 'rg8unorm', 'R8G8_SNORM': 'rg8snorm',
    'R8G8B8A8_UNORM': 'rgba8unorm', 'R8G8B8A8_SNORM': 'rgba8snorm',
    'R8G8B8A8_UINT': 'rgba8uint', 'R8G8B8A8_SINT': 'rgba8sint',
    'R8G8B8A8_SRGB': 'rgba8unorm-srgb',
    'R16_FLOAT': 'r16float', 'R16G16_FLOAT': 'rg16float', 'R16G16B16A16_FLOAT': 'rgba16float',
    'R16_UNORM': 'r16unorm', 'R16G16_UNORM': 'rg16unorm', 'R16G16B16A16_UNORM': 'rgba16unorm',
    'R16_UINT': 'r16uint', 'R16G16_UINT': 'rg16uint', 'R16G16B16A16_UINT': 'rgba16uint',
    'R16_SNORM': 'r16snorm', 'R16G16_SNORM': 'rg16snorm', 'R16G16B16A16_SNORM': 'rgba16snorm',
    'R32_FLOAT': 'r32float', 'R32G32_FLOAT': 'rg32float', 'R32G32B32A32_FLOAT': 'rgba32float',
    'R32_UINT': 'r32uint', 'R32G32_UINT': 'rg32uint', 'R32G32B32A32_UINT': 'rgba32uint',
    'R32_SINT': 'r32sint', 'R32G32_SINT': 'rg32sint', 'R32G32B32A32_SINT': 'rgba32sint',
    'R11G11B10_FLOAT': 'rg11b10ufloat',
    'RGB10A2_UNORM': 'rgb10a2unorm',
    'BC1_UNORM': 'bc1-rgba-unorm', 'BC3_UNORM': 'bc3-rgba-unorm', 'BC7_UNORM': 'bc7-rgba-unorm',
};

/** WebGPU core（本 Dawn 建置）支援 write-only storage 的格式（_storage_probe.mjs 實測） */
const STORAGE_CAPABLE = new Set([
    'rgba8unorm', 'rgba8snorm', 'rgba8uint', 'rgba8sint',
    'rgba16float', 'rgba16uint', 'rgba16sint',
    'r32float', 'rg32float', 'rgba32float',
    'r32uint', 'rg32uint', 'rgba32uint',
    'r32sint', 'rg32sint', 'rgba32sint',
]);

/** 不支援 storage 的格式 → 最接近的可 storage 替代（與 WGSLCodeGenerator 一致） */
const STORAGE_FORMAT_SUBSTITUTE = {
    'r8unorm': 'rgba8unorm', 'r8snorm': 'rgba8snorm', 'r8uint': 'rgba8uint', 'r8sint': 'rgba8sint',
    'rg8unorm': 'rgba8unorm', 'rg8snorm': 'rgba8snorm', 'rg8uint': 'rgba8uint', 'rg8sint': 'rgba8sint',
    'rgba8unorm-srgb': 'rgba8unorm',
    'r16float': 'rgba16float', 'rg16float': 'rgba16float',
    'r16uint': 'rgba16uint', 'rg16uint': 'rgba16uint',
    'r16sint': 'rgba16sint', 'rg16sint': 'rgba16sint',
    'r16unorm': 'rgba16float', 'rg16unorm': 'rgba16float', 'rgba16unorm': 'rgba16float',
    'rg11b10ufloat': 'rgba16float', 'rgb9e5ufloat': 'rgba16float', 'rgb10a2unorm': 'rgba16float',
};

/** 已是 WebGPU 風格的名稱（小寫含 - 或純小寫）原樣映射 */
function toWebGPUFormat(value) {
    const v = (value || '').trim();
    if (FORMAT_TABLE[v.toUpperCase()]) return FORMAT_TABLE[v.toUpperCase()];
    return v; // 假設已是 WebGPU 名稱
}

/** storage-capable 格式（與框架 storageCapableFormat 相同映射） */
function storageCapableFormat(webgpuFormat) {
    const f = (webgpuFormat || '').toLowerCase();
    if (STORAGE_CAPABLE.has(f)) return f;
    return STORAGE_FORMAT_SUBSTITUTE[f] || f;
}

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

// ---------------------------------------------------------------------------
// 3b. per-pass 巨集展開（magpie-prescalers 模板的定義語意）
// ---------------------------------------------------------------------------

/** 依括號深度切分字串（沿用頂層定義） */

/**
 * 解析 per-pass `#define`（連續行感知），回傳 { macros, stripped }。
 * macros: name → { params: string[] | null, body: string }
 */
function parseMacros(code) {
    const macros = new Map();
    const kept = [];
    const lines = code.split('\n');
    for (let i = 0; i < lines.length; i++) {
        const t = lines[i].trim();
        const m = t.match(/^#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\(([^)]*)\))?\s*(.*)$/);
        if (m) {
            let body = m[4] || '';
            // 行尾 `\` 接續
            while (/\\\s*$/.test(lines[i]) && i + 1 < lines.length) {
                i++;
                body = body.replace(/\\\s*$/, '') + ' ' + lines[i].trim();
            }
            body = body.replace(/\\\s*$/, '').trim();
            const params = m[3] !== undefined
                ? m[3].split(',').map((s) => s.trim()).filter((s) => s.length)
                : null;
            macros.set(m[1], { params, body });
            continue; // 定義行從程式碼移除
        }
        kept.push(lines[i]);
    }
    return { macros, stripped: kept.join('\n') };
}

/**
 * 展開 magpie-prescalers 模板 per-pass 定義的**已知語意**：
 *
 * 1. 存取器定義 `#define X_tex(p) BODY`：
 *    - `GET_SAMPLE(vec4(texture(X, p)))`      → 純量（GET_SAMPLE 語意,此模板為 luma）
 *    - `(float(texture(X, p).x))`             → 純量（x 通道）
 *    - `(vec4(texture(X, p)))`                → vec4
 *    產生 WGSL fn（textureSampleLevel(X, sam_X, p, 0.0)）並把呼叫端的 `.x` 後綴
 *    剝除（HLSL 純量 swizzle 在 WGSL 不合法）。
 * 2. 別名 `#define HOOKED_tex(p) X_tex(p)` / `#define HOOKED_pt X_pt` →
 *    識別字改寫到實際名稱。
 * 3. `#define imageStore(out_image, pos, val) BODY` → 呼叫端以位置參數代入 BODY。
 * 4. `HOOKED_map(id)`（prescalers.hlsli 的 frag_map）→ 呼叫端包 vec2<f32>(...)，
 *    本體由 COMMON prelude 的 fn HOOKED_map 提供（scene.outputPt * (id + 0.5)）。
 *
 * 僅在「存取器確實被使用」時輸出 fn，避免不必要的 sampler 綁定需求。
 *
 * @param {string} code - 前處理後的 pass HLSL（仍含 #define 行）
 * @param {{samplers: Set<string>, textures: Set<string>}} env - 資源名稱集合
 * @returns {string} 展開後的程式碼（已移除 #define 行、已插入 WGSL 存取器）
 */
function expandPassDefines(code, env) {
    const { macros, stripped } = parseMacros(code);

    // --- 1. 分類存取器定義 ---
    // name → { tex, param, kind: 'luma'|'chanX'|'vec4' }
    const accessors = new Map();
    // name → 別名目標（fn 別名）
    const fnAliases = new Map();
    // name → 純識別字別名
    const idAliases = new Map();
    let imageStoreMacro = null;
    let getSampleBody = null;

    for (const [name, def] of macros) {
        const body = def.body || '';
        if (name === 'GET_SAMPLE' && def.params && def.params.length === 1) {
            getSampleBody = { arg: def.params[0], body };
            continue;
        }
        if (name === 'imageStore' && def.params && def.params.length === 3) {
            imageStoreMacro = { params: def.params, body };
            continue;
        }
        if (/_tex$/.test(name) && def.params && def.params.length === 1) {
            const param = def.params[0];
            let m = body.match(
                new RegExp(`^GET_SAMPLE\\s*\\(\\s*vec4\\s*\\(\\s*texture\\s*\\(\\s*(\\w+)\\s*,\\s*${param}\\s*\\)\\s*\\)\\s*\\)$`));
            if (m) { accessors.set(name, { tex: m[1], param, kind: 'luma' }); continue; }
            m = body.match(
                new RegExp(`^\\(?\\s*float\\s*\\(\\s*texture\\s*\\(\\s*(\\w+)\\s*,\\s*${param}\\s*\\)\\s*\\.x\\s*\\)\\s*\\)?$`));
            if (m) { accessors.set(name, { tex: m[1], param, kind: 'chanX' }); continue; }
            m = body.match(
                new RegExp(`^\\(?\\s*vec4\\s*\\(\\s*texture\\s*\\(\\s*(\\w+)\\s*,\\s*${param}\\s*\\)\\s*\\)\\s*\\)?$`));
            if (m) { accessors.set(name, { tex: m[1], param, kind: 'vec4' }); continue; }
            // fn 別名形狀：`#define HOOKED_tex(p) X_tex(p)`（指向另一個存取器）
            m = body.match(new RegExp(`^(\\w+)\\s*\\(\\s*${param}\\s*\\)$`));
            if (m && (accessors.has(m[1]) || /_tex$/.test(m[1]))) {
                fnAliases.set(name, m[1]);
                continue;
            }
            // 不認得的形狀：留給一般 #define 移除（呼叫端會得到明確的 WGSL 錯誤）
            continue;
        }
        // fn 別名：`#define HOOKED_tex(p) X_tex(p)`
        if (def.params && def.params.length === 1 && /_tex$/.test(name)) {
            const m = body.match(new RegExp(`^(\\w+)\\s*\\(\\s*${def.params[0]}\\s*\\)$`));
            if (m && (accessors.has(m[1]) || /_tex$/.test(m[1]))) {
                fnAliases.set(name, m[1]);
                continue;
            }
        }
        // 純識別字別名（HOOKED_pt → temp_pt 等）
        if (!def.params && /^[A-Za-z_]\w*$/.test(body) && /_(pt|size)$/.test(name)) {
            idAliases.set(name, body);
            continue;
        }
    }

    // --- 2. fn 別名先解析到最終存取器（可鏈式） ---
    let changed = true;
    while (changed) {
        changed = false;
        for (const [from, to] of fnAliases) {
            const resolved = fnAliases.get(to);
            if (resolved) { fnAliases.set(from, resolved); changed = true; }
        }
    }

    // --- 3. 改寫識別字別名（HOOKED_pt → temp_pt 等） ---
    let s = stripped;
    for (const [from, to] of idAliases) {
        s = s.replace(new RegExp(`\\b${from}\\b`, 'g'), to);
    }

    // --- 4. 改寫 fn 別名呼叫端（HOOKED_tex( → X_tex(） ---
    for (const [from, to] of fnAliases) {
        s = s.replace(new RegExp(`\\b${from}\\s*\\(`, 'g'), `${to}(`);
    }

    // --- 5. imageStore 巨集呼叫端代入（imageStore(a, b, c) → BODY 代入） ---
    if (imageStoreMacro) {
        const { params, body } = imageStoreMacro;
        s = replaceCallSites(s, 'imageStore', 3, (args) => {
            let out = body;
            params.forEach((p, i) => {
                out = out.replace(new RegExp(`\\b${p}\\b`, 'g'), args[i]);
            });
            return out;
        });
    }

    // --- 6. HOOKED_map(id) 呼叫端包 vec2<f32>(...); 本體在 COMMON prelude ---
    s = replaceCallSites(s, 'HOOKED_map', 1, (args) => `HOOKED_map(vec2<f32>(${args[0]}))`);

    // --- 7. 純量存取器的 `.x` 後綴剝除（HLSL 純量 swizzle） ---
    for (const [name, acc] of accessors) {
        if (acc.kind === 'vec4') continue;
        s = stripScalarXSuffix(s, name);
    }

    // --- 8. 產生「實際被使用」的存取器 fn（模組層級） ---
    const accessorFns = [];
    for (const [name, acc] of accessors) {
        const usedRe = new RegExp(`\\b${name}\\s*\\(`);
        if (!usedRe.test(s)) continue;
        const sampler = `sam_${acc.tex}`;
        if (env.samplers && !env.samplers.has(sampler)) {
            console.warn(`[prescaler2wgsl] 存取器 ${name} 需要 ${sampler} 但效果未宣告 sampler,略過該存取器`);
            continue;
        }
        if (env.textures && !env.textures.has(acc.tex)) {
            console.warn(`[prescaler2wgsl] 存取器 ${name} 引用未宣告紋理 ${acc.tex},略過`);
            continue;
        }
        const sample = `textureSampleLevel(${acc.tex}, ${sampler}, ${acc.param}, 0.0)`;
        if (acc.kind === 'luma') {
            if (!getSampleBody) {
                throw new Error(`存取器 ${name} 使用 GET_SAMPLE 但該 pass 未定義 GET_SAMPLE`);
            }
            const expr = getSampleBody.body.replace(
                new RegExp(`\\b${getSampleBody.arg}\\b`, 'g'), sample);
            accessorFns.push(`fn ${name}(${acc.param}: vec2<f32>) -> f32 {\n    return ${expr};\n}`);
        } else if (acc.kind === 'chanX') {
            accessorFns.push(`fn ${name}(${acc.param}: vec2<f32>) -> f32 {\n    return ${sample}.x;\n}`);
        } else {
            accessorFns.push(`fn ${name}(${acc.param}: vec2<f32>) -> vec4<f32> {\n    return ${sample};\n}`);
        }
    }

    if (accessorFns.length) {
        s = accessorFns.join('\n') + '\n\n' + s;
    }
    return s;
}

/**
 * 改寫具名函式的呼叫端（括號感知）：name(a0, a1, …) → fn(args)
 * fn 回傳 null 時原樣保留該呼叫。僅處理頂層程式碼（WGSL 程式碼沒有字串）。
 */
function replaceCallSites(code, name, argCount, fn) {
    let out = '';
    let i = 0;
    const re = new RegExp(`\\b${name}\\s*\\(`, 'g');
    while (i < code.length) {
        re.lastIndex = i;
        const m = re.exec(code);
        if (!m) { out += code.slice(i); break; }
        const open = m.index + m[0].length - 1;
        let depth = 0;
        let end = -1;
        for (let j = open; j < code.length; j++) {
            if (code[j] === '(') depth++;
            else if (code[j] === ')') { depth--; if (depth === 0) { end = j; break; } }
        }
        if (end < 0) { out += code.slice(i); break; }
        const args = splitTopLevel(code.slice(open + 1, end));
        const repl = (args.length === argCount) ? fn(args) : null;
        if (repl === null || repl === undefined) {
            out += code.slice(i, end + 1); // 參數數不符或 fn 放棄：原樣保留
            i = end + 1;
            continue;
        }
        out += code.slice(i, m.index) + repl;
        i = end + 1;
    }
    return out;
}

/**
 * HLSL mul(a, b) → ((a) * (b))：括號平衡掃描,任意巢狀深度
 * （`mul(yuv2rgb, float3(value.x, UV))`、mul 內再含 mul 皆可;迴圈到不再出現為止）。
 */
function expandHlslMul(code) {
    const re = /\bmul\s*\(/;
    let s = code;
    for (let iter = 0; iter < 256; iter++) {
        const m = re.exec(s);
        if (!m) break;
        const open = m.index + m[0].length - 1;
        let depth = 0;
        let end = -1;
        for (let j = open; j < s.length; j++) {
            if (s[j] === '(') depth++;
            else if (s[j] === ')') { depth--; if (depth === 0) { end = j; break; } }
        }
        if (end < 0) break;
        const args = splitTopLevel(s.slice(open + 1, end));
        if (args.length !== 2) break; // 不認得的形狀：原樣保留（會得到明確編譯錯誤）
        s = s.slice(0, m.index) + `((${args[0]}) * (${args[1]}))` + s.slice(end + 1);
    }
    return s;
}

/**
 * 剝除純量函式呼叫後的 `.x` 後綴：`X_tex(...).x` → `X_tex(...)`
 * （HLSL 純量 swizzle;WGSL 的 f32 沒有分量存取）
 */
function stripScalarXSuffix(code, name) {
    let out = '';
    let i = 0;
    const re = new RegExp(`\\b${name}\\s*\\(`, 'g');
    while (i < code.length) {
        re.lastIndex = i;
        const m = re.exec(code);
        if (!m) { out += code.slice(i); break; }
        const open = m.index + m[0].length - 1;
        let depth = 0;
        let end = -1;
        for (let j = open; j < code.length; j++) {
            if (code[j] === '(') depth++;
            else if (code[j] === ')') { depth--; if (depth === 0) { end = j; break; } }
        }
        if (end < 0) { out += code.slice(i); break; }
        // 檢查後綴 .x
        const tail = code.slice(end + 1);
        const sm = tail.match(/^\s*\.\s*x\b/);
        if (sm) {
            out += code.slice(i, end + 1);
            i = end + 1 + sm[0].length;
        } else {
            out += code.slice(i, end + 1);
            i = end + 1;
        }
    }
    return out;
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
// Magpie prescalers.hlsli 的 HOOKED_map(id) = frag_map(id) = OUTPUT_pt * (vec2(id) + 0.5)
//（呼叫端已包 vec2<f32>(...);pos 型別 i32/u32 皆可正確轉換）。
fn HOOKED_map(id: vec2<f32>) -> vec2<f32> {
    return scene.outputPt * (id + vec2<f32>(0.5));
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

    // 4b2. 紋理/取樣器名稱（供 storage texture 寫入規則與存取器產生辨識）
    const textureNames = new Set();
    const samplerNames = new Set();
    for (const r of blocks.resources) {
        if (r.kind === 'TEXTURE') {
            for (const line of r.code) {
                const m = line.match(/^\s*(?:Texture2D|var)\s+([A-Za-z_][A-Za-z0-9_]*)/);
                if (m) textureNames.add(m[1]);
            }
        } else if (r.kind === 'SAMPLER') {
            for (const line of r.code) {
                const m = line.match(/^\s*(?:SamplerState|var)\s+([A-Za-z_][A-Za-z0-9_]*)/);
                if (m) samplerNames.add(m[1]);
            }
        }
    }

    // 4b3. 被任一 pass 寫入（//! OUT）的紋理 → 其 FORMAT 需為 storage-capable
    //      （Dawn 拒絕 r16float 等的 storage 存取,見 tools/_storage_probe.mjs）
    const writtenTextures = new Set();
    for (const p of prepared) {
        for (const d of p.directives) {
            const m = d.match(/^\/\/!\s*OUT\s+(.*)$/i);
            if (m) {
                m[1].split(',').forEach((s) => {
                    const n = s.trim();
                    if (n) writtenTextures.add(n.replace(/_(sampled|storaged)$/, ''));
                });
            }
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
    out.push('//   - HLSL `mul(a, b)` → WGSL `(a) * (b)`；HLSL typed local declaration →');
    out.push('//     `var NAME: TYPE = EXPR;`（模板 per-pass 定義 X_tex/HOOKED_* 已展開為');
    out.push('//     具名 WGSL 存取函式,GET_SAMPLE 語意保留）。');
    out.push('//   - HLSL `float2x3` / `float3x3` 為列優先，WGSL 為欄優先；此處以「欄 = HLSL 的欄」');
    out.push('//     重建，`M * v` 的結果與 HLSL `mul(M, v)` 完全相同。');
    out.push('//   - `shared` 陣列 → `var<workgroup>`；`barrier()` → `workgroupBarrier()`。');
    out.push('//   - 跨 pass 共用的權重函式移入 //! COMMON（每個 pass 模組各自一份）。');
    out.push('//   - 被 pass 寫入的紋理 FORMAT 映射為 storage-capable 格式（Dawn 限制）。');
    out.push('');
    out.push(...blocks.meta.filter((l) => l.trim().length > 0));

    // 資源區塊（FORMAT 正規化為 WebGPU 名稱;storage 寫入者映射為 storage-capable）
    for (const r of blocks.resources) {
        const texNameM = r.code.map((l) => l.match(/^\s*(?:Texture2D|var)\s+([A-Za-z_][A-Za-z0-9_]*)/))
            .find((m) => m);
        const texName = texNameM ? texNameM[1].replace(/_(sampled|storaged)$/, '') : null;
        const fmtIdx = r.directives.findIndex((d) => /^\/\/!\s*FORMAT\s+/i.test(d));
        if (fmtIdx >= 0) {
            const declared = r.directives[fmtIdx].replace(/^\/\/!\s*FORMAT\s+/i, '').trim();
            let webgpu = toWebGPUFormat(declared);
            if (texName && (writtenTextures.has(texName))) {
                webgpu = storageCapableFormat(webgpu);
            }
            r.directives[fmtIdx] = `//! FORMAT ${webgpu}`;
        }
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
        // per-pass 模板定義展開（X_tex 存取器 / HOOKED_* 別名 / imageStore /
        // HOOKED_map 包裝;必須在 translateCode 之前,因為要吃 HLSL 形狀的定義）
        code = expandPassDefines(code, { samplers: samplerNames, textures: textureNames });
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
