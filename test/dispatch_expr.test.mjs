// dispatch_expr.test.mjs — `//! DISPATCH` 求值器的等價性與正確性測試
//
// 目標:證明手寫的 AST 求值器（src/runtime/DispatchExpr.ts）與原本的
// `new Function` 實作**對每一條現存運算式都給出相同結果**，且不再依賴
// 動態程式碼生成（Chrome MV3 擴充套件的預設 CSP 禁止）。
//
// 測試內容：
//   1. 等價性：對全庫 21 條 distinct `//! DISPATCH` 運算式 × 多組上下文，
//      比較 AST 求值器與 `new Function` 的結果（int32 語意）。
//   2. 語法錯誤處理：錯誤訊息格式與歷史一致（由 PipelineManager 串接）。
//   3. 未定義變數：明確報錯。
//   4. 運算子覆蓋：位移、位元、餘數、單元運算。
//   5. CSP 迴歸：dist bundle 內不得含有 `new Function` / `eval(` 的實際呼叫。
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..');
const DIST = path.join(REPO, 'dist', 'wgfx.esm.js');
const EFFECTS = path.join(REPO, 'effects');

const { parseDispatchExpression, DispatchExprError } =
    await import(pathToFileURL(path.join(REPO, 'dist', 'wgfx.esm.js')).href);

let pass = 0, fail = 0;
function check(name, ok, detail = '') {
    if (ok) { pass++; console.log(`  [PASS] ${name}`); }
    else { fail++; console.log(`  [FAIL] ${name}${detail ? ' — ' + detail : ''}`); }
}

/** 與 JavaScript `| 0` 等價的 ToInt32（用於比對 new Function 的結果）。 */
function toInt32(v) {
    if (!Number.isFinite(v)) return 0;
    return Math.trunc(v) | 0;
}

/** 舊實作：new Function（僅用於測試比對，不在產品碼中使用）。 */
function legacyEval(expr, ctx) {
    const fn = new Function(...Object.keys(ctx), `return (${expr});`);
    return fn(...Object.values(ctx));
}

// ---------- 收集全庫的 distinct 運算式 ----------
function collectDispatchExprs(dir, out = new Set()) {
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return out; }
    for (const ent of entries) {
        const full = path.join(dir, ent.name);
        if (ent.isDirectory()) collectDispatchExprs(full, out);
        else if (ent.name.endsWith('.wgsl')) {
            const code = fs.readFileSync(full, 'utf8');
            for (const m of code.matchAll(/^\/\/!\s*DISPATCH\s+(.+)$/gm)) {
                // 一行可能有兩個運算式（以逗號分隔）
                for (const part of m[1].split(',')) {
                    const e = part.trim();
                    if (e) out.add(e);
                }
            }
        }
    }
    return out;
}

const allExprs = [...collectDispatchExprs(EFFECTS)].sort();
console.log(`=== 1. 等價性:全庫 ${allExprs.length} 條 distinct 運算式 ===`);

// 多組上下文:涵蓋不同尺寸與通道數
const contexts = [
    { NW: 1, NH: 1, N_GROUPS: 1, INPUT_WIDTH: 1, INPUT_HEIGHT: 1, OUTPUT_WIDTH: 1, OUTPUT_HEIGHT: 1 },
    { NW: 16, NH: 16, N_GROUPS: 1, INPUT_WIDTH: 16, INPUT_HEIGHT: 16, OUTPUT_WIDTH: 16, OUTPUT_HEIGHT: 16 },
    { NW: 128, NH: 72, N_GROUPS: 4, INPUT_WIDTH: 128, INPUT_HEIGHT: 72, OUTPUT_WIDTH: 256, OUTPUT_HEIGHT: 144 },
    { NW: 1920, NH: 1080, N_GROUPS: 1024, INPUT_WIDTH: 1920, INPUT_HEIGHT: 1080, OUTPUT_WIDTH: 3840, OUTPUT_HEIGHT: 2160 },
    { NW: 1919, NH: 1079, N_GROUPS: 511, INPUT_WIDTH: 1919, INPUT_HEIGHT: 1079, OUTPUT_WIDTH: 3838, OUTPUT_HEIGHT: 2158 },
];

let mismatch = 0;
for (const expr of allExprs) {
    for (const ctx of contexts) {
        let legacyVal, legacyErr = null;
        try { legacyVal = toInt32(legacyEval(expr, ctx)); }
        catch (e) { legacyErr = e.message; }

        let newVal, newErr = null;
        try { newVal = toInt32(parseDispatchExpression(expr).evaluate(ctx)); }
        catch (e) { newErr = e.message; }

        if (legacyErr || newErr) {
            if (!!legacyErr !== !!newErr) {
                mismatch++;
                console.log(`    MISMATCH(err) "${expr}" ctx=${JSON.stringify(ctx)}`);
                console.log(`      legacy: ${legacyErr} / new: ${newErr}`);
            }
            continue;
        }
        if (legacyVal !== newVal) {
            mismatch++;
            console.log(`    MISMATCH "${expr}" ctx NW=${ctx.NW} NH=${ctx.NH}: legacy=${legacyVal} new=${newVal}`);
        }
    }
}
check(`${allExprs.length} 條運算式 × ${contexts.length} 組上下文 與 new Function 等價`, mismatch === 0,
    mismatch > 0 ? `${mismatch} 處不一致` : '');

// ---------- 2. 語法錯誤訊息格式 ----------
console.log('\n=== 2. 錯誤處理 ===');
{
    let threw = null;
    try { parseDispatchExpression('1 +'); } catch (e) { threw = e; }
    check('語法錯誤拋出 DispatchExprError', threw instanceof DispatchExprError, threw ? threw.name : 'no throw');
    check('語法錯誤帶位置資訊', threw && typeof threw.position === 'number');
}
{
    let threw = null;
    try { parseDispatchExpression('1 @ 2'); } catch (e) { threw = e; }
    check('非法字元被拒絕', threw instanceof DispatchExprError);
}
{
    let threw = null;
    try { parseDispatchExpression('UNKNOWN_VAR + 1').evaluate({ NW: 1 }); } catch (e) { threw = e; }
    check('未定義變數明確報錯', threw !== null && /Undefined variable/.test(threw.message),
        threw ? threw.message : 'no throw');
}
{
    let threw = null;
    try { parseDispatchExpression('1 2'); } catch (e) { threw = e; }
    check('多餘 token 被拒絕', threw instanceof DispatchExprError);
}

// ---------- 3. 運算子覆蓋 ----------
console.log('\n=== 3. 運算子覆蓋 ===');
const opCases = [
    ['1 + 2', {}, 3],
    ['5 - 3', {}, 2],
    ['4 * 6', {}, 24],
    ['7 / 2', {}, 3.5],
    ['7 % 3', {}, 1],
    ['1 << 4', {}, 16],
    ['256 >> 4', {}, 16],
    ['-1 >>> 28', {}, 15],
    ['12 & 10', {}, 8],
    ['12 | 10', {}, 14],
    ['12 ^ 10', {}, 6],
    ['-5', {}, -5],
    ['~0', {}, -1],
    ['+7', {}, 7],
    ['(2 + 3) * 4', {}, 20],
    ['2 + 3 * 4', {}, 14],
    ['NW + 1', { NW: 41 }, 42],
    ['(NW >> 1) * 2', { NW: 16 }, 16],
];
for (const [expr, ctx, expected] of opCases) {
    const got = parseDispatchExpression(expr).evaluate(ctx);
    check(`"${expr}" = ${expected}`, got === expected, `got ${got}`);
}

// ---------- 4. int32 回繞語意 ----------
console.log('\n=== 4. 32 位元整數語意(與 JS | 0 一致) ===');
{
    const got = parseDispatchExpression('2147483647 + 1').evaluate({});
    check('溢出回繞為負數', got === -2147483648, `got ${got}`);
}
{
    const got = parseDispatchExpression('1 << 31').evaluate({});
    check('左移 31 位為 INT_MIN', got === -2147483648, `got ${got}`);
}

// ---------- 5. CSP 迴歸:dist 不得含動態程式碼生成 ----------
console.log('\n=== 5. CSP 迴歸檢查(MV3 相容性) ===');
{
    const bundle = fs.readFileSync(DIST, 'utf8');
    // 移除註解後檢查是否還有實際的 new Function / eval 呼叫
    const stripped = bundle
        .replace(/\/\*[\s\S]*?\*\//g, '')
        .replace(/\/\/.*$/gm, '');
    const hasNewFunction = /new\s+Function\s*\(/.test(stripped);
    const hasEvalCall = /(^|[^.\w])eval\s*\(/.test(stripped);
    check('bundle 內無 new Function 呼叫', !hasNewFunction);
    check('bundle 內無 eval 呼叫', !hasEvalCall);
}

// ---------- 6. 瀏覽器 bundle 也要乾淨 ----------
{
    const browserBundle = fs.readFileSync(path.join(REPO, 'dist', 'wgfx.browser.esm.js'), 'utf8');
    const stripped = browserBundle
        .replace(/\/\*[\s\S]*?\*\//g, '')
        .replace(/\/\/.*$/gm, '');
    check('browser bundle 內無 new Function', !/new\s+Function\s*\(/.test(stripped));
    check('browser bundle 內無 eval 呼叫', !/(^|[^.\w])eval\s*\(/.test(stripped));
}

console.log('');
console.log(`=== 總結: ${pass}/${pass + fail} tests passed ===`);
process.exit(fail === 0 ? 0 : 1);
