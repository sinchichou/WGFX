// magpie_mix_select.test.mjs — HLSL `mix(a, b, bool)` → WGSL `select(a, b, bool)` 迴歸測試
//
// 背景(2026-11-15 修復的缺陷):
//   Magpie 的 boolean `mix(a, b, c)` = `c ? b : a`
//   (見 effects_src/NNEDI3/prescalers.hlsli 的 `#define` 說明)。
//   舊版轉換器誤輸出 `select(b, a, c)`——WGSL `select(false, true, cond)` 的
//   false/true 參數位置與 HLSL 相反 → 條件被實質反轉。NNEDI3 的
//       float mstd2 = mix(0.0, inversesqrt(mstd1), mstd1 >= 1.192092896e-7F);
//   因此在一般(有紋理的)鄰域變成 mstd2 = 0 → mstd1 *= 0 → 輸出塌縮成
//   鄰域均值(模糊/變暗),nns 權重完全失效;均勻鄰域則會算出 0 * Inf = NaN。
//   修復:轉換器改輸出 `select(a, b, c)`(tools/magpie_prescaler2wgsl.mjs),
//   RAVU 的獨立產生器(effects/RAVU/_gen_ravu.mjs)同步修正。
//
// 防線:
//   1. 用轉換器重新生成 NNEDI3_nns16_win8x4,斷言 mstd2 為正確順序。
//   2. 掃描所有已提交的 effects/NNEDI3/*.wgsl,斷言無反向 select。
//   3. 斷言 RAVU 產生器的 select 模板參數順序正確。
// 執行:node test/magpie_mix_select.test.mjs
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..');

let pass = 0, fail = 0;
function check(name, ok, detail = '') {
    if (ok) { pass++; console.log(`  [PASS] ${name}`); }
    else { fail++; console.log(`  [FAIL] ${name}${detail ? ' — ' + detail : ''}`); }
}

// HLSL `mix(0.0, inversesqrt(mstd1), cond)` 的兩種 WGSL 形態:
const RE_CORRECT = /select\(\s*0\.0\s*,\s*inverseSqrt\(mstd1\)\s*,\s*mstd1\s*>=/;
const RE_REVERSED = /select\(\s*inverseSqrt\(mstd1\)\s*,\s*0\.0\s*,/;

// ---------- 1. 轉換器輸出 ----------
{
    const outDir = path.join(REPO, '.tmp', 'mix_select_regen');
    fs.mkdirSync(outDir, { recursive: true });
    const out = path.join(outDir, 'NNEDI3_nns16_win8x4.wgsl');
    // stdio 用 inherit:沙箱環境下以 pipe 捕捉子程序輸出會 EPERM
    execFileSync(process.execPath, [
        path.join(REPO, 'tools', 'magpie_prescaler2wgsl.mjs'),
        path.join(REPO, 'effects_src', 'NNEDI3', 'NNEDI3_nns16_win8x4.hlsl'),
        out,
    ], { stdio: 'inherit' });
    const text = fs.readFileSync(out, 'utf8');
    check('轉換器輸出 select(0.0, inverseSqrt(mstd1), …) 正確順序', RE_CORRECT.test(text));
    check('轉換器輸出不含反向 select(inverseSqrt(mstd1), 0.0, …)', !RE_REVERSED.test(text));
}

// ---------- 2. 已提交的 NNEDI3 WGSL ----------
{
    const dir = path.join(REPO, 'effects', 'NNEDI3');
    const files = fs.readdirSync(dir).filter(f => f.endsWith('.wgsl'));
    const reversed = [];
    let withCorrect = 0;
    for (const f of files) {
        const t = fs.readFileSync(path.join(dir, f), 'utf8');
        if (RE_REVERSED.test(t)) reversed.push(f);
        if (RE_CORRECT.test(t)) withCorrect++;
    }
    check(`effects/NNEDI3 共 ${files.length} 檔皆無反向 mstd2 select`, reversed.length === 0,
        reversed.join(', '));
    check('每個 NNEDI3 變體都含正確順序的 mstd2 select', withCorrect === files.length,
        `${withCorrect}/${files.length}`);
}

// ---------- 3. RAVU 產生器的 select 模板 ----------
{
    const t = fs.readFileSync(path.join(REPO, 'effects', 'RAVU', '_gen_ravu.mjs'), 'utf8');
    check('RAVU 產生器 select(a, b, cond) 模板參數順序正確',
        t.includes('select(${args[0]}, ${args[1]}, ${args[2]})'));
    check('RAVU 產生器無 select(b, a, cond) 反向模板',
        !t.includes('select(${args[1]}, ${args[0]}, ${args[2]})'));
}

console.log('');
console.log(`=== 總結: ${pass}/${pass + fail} tests passed ===`);
process.exit(fail === 0 ? 0 : 1);
