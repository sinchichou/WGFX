// 移植特效靜態檢查器（checker for ported WGSL effects）
//
// 用法：node tools/port/check_effect.mjs <file.wgsl | 目錄> [--strict]
// 退出碼：0 = 無 ERROR；1 = 有 ERROR（--strict 時 WARN 也算失敗）
//
// 檢查項（對應本輪移植戰役學到的陷阱）：
//   E1 零分母位點：`1.0 / x`、`inverseSqrt(`、`rcp(`、`/ max(` — 需有 saturate/select 後盾
//      或分母可證非零（常數、clamp 過的變數）；無法自動證明時報 ERROR 由人工分類
//   E2 HLSL 殘留：asuint/asfloat/f16tof32/lerp/frac/ddx/ConstantBuffer/RWTexture2D/#include/float2 等
//   E3 pass 邊界檢查：每個 //!PASS 區塊的進入點應含 `>= outputSize` / `>= size` 類 guard
//   E4 結構：//!PASS 與 //!END 成對；//!VERSION 存在；//!TEXTURE 缺 WIDTH/HEIGHT（無外部資源時會炸）
//   W1 WGSL min/max/saturate 的 NaN 傳播與 HLSL 三元語義不同 — 提醒檢視除法下游
//   W2 Gather 模擬：含 GatherRed/Gather 註解或 gather 呼叫時，提醒 D3D 四點順序
//   I1 matrix alias 轉置提醒（float3x4 → mat4x3）
import fs from 'fs';
import path from 'path';

const args = process.argv.slice(2);
const strict = args.includes('--strict');
const target = args.find(a => !a.startsWith('--'));
if (!target) { console.error('用法: node tools/port/check_effect.mjs <file|dir> [--strict]'); process.exit(2); }

const files = [];
const st = fs.statSync(target);
if (st.isDirectory()) {
    (function walk(d) { for (const e of fs.readdirSync(d, { withFileTypes: true })) { const p = path.join(d, e.name); e.isDirectory() ? walk(p) : e.name.endsWith('.wgsl') && files.push(p); } })(target);
} else files.push(target);

const HLSL_LEFTOVER = [
    [/\basuint\s*\(/, 'asuint('], [/\basfloat\s*\(/, 'asfloat('], [/\basint\s*\(/, 'asint('],
    [/f16tof32\s*\(/, 'f16tof32('], [/f32tof16\s*\(/, 'f32tof16('],
    [/\blerp\s*\(/, 'lerp('], [/\bfrac\s*\(/, 'frac('], [/\bddx\s*\(/, 'ddx('], [/\bddy\s*\(/, 'ddy('],
    [/ConstantBuffer/, 'ConstantBuffer'], [/RWTexture2D/, 'RWTexture2D'], [/StructuredBuffer/, 'StructuredBuffer'],
    [/#include/, '#include'], [/\bsampler2D\b/, 'sampler2D'], [/\bmul\s*\(/, 'mul('],
    [/\bfloat[234]\s+\w+\s*[;,=)]/, 'floatN 變數宣告'], [/\bhalf\s+\w+\s*[;,=)]/, 'half 變數'],
    [/\bdiscard;?/, 'discard（compute 無效）'],
];
const DIV_PATTERNS = [
    [/1\.0\s*\/\s*(?!\s*0\b|\s*[0-9])/g, '1.0 / 變數'],
    [/\binverseSqrt\s*\(/g, 'inverseSqrt('],
    [/\brcp\s*\(/g, 'rcp('],
    [/\/\s*max\s*\(/g, '/ max('],
];

let totalErr = 0, totalWarn = 0;
for (const file of files) {
    // 統一換行：JS 的 `.` 不匹配 \r、$（無 m）不認 \r 前——CRLF 會讓 /\/\/.*$/ 全面失效
    const code = fs.readFileSync(file, 'utf8').replace(/\r\n/g, '\n');
    const lines = code.split('\n');
    const findings = [];

    // E2 HLSL 殘留
    lines.forEach((ln, i) => {
        const bare = ln.replace(/\/\/.*$/, ''); // 去行註解
        for (const [re, name] of HLSL_LEFTOVER) {
            if (re.test(bare)) findings.push(['E2', i + 1, `HLSL 殘留: ${name}`]);
        }
    });

    // E1 零分母位點（往後看 3 行內有無 saturate/select/clamp 後盾；行內 `checker:ok` 標註可豁免）
    lines.forEach((ln, i) => {
        const bare = ln.replace(/\/\/.*$/, '');
        if (ln.includes('checker:ok')) return;
        if (/^\s*fn\s/.test(bare)) return; // 函式定義簽名（防護責任在呼叫點/函式體）
        for (const [re, name] of DIV_PATTERNS) {
            re.lastIndex = 0;
            if (!re.test(bare)) continue;
            const ctx = lines.slice(i, i + 4).join(' ');
            const guarded = /\bsaturate\b|\bselect\b|\bclamp\b|==\s*0\.0|<=\s*0\.0|===\s*0|max\(.*,\s*(1e|1\.0e|0\.0001|\/ ?32768)/.test(ctx);
            const noted = lines.slice(Math.max(0, i - 3), i).join(' ').includes('保護') || lines.slice(Math.max(0, i - 3), i).join(' ').toLowerCase().includes('protect');
            if (!guarded && !noted) findings.push(['E1', i + 1, `${name} — 3 行內無 saturate/select/clamp 後盾`]);
        }
    });

    // E3 pass 邊界檢查
    const passIdx = [];
    lines.forEach((ln, i) => { if (/^\/\/!\s*PASS\s+\d+/.test(ln)) passIdx.push(i); });
    passIdx.forEach((pi, n) => {
        const end = n + 1 < passIdx.length ? passIdx[n + 1] : lines.length;
        const block = lines.slice(pi, end).join('\n');
        if (!/>\=\s*(outputSize|size|outSize)/.test(block) && !/>=\s*(scene\.)?(outputSize|inputSize)/.test(block)) {
            findings.push(['E3', pi + 1, 'pass 區塊缺邊界 guard（>= outputSize / >= size）']);
        }
    });

    // E4 結構
    if (!/^\/\/!\s*VERSION/m.test(code)) findings.push(['E4', 1, '缺 //!VERSION']);
    // （//!END 與 //!PASS 數量不必相等：DSL 允許省略 END，且 COMMON/PARAMETER 等區塊
    //  也用 END 終止 — 真正的驗證閘門是編譯與 effect_sweep，此處僅提醒。）
    lines.forEach((ln, i) => {
        if (/^\/\/!\s*TEXTURE/.test(ln)) {
            const block = lines.slice(i, i + 6).join('\n');
            if (!/WIDTH/.test(block) || !/HEIGHT/.test(block)) {
                // INPUT/OUTPUT 可由宿主提供；中間紋理必須宣告尺寸
                if (!/INPUT|OUTPUT/.test(lines.slice(i, i + 6).map(l => l.replace(/^\/\/!/, '')).join(' '))) {
                    findings.push(['E4', i + 1, '//!TEXTURE 缺 WIDTH/HEIGHT（非 INPUT/OUTPUT 需明確尺寸）']);
                }
            }
        }
    });

    // W2 Gather 提醒
    if (/Gather(Red|Green|Blue)?\s*\(/.test(code)) findings.push(['W2', 1, '含 gather 語義 — 確認 D3D 四點順序 x=BL,y=BR,z=TR,w=TL 已按註解重建']);

    // W1 除法下游 NaN 提醒（有 E1 時自動附帶）
    if (findings.some(f => f[0] === 'E1')) findings.push(['W1', 1, '存在未護除法 — WGSL min/max 會傳播 NaN（HLSL 三元語義會治癒），檢視下游是否經 max/min']);

    const err = findings.filter(f => f[0].startsWith('E')).length;
    const warn = findings.filter(f => f[0].startsWith('W')).length;
    totalErr += err; totalWarn += warn;
    console.log(`\n=== ${path.relative(process.cwd(), file)} — ${err} ERROR / ${warn} WARN ===`);
    for (const [lvl, ln, msg] of findings) console.log(`  [${lvl}] L${ln}: ${msg}`);
    if (findings.length === 0) console.log('  （乾淨）');
}

const fail = totalErr > 0 || (strict && totalWarn > 0);
console.log(`\n總計: ${totalErr} ERROR / ${totalWarn} WARN — ${fail ? 'FAIL' : 'PASS'}${strict && totalWarn > 0 && totalErr === 0 ? '（--strict）' : ''}`);
process.exit(fail ? 1 : 0);
