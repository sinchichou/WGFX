// OpenNR CPU 參考實作 smoke：載入真實權重，跑完整 19 層前向，檢查數值健全性與決定性
import path from 'path';
import { fileURLToPath } from 'url';
import { loadModel, forward, alignExtent, FeatureMap, conv3x3, gelu, f32tof16, f16tof32, p4, h4 } from './opennr_ref.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const MODEL = path.join(HERE, '..', 'effects_src', 'OpenNR-ReShade', 'upstream', 'models', '1pass');

let fail = 0;
const check = (name, ok, detail = '') => {
    console.log(`  ${ok ? '[PASS]' : '[FAIL]'} ${name}${detail ? ' — ' + detail : ''}`);
    if (!ok) fail++;
};

console.log('=== 1. f16 round-trip ===');
{
    const vals = [0, 1, -1, 0.5, 0.25, 65504, 1e-5, 6.103515625e-5, -2.5, 0.1];
    const F16_MIN_NORMAL = 2 ** -14;
    let worstRel = 0, worstAbs = 0;
    for (const v of vals) {
        const r = f16tof32(f32tof16(v));
        if (Math.abs(v) >= F16_MIN_NORMAL) worstRel = Math.max(worstRel, Math.abs(r - v) / Math.abs(v));
        else worstAbs = Math.max(worstAbs, Math.abs(r - v));
    }
    // f16 有兩個精度體制：正規數相對誤差 ≤ 2^-11、subnormal 絕對誤差 ≤ 2^-25（半個量子）
    check('f16 round-trip（正規數 rel ≤ 1e-3；subnormal abs ≤ 2^-25）',
        worstRel < 1e-3 && worstAbs <= 2 ** -25,
        `worstRel=${worstRel.toExponential(2)} worstAbs=${worstAbs.toExponential(2)}`);
    // 位元級對照已知值
    check('f32tof16(1.0) === 0x3C00', f32tof16(1.0) === 0x3c00, `got 0x${f32tof16(1.0).toString(16)}`);
    check('f32tof16(-2.0) === 0xC000', f32tof16(-2.0) === 0xc000, `got 0x${f32tof16(-2.0).toString(16)}`);
    const packed = p4([1, -2, 0.5, 0]);
    const un = h4(packed);
    check('p4/h4 round-trip', un[0] === 1 && un[1] === -2 && un[2] === 0.5 && un[3] === 0,
        `[${un.join(', ')}]`);
}

console.log('=== 2. GELU 對照已知值 ===');
{
    // gelu(0)=0, gelu(1)≈0.841192, gelu(-1)≈-0.158808
    check('gelu(0) === 0', Math.abs(gelu(0)) < 1e-7, `got ${gelu(0)}`);
    check('gelu(1) ≈ 0.841192', Math.abs(gelu(1) - 0.841192) < 1e-5, `got ${gelu(1).toFixed(6)}`);
    check('gelu(-1) ≈ -0.158808', Math.abs(gelu(-1) + 0.158808) < 1e-5, `got ${gelu(-1).toFixed(6)}`);
    check('gelu 大輸入不 NaN', Number.isFinite(gelu(1000)) && Number.isFinite(gelu(-1000)), `gelu(1000)=${gelu(1000)}`);
}

console.log('=== 3. 載入真實權重 ===');
const model = loadModel(MODEL);
check('層數 19', model.convs.length === 19, `got ${model.convs.length}`);
check('header cin=16 base=16 levels=4 cout=4',
    model.cin === 16 && model.base === 16 && model.levels === 4 && model.cout === 4,
    `cin=${model.cin} base=${model.base} levels=${model.levels} cout=${model.cout}`);
check('global 段存在', !!model.glob, model.glob ? `g_in=${model.glob.g_in} g_hidden=${model.glob.g_hidden} g_out=${model.glob.g_out}` : 'none');
{
    const names = model.convs.map(c => c.name);
    const expect = ['enc0a','enc0b','enc1a','enc1b','enc2a','enc2b','enc3a','enc3b','mid_a','mid_b',
                    'dec3a','dec3b','dec2a','dec2b','dec1a','dec1b','dec0a','dec0b','out'];
    check('層名順序與 model.txt 一致', JSON.stringify(names) === JSON.stringify(expect), names.join(','));
    const acts = model.convs.filter(c => c.act).map(c => c.name);
    check('GELU 層 = 18（除 out）', acts.length === 18 && !model.convs[18].act, `act=true 共 ${acts.length}`);
    const out = model.convs[18];
    check('out 層 cout=4 pad=16', out.cout === 4 && out.pad === 16, `cout=${out.cout} pad=${out.pad}`);
    check('dec3a pad=128（cin=256）', model.convs[10].pad === 128 && model.convs[10].cin === 256,
        `cin=${model.convs[10].cin} pad=${model.convs[10].pad}`);
    // 權重為 packed uint2（每個 u32 = 2 個 half）→ w.length*4 = 檔案的位元組數 = 9*cin*pad*2
    let wOK = true;
    for (const c of model.convs) if (c.w.length * 4 !== 9 * c.cin * c.pad * 2) { wOK = false; console.log(`    ${c.name} w bytes=${c.w.length*4} 期望=${9*c.cin*c.pad*2}`); }
    check('每層 w 位元組 = 9*cin*pad*2', wOK);
    let bOK = true;
    for (const c of model.convs) if (c.b.length !== c.pad) { bOK = false; console.log(`    ${c.name} b=${c.b.length} 期望=${c.pad}`); }
    check('每層 b 長度 = pad（f32）', bOK);
}

console.log('=== 4. 完整前向（小解析度 320x320）===');
const W = alignExtent(320), H = alignExtent(320);
console.log(`  網路解析度 ${W}x${H}`);
const t0 = Date.now();
let nanLayer = null;
const trace = forward(model, W, H, {
    frame: 0, tone: 1.0, structure: 1.0,
    onLayer: (name, fm) => {
        if (nanLayer) return;
        for (let i = 0; i < fm.data.length; i++) {
            const v = h4([fm.data[i], 0])[0];
            if (!Number.isFinite(v)) { nanLayer = `${name} @u32[${i}]`; break; }
        }
    }
});
const dt = Date.now() - t0;
check('19 層全部執行', trace.layerCount === 19, `layerCount=${trace.layerCount}`);
check('無 NaN/Inf 中間值', nanLayer === null, nanLayer || 'clean');
console.log(`  前向耗時 ${dt} ms（${W}x${H}）`);

console.log('=== 5. 輸出值域與統計 ===');
{
    const head = trace.head;
    check('head 維度 = W×H×4(pad16)', head.w === W && head.h === H && head.ch === 4 && head.pad === 16,
        `${head.w}x${head.h} ch=${head.ch} pad=${head.pad}`);
    let mn = Infinity, mx = -Infinity, sum = 0, cnt = 0;
    for (let p = 0; p < head.npix; p++) {
        const v = head.get4(p, 0);
        for (let k = 0; k < 4; k++) { mn = Math.min(mn, v[k]); mx = Math.max(mx, v[k]); sum += v[k]; cnt++; }
    }
    console.log(`  head 值域 [${mn.toFixed(4)}, ${mx.toFixed(4)}] 均值 ${(sum / cnt).toFixed(5)}`);
    check('輸出有限且非全零', Number.isFinite(mn) && Number.isFinite(mx) && (mx - mn) > 1e-6,
        `range=${(mx - mn).toExponential(2)}`);
    check('輸出量級合理（|v| < 10）', Math.abs(mn) < 10 && Math.abs(mx) < 10, `[${mn.toFixed(3)}, ${mx.toFixed(3)}]`);
    // pad 區（通道 4..15）應被寫入但我們只讀 0..3；確認 buffer 寬度正確
    check('pad 區 buffer 存在（ch4=4）', head.ch4 === 4, `ch4=${head.ch4}`);
}

console.log('=== 6. 決定性（同輸入兩次結果相同）===');
{
    const t1 = forward(model, W, H, { frame: 0, tone: 1.0, structure: 1.0 });
    let same = t1.head.data.length === trace.head.data.length;
    if (same) for (let i = 0; i < t1.head.data.length; i++) if (t1.head.data[i] !== trace.head.data[i]) { same = false; break; }
    check('兩次前向 bit-identical', same);
}

console.log('=== 7. 噪聲時域（frame 進種子）===');
{
    const t0f = forward(model, W, H, { frame: 0, tone: 1.0, structure: 1.0 });
    const t1f = forward(model, W, H, { frame: 1, tone: 1.0, structure: 1.0 });
    const c0 = t0f.chans, c1 = t1f.chans;
    let diff = 0, total = 0;
    for (let p = 0; p < c0.npix; p++) {
        const a = c0.get4(p, 0), b = c1.get4(p, 0);   // 通道 0..3 為種子噪聲
        for (let k = 0; k < 4; k++) { total++; if (a[k] !== b[k]) diff++; }
    }
    const frac = diff / total;
    console.log(`  frame 0 vs 1：噪聲通道差異比例 ${(frac * 100).toFixed(1)}%（${diff}/${total}）`);
    // 門檻為 50%：f16 只有 10-bit mantissa，不同 u01 抽樣常四捨五入到同一個 f16 值，
    // 因此並非 100% 位元改變；但絕大多數像素確實隨 frame 重抽（實測 ~75%）。
    check('噪聲隨 frame 變化（>50% 位元不同）', frac > 0.5, `frac=${frac.toFixed(4)}`);
    // 色彩通道（4..9）不應受 frame 影響
    let cdiff = 0, ctot = 0;
    for (let p = 0; p < c0.npix; p++) {
        const a = c0.get4(p, 1), b = c1.get4(p, 1);
        for (let k = 0; k < 4; k++) { ctot++; if (a[k] !== b[k]) cdiff++; }
    }
    check('色彩通道不受 frame 影響', cdiff === 0, `diff=${cdiff}/${ctot}`);
}

console.log('=== 8. 池化與 upcat 幾何 ===');
{
    const f = new FeatureMap(8, 8, 4);
    for (let p = 0; p < 64; p++) f.set4(p, 0, [p, p, p, p]);
    const o = new FeatureMap(4, 4, 4);
    const { pool2x2, upcat } = await import('./opennr_ref.mjs');
    pool2x2(f, o, 4);
    // p=0 應為 (0 + 1 + 8 + 9)/4 = 4.5
    const v = o.get4(0, 0);
    check('pool2x2 首像素 = 4.5', Math.abs(v[0] - 4.5) < 1e-3, `got ${v[0]}`);
    // upcat scale=2：lo 4x4 → out 8x8
    const lo = new FeatureMap(4, 4, 4);
    const sk = new FeatureMap(8, 8, 4);
    lo.set4(0, 0, [1, 1, 1, 1]);
    sk.set4(0, 0, [7, 7, 7, 7]);
    const oc = new FeatureMap(8, 8, 8);
    upcat(lo, sk, oc, 4, 4, 2);
    const a = oc.get4(0, 0), b = oc.get4(0, 1);
    check('upcat lo 通道放大正確', Math.abs(a[0] - 1) < 1e-6, `got ${a[0]}`);
    check('upcat skip 通道正確', Math.abs(b[0] - 7) < 1e-6, `got ${b[0]}`);
}

console.log(`\n=== 總結: ${fail === 0 ? 'ALL PASS' : fail + ' FAILED'} ===`);
process.exit(fail === 0 ? 0 : 1);
