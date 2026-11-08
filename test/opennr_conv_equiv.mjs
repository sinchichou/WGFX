// 交叉驗證：conv3x3（可讀參考版）與 conv3x3Fast（typed-array 快速版）必須產生位元相同的輸出
// 這是驗收可信度的基礎 —— 若兩版等價，全網驗收用快速版即等同用參考版。
import path from 'path';
import { fileURLToPath } from 'url';
import { loadModel, conv3x3, conv3x3Fast, FeatureMap, p4, h4 } from './opennr_ref.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const MODEL = path.join(HERE, '..', 'effects_src', 'OpenNR-ReShade', 'upstream', 'models', '1pass');
const model = loadModel(MODEL);

// 決定性偽隨機
let seed = 12345;
const rnd = () => { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return (seed >>> 8) / 16777216; };

let fail = 0;
const check = (n, ok, d = '') => { console.log(`  ${ok ? '[PASS]' : '[FAIL]'} ${n}${d ? ' — ' + d : ''}`); if (!ok) fail++; };

// 測所有「獨特形狀」的層，外加 zeroPad 與 act 的組合
const shapes = new Map();
for (const c of model.convs) shapes.set(`${c.cin}_${c.pad}`, c);
const layers = [...shapes.values()];
console.log(`=== 交叉驗證 ${layers.length} 種獨特層形狀 × {act,zeroPad} 組合 ===`);

for (const c of layers) {
    for (const act of [true, false]) {
        for (const zeroPad of [false, true]) {
            const W = 8, H = 8;
            const x = new FeatureMap(W, H, c.cin);
            // 填入可重現的 f16 值（含負數與小數，覆蓋 subnormal 邊界）
            for (let i = 0; i < x.data.length; i++) {
                const v = (rnd() * 2 - 1) * 2;
                const pk = p4([v, v * 0.5, -v, v + 0.25]);
                x.data[i] = pk[i & 1];
            }
            const o1 = new FeatureMap(W, H, c.cout, c.pad);
            const o2 = new FeatureMap(W, H, c.cout, c.pad);
            conv3x3(x, o1, c, act, zeroPad);
            conv3x3Fast(x, o2, c, act, zeroPad);
            let diff = 0;
            for (let i = 0; i < o1.data.length; i++) if (o1.data[i] !== o2.data[i]) diff++;
            const tag = `${c.name} cin=${c.cin} pad=${c.pad} act=${act ? 1 : 0} zp=${zeroPad ? 1 : 0}`;
            check(tag, diff === 0, diff === 0 ? 'bit-identical' : `${diff}/${o1.data.length} u32 不同`);
        }
    }
}

console.log(`\n=== 總結: ${fail === 0 ? 'ALL PASS（快速版可信）' : fail + ' FAILED'} ===`);
process.exit(fail === 0 ? 0 : 1);
