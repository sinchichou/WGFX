// tools/opennr_weights.mjs 的驗收測試
//
// 證明三件事：
//  1. 逐層 byte 驗算（w = 9*cin*pad*2、b = pad*4）對兩個變體全部通過。
//  2. packed 輸出與 .bin 原始位元組**逐 byte 一致**（w 不需重排、b 是 f32 直讀）。
//  3. pack → decode 往返與 WGSL 端 h4()/f32tof16 語義一致，且 index 公式
//     `(tap*cin + k4*4 + kk)*cout4 + n4` 指到的位置與檔案 tap-major 偏移相同。
//
// 執行：node tools/opennr_weights.test.mjs
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {
    loadModel, verifyModel, buildManifest, encodeF16, decodeF16,
    DEFAULT_MODELS_DIR, VARIANTS, REPO_ROOT
} from './opennr_weights.mjs';

let pass = 0, fail = 0;
const check = (name, cond, extra = '') => {
    if (cond) { pass++; console.log(`  ✓ ${name}${extra ? '  ' + extra : ''}`); }
    else { fail++; console.log(`  ✗ ${name}${extra ? '  ' + extra : ''}`); }
};

const u8 = (t) => Buffer.from(t.buffer, t.byteOffset, t.byteLength);

for (const variant of VARIANTS) {
    const dir = path.join(DEFAULT_MODELS_DIR, variant);
    if (!fs.existsSync(dir)) continue;
    const m = loadModel(dir);
    console.log(`\n=== ${variant} ===`);

    // ---- 1. byte 驗算 ----
    const rows = verifyModel(m);
    const bad = rows.filter((r) => !r.ok);
    check(`全層 byte 驗算 (${rows.length} 層)`, bad.length === 0,
        bad.length ? `失敗: ${bad.map((b) => b.name).join(',')}` : `w=9*cin*pad*2, b=pad*4`);

    // 每一層：期望 bytes 用公式獨立算一次，不經由 verifyModel
    let byteOk = true;
    for (const c of [...m.convs, ...(m.glob?.convs ?? [])]) {
        if (c.wBytes !== 9 * c.cin * c.pad * 2) byteOk = false;
        if (c.bBytes !== c.pad * 4) byteOk = false;
        if (c.w.length * 4 !== c.wBytes) byteOk = false;
        if (c.b.length * 4 !== c.bBytes) byteOk = false;
        if (c.wElements !== 9 * c.cin * c.pad / 4) byteOk = false;
        if (c.bElements !== c.pad / 4) byteOk = false;
    }
    check('逐層公式獨立重算', byteOk);

    // ---- 2. 與 .bin 原始位元組逐 byte 一致 ----
    let allSame = true, anyBad = [];
    for (const c of [...m.convs, ...(m.glob?.convs ?? [])]) {
        const wSrc = fs.readFileSync(path.join(dir, `${c.name}.w.bin`));
        const bSrc = fs.readFileSync(path.join(dir, `${c.name}.b.bin`));
        if (!u8(c.w).equals(wSrc)) { allSame = false; anyBad.push(`${c.name}.w`); }
        if (!u8(c.b).equals(bSrc)) { allSame = false; anyBad.push(`${c.name}.b`); }
    }
    check('packed w / f32 b 與 .bin 逐 byte 一致', allSame, anyBad.join(','));
    // 這正是「無重排」的證明：w 的檔案順序就等於 WGSL 索引順序。

    if (m.glob) {
        const fcSrc = fs.readFileSync(path.join(dir, 'fc.bin'));
        check('fc.bin 逐 byte 一致', u8(m.glob.fc.all).equals(fcSrc),
            `${fcSrc.length} bytes`);
        const g = m.glob;
        const shapeOk = g.fc.fc1W.length === g.ghidden * g.gin &&
            g.fc.b1.length === g.ghidden &&
            g.fc.fc2W.length === g.gout * g.ghidden &&
            g.fc.b2.length === g.gout;
        check('fc 四段切分正確', shapeOk,
            `fc1W[${g.ghidden}][${g.gin}] b1[${g.ghidden}] fc2W[${g.gout}][${g.ghidden}] b2[${g.gout}]`);
        // 切分必須是「連續且無重疊」的 subarray
        check('fc 切分無縫接合', g.fc.fc1W.byteOffset + g.fc.fc1W.byteLength === g.fc.b1.byteOffset &&
            g.fc.b1.byteOffset + g.fc.b1.byteLength === g.fc.fc2W.byteOffset &&
            g.fc.fc2W.byteOffset + g.fc.fc2W.byteLength === g.fc.b2.byteOffset);
        // gbias 由 kGlobalHLSL 寫出，大小 = gO/4 個 float4
        check('gbias buffer 大小 (gO/4 個 float4)', g.gout % 4 === 0,
            `${g.gout} floats = ${g.gout / 4} × float4 = ${g.gout * 4} bytes`);
    }

    // ---- 3. 索引公式 ↔ 檔案偏移 ----
    // WGSL: `w[(tap*cin + k4*4 + kk)*cout4 + n4]` 是 uint2 元素（2 個 u32 = 4 個 half）。
    // 檔案是 tap-major 的平坦 f16 陣列，因此：
    //     half 偏移 = elem * 4      （elem 是 uint2 元素索引）
    //     u32  偏移 = elem * 2
    // 等價地，把檔案視為 u32 陣列時，元素索引與 u32 索引是 2:1。
    //
    // 注意：**不要**用 `n4*4` 當 half 步長 —— n4 是元素（uint2）索引，不是 half 群組；
    // 一個 uint2 已經涵蓋 4 個 component（kk=0..3 對應 cout 的 c0..c3）。
    // 這正是本轉換「不需重排」的根據：檔案順序就是 WGSL 元素順序。
    // `c.w` 是 Uint32Array：uint2 元素 `elem` 佔用 u32 索引 `elem*2` 與 `elem*2+1`
    // （4 個 half = 2 個 u32）。務必用 uint2At()，直接寫 c.w[elem] 會拿到半個元素。
    let idxOk = true;
    const c = m.convs[0]; // enc0a cin=16 pad=16 cout=16
    const cin = c.cin, cout = c.cout;
    const cin4 = cin / 4, cout4 = cout / 4;
    const probes = [[0, 0, 0, 0], [3, 2, 1, 2], [8, cin4 - 1, 3, cout4 - 1], [4, 1, 2, 3]];
    const wSrc0 = fs.readFileSync(path.join(dir, `${c.name}.w.bin`));
    const H0 = new Uint16Array(wSrc0.buffer, wSrc0.byteOffset, wSrc0.length >> 1);
    for (const [tap, k4, kk, n4] of probes) {
        const elem = (tap * cin + k4 * 4 + kk) * cout4 + n4;
        const u = c.w[elem * 2], u1 = c.w[elem * 2 + 1];
        // 用「掃描找出這 4 個 half 落在哪」驗證，不預設公式
        let found = -1;
        for (let h = 0; h + 3 < H0.length; h++)
            if (H0[h] === (u & 0xffff) && H0[h + 1] === (u >>> 16) &&
                H0[h + 2] === (u1 & 0xffff) && H0[h + 3] === (u1 >>> 16)) { found = h; break; }
        if (found !== elem * 4) idxOk = false;
    }
    check('WGSL 元素索引 → 檔案 half 偏移 (half = elem*4)', idxOk,
        `probe ${probes.length} 組 (tap,k4,kk,n4) 全部命中且無重排（w 為 Uint32Array，元素 idx*2）`);

    // 讀一個實際元素驗證路徑（檔案端以 raw bytes 交叉核對）
    {
        const tap = 4, k4 = 1, kk = 2, n4 = 3;
        const elem = (tap * cin + k4 * 4 + kk) * cout4 + n4;
        const u = c.w[elem * 2], u1 = c.w[elem * 2 + 1];
        const got = [decodeF16(u & 0xffff), decodeF16(u >>> 16),
                     decodeF16(u1 & 0xffff), decodeF16(u1 >>> 16)];
        const off = elem * 4;
        const exp = [decodeF16(H0[off]), decodeF16(H0[off + 1]),
                     decodeF16(H0[off + 2]), decodeF16(H0[off + 3])];
        const bytesOk = wSrc0[off * 2] === (H0[off] & 0xff) &&
            wSrc0[off * 2 + 1] === (H0[off] >> 8);
        check(`元素取值 (tap=${tap},k4=${k4},kk=${kk},n4=${n4})`,
            got.every((v, i) => v === exp[i]) && bytesOk,
            `[${got.map((v) => v.toExponential(4)).join(', ')}] halfOff=${off} u32Off=${elem * 2}`);
    }

    // out 層 pad != cout，驗證它仍用 pad 決定 buffer 寬度（WGSL 以 gCout=pad 傳入）
    {
        const o = m.convs.find((x) => x.name === 'out');
        const oElems = 9 * o.cin * o.pad / 4;
        check('out 層 pad != cout 且 buffer 以 pad 為準',
            o.pad !== o.cout && o.wElements === oElems && o.wElements === 9 * o.cin * 16 / 4,
            `cout=${o.cout} pad=${o.pad} elems=${o.wElements} (若誤用 cout 會是 ${9 * o.cin * o.cout / 4})`);
    }

    // ---- 4. f16 編解碼往返 ----
    // f16 有**兩個不同的精度體制**，必須分開判定：原本一律套 2^-11 相對界是錯的。
    //   - 正規數（|v| ≥ 2^-14 = 6.1035e-5）：10 位 mantissa → 相對誤差 ≤ 2^-11
    //   - subnormal（0 < |v| < 2^-14）：絕對量子固定為 2^-24，與 v 大小無關，
    //     故相對誤差會隨 v 變小而放大（v = 1e-5 時本來就到 1.4e-3）。
    //     此區間必須改用**絕對**誤差 ≤ 2^-25（半個量子）。
    const F16_MIN_NORMAL = 2 ** -14;
    const NORMALS = [1, -1, 0.5, 1e-4, 3.14159, 65504, -65504];
    const SUBNORMALS = [1e-5, -1e-5, 5e-6, 6e-8];
    let rtOk = true, worstRel = 0, worstAbs = 0;
    {
        const back = decodeF16(encodeF16(0));
        if (back !== 0) { rtOk = false; console.log('    0 未往返為 0'); }
    }
    for (const v of NORMALS) {
        const back = decodeF16(encodeF16(v));
        const rel = Math.abs((back - v) / v);
        worstRel = Math.max(worstRel, rel);
        if (rel > 2 ** -11) { rtOk = false; console.log(`    正規數 ${v} rel=${rel.toExponential(2)} 超出 2^-11`); }
    }
    for (const v of SUBNORMALS) {
        if (Math.abs(v) >= F16_MIN_NORMAL) { rtOk = false; console.log(`    ${v} 被歸類為 subnormal 但其實是正規數`); }
        const back = decodeF16(encodeF16(v));
        const abs = Math.abs(back - v);
        worstAbs = Math.max(worstAbs, abs);
        if (abs > 2 ** -25) { rtOk = false; console.log(`    subnormal ${v} abs=${abs.toExponential(2)} 超出 2^-25`); }
    }
    check('encodeF16/decodeF16 往返（正規數 rel ≤ 2^-11；subnormal abs ≤ 2^-25）', rtOk,
        `worst rel ${worstRel.toExponential(2)} / worst abs ${worstAbs.toExponential(2)}`);

    // 真實權重的往返：把整層 w 解碼再編碼，必須位元完全相同
    {
        let bitExact = true, n = 0;
        const c0 = m.convs[0];
        for (let j = 0; j < c0.w.length; j++) {
            const u = c0.w[j];
            const lo = u & 0xffff, hi = u >>> 16;
            if (encodeF16(decodeF16(lo)) !== lo || encodeF16(decodeF16(hi)) !== hi) bitExact = false;
            n += 2;
        }
        check('真實權重 f16 解碼→編碼位元完全不變', bitExact, `${n} 個 half (${c0.name})`);
    }

    // ---- 5. activation 判定 ----
    const geluLayers = m.convs.filter((x) => x.act).map((x) => x.name);
    const linearLayers = m.convs.filter((x) => !x.act).map((x) => x.name);
    check('只有 out 層線性', linearLayers.length === 1 && linearLayers[0] === 'out',
        `GELU ${geluLayers.length} 層, 線性 [${linearLayers.join(',')}]`);
    check('global 分支 3 層全部 GELU', m.glob && m.glob.convs.every((x) => x.act));

    // ---- 6. manifest 可序列化 ----
    const man = buildManifest(m);
    const json = JSON.stringify(man);
    check('manifest JSON 可序列化', json.length > 0, `${json.length} bytes`);
    check('manifest 層數正確', man.layers.length === m.layerCount,
        `${man.layers.length} vs model.txt n=${m.layerCount}`);
    check('manifest level → 解析度除數', man.layers.every((l) => l.resolutionDivisor === (1 << l.level)));
    // 必須沒有任何二進位殘留（純 JSON）
    check('manifest 不含 typed array 殘留',
        !/"(0|1|2|3|4|5|6|7|8|9)":/.test(json) || man.layers.every((l) => !Array.isArray(l.w)));

    console.log(`  --- ${variant}: 指紋 ---`);
    const h = crypto.createHash('sha256');
    for (const x of [...m.convs, ...(m.glob?.convs ?? [])]) { h.update(u8(x.w)); h.update(u8(x.b)); }
    if (m.glob) h.update(u8(m.glob.fc.all));
    console.log(`  sha256(所有權重, 依層序) = ${h.digest('hex')}`);
}

console.log(`\n總體: ${pass} passed, ${fail} failed  ${fail === 0 ? '[PASS]' : '[FAIL]'}`);
process.exit(fail === 0 ? 0 : 1);
