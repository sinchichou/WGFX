// split_passes.test.mjs — pass 自動拆分器驗證(tools/pack/split_passes.mjs)
//
// 三層驗證:
//   1. 單元測試(無 GPU):storage 數量偵測、門檻判斷、分塊、DESC 標記、
//      進入點重編號、白名單 throw 行為、merge/split 往返不變性。
//   2. 等價性(真跑 GPU):對 CuNNy-4x24-NVL / 4x32-NVL / 8x32-NVL 三檔——
//      「人工拆分版(現存 committed 檔)」merge 回未拆分 → 自動拆分 →
//      (a) 與人工版逐行比對;(b) 兩版以同一輸入真跑、OUTPUT 逐像素比對。
//   3. 驗收:自動拆分產物以 node test/port_check.mjs(一檔一 process)ALL PASS。
//
// 執行:node test/split_passes.test.mjs(需先 npm run build;編譯 CuNNy2 大檔較慢)
import 'webgpu';
import { create, globals } from 'webgpu';
Object.assign(globalThis, globals);
const nav = { gpu: create([]) };

import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.join(HERE, '..');
const DIST = path.join(REPO, 'dist', 'wgfx.esm.js');
const TMP = path.join(REPO, '.tmp');

const { parseEffect, storageTargetsOf, storeStatementsOf, splitEffect, mergeSplitPasses } =
    await import('file:///' + path.join(REPO, 'tools', 'pack', 'split_passes.mjs').replace(/\\/g, '/'));

const results = [];
function record(name, pass, detail = '') {
    results.push({ name, pass });
    console.log(`  [${pass ? 'PASS' : 'FAIL'}] ${name}${detail ? ' — ' + detail : ''}`);
}
async function runTest(name, fn) {
    try { await fn(); record(name, true); }
    catch (err) { record(name, false, (err && err.message) || String(err)); }
}
function assert(cond, msg) { if (!cond) throw new Error(msg); }
function assertThrows(fn, matchRe, label) {
    let thrown = null;
    try { fn(); } catch (e) { thrown = e; }
    assert(thrown !== null, `${label}: 未拋出錯誤`);
    if (matchRe) assert(matchRe.test(thrown.message), `${label}: 錯誤訊息不符 → ${thrown.message}`);
}

// ---- f16 解碼(readback 用) ----
function f16(h) {
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x3ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

// ---- 合成 mini 效果(單元測試用;不需可編譯,只需可被拆分器解析) ----
function miniEffect(storeTargets, { extraOut = null, selfRead = false, dispatch = false, storeShape = 'line' } = {}) {
    const storeLines = storeTargets.map((t, i) => {
        if (storeShape === 'multiline' && i === 0) {
            return `\ttextureStore(${t}, vec2<i32>(gxy),\n\t    v${i});`;
        }
        return `\tv${i} = max(v${i}, vec4<f32>(0.0));\n\ttextureStore(${t}, vec2<i32>(gxy), v${i});`;
    }).join('\n');
    const outList = [...storeTargets, ...(extraOut ? [extraOut] : [])];
    const read = selfRead ? `\tlet s = textureSampleLevel(${storeTargets[0].replace('_storaged', '')}_sampled, SP, uv, 0.0);\n` : '';
    return `//!VERSION 4
//!SORT_NAME mini
//!TEXTURE
//!WIDTH INPUT_WIDTH
//!HEIGHT INPUT_HEIGHT
Texture2D INPUT;
//!PASS 1
//!DESC base
//!BLOCK_SIZE 8
//!NUM_THREADS 64
//!IN INPUT
//!OUT ${outList.join(', ')}${dispatch ? '\n//!DISPATCH 4, 4' : ''}
fn Pass1(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let gxy = wg.xy * 8u + lid.xy;
    let uv = (vec2<f32>(gxy) + vec2<f32>(0.5)) * scene.inputPt;
${read}    var v0: vec4<f32> = vec4<f32>(0.0); var v1: vec4<f32> = vec4<f32>(0.0); var v2: vec4<f32> = vec4<f32>(0.0); var v3: vec4<f32> = vec4<f32>(0.0); var v4: vec4<f32> = vec4<f32>(0.0); var v5: vec4<f32> = vec4<f32>(0.0); var v6: vec4<f32> = vec4<f32>(0.0); var v7: vec4<f32> = vec4<f32>(0.0);
${storeLines}
}
`;
}
const SIX = ['A_storaged', 'B_storaged', 'C_storaged', 'D_storaged', 'E_storaged', 'F_storaged'];
const EIGHT = [...SIX, 'G_storaged', 'H_storaged'];

async function main() {
    console.log('=== pass 自動拆分器驗證 ===');
    fs.accessSync(DIST);

    // ===== 1. 單元測試 =====
    await runTest('U1: storageTargetsOf 依 //!OUT 順序偵測 _storaged 目標', () => {
        const { passes } = parseEffect(miniEffect(SIX));
        assert(storageTargetsOf(passes[0]).join(',') === SIX.join(','),
            `偵測結果: ${storageTargetsOf(passes[0]).join(',')}`);
    });

    await runTest('U2: 未達門檻(4 張)→ 原樣保留不拆分', () => {
        const four = SIX.slice(0, 4);
        const { text, report } = splitEffect(miniEffect(four));
        const { passes } = parseEffect(text);
        assert(passes.length === 1, `pass 數 ${passes.length}`);
        assert(report[0].action === 'keep', `action: ${report[0].action}`);
        assert(passes[0].desc === 'base', `DESC 被改動: ${passes[0].desc}`);
    });

    await runTest('U3: 6 張 → 拆 2 個子 pass([4,2]),DESC 標記 (part k/n),store 正確分組', () => {
        const { text } = splitEffect(miniEffect(SIX));
        const { passes } = parseEffect(text);
        assert(passes.length === 2, `pass 數 ${passes.length}`);
        assert(passes[0].desc === 'base (part 1/2)', `DESC1: ${passes[0].desc}`);
        assert(passes[1].desc === 'base (part 2/2)', `DESC2: ${passes[1].desc}`);
        assert(storageTargetsOf(passes[0]).join(',') === SIX.slice(0, 4).join(','), `OUT1: ${passes[0].out}`);
        assert(storageTargetsOf(passes[1]).join(',') === SIX.slice(4).join(','), `OUT2: ${passes[1].out}`);
        const s0 = storeStatementsOf(passes[0]).map((s) => s.target);
        const s1 = storeStatementsOf(passes[1]).map((s) => s.target);
        assert(s0.join(',') === SIX.slice(0, 4).join(','), `part1 stores: ${s0}`);
        assert(s1.join(',') === SIX.slice(4).join(','), `part2 stores: ${s1}`);
        // 進入點重編號
        assert(/^fn Pass1\(/m.test(passes[0].codeLines.join('\n')), 'part1 應含 fn Pass1(');
        assert(/^fn Pass2\(/m.test(passes[1].codeLines.join('\n')), 'part2 應含 fn Pass2(');
        assert(!/fn Pass\d+\(/.test(passes[1].codeLines.join('\n').replace(/fn Pass2\(/, '')), 'part2 不得殘留舊進入點名');
    });

    await runTest('U4: 8 張 → 拆 2 個子 pass([4,4])', () => {
        const { text } = splitEffect(miniEffect(EIGHT));
        const { passes } = parseEffect(text);
        assert(passes.length === 2, `pass 數 ${passes.length}`);
        assert(storageTargetsOf(passes[0]).length === 4, `part1: ${storageTargetsOf(passes[0])}`);
        assert(storageTargetsOf(passes[1]).length === 4, `part2: ${storageTargetsOf(passes[1])}`);
    });

    await runTest('U5: --max-storage-textures=3 → [3,3];=8 → 不拆', () => {
        const a = splitEffect(miniEffect(SIX), { maxStorageTextures: 3 });
        const pa = parseEffect(a.text).passes;
        assert(pa.length === 2 && storageTargetsOf(pa[0]).length === 3 && storageTargetsOf(pa[1]).length === 3,
            `limit=3 → ${pa.length} passes, [${storageTargetsOf(pa[0]).length}, ${storageTargetsOf(pa[1]).length}]`);
        const b = splitEffect(miniEffect(SIX), { maxStorageTextures: 8 });
        assert(parseEffect(b.text).passes.length === 1, 'limit=8 不應拆分');
    });

    await runTest('U6: 白名單——跨行 textureStore → throw', () => {
        assertThrows(() => splitEffect(miniEffect(SIX, { storeShape: 'multiline' })), /單行完整陳述式/, '跨行 store');
    });

    await runTest('U7: 白名單——//!OUT 有目標沒有 store(或反之)→ throw', () => {
        const missing = miniEffect(SIX).replace(/^\s*textureStore\(F_storaged,[^\n]*\n/m, '');
        assertThrows(() => splitEffect(missing), /沒有對應的 textureStore/, 'OUT 有但 store 缺');
        const extra = miniEffect(SIX).replace(
            'textureStore(F_storaged, vec2<i32>(gxy), v5);',
            'textureStore(Z_storaged, vec2<i32>(gxy), v5);');
        assertThrows(() => splitEffect(extra), /textureStore\(Z_storaged[,)]|不在 .*OUT 清單中|沒有對應的 textureStore/, 'store 有但 OUT 缺');
    });

    await runTest('U8: 白名單——pass 讀取自己要寫的紋理 → throw', () => {
        assertThrows(() => splitEffect(miniEffect(SIX, { selfRead: true })), /讀取.*_sampled/, '自我讀取');
    });

    await runTest('U9: 白名單——//!DISPATCH pass → throw', () => {
        assertThrows(() => splitEffect(miniEffect(SIX, { dispatch: true })), /DISPATCH/, 'DISPATCH');
    });

    await runTest('U10: 超限且 OUT 混非 _storaged → throw;未超限混 OUTPUT → 保留', () => {
        assertThrows(() => splitEffect(miniEffect(SIX, { extraOut: 'OUTPUT' })), /混有非 _storaged/, '超限混 OUTPUT');
        const four = miniEffect(SIX.slice(0, 4), { extraOut: 'OUTPUT' });
        const { passes } = parseEffect(splitEffect(four).text);
        assert(passes.length === 1, '未超限混 OUTPUT 不應拆分');
    });

    await runTest('U11: 往返不變性 split → merge → split 與單次 split 結果一致', () => {
        const once = splitEffect(miniEffect(EIGHT)).text;
        const back = mergeSplitPasses(once).text;
        const twice = splitEffect(back).text;
        assert(twice === once, 'split(merge(split(x))) != split(x)');
    });

    // ===== 2. 三檔等價性(merge 重建未拆分 → 自動拆分 → 比對) =====
    const bundle = await import('file:///' + DIST.replace(/\\/g, '/'));
    const { WGFXRuntime } = bundle;
    const adapter = await nav.gpu.requestAdapter();
    assert(adapter, '無法取得 WebGPU adapter');
    const device = await adapter.requestDevice();
    const uncaptured = [];
    device.onuncapturederror = (e) => uncaptured.push(String((e.error && e.error.message) || e));

    const SIZE = 256;
    const TARGETS = ['CuNNy-4x24-NVL', 'CuNNy-4x32-NVL', 'CuNNy-8x32-NVL'];
    fs.mkdirSync(TMP, { recursive: true });

    /** 以 port_check 同款流程真跑一個效果,回傳 OUTPUT 像素(Uint16 原值陣列) */
    async function runEffect(code) {
        const runtime = new WGFXRuntime(device);
        try {
            device.pushErrorScope('validation');
            await runtime.compile(code, {
                defines: { INPUT_WIDTH: SIZE, INPUT_HEIGHT: SIZE },
                textures: {
                    INPUT: {
                        size: [SIZE, SIZE], format: 'rgba8unorm',
                        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
                    },
                    OUTPUT: {
                        size: [SIZE, SIZE], format: 'rgba16float',
                        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
                    },
                },
            });
            const compileErr = await device.popErrorScope();
            if (compileErr) throw new Error('compile: ' + compileErr.message);
            const rm = runtime.resourceManager;
            const outTex = rm.getTexture('OUTPUT');
            const ow = outTex.width, oh = outTex.height;
            rm.updateSceneBuffer(SIZE, SIZE, ow, oh);
            // 與 port_check 完全相同的輸入圖樣
            const data = new Uint8Array(SIZE * SIZE * 4);
            for (let y = 0; y < SIZE; y++) {
                for (let x = 0; x < SIZE; x++) {
                    const i = (y * SIZE + x) * 4;
                    const v = (x * 7 + y * 13 + ((x * y) % 97)) % 256;
                    data[i] = v; data[i + 1] = 255 - v; data[i + 2] = (v * 3) % 256; data[i + 3] = 255;
                }
            }
            device.queue.writeTexture({ texture: rm.getTexture('INPUT') }, data,
                { bytesPerRow: SIZE * 4, rowsPerImage: SIZE }, [SIZE, SIZE, 1]);
            device.pushErrorScope('validation');
            const enc = device.createCommandEncoder();
            for (const p of runtime.shaderInfo.passes) runtime.dispatchPass(`PASS_${p.index}`, enc);
            device.queue.submit([enc.finish()]);
            const runErr = await device.popErrorScope();
            if (runErr) throw new Error('dispatch: ' + runErr.message);
            // 讀回(單次 mapAsync)
            const bytesPerRow = Math.ceil((ow * 8) / 256) * 256;
            const staging = device.createBuffer({
                size: bytesPerRow * oh,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });
            const enc2 = device.createCommandEncoder();
            enc2.copyTextureToBuffer({ texture: outTex }, { buffer: staging, bytesPerRow, rowsPerImage: oh }, [ow, oh, 1]);
            device.queue.submit([enc2.finish()]);
            await staging.mapAsync(GPUMapMode.READ);
            const raw = new Uint16Array(staging.getMappedRange().slice(0));
            staging.unmap();
            staging.destroy();
            return { raw, ow, oh, passes: runtime.shaderInfo.passes.length };
        } finally {
            runtime.resourceManager.dispose();
        }
    }

    for (const name of TARGETS) {
        await runTest(`E1[${name}]: merge 重建未拆分 → 自動拆分,每個 pass storage ≤ 4`, () => {
            const committed = fs.readFileSync(path.join(REPO, 'effects', 'CuNNy2', `${name}.wgsl`), 'utf8');
            const merged = mergeSplitPasses(committed);
            assert(merged.report.length >= 1, `merge 未合併任何群組(${merged.report.length})`);
            const { passes } = parseEffect(merged.text);
            // 重建的未拆分版:每個合併群組的 storage 數 = 原始 conv pass 的 6–8 張
            const overLimit = passes.filter((p) => storageTargetsOf(p).length > 4);
            assert(overLimit.length >= 1, `重建的未拆分版應有超過 4 張 storage 的 pass(實際 ${overLimit.length})`);
            const auto = splitEffect(merged.text);
            for (const p of auto.passes) {
                const n = storageTargetsOf(p).length;
                assert(n <= 4, `拆分後 pass「${p.desc}」仍有 ${n} 張 storage texture`);
            }
        });

        await runTest(`E2[${name}]: 自動拆分結果與人工拆分版**逐行一致**`, () => {
            const committed = fs.readFileSync(path.join(REPO, 'effects', 'CuNNy2', `${name}.wgsl`), 'utf8')
                .replace(/\r\n/g, '\n');
            const merged = mergeSplitPasses(committed.replace(/\r\n/g, '\n'));
            const auto = splitEffect(merged.text).text;
            const a = auto.split('\n');
            const b = committed.split('\n');
            let diffs = 0;
            for (let i = 0; i < Math.max(a.length, b.length); i++) {
                if (a[i] !== b[i]) diffs++;
            }
            assert(diffs === 0, `逐行比對有 ${diffs} 行差異(auto=${a.length} 行,manual=${b.length} 行)`);
        });

        await runTest(`E3[${name}]: 自動拆分 vs 人工版真跑 OUTPUT **逐像素相同**`, async () => {
            const committed = fs.readFileSync(path.join(REPO, 'effects', 'CuNNy2', `${name}.wgsl`), 'utf8');
            const merged = mergeSplitPasses(committed);
            const auto = splitEffect(merged.text).text;
            const manual = await runEffect(committed);
            const autoRun = await runEffect(auto);
            assert(manual.ow === autoRun.ow && manual.oh === autoRun.oh,
                `OUTPUT 尺寸不符: ${manual.ow}x${manual.oh} vs ${autoRun.ow}x${autoRun.oh}`);
            assert(manual.raw.length === autoRun.raw.length, `讀回長度不符`);
            // 逐像素(f16 精度)比對:兩版數學完全相同 → 期望 0 差異
            let maxDiff = 0, diffCount = 0, nanCount = 0;
            for (let i = 0; i < manual.raw.length; i++) {
                const a = f16(manual.raw[i]);
                const b = f16(autoRun.raw[i]);
                if (Number.isNaN(a) || Number.isNaN(b)) { nanCount++; continue; }
                const d = Math.abs(a - b);
                if (d > maxDiff) maxDiff = d;
                if (d > 1e-6) diffCount++;
            }
            assert(nanCount === 0, `出現 ${nanCount} 個 NaN 分量`);
            assert(diffCount === 0, `${diffCount} 個像素分量差異超過 1e-6(最大差 ${maxDiff})`);
            console.log(`        max|Δ| = ${maxDiff},像素分量數 = ${manual.raw.length}`);
        });

        await runTest(`E4[${name}]: 自動拆分產物 port_check ALL PASS(一檔一 process)`, () => {
            const committed = fs.readFileSync(path.join(REPO, 'effects', 'CuNNy2', `${name}.wgsl`), 'utf8');
            const auto = splitEffect(mergeSplitPasses(committed).text).text;
            const outPath = path.join(TMP, `split-check-${name}.wgsl`);
            fs.writeFileSync(outPath, auto);
            const out = execFileSync(process.execPath, ['test/port_check.mjs', outPath, '--quiet'], {
                cwd: REPO, encoding: 'utf8',
            });
            assert(/ALL PASS/.test(out), `port_check 未 ALL PASS:\n${out}`);
        });
    }

    if (uncaptured.length) {
        record('GPU 未捕捉錯誤檢查', false, uncaptured.join(' | '));
    } else {
        record('GPU 未捕捉錯誤檢查', true);
    }

    const passed = results.filter((r) => r.pass).length;
    console.log(`\n=== 總結: ${passed}/${results.length} tests passed ===`);
    device.destroy?.();
    process.exit(passed === results.length ? 0 : 1);
}

try {
    await main();
} catch (err) {
    console.error('測試套件執行失敗:', err);
    process.exit(1);
}
