// 動態批次掃描（dynamic effect sweep）— 移植品質閘門
//
// 用法：node test/effect_sweep.mjs [目錄=examples] [--json]
// 每個特效一個子行程（Dawn node polyfill 多特效累積會原生崩潰），驅動器彙總。
//
// 每特效檢查：
//   1. 編譯成功（WGFXRuntime，真實 WebGPU 裝置）
//   2. 兩張測試幀（平坦 + 決定性雜訊）全 pass dispatch
//   3. 無未捕捉 GPU 驗證錯誤（pushErrorScope）
//   4. 輸出非全零、無 NaN/Inf、worstChroma 統計
//   5. 耗時（編譯 + 每幀 dispatch）
import { create, globals } from 'webgpu';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawnSync } from 'child_process';
Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
const HERE = path.dirname(fileURLToPath(import.meta.url));
const SELF = fileURLToPath(import.meta.url);
const { WGFXRuntime } = await import('../dist/wgfx.esm.js');

const args = process.argv.slice(2);
const asJson = args.includes('--json');
const target = args.find(a => !a.startsWith('--')) || 'examples';
const root = path.isAbsolute(target) ? target : path.join(process.cwd(), target);

const files = [];
if (!process.argv.includes('--effect')) {
    (function walk(d) {
        for (const e of fs.readdirSync(d, { withFileTypes: true })) {
            const p = path.join(d, e.name);
            if (e.isDirectory()) walk(p);
            else if (e.name.endsWith('.wgsl') && !e.name.includes('debug')) files.push(p);
        }
    })(root);
} else {
    files.push(path.resolve(process.argv[process.argv.indexOf('--effect') + 1]));
}
files.sort();

const W = 128, H = 128;

function makeFrame(kind) {
    const px = new Uint8Array(W * H * 4);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const o = (y * W + x) * 4;
        if (kind === 'flat') { px[o] = 120; px[o + 1] = 90; px[o + 2] = 60; }
        else {
            const h = ((x * 374761393 + y * 668265263) ^ 0x5bf03635) >>> 0;
            px[o] = (h >>> 4) & 0xff; px[o + 1] = (h >>> 12) & 0xff; px[o + 2] = (h >>> 20) & 0xff;
        }
        px[o + 3] = 255;
    }
    return px;
}

const f16d = h => { const s = (h & 0x8000) ? -1 : 1, e = (h & 0x7c00) >> 10, m = h & 0x03ff; return e === 0 ? s * 2 ** -14 * (m / 1024) : e === 31 ? (m ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + m / 1024); };

async function sweepOne(relFile) {
    const adapter = await nav.gpu.requestAdapter();
    const dev = await adapter.requestDevice({ requiredLimits: { maxStorageTexturesPerShaderStage: 8 } });
    const t0 = performance.now();
    const code = fs.readFileSync(relFile, 'utf8');
    const r = new WGFXRuntime(dev);
    const U = GPUTextureUsage;
    const baseOpts = {
        defines: { INPUT_WIDTH: W, INPUT_HEIGHT: H },
        textures: { INPUT: { size: [W, H], format: 'rgba8unorm', usage: U.TEXTURE_BINDING | U.COPY_DST } }
    };
    try {
        await r.compile(code, baseOpts);
    } catch (e) {
        // 效果未宣告 OUTPUT 尺寸（低階 compile 需外部提供）→ 以輸入尺寸退回重試
        if (/OUTPUT.*WIDTH\/HEIGHT/.test(e?.message || '')) {
            await r.compile(code, {
                ...baseOpts,
                textures: {
                    ...baseOpts.textures,
                    OUTPUT: { size: [W, H], format: 'rgba16float', usage: U.STORAGE_BINDING | U.COPY_SRC | U.TEXTURE_BINDING }
                }
            });
        } else throw e;
    }
    const tCompile = performance.now() - t0;
    const rm = r.resourceManager;
    const outTex = rm.getTexture('OUTPUT');
    const OW = outTex ? outTex.width : W, OH = outTex ? outTex.height : H;
    rm.updateSceneBuffer(W, H, OW, OH);

    async function runFrame(kind) {
        dev.pushErrorScope('validation');
        const px = makeFrame(kind);
        dev.queue.writeTexture({ texture: rm.getTexture('INPUT') }, px,
            { bytesPerRow: W * 4, rowsPerImage: H }, [W, H, 1]);
        const t1 = performance.now();
        const cmd = dev.createCommandEncoder();
        r.dispatchAllPasses(cmd);
        dev.queue.submit([cmd.finish()]);
        await dev.queue.onSubmittedWorkDone();
        const tFrame = performance.now() - t1;
        const gpuErr = await dev.popErrorScope();
        if (gpuErr) throw new Error(`GPU 驗證錯誤: ${gpuErr.message}`);
        const bpr = Math.ceil(OW * 8 / 256) * 256;
        const rb = dev.createBuffer({ size: bpr * OH, usage: 0x8 | 0x1 });
        const cmd2 = dev.createCommandEncoder();
        cmd2.copyTextureToBuffer({ texture: outTex }, { buffer: rb, bytesPerRow: bpr, rowsPerImage: OH }, [OW, OH, 1]);
        dev.queue.submit([cmd2.finish()]);
        await rb.mapAsync(1);
        const u16 = new Uint16Array(rb.getMappedRange()).slice();
        rb.unmap(); rb.destroy();
        const stride = bpr / 2;
        let nonZero = 0, nonFinite = 0, worstChroma = 0;
        for (let i = 0; i < OW * OH; i++) {
            const v = [f16d(u16[i * 4]), f16d(u16[i * 4 + 1]), f16d(u16[i * 4 + 2])];
            if (!Number.isFinite(v[0]) || !Number.isFinite(v[1]) || !Number.isFinite(v[2])) { nonFinite++; continue; }
            if (v[0] !== 0 || v[1] !== 0 || v[2] !== 0) nonZero++;
            worstChroma = Math.max(worstChroma, v[0] - Math.max(v[1], v[2]), v[1] - Math.max(v[0], v[2]), v[2] - Math.max(v[0], v[1]));
        }
        return { tFrame, nonZero, nonFinite, worstChroma: +worstChroma.toFixed(3) };
    }

    const flat = await runFrame('flat');
    const noise = await runFrame('noise');
    dev.destroy?.();
    return {
        file: path.basename(relFile), passes: r.shaderInfo.passes.length,
        out: `${OW}x${OH}`, tCompile: Math.round(tCompile),
        flat: { ...flat, tFrame: +flat.tFrame.toFixed(2) },
        noise: { ...noise, tFrame: +noise.tFrame.toFixed(2) }
    };
}

if (process.argv.includes('--effect')) {
    const rel = process.argv[process.argv.indexOf('--effect') + 1];
    try {
        const res = await sweepOne(rel);
        console.log('@@JSON@@' + JSON.stringify(res));
    } catch (e) {
        console.log('@@JSON@@' + JSON.stringify({ file: path.basename(rel), error: (e?.message || String(e)).slice(0, 300) }));
    }
    process.exit(0);
}

// ---- 驅動器 ----
console.log(`掃描 ${files.length} 個特效（每特效子行程隔離）...\n`);
const results = [];
for (const f of files) {
    const p = spawnSync(process.execPath, [SELF, '--effect', f], { encoding: 'utf8', timeout: 120000 });
    const jsonLine = (p.stdout || '').split('\n').find(l => l.startsWith('@@JSON@@'));
    if (jsonLine) results.push(JSON.parse(jsonLine.slice(8)));
    else results.push({ file: path.basename(f), error: (p.stderr || p.stdout || 'no output').split('\n').filter(Boolean).slice(-2).join(' | ').slice(0, 200) });
    const last = results[results.length - 1];
    console.log(`  ${last.error ? '✗' : '✓'} ${last.file}${last.error ? ' — ' + last.error : ''}`);
}

console.log('\n' + '='.repeat(110));
console.log('特效'.padEnd(36) + 'pass'.padEnd(5) + '輸出'.padEnd(11) + '編譯ms'.padEnd(8) + '幀ms'.padEnd(8) + '非零'.padEnd(8) + 'NaN'.padEnd(5) + 'worstChroma');
let fail = 0;
for (const r of results) {
    if (r.error) { fail++; console.log(`${r.file.padEnd(36)} FAILED — ${r.error}`); continue; }
    const bad = r.nonFinite > 0 || r.flat.nonZero === 0 || r.noise.nonZero === 0;
    if (bad) fail++;
    console.log(
        (bad ? '⚠ ' : '  ') + r.file.padEnd(34) +
        String(r.passes).padEnd(5) + r.out.padEnd(11) +
        String(r.tCompile).padEnd(8) +
        `${r.flat.tFrame}/${r.noise.tFrame}`.padEnd(10) +
        `${r.flat.nonZero}/${r.noise.nonZero}`.padEnd(9) +
        `${r.flat.nonFinite}/${r.noise.nonFinite}`.padEnd(6) +
        r.noise.worstChroma
    );
}
console.log(`\n${fail === 0 ? '全部通過' : fail + ' 個失敗'} / ${results.length}`);
if (asJson) console.log('@@JSON@@' + JSON.stringify(results));
process.exit(fail === 0 ? 0 : 1);
