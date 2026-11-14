// test/bench_cnn_caseE.mjs — WGFX CNN 效能基準（審計 M1/M2，見 docs/audit/perf-audit.md §5）
//
// 執行：node test/bench_cnn_caseE.mjs [--size N] [--rounds N] [--frames N] [--effect <rel>] [--no-upload]
//
// 目的：把「每幀真實 GPU 時間」與「框架 encode+submit CPU 時間」分開量，
//       判定 CuNNy 掉 FPS 究竟是 GPU bound 還是框架 overhead 主導。
//
// 量測項（每幀 = 上傳 INPUT → 單一 encoder 逐 pass dispatch → 一次 submit）：
//   1. encode+submit only   : 只量 CPU 端（createCommandEncoder → dispatchAllPasses → finish → submit），不等 GPU
//   2. total frame time      : 每幀 await queue.onSubmittedWorkDone() 的完整牆鐘（含上傳與同步延遲）
//   3. GPU saturated         : N 幀連續 submit 後只 await 一次（飽和吞吐 = N 幀總時間 / N）
//   4. GPU execution         : timestamp-query 量到的 GPU 端執行時間（整幀 span + 逐 pass）
//   5. framework overhead    : total frame time − GPU execution（CPU 上傳 + encode + 等待延遲）
//
// 裝置建立與效果載入沿用 test/bench_hotpath.mjs 的做法（webgpu polyfill + dist/wgfx.esm.js）。
// 已知限制：node-webgpu 的 Dawn 傳輸層在「單一 command buffer 極大」或「大量 writeTexture 突發」
//          時會丟出無訊息的 Error（實測於 --size >= 512）；預設 256×256 穩定可重現，
//          大尺寸請用 --rounds/--frames 降低突發量。
import { create, globals } from 'webgpu';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

Object.assign(globalThis, globals);
const nav = { gpu: create([]) };
// 讓非同步的 polyfill 例外也能印出可讀訊息（否則只會看到 "<anonymous_script>:0 [Error]"）
process.on('uncaughtException', (e) => {
    console.error('FATAL (uncaught):', (e && (e.stack || e.message)) || e);
    console.error('若此錯誤出現在較大尺寸（--size >= 512），多半是 node-webgpu/Dawn 的傳輸層在大量 '
        + 'command buffer / 大筆 writeTexture 下失穩：請改用預設 --size 256，或降低 --rounds / --frames。');
    process.exit(1);
});
process.on('unhandledRejection', (e) => {
    console.error('FATAL (unhandled):', (e && (e.stack || e.message)) || e);
    process.exit(1);
});
const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.join(HERE, '..');
const { WGFXRuntime, Logger, LogLevel } = await import('../dist/wgfx.esm.js');
// 只保留錯誤輸出，讓基準表格乾淨（Logger 預設會印 INFO）。
Logger.setLevel(LogLevel.ERROR);

// ---------------------------------------------------------------- CLI
const argv = process.argv.slice(2);
let SIZE = 256;
let ROUNDS = 5;
let FRAMES = 30;
let EFFECT_REL = 'effects/CuNNy/CuNNy-2x4C-NVL.wgsl';
let UPLOAD = true;
let ADAPTER = 'auto';
for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--size') SIZE = Math.max(16, Number(argv[++i]) | 0);
    else if (argv[i] === '--rounds') ROUNDS = Math.max(1, Number(argv[++i]) | 0);
    else if (argv[i] === '--frames') FRAMES = Math.max(1, Number(argv[++i]) | 0);
    else if (argv[i] === '--effect') EFFECT_REL = argv[++i];
    else if (argv[i] === '--no-upload') UPLOAD = false;
    else if (argv[i] === '--adapter') ADAPTER = argv[++i];
}

// ---------------------------------------------------------------- effect file
// 優先使用 effects/CuNNy/CuNNy-2x4C-NVL.wgsl；若不存在，改用 effects/CuNNy/ 中
// 實際存在的最小 .wgsl（不自行捏造效果檔），並在輸出中標明實際使用的檔案。
function pickEffect() {
    const primary = path.join(REPO, EFFECT_REL);
    if (fs.existsSync(primary)) {
        return { rel: EFFECT_REL.replace(/\\/g, '/'), abs: primary, fallback: false };
    }
    const dir = path.join(REPO, 'effects', 'CuNNy');
    const files = fs.readdirSync(dir)
        .filter((f) => f.toLowerCase().endsWith('.wgsl'))
        .map((f) => {
            const abs = path.join(dir, f);
            return { rel: `effects/CuNNy/${f}`, abs, size: fs.statSync(abs).size, fallback: true };
        })
        .sort((a, b) => a.size - b.size);
    if (files.length === 0) throw new Error(`No .wgsl effect found in ${dir}`);
    return files[0];
}
const effect = pickEffect();
const shaderText = fs.readFileSync(effect.abs, 'utf8');

// ---------------------------------------------------------------- device
// --adapter auto | high-performance | low-power：本機雙 GPU（RTX 4060 dGPU /
// Radeon 780M iGPU）以 powerPreference 切換；auto 交由驅動決定。
const adapter = await nav.gpu.requestAdapter(
    ADAPTER === 'auto' ? {} : { powerPreference: ADAPTER });
if (!adapter) {
    console.error(`FATAL: requestAdapter(${ADAPTER}) returned null — no GPU adapter available `
        + `(--adapter ${ADAPTER} 不存在？可用值：auto | high-performance | low-power).`);
    process.exit(2);
}
const adapterInfo = adapter.info || {};
const hasTimestamp = adapter.features.has('timestamp-query');
const al = adapter.limits;
const dev = await adapter.requestDevice({
    requiredFeatures: hasTimestamp ? ['timestamp-query'] : [],
    requiredLimits: {
        maxStorageTexturesPerShaderStage: Math.min(8, al.maxStorageTexturesPerShaderStage),
        maxComputeWorkgroupStorageSize: Math.min(32768, al.maxComputeWorkgroupStorageSize),
    },
});
const uncaptured = [];
dev.onuncapturederror = (e) => uncaptured.push(String((e.error && e.error.message) || e));
if (dev.lost) dev.lost.then((info) => console.error(`[GPU device lost] reason=${info.reason} message=${info.message}`));

// ---------------------------------------------------------------- compile effect
const W = SIZE, H = SIZE;
const externalResources = {
    defines: { INPUT_WIDTH: W, INPUT_HEIGHT: H, NW: W, NH: H },
    textures: {
        INPUT: {
            size: [W, H], format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        },
        OUTPUT: {
            // CuNNy 系列是 2× 上取樣；honorDeclaredOutput 讓效果檔宣告優先。
            size: [2 * W, 2 * H], format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
        },
    },
};
const rt = new WGFXRuntime(dev);
await rt.compile(shaderText, externalResources, { honorDeclaredOutput: true });
const outTex = rt.resourceManager.getTexture('OUTPUT');
const inTex = rt.resourceManager.getTexture('INPUT');
if (!outTex || !inTex) throw new Error('INPUT/OUTPUT texture missing after compile');
rt.resourceManager.updateSceneBuffer(W, H, outTex.width, outTex.height);

// ---------------------------------------------------------------- helpers
function stats(vals) {
    const s = [...vals].sort((a, b) => a - b);
    const med = s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
    return { min: s[0], med, max: s[s.length - 1] };
}
const f3 = (v) => v.toFixed(3);
const fInt = (v) => Math.round(v).toString();

// 合成輸入影像（256×256 RGBA8），每幀上傳一次以模擬 copyExternalImageToTexture。
const inputData = new Uint8Array(W * H * 4);
for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        inputData[i] = x & 255; inputData[i + 1] = y & 255;
        inputData[i + 2] = (x ^ y) & 255; inputData[i + 3] = 255;
    }
}
function uploadInput() {
    // Node 無 ImageBitmap/VideoFrame；queue.writeTexture 是同類的 CPU→GPU 像素上傳。
    dev.queue.writeTexture(
        { texture: inTex }, inputData,
        { bytesPerRow: W * 4, rowsPerImage: H },
        { width: W, height: H }
    );
}
function encodeAndSubmit() {
    const e = dev.createCommandEncoder({ label: 'WGFX Frame Processing' });
    rt.dispatchAllPasses(e);
    dev.queue.submit([e.finish()]);
}
function fullFrame() {
    if (UPLOAD) uploadInput();
    encodeAndSubmit();
}

// ---------------------------------------------------------------- M0: sync baseline (empty submit + await)
// 量出「每幀 await queue.onSubmittedWorkDone()」本身的喚醒延遲（與 CNN 無關的固定成本）。
async function measureSyncBaseline() {
    const perRound = [];
    for (let r = 0; r < ROUNDS; r++) {
        const t0 = performance.now();
        for (let i = 0; i < FRAMES; i++) {
            const e = dev.createCommandEncoder();
            dev.queue.submit([e.finish()]);
            await dev.queue.onSubmittedWorkDone();
        }
        const t1 = performance.now();
        perRound.push((t1 - t0) / FRAMES);
    }
    return stats(perRound);
}

// ---------------------------------------------------------------- M1: encode+submit only (no GPU wait)
async function measureEncodeSubmit() {
    encodeAndSubmit();
    await dev.queue.onSubmittedWorkDone();
    const perRound = [];
    for (let r = 0; r < ROUNDS; r++) {
        const t0 = performance.now();
        for (let i = 0; i < FRAMES; i++) encodeAndSubmit();
        const t1 = performance.now();
        perRound.push((t1 - t0) / FRAMES);
        await dev.queue.onSubmittedWorkDone(); // drain，避免下一輪被 GPU 背壓影響
    }
    return stats(perRound);
}

// ---------------------------------------------------------------- M2: real frame time (await per frame)
async function measureRealFrame() {
    for (let i = 0; i < 3; i++) { fullFrame(); await dev.queue.onSubmittedWorkDone(); }
    const perRound = [], uploadPerRound = [];
    for (let r = 0; r < ROUNDS; r++) {
        const t0 = performance.now();
        let upNs = 0;
        for (let i = 0; i < FRAMES; i++) {
            const u0 = performance.now();
            if (UPLOAD) uploadInput();
            upNs += performance.now() - u0;
            const e = dev.createCommandEncoder({ label: 'WGFX Frame Processing' });
            rt.dispatchAllPasses(e);
            dev.queue.submit([e.finish()]);
            await dev.queue.onSubmittedWorkDone();
        }
        const t1 = performance.now();
        perRound.push((t1 - t0) / FRAMES);
        uploadPerRound.push(upNs / FRAMES);
    }
    return { frame: stats(perRound), upload: stats(uploadPerRound) };
}

// ---------------------------------------------------------------- M3: saturated throughput (N frames / N)
async function measureSaturated() {
    for (let i = 0; i < 3; i++) { fullFrame(); await dev.queue.onSubmittedWorkDone(); }
    // 一次掛太多幀會讓 Dawn 的 wire 傳輸緩衝區爆掉（大尺寸 + writeTexture 尤其明顯），
    // 故把 burst 長度限制在 ~8MB 的上傳量以內。
    const burst = Math.max(4, Math.min(FRAMES, Math.floor(8_000_000 / (W * H * 4))));
    const perRound = [];
    for (let r = 0; r < ROUNDS; r++) {
        const t0 = performance.now();
        for (let i = 0; i < burst; i++) fullFrame();
        await dev.queue.onSubmittedWorkDone();
        const t1 = performance.now();
        perRound.push((t1 - t0) / burst);
    }
    return stats(perRound);
}

// ---------------------------------------------------------------- M4: timestamp-query per-pass GPU time
// 逐 pass 的 compute pass 以 timestampWrites 包住；PipelineManager.dispatchPass 會自行
// beginComputePass()（無法注入 timestamp），故此處以已建立的 pipeline/bindGroup 重放
// 相同的 dispatch（CuNNy 無 //! DISPATCH，網格由 BLOCK_SIZE 推導）。
function passLabels(text, passes) {
    const desc = new Map();
    let cur = null;
    for (const line of text.split(/\r?\n/)) {
        const pm = /^\s*\/\/!PASS\s+(\d+)\s*$/.exec(line);
        if (pm) { cur = Number(pm[1]); if (!desc.has(cur)) desc.set(cur, ''); continue; }
        const dm = /^\s*\/\/!DESC\s+(.*)$/.exec(line);
        if (dm && cur !== null && !desc.get(cur)) desc.set(cur, dm[1].trim());
    }
    return passes.map((p) => `PASS_${p.index}${desc.get(p.index) ? ' ' + desc.get(p.index) : ''}`);
}

async function measureTimestamps() {
    const passes = rt.shaderInfo.passes;
    const n = passes.length;
    if (n === 0) return null;
    if (passes.some((p) => p.dispatch && p.dispatch.length)) {
        console.log('note: explicit //! DISPATCH present — per-pass timestamp breakdown skipped.');
        return null;
    }
    // 暖機：讓 bindGroup / 網格快取就緒
    encodeAndSubmit();
    await dev.queue.onSubmittedWorkDone();

    const labels = passLabels(shaderText, passes);
    // D3D12 的 timestamp 計數器解析度只有 65536 ns（1 tick），單一 pass（~10~30µs）
    // 低於一個 tick 會被量化成 0。因此同一個 command buffer 內重複 K 次 dispatch，
    // 以「總 span / K」取得遠細於 1 tick 的平均時間（仍以 timestamp-query 量測）。
    // K 以「總像素工作量 ≤ 12M」為上限；且每個 command buffer 只放 CHUNK 次重複
    // （單一 command buffer 太大時 Dawn 會直接丟出無訊息的 Error）。
    const K = Math.max(4, Math.min(64, Math.floor(12_000_000 / (W * H))));
    const CHUNK = 8;
    const querySet = dev.createQuerySet({ type: 'timestamp', count: 2 });
    const resolveBuf = dev.createBuffer({
        size: 256, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const readBuf = dev.createBuffer({
        size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    function encodePass(enc, passIndex, ts) {
        const pipeline = rt.pipelineManager.pipelines.get(passIndex);
        if (!pipeline) throw new Error(`pipeline for pass ${passIndex} missing`);
        const pe = enc.beginComputePass(ts ? { timestampWrites: ts } : undefined);
        pe.setPipeline(pipeline.computePipeline);
        pe.setBindGroup(0, pipeline.bindGroup);
        pe.dispatchWorkgroups(pipeline.gridDx, pipeline.gridDy, 1);
        pe.end();
    }

    // 送出一段 command buffer，回傳兩個 timestamp 之間的毫秒數
    async function timeSpan(encodeFn) {
        const enc = dev.createCommandEncoder({ label: 'caseE timestamp span' });
        encodeFn(enc);
        dev.queue.submit([enc.finish()]);
        await dev.queue.onSubmittedWorkDone();
        const enc2 = dev.createCommandEncoder();
        enc2.resolveQuerySet(querySet, 0, 2, resolveBuf, 0);
        enc2.copyBufferToBuffer(resolveBuf, 0, readBuf, 0, 16);
        dev.queue.submit([enc2.finish()]);
        await readBuf.mapAsync(GPUMapMode.READ);
        // D3D12 timestamp 帶很大的 epoch offset（> 2^53），先在 BigInt 域做差再轉 Number。
        const raw = new BigInt64Array(readBuf.getMappedRange().slice(0));
        readBuf.unmap();
        return Number(raw[1] - raw[0]) / 1e6;
    }

    // 把 K 次重複切成 CHUNK 個 command buffer，各自量 span 後相加
    async function timeSpanChunked(makeChunkFn) {
        let total = 0;
        for (let c = 0; c < K; c += CHUNK) {
            const count = Math.min(CHUNK, K - c);
            total += await timeSpan((enc) => makeChunkFn(enc, c, count));
        }
        return total;
    }

    // (a) 整幀 GPU 時間：K 個完整 4-pass 幀連續執行（純 GPU 吞吐，不含 CPU）
    const frameSamples = [];
    for (let r = 0; r < ROUNDS; r++) {
        if (process.env.WGFX_TRACE) console.error(`[trace] ts frame round ${r} (K=${K})`);
        const total = await timeSpanChunked((enc, c, count) => {
            for (let k = 0; k < count; k++) {
                for (let i = 0; i < n; i++) {
                    const ts = (k === 0 && i === 0)
                        ? { querySet, beginningOfPassWriteIndex: 0 }
                        : (k === count - 1 && i === n - 1)
                            ? { querySet, endOfPassWriteIndex: 1 }
                            : undefined;
                    encodePass(enc, passes[i].index, ts);
                }
            }
        });
        frameSamples.push(total / K);
    }

    // (b) 逐 pass GPU 時間：只重複 dispatch 該 pass K 次
    const passSamples = [];
    for (let i = 0; i < n; i++) {
        const s = [];
        for (let r = 0; r < ROUNDS; r++) {
            if (process.env.WGFX_TRACE) console.error(`[trace] ts pass ${i} round ${r}`);
            const total = await timeSpanChunked((enc, c, count) => {
                for (let k = 0; k < count; k++) {
                    const ts = (k === 0)
                        ? { querySet, beginningOfPassWriteIndex: 0 }
                        : (k === count - 1)
                            ? { querySet, endOfPassWriteIndex: 1 }
                            : undefined;
                    encodePass(enc, passes[i].index, ts);
                }
            });
            s.push(total / K);
        }
        passSamples.push(s);
    }

    resolveBuf.destroy();
    readBuf.destroy();
    querySet.destroy();
    const passStats = passSamples.map((v) => stats(v));
    // sum = 各 pass 的時間相加（不是 median of medians）
    const sumStats = {
        min: passStats.reduce((a, s) => a + s.min, 0),
        med: passStats.reduce((a, s) => a + s.med, 0),
        max: passStats.reduce((a, s) => a + s.max, 0),
    };
    const gpu = stats(frameSamples);
    return { labels, passStats, gpu, sumStats, fromFrameSpan: true };
}

// ---------------------------------------------------------------- run
const adapterDesc = [
    adapterInfo.vendor || 'unknown', adapterInfo.architecture || '',
    adapterInfo.device || adapterInfo.description || '',
].filter(Boolean).join('/');
console.log('=== WGFX CNN case E benchmark ===');
console.log(`effect: ${effect.rel} (${rt.shaderInfo.passes.length} passes)${effect.fallback ? ' [fallback: primary missing]' : ''} | ${W}x${H} -> ${outTex.width}x${outTex.height} | rounds=${ROUNDS} frames=${FRAMES}`);
console.log(`adapter: ${adapterDesc} | timestamp-query=${hasTimestamp ? 'on' : 'off'} shader-f16=${adapter.features.has('shader-f16') ? 'on' : 'off'} | upload=${UPLOAD ? 'writeTexture' : 'off'}`);

const encStats = await measureEncodeSubmit();
if (process.env.WGFX_TRACE) console.error('[trace] encode+submit done');
const syncStats = await measureSyncBaseline();
if (process.env.WGFX_TRACE) console.error('[trace] sync baseline done');
const real = await measureRealFrame();
if (process.env.WGFX_TRACE) console.error('[trace] real frame done');
const satStats = await measureSaturated();
if (process.env.WGFX_TRACE) console.error('[trace] saturated done');
let ts = null;
if (hasTimestamp) {
    try { ts = await measureTimestamps(); }
    catch (e) { console.log(`note: timestamp-query measurement failed (${e.message}); skipping per-pass breakdown`); }
} else {
    console.log('timestamp-query unavailable, skipping per-pass breakdown');
}

const rows = [];
rows.push(['encode+submit only (pure framework CPU)', encStats, false]);
rows.push(['sync baseline (empty submit + await)', syncStats, true]);
rows.push(['total frame time/frame (await + onDone)', real.frame, true]);
rows.push(['GPU saturated (N frames submitted / N)', satStats, true]);
if (ts) rows.push(['GPU execution (timestamp-query, K-repeated)', ts.gpu, true]);
console.log('metric                                          min(ms)  med(ms)   FPS(min)  FPS(med)');
for (const [label, s, withFps] of rows) {
    console.log(`${label.padEnd(46)} ${f3(s.min).padStart(7)} ${f3(s.med).padStart(8)}   ${withFps ? fInt(1000 / s.min).padStart(8) : '       -'}  ${withFps ? fInt(1000 / s.med).padStart(8) : '      -'}`);
}
const diff = { min: real.frame.min - encStats.min, med: real.frame.med - encStats.med };
console.log(`${'difference (total frame - encode+submit)'.padEnd(46)} ${f3(diff.min).padStart(7)} ${f3(diff.med).padStart(8)}          -         -`);
if (ts) {
    const ovh = { min: real.frame.min - ts.gpu.min, med: real.frame.med - ts.gpu.med };
    console.log(`${'framework overhead (total frame - GPU exec)'.padEnd(46)} ${f3(ovh.min).padStart(7)} ${f3(ovh.med).padStart(8)}          -         -`);
    const per = ts.labels.map((l, i) => `${l}=${f3(ts.passStats[i].med)}`).join(' | ');
    console.log(`per-pass GPU ms (timestamp-query, median): ${per} | sum=${f3(ts.sumStats.med)}`);
}
console.log(`upload CPU (queue.writeTexture) per frame: ${f3(real.upload.med)} ms (included in total frame time)`);

if (uncaptured.length > 0) {
    console.error(`[GPU uncaptured error] ${uncaptured.join(' | ')}`);
    process.exit(1);
}
console.log('=== done ===');
process.exit(0);
