// diff_core.test.mjs — Pure-Node tests for src/verification/diff-core.ts.
//
// No GPU, no `webgpu` polyfill and no build step are required. The TypeScript
// source is transpiled in memory with the already-installed `typescript`
// dev-dependency and imported from a `data:` URL (fallback: Node's native
// `--experimental-strip-types`).

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import assert from 'node:assert/strict';

const DIFF_CORE_URL = new URL('../src/verification/diff-core.ts', import.meta.url);

/**
 * Load the TypeScript module without touching dist/ or running a build.
 */
async function loadDiffCore() {
    const ts = (await import('typescript')).default;
    const source = readFileSync(DIFF_CORE_URL, 'utf8');
    const { outputText } = ts.transpileModule(source, {
        fileName: fileURLToPath(DIFF_CORE_URL),
        compilerOptions: {
            target: ts.ScriptTarget.ES2022,
            module: ts.ModuleKind.ESNext,
        },
    });
    const dataUrl = 'data:text/javascript;base64,' + Buffer.from(outputText, 'utf8').toString('base64');
    return import(dataUrl);
}

let diffCore;
if (process.env.DIFF_CORE_NATIVE_TS === '1') {
    diffCore = await import('../src/verification/diff-core.ts');
} else {
    try {
        diffCore = await loadDiffCore();
    } catch {
        // Fallback: re-run this file with Node's built-in TypeScript stripping.
        const child = spawnSync(
            process.execPath,
            ['--experimental-strip-types', '--no-warnings', fileURLToPath(import.meta.url)],
            { stdio: 'inherit', env: { ...process.env, DIFF_CORE_NATIVE_TS: '1' } }
        );
        process.exit(child.status ?? 1);
    }
}

const { comparePixels, generateDiffHeatmap } = diffCore;

let passed = 0;
let failed = 0;

function check(name, fn) {
    try {
        fn();
        passed++;
        console.log(`  ✅ ${name}`);
    } catch (error) {
        failed++;
        console.error(`  ❌ ${name}\n     ${error.message}`);
    }
}

console.log('\n🧪 diff-core pure-Node tests (no GPU, no webgpu polyfill)\n');

// ---------------------------------------------------------------------------
// 1) Two identical images.
// ---------------------------------------------------------------------------
check('identical images → MAE=0, PSNR=Infinity, diff ratio=0', () => {
    const image = new Uint8Array([
        0, 64, 128, 255,
        10, 20, 30, 40,
        200, 150, 100, 50,
    ]);

    const metrics = comparePixels(image, image, 3, 1);

    assert.equal(metrics.mae, 0);
    assert.equal(metrics.maxAbsDiff, 0);
    assert.equal(metrics.psnr, Infinity);
    assert.equal(metrics.diffPixelCount, 0);
    assert.equal(metrics.diffPixelRatio, 0);
    assert.equal(metrics.diffPixelPercent, 0);
    assert.equal(metrics.perChannelMAE.R, 0);
    assert.equal(metrics.perChannelMAE.G, 0);
    assert.equal(metrics.perChannelMAE.B, 0);
    assert.equal(metrics.perChannelMAE.A, 0);
});

// ---------------------------------------------------------------------------
// 2) Two images with a known difference.
//
//    2x2 pixels, RGBA order:
//      pixel 0: ref [10,20,30,40]  test [10,20,30,44] → only A differs by 4
//      pixel 1: identical
//      pixel 2: identical
//      pixel 3: ref [130,140,150,160] test [130,140,150,100] → A differs by 60
//
//    sum(|diff|) = 4 + 60 = 64 over 4 pixels * 4 channels = 16 samples
//    MAE = 64 / 16 = 4, max abs diff = 60, differing pixels = 2 / 4 = 50%
//    MSE = (4^2 + 60^2) / 16 = 3616 / 16 = 226
// ---------------------------------------------------------------------------
check('known differences → MAE=4, max=60, diff=50%', () => {
    const reference = new Uint8Array([
        10, 20, 30, 40,
        50, 60, 70, 80,
        90, 100, 110, 120,
        130, 140, 150, 160,
    ]);
    const test = new Uint8Array([
        10, 20, 30, 44,
        50, 60, 70, 80,
        90, 100, 110, 120,
        130, 140, 150, 100,
    ]);

    const metrics = comparePixels(reference, test, 2, 2);

    assert.equal(metrics.mae, 4);
    assert.equal(metrics.maxAbsDiff, 60);
    assert.equal(metrics.diffPixelCount, 2);
    assert.equal(metrics.diffPixelRatio, 0.5);
    assert.equal(metrics.diffPixelPercent, 50);
    assert.equal(metrics.perChannelMAE.A, 16);
    assert.equal(metrics.perChannelMAE.R, 0);
    assert.equal(metrics.perChannelMAE.G, 0);
    assert.equal(metrics.perChannelMAE.B, 0);
    assert.equal(metrics.perChannelMaxAbsDiff.A, 60);
    assert.equal(metrics.mse, 226);
    assert.equal(metrics.rmse, Math.sqrt(226));

    const expectedPsnr = 20 * Math.log10(255 / Math.sqrt(226));
    assert.ok(Math.abs(metrics.psnr - expectedPsnr) < 1e-10);

    // Uint8ClampedArray inputs must produce the exact same numbers.
    const clamped = comparePixels(
        new Uint8ClampedArray(reference),
        new Uint8ClampedArray(test),
        2,
        2
    );
    assert.deepEqual(clamped, metrics);
});

// ---------------------------------------------------------------------------
// 3) Heatmap: 3x3 image, only the center pixel differs (R channel, delta 200).
// ---------------------------------------------------------------------------
check('heatmap → 3x3 RGBA output, center red = 200', () => {
    const reference = new Uint8Array(3 * 3 * 4);
    const test = new Uint8Array(3 * 3 * 4);
    const center = (1 * 3 + 1) * 4;
    test[center] = 200; // center pixel, red channel

    const heatmap = generateDiffHeatmap(reference, test, 3, 3);

    assert.ok(heatmap instanceof Uint8Array);
    assert.equal(heatmap.length, 3 * 3 * 4);

    assert.equal(heatmap[center], 200);
    assert.equal(heatmap[center + 1], 0);
    assert.equal(heatmap[center + 2], 0);
    assert.equal(heatmap[center + 3], 255);

    // Every other pixel must be black (but opaque).
    for (let pixel = 0; pixel < 9; pixel++) {
        if (pixel === 4) continue;
        const base = pixel * 4;
        assert.equal(heatmap[base], 0);
        assert.equal(heatmap[base + 3], 255);
    }
});

// ---------------------------------------------------------------------------
// 4) Guard rails: mismatched buffer sizes are rejected.
// ---------------------------------------------------------------------------
check('mismatched buffer size → throws', () => {
    assert.throws(
        () => comparePixels(new Uint8Array(8), new Uint8Array(8), 2, 2),
        /Size mismatch/
    );
});

console.log(`\n${failed === 0 ? '✅' : '❌'} ${passed} passed, ${failed} failed\n`);
process.exit(failed === 0 ? 0 : 1);
