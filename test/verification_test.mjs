// verification_test.mjs — Unit tests for shader verification module

import { create, globals } from 'webgpu';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

import { EffectPairResolver, HLSTranspiler, GPUExecutor, PixelComparator, StatisticsEngine, DiffVisualizer, ReportGenerator, VerificationRunner } from '../dist/verification.esm.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const WORKSPACE_ROOT = resolve(__dirname, '..');

Object.assign(globalThis, globals);

let passCount = 0;
let failCount = 0;

function assert(condition, message) {
    if (condition) {
        console.log(`  ✅ ${message}`);
        passCount++;
    } else {
        console.error(`  ❌ ${message}`);
        failCount++;
    }
}

function assertEqual(actual, expected, message, tolerance = 1e-6) {
    const pass = typeof actual === 'number' && typeof expected === 'number'
        ? Math.abs(actual - expected) < tolerance
        : actual === expected;
    if (pass) {
        console.log(`  ✅ ${message}`);
        passCount++;
    } else {
        console.error(`  ❌ ${message} (expected: ${expected}, actual: ${actual})`);
        failCount++;
    }
}

async function runTests() {
    console.log('\n🧪 Running Shader Verification Tests\n');

    // Initialize WebGPU
    const nav = { gpu: create([]) };
    const adapter = await nav.gpu.requestAdapter();
    if (!adapter) {
        console.error('❌ No WebGPU adapter found - skipping GPU tests');
        return;
    }
    const device = await adapter.requestDevice();

    // Test 1: EffectPairResolver
    console.log('Test Suite: EffectPairResolver');
    const resolver = new EffectPairResolver({ workspaceRoot: WORKSPACE_ROOT });

    const pairs = await resolver.discover();
    assert(pairs.length > 0, `Discovered ${pairs.length} effect pairs`);

    const bilinear = await resolver.resolveByName('Bilinear');
    console.log(`  Bilinear resolved:`, bilinear ? `${bilinear.category}/${bilinear.name}` : 'NOT FOUND');
    assert(bilinear !== null, 'Resolved Bilinear by name');
    if (bilinear) {
        assert(bilinear.name === 'Bilinear', 'Bilinear name matches');
        assert(bilinear.category === 'Scalers', 'Bilinear category is Scalers');
    }

    const categories = await resolver.listCategories();
    console.log(`  Categories found: ${categories.join(', ')}`);
    assert(categories.includes('Scalers'), `Categories includes Scalers (found: ${categories.join(', ')})`);
    assert(categories.includes('Anime4K'), 'Categories includes Anime4K');

    const scalersEffects = await resolver.listEffects('Scalers');
    console.log(`  Scalers effects: ${scalersEffects.join(', ')}`);
    assert(scalersEffects.includes('Bilinear'), `Scalers effects includes Bilinear (found: ${scalersEffects.join(', ')})`);
    assert(scalersEffects.includes('Nearest'), 'Scalers effects includes Nearest');

    console.log();

    // Test 2: HLSTranspiler
    console.log('Test Suite: HLSTranspiler');
    const transpiler = new HLSTranspiler();

    const fs = await import('fs');
    const hlslSource = fs.readFileSync(resolve(WORKSPACE_ROOT, 'effects_src/Bilinear.hlsl'), 'utf-8');
    console.log(`  HLSL source length: ${hlslSource.length}`);
    const transpileResult = await transpiler.transpile(hlslSource);
    
    console.log(`  Modules: ${transpileResult.modules.length}`);
    console.log(`  Passes: ${transpileResult.shaderInfo.passes.length}`);
    console.log(`  Diagnostics: ${transpileResult.diagnostics.length}`);
    for (const d of transpileResult.diagnostics) {
        console.log(`    ${d.severity}: ${d.message}`);
    }
    
    // Debug: print generated WGSL
    for (const m of transpileResult.modules) {
        console.log(`\n--- Pass ${m.passIndex} WGSL ---`);
        console.log(m.wgslCode);
    }

    assert(transpileResult.modules.length > 0, 'Transpiled at least one module');
    assert(transpileResult.shaderInfo.passes.length === 1, 'Bilinear has 1 pass');
    assert(transpileResult.modules[0].wgslCode.includes('textureSampleLevel'), 'Generated WGSL uses textureSampleLevel');
    assert(!transpileResult.diagnostics.some(d => d.severity === 'error'), 'No transpilation errors');

    console.log();

    // Test 3: GPUExecutor
    console.log('Test Suite: GPUExecutor');
    const executor = new GPUExecutor(device);

    const inputWidth = 64, inputHeight = 64;
    const inputData = new Uint8Array(inputWidth * inputHeight * 4);
    for (let i = 0; i < inputData.length; i += 4) {
        inputData[i] = 128; inputData[i+1] = 64; inputData[i+2] = 192; inputData[i+3] = 255;
    }

    const inputTexture = { data: inputData, width: inputWidth, height: inputHeight, format: 'rgba8unorm' };

    // Use runHLSL which goes through the full compilation pipeline
    const hlslExec = await executor.runHLSL(
        hlslSource, // Already read in HLSTranspiler test
        inputTexture,
        { 
            outputWidth: inputWidth, 
            outputHeight: inputHeight, 
            defines: { INPUT_WIDTH: inputWidth, INPUT_HEIGHT: inputHeight },
            externalResources: {
                textures: {
                    INPUT: { size: [inputWidth, inputHeight], format: 'rgba8unorm', usage: globals.GPUTextureUsage.TEXTURE_BINDING | globals.GPUTextureUsage.COPY_DST },
                    OUTPUT: { size: [inputWidth, inputHeight], format: 'rgba16float', usage: globals.GPUTextureUsage.STORAGE_BINDING | globals.GPUTextureUsage.COPY_SRC }
                }
            }
        }
    );

    assert(hlslExec.output instanceof Float32Array, 'Output is Float32Array');
    assert(hlslExec.output.length === inputWidth * inputHeight * 4, 'Output has correct size');
    assert(hlslExec.width === inputWidth && hlslExec.height === inputHeight, 'Output dimensions match');
    assert(hlslExec.passCount === 1, 'Executed 1 pass');
    assert(hlslExec.totalTime > 0, 'Execution time recorded');

    // Test WGSL native execution
    const wgslSource = fs.readFileSync(resolve(WORKSPACE_ROOT, 'effects/Scalers/Bilinear.wgsl'), 'utf-8');
    const wgslExec = await executor.runWGSL(wgslSource, inputTexture, { 
        outputWidth: inputWidth, 
        outputHeight: inputHeight, 
        defines: { INPUT_WIDTH: inputWidth, INPUT_HEIGHT: inputHeight },
        externalResources: {
            textures: {
                INPUT: { size: [inputWidth, inputHeight], format: 'rgba8unorm', usage: globals.GPUTextureUsage.TEXTURE_BINDING | globals.GPUTextureUsage.COPY_DST },
                OUTPUT: { size: [inputWidth, inputHeight], format: 'rgba16float', usage: globals.GPUTextureUsage.STORAGE_BINDING | globals.GPUTextureUsage.COPY_SRC }
            }
        }
    });

    assert(wgslExec.output instanceof Float32Array, 'WGSL output is Float32Array');
    assert(wgslExec.output.length === inputWidth * inputHeight * 4, 'WGSL output has correct size');

    console.log();

    // Test 4: PixelComparator
    console.log('Test Suite: PixelComparator');
    const comparator = new PixelComparator();

    // Identical images should pass
    const identical = comparator.compare(hlslExec.output, hlslExec.output, inputWidth, inputHeight, { tolerance: 'standard' });
    assert(identical.passed === true, 'Identical images pass');
    assert(identical.aggregate.maxAbsError === 0, 'Max error is 0 for identical');
    assert(identical.aggregate.outlierCount === 0, 'No outliers for identical');

    // Test with known difference
    // 0.0625 = 2^-4 is exactly representable in f32 (and f16), so the injected
    // delta survives the GPU readback quantization without float-epsilon drift
    // (0.01 as f32 is 0.009999999776…, which would break a `>= 0.01` assertion).
    const testData = new Float32Array(hlslExec.output);
    testData[0] = testData[0] + 0.0625; // Small difference in first pixel R channel
    const diffResult = comparator.compare(hlslExec.output, testData, inputWidth, inputHeight, { tolerance: 'standard' });
    assert(diffResult.passed === false, 'Different images fail with standard tolerance');
    assert(diffResult.aggregate.maxAbsError >= 0.0625, 'Max error detects difference');
    assert(diffResult.aggregate.outlierCount > 0, 'Outliers detected');

    // Test strict tolerance
    const strictResult = comparator.compare(hlslExec.output, testData, inputWidth, inputHeight, { tolerance: 'strict' });
    assert(strictResult.aggregate.outlierCount > 0, 'Strict tolerance detects small difference');

    // Test lenient tolerance
    const largeDiffData = new Float32Array(hlslExec.output);
    largeDiffData[0] = largeDiffData[0] + 0.02;
    const lenientResult = comparator.compare(hlslExec.output, largeDiffData, inputWidth, inputHeight, { tolerance: 'lenient' });
    assert(lenientResult.aggregate.outlierCount > 0, 'Lenient tolerance still detects larger difference');

    console.log();

    // Test 5: StatisticsEngine
    console.log('Test Suite: StatisticsEngine');
    const statsEngine = new StatisticsEngine();

    const refChannel = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]);
    const testChannel = new Float32Array([0.11, 0.19, 0.31, 0.39, 0.51]);
    
    const chStats = statsEngine.computeChannelStats(refChannel, testChannel, 5, 1, 'R', { tolerance: 0.01 });
    
    assert(chStats.channel === 'R', 'Channel name is R');
    assert(chStats.maxAbsError > 0, 'Max error > 0');
    assert(chStats.mae > 0, 'MAE > 0');
    assert(chStats.rmse > 0, 'RMSE > 0');
    assert(chStats.psnr > 0, 'PSNR > 0');
    assert(chStats.ssim >= 0 && chStats.ssim <= 1, 'SSIM in range [0,1]');
    assert(chStats.histogram.length === 64, 'Histogram has 64 bins');
    assert(chStats.percentiles.p50 >= 0, 'P50 computed');
    assert(chStats.percentiles.p99 >= chStats.percentiles.p50, 'P99 >= P50');

    const aggStats = statsEngine.computeAggregateStats([chStats, chStats, chStats, chStats], 0.01);
    assert(aggStats.maxAbsError === chStats.maxAbsError, 'Aggregate max error matches');
    assert(aggStats.mae === chStats.mae, 'Aggregate MAE matches');
    assert(aggStats.worstChannel >= 0 && aggStats.worstChannel <= 3, 'Worst channel in range');

    console.log();

    // Test 6: DiffVisualizer
    console.log('Test Suite: DiffVisualizer');
    const visualizer = new DiffVisualizer();

    const errorMap = new Float32Array(inputWidth * inputHeight);
    for (let i = 0; i < errorMap.length; i++) {
        errorMap[i] = Math.random() * 0.1;
    }

    const heatmap = visualizer.generateHeatmap(errorMap, inputWidth, inputHeight);
    assert(heatmap.buffer instanceof Buffer, 'Heatmap returns Buffer');
    assert(heatmap.width >= inputWidth, 'Heatmap width correct');
    assert(heatmap.height === inputHeight, 'Heatmap height correct');
    assert(heatmap.mode === 'heatmap', 'Heatmap mode correct');

    const sideBySide = visualizer.generateSideBySide(hlslExec.output, wgslExec.output, inputWidth, inputHeight);
    assert(sideBySide.buffer instanceof Buffer, 'Side-by-side returns Buffer');
    assert(sideBySide.width === inputWidth * 3 + 20, 'Side-by-side width correct (3 panels + padding)');

    const difference = visualizer.generateDifference(hlslExec.output, wgslExec.output, inputWidth, inputHeight);
    assert(difference.buffer instanceof Buffer, 'Difference returns Buffer');

    const overlay = visualizer.generateOverlay(hlslExec.output, wgslExec.output, inputWidth, inputHeight);
    assert(overlay.buffer instanceof Buffer, 'Overlay returns Buffer');

    console.log();

    // Test 7: ReportGenerator
    console.log('Test Suite: ReportGenerator');
    const reportGen = new ReportGenerator();

    const mockReport = {
        pair: {
            name: 'TestEffect',
            category: 'Test',
            hlslPath: '/test/hlsl',
            wgslPath: '/test/wgsl',
            hlslRelativePath: 'Test.hlsl',
            wgslRelativePath: 'Test.wgsl',
        },
        comparison: identical,
        timestamp: new Date().toISOString(),
        metadata: {
            hlslExecutionTime: 10.5,
            wgslExecutionTime: 11.2,
            totalTime: 21.7,
            inputSize: [64, 64],
            outputSize: [64, 64],
        },
    };

    const outputDir = resolve(WORKSPACE_ROOT, 'verification_test_output');
    if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

    const jsonPaths = await reportGen.generate(mockReport, { format: 'json', outputDir, baseName: 'test_report' });
    assert(jsonPaths.length === 1, 'JSON report generated');
    assert(fs.existsSync(jsonPaths[0]), 'JSON file exists');

    const htmlPaths = await reportGen.generate(mockReport, { format: 'html', outputDir, baseName: 'test_report' });
    assert(htmlPaths.length === 1, 'HTML report generated');
    assert(fs.existsSync(htmlPaths[0]), 'HTML file exists');

    const mdPaths = await reportGen.generate(mockReport, { format: 'markdown', outputDir, baseName: 'test_report' });
    assert(mdPaths.length === 1, 'Markdown report generated');
    assert(fs.existsSync(mdPaths[0]), 'Markdown file exists');

    // Clean up test output
    fs.rmSync(outputDir, { recursive: true, force: true });

    console.log();

    // Test 8: HLSL vs WGSL Bilinear comparison (integration test)
    console.log('Test Suite: Integration - Bilinear HLSL vs WGSL');
    const bilinearPair = await resolver.resolveByName('Bilinear', 'Scalers');
    assert(bilinearPair !== null, 'Found Bilinear pair');

    const hlslSrc = fs.readFileSync(bilinearPair.hlslPath, 'utf-8');
    const wgslSrc = fs.readFileSync(bilinearPair.wgslPath, 'utf-8');

    const hlslTranspile = await transpiler.transpile(hlslSrc);

    const execOptions = { 
        outputWidth: inputWidth, 
        outputHeight: inputHeight, 
        defines: { INPUT_WIDTH: inputWidth, INPUT_HEIGHT: inputHeight },
        externalResources: {
            textures: {
                INPUT: { size: [inputWidth, inputHeight], format: 'rgba8unorm', usage: globals.GPUTextureUsage.TEXTURE_BINDING | globals.GPUTextureUsage.COPY_DST },
                OUTPUT: { size: [inputWidth, inputHeight], format: 'rgba16float', usage: globals.GPUTextureUsage.STORAGE_BINDING | globals.GPUTextureUsage.COPY_SRC }
            }
        }
    };

    const hlslResult = await executor.runModules(
        hlslTranspile,
        inputTexture,
        execOptions
    );
    const wgslResult = await executor.runWGSL(wgslSrc, inputTexture, execOptions);

    const bilinearCompare = comparator.compare(hlslResult.output, wgslResult.output, inputWidth, inputHeight, { tolerance: 'standard' });

    console.log(`  Bilinear HLSL vs WGSL: ${bilinearCompare.passed ? 'PASS' : 'FAIL'}`);
    console.log(`  Max Error: ${bilinearCompare.aggregate.maxAbsError.toExponential(4)}`);
    console.log(`  PSNR: ${bilinearCompare.aggregate.psnr === Infinity ? '∞' : bilinearCompare.aggregate.psnr.toFixed(2)} dB`);
    console.log(`  SSIM: ${bilinearCompare.aggregate.ssim.toFixed(6)}`);

    // Bilinear @1:1 samples the input at output texel centers in both versions —
    // identical math → bit-identical output (f16 readback included).
    assert(bilinearCompare.passed === true, 'Bilinear HLSL (transpiled) matches ported WGSL pixel-for-pixel');
    assert(bilinearCompare.aggregate.ssim > 0.99, 'Bilinear SSIM > 0.99');

    console.log();

    // Summary
    console.log('='.repeat(50));
    console.log(`Tests: ${passCount} passed, ${failCount} failed`);
    console.log('='.repeat(50));

    device.destroy();

    if (failCount > 0) {
        process.exit(1);
    }
}

runTests().catch(err => {
    console.error('❌ Test runner error:', err);
    process.exit(1);
});