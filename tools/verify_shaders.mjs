#!/usr/bin/env node
/**
 * verify_shaders.mjs — Single effect pair verification CLI.
 * 
 * Usage:
 *   npm run verify -- --effect Bilinear
 *   npm run verify -- --effect Bilinear --category Scalers
 *   npm run verify -- --hlsl effects_src/Bilinear.hlsl --wgsl effects/Scalers/Bilinear.wgsl
 *   npm run verify -- --effect Bilinear --tolerance strict --visualize --format html,json
 */

import { create, globals } from 'webgpu';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { program } from 'commander';
import fs from 'fs';

import { EffectPairResolver, HLSTranspiler, GPUExecutor, PixelComparator, DiffVisualizer, ReportGenerator } from '../dist/verification.esm.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const WORKSPACE_ROOT = resolve(__dirname, '..');

program
    .name('verify_shaders')
    .description('Verify HLSL/WGSL shader pair output equivalence')
    .version('1.0.0');

program
    .option('-e, --effect <name>', 'Effect name to verify (e.g., Bilinear)')
    .option('-c, --category <name>', 'Category to search in (e.g., Scalers, Anime4K)')
    .option('--hlsl <path>', 'Explicit HLSL file path')
    .option('--wgsl <path>', 'Explicit WGSL file path')
    .option('-w, --width <number>', 'Input width', '64')
    .option('-h, --height <number>', 'Input height', '64')
    .option('--output-width <number>', 'Output width (defaults to input width)')
    .option('--output-height <number>', 'Output height (defaults to input height)')
    .option('-t, --tolerance <level>', 'Tolerance: strict | standard | lenient', 'standard')
    .option('--visualize', 'Generate visualization images')
    .option('--viz-modes <modes>', 'Comma-separated visualization modes', 'heatmap,side-by-side,difference')
    .option('-f, --format <formats>', 'Comma-separated output formats: json,html,markdown', 'json,html')
    .option('-o, --output-dir <path>', 'Output directory', './verification_output')
    .option('--define <defines...>', 'Macro defines as KEY=VALUE', [])
    .option('--list', 'List all available effect pairs')
    .option('--list-categories', 'List all categories')
    .option('-v, --verbose', 'Verbose output')
    .parse();

const options = program.opts();

async function main() {
    // Initialize WebGPU
    const nav = { gpu: create([]) };
    Object.assign(globalThis, globals);
    
    const adapter = await nav.gpu.requestAdapter();
    if (!adapter) {
        console.error('❌ No WebGPU adapter found');
        process.exit(1);
    }
    const device = await adapter.requestDevice();

    const resolver = new EffectPairResolver({ workspaceRoot: WORKSPACE_ROOT });

    // Handle list commands
    if (options.list) {
        const pairs = await resolver.discover();
        console.log(`\nFound ${pairs.length} effect pairs:\n`);
        for (const pair of pairs) {
            console.log(`  ${pair.category}/${pair.name}`);
            console.log(`    HLSL: ${pair.hlslRelativePath}`);
            console.log(`    WGSL: ${pair.wgslRelativePath}`);
        }
        device.destroy();
        return;
    }

    if (options.listCategories) {
        const categories = await resolver.listCategories();
        console.log('\nAvailable categories:');
        for (const cat of categories) {
            const effects = await resolver.listEffects(cat);
            console.log(`  ${cat} (${effects.length} effects)`);
        }
        device.destroy();
        return;
    }

    // Resolve effect pair
    let pair;
    if (options.hlsl && options.wgsl) {
        // Explicit paths provided
        const name = fs.basename(options.hlsl, '.hlsl');
        const category = fs.basename(fs.dirname(options.hlsl));
        pair = {
            name,
            category,
            hlslPath: resolve(WORKSPACE_ROOT, options.hlsl),
            wgslPath: resolve(WORKSPACE_ROOT, options.wgsl),
            hlslRelativePath: options.hlsl,
            wgslRelativePath: options.wgsl,
        };
    } else if (options.effect) {
        pair = await resolver.resolveByName(options.effect, options.category);
        if (!pair) {
            console.error(`❌ Effect pair not found: ${options.effect}${options.category ? ` (category: ${options.category})` : ''}`);
            device.destroy();
            process.exit(1);
        }
    } else {
        console.error('❌ Must specify --effect or both --hlsl and --wgsl');
        program.help();
        device.destroy();
        process.exit(1);
    }

    console.log(`\n🔍 Verifying: ${pair.category}/${pair.name}`);
    console.log(`   HLSL: ${pair.hlslRelativePath}`);
    console.log(`   WGSL: ${pair.wgslRelativePath}`);

    // Read source files
    const hlslSource = fs.readFileSync(pair.hlslPath, 'utf-8');
    const wgslSource = fs.readFileSync(pair.wgslPath, 'utf-8');

    // Parse defines
    const defines = {};
    for (const def of options.define) {
        const [key, value] = def.split('=');
        if (key && value !== undefined) {
            defines[key] = parseFloat(value);
        }
    }

    // Prepare input texture
    const inputWidth = parseInt(options.width);
    const inputHeight = parseInt(options.height);
    const outputWidth = options.outputWidth ? parseInt(options.outputWidth) : inputWidth;
    const outputHeight = options.outputHeight ? parseInt(options.outputHeight) : inputHeight;

    const inputData = generateTestPattern(inputWidth, inputHeight);
    const inputTexture = {
        data: inputData,
        width: inputWidth,
        height: inputHeight,
        format: 'rgba8unorm',
    };

    // Transpile HLSL
    if (options.verbose) console.log('\n📝 Transpiling HLSL...');
    const transpiler = new HLSTranspiler();
    const hlslTranspileResult = await transpiler.transpile(hlslSource, { defines });

    if (hlslTranspileResult.diagnostics.some(d => d.severity === 'error')) {
        for (const diag of hlslTranspileResult.diagnostics) {
            if (diag.severity === 'error') console.error(`   ❌ ${diag.message}`);
        }
        device.destroy();
        process.exit(1);
    }

    if (options.verbose) {
        for (const diag of hlslTranspileResult.diagnostics) {
            if (diag.severity === 'warning') console.warn(`   ⚠️  ${diag.message}`);
        }
    }

    // Execute both
    const executor = new GPUExecutor(device);
    const execOptions = {
        outputWidth,
        outputHeight,
        outputFormat: 'rgba16float',
        defines,
        captureTimings: options.verbose,
        externalResources: {
            textures: {
                INPUT: { size: [inputWidth, inputHeight], format: 'rgba8unorm', usage: globals.GPUTextureUsage.TEXTURE_BINDING | globals.GPUTextureUsage.COPY_DST },
                OUTPUT: { size: [outputWidth, outputHeight], format: 'rgba16float', usage: globals.GPUTextureUsage.STORAGE_BINDING | globals.GPUTextureUsage.COPY_SRC }
            }
        }
    };

    if (options.verbose) console.log('\n🚀 Executing HLSL (transpiled)...');
    // 生成的模組是獨立 compute shader，須直接 dispatch（runWGSL 會把無 //!PASS 的
    // 生成碼 parse 成 0 個 pass）
    const hlslExec = await executor.runModules(hlslTranspileResult, inputTexture, execOptions);

    if (options.verbose) console.log('🚀 Executing WGSL (native)...');
    const wgslExec = await executor.runWGSL(wgslSource, inputTexture, execOptions);

    // Compare
    if (options.verbose) console.log('\n📊 Comparing outputs...');
    const comparator = new PixelComparator();
    const tolerance = options.tolerance;
    const comparison = comparator.compare(
        hlslExec.output,
        wgslExec.output,
        outputWidth,
        outputHeight,
        { tolerance }
    );

    // Print results
    printComparisonResult(comparison, tolerance, options.verbose);

    // Generate visualizations
    let visualizations;
    if (options.visualize && comparison.errorMap) {
        if (options.verbose) console.log('\n🎨 Generating visualizations...');
        const visualizer = new DiffVisualizer();
        const modes = options.vizModes.split(',');
        visualizations = new Map();

        for (const mode of modes) {
            let result;
            switch (mode) {
                case 'heatmap':
                    result = visualizer.generateHeatmap(comparison.errorMap, outputWidth, outputHeight);
                    break;
                case 'side-by-side':
                    result = visualizer.generateSideBySide(hlslExec.output, wgslExec.output, outputWidth, outputHeight);
                    break;
                case 'difference':
                    result = visualizer.generateDifference(hlslExec.output, wgslExec.output, outputWidth, outputHeight);
                    break;
                case 'overlay':
                    result = visualizer.generateOverlay(hlslExec.output, wgslExec.output, outputWidth, outputHeight);
                    break;
                default:
                    console.warn(`   Unknown visualization mode: ${mode}`);
                    continue;
            }
            visualizations.set(mode, result);
            
            // Save to file
            const outputDir = resolve(WORKSPACE_ROOT, options.outputDir);
            if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });
            const fileName = `${pair.name}_${mode}.png`;
            const filePath = resolve(outputDir, fileName);
            fs.writeFileSync(filePath, result.buffer);
            if (options.verbose) console.log(`   Saved: ${filePath}`);
        }
    }

    // Generate reports
    const reportGen = new ReportGenerator();
    const outputDir = resolve(WORKSPACE_ROOT, options.outputDir);
    if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

    const formats = options.format.split(',');
    const baseName = `verify_${pair.name}`;

    const report = {
        pair,
        comparison,
        timestamp: new Date().toISOString(),
        metadata: {
            hlslExecutionTime: hlslExec.totalTime,
            wgslExecutionTime: wgslExec.totalTime,
            totalTime: hlslExec.totalTime + wgslExec.totalTime,
            inputSize: [inputWidth, inputHeight],
            outputSize: [outputWidth, outputHeight],
        },
        visualizations,
    };

    console.log('\n📄 Generating reports...');
    for (const format of formats) {
        try {
            const paths = await reportGen.generate(report, {
                format,
                outputDir,
                baseName,
                detailed: true,
                embedImages: true,
            });
            for (const p of paths) {
                console.log(`   ${format.toUpperCase()}: ${p}`);
            }
        } catch (e) {
            console.error(`   ❌ ${format} report failed: ${e.message}`);
        }
    }

    device.destroy();

    console.log(`\n${comparison.passed ? '✅' : '❌'} Verification ${comparison.passed ? 'PASSED' : 'FAILED'}`);
    process.exit(comparison.passed ? 0 : 1);
}

function printComparisonResult(
    comparison,
    tolerance,
    verbose
) {
    const agg = comparison.aggregate;
    const tolValue = { strict: 0.5/65535, standard: 1e-3, lenient: 1e-2 }[tolerance];

    console.log('\n' + '='.repeat(50));
    console.log('COMPARISON RESULTS');
    console.log('='.repeat(50));
    console.log(`Status:         ${comparison.passed ? '✅ PASS' : '❌ FAIL'}`);
    console.log(`Tolerance:      ${tolerance} (${tolValue})`);
    console.log(`Resolution:     ${comparison.width}×${comparison.height} (${comparison.pixelCount} pixels)`);
    console.log('-'.repeat(50));
    console.log(`Max Abs Error:  ${agg.maxAbsError.toExponential(4)}`);
    console.log(`MAE:            ${agg.mae.toExponential(4)}`);
    console.log(`RMSE:           ${agg.rmse.toExponential(4)}`);
    console.log(`PSNR:           ${agg.psnr === Infinity ? '∞' : agg.psnr.toFixed(2) + ' dB'}`);
    console.log(`SSIM:           ${agg.ssim.toFixed(6)}`);
    console.log(`Outliers:       ${agg.outlierCount} / ${comparison.pixelCount * 4}`);
    console.log(`Worst Channel:  ${['R', 'G', 'B', 'A'][agg.worstChannel]}`);

    if (verbose) {
        console.log('\nPer-Channel Details:');
        console.log('-'.repeat(50));
        for (const ch of comparison.perChannel) {
            console.log(`  ${ch.channel}: max=${ch.maxAbsError.toExponential(3)} MAE=${ch.mae.toExponential(3)} RMSE=${ch.rmse.toExponential(3)} PSNR=${ch.psnr===Infinity?'∞':ch.psnr.toFixed(1)} SSIM=${ch.ssim.toFixed(4)} outliers=${ch.outlierCount}`);
            console.log(`     Percentiles: P50=${ch.percentiles.p50.toExponential(2)} P90=${ch.percentiles.p90.toExponential(2)} P99=${ch.percentiles.p99.toExponential(2)}`);
        }
    }
}

function generateTestPattern(width, height) {
    const data = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4;
            const v = (x * 7 + y * 13 + ((x * y) % 97)) % 256;
            data[i] = v;
            data[i + 1] = 255 - v;
            data[i + 2] = (v * 3) % 255;
            data[i + 3] = 255;
        }
    }
    return data;
}

main().catch(err => {
    console.error('❌ Fatal error:', err);
    process.exit(1);
});