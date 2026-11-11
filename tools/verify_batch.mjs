#!/usr/bin/env node
/**
 * verify_batch.mjs — Batch verification CLI for multiple effects.
 * 
 * Usage:
 *   npm run verify:batch -- --category Scalers
 *   npm run verify:batch -- --all
 *   npm run verify:batch -- --category Anime4K --parallel 4
 *   npm run verify:batch -- --filter "Scalers/*,FSR/*" --tolerance strict
 */

import { create, globals } from 'webgpu';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { program } from 'commander';
import fs from 'fs';

import { EffectPairResolver, VerificationRunner } from '../dist/verification.esm.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const WORKSPACE_ROOT = resolve(__dirname, '..');

program
    .name('verify_batch')
    .description('Batch verify multiple HLSL/WGSL shader pairs')
    .version('1.0.0');

program
    .option('-c, --category <name>', 'Category to verify (e.g., Scalers, Anime4K)')
    .option('-a, --all', 'Verify all discovered effect pairs')
    .option('--filter <patterns>', 'Comma-separated glob patterns (e.g., "Scalers/*,FSR/*")')
    .option('--exclude <patterns>', 'Comma-separated glob patterns to exclude')
    .option('-p, --parallel <n>', 'Parallel executions', '1')
    .option('-w, --width <number>', 'Input width', '64')
    .option('-h, --height <number>', 'Input height', '64')
    .option('--output-width <number>', 'Output width')
    .option('--output-height <number>', 'Output height')
    .option('-t, --tolerance <level>', 'Tolerance: strict | standard | lenient', 'standard')
    .option('--visualize', 'Generate visualizations for failed effects')
    .option('-f, --format <formats>', 'Comma-separated output formats', 'json,html,markdown')
    .option('-o, --output-dir <path>', 'Output directory', './verification_output')
    .option('--define <defines...>', 'Macro defines as KEY=VALUE')
    .option('--baseline-dir <path>', 'Directory with baseline outputs for regression')
    .option('--summary-only', 'Only generate summary report, skip individual reports')
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
    const runner = new VerificationRunner(device);

    // Discover effects
    let pairs;
    if (options.all) {
        pairs = await resolver.discover();
    } else if (options.category) {
        pairs = await resolver.getByCategory(options.category);
    } else if (options.filter) {
        const allPairs = await resolver.discover();
        const patterns = options.filter.split(',').map(p => p.trim());
        pairs = allPairs.filter(pair => {
            const relPath = `${pair.category}/${pair.name}`;
            return patterns.some(pattern => matchGlob(relPath, pattern));
        });
    } else {
        console.error('❌ Must specify --all, --category, or --filter');
        program.help();
        device.destroy();
        process.exit(1);
    }

    // Apply exclusions
    if (options.exclude) {
        const excludePatterns = options.exclude.split(',').map(p => p.trim());
        pairs = pairs.filter(pair => {
            const relPath = `${pair.category}/${pair.name}`;
            return !excludePatterns.some(pattern => matchGlob(relPath, pattern));
        });
    }

    if (pairs.length === 0) {
        console.log('No effects matched the criteria.');
        device.destroy();
        return;
    }

    console.log(`\n🔍 Batch verification: ${pairs.length} effect(s)`);
    if (options.verbose) {
        for (const pair of pairs) {
            console.log(`   ${pair.category}/${pair.name}`);
        }
    }

    // Parse defines
    const defines = {};
    for (const def of options.define || []) {
        const [key, value] = def.split('=');
        if (key && value !== undefined) {
            defines[key] = parseFloat(value);
        }
    }

    // Prepare base config
    const inputWidth = parseInt(options.width);
    const inputHeight = parseInt(options.height);
    const outputWidth = options.outputWidth ? parseInt(options.outputWidth) : inputWidth;
    const outputHeight = options.outputHeight ? parseInt(options.outputHeight) : inputHeight;

    const baseConfig = {
        input: { width: inputWidth, height: inputHeight },
        outputSize: [outputWidth, outputHeight],
        tolerance: options.tolerance,
        visualize: options.visualize,
        reportFormats: options.summaryOnly ? [] : options.format.split(','),
        outputDir: resolve(WORKSPACE_ROOT, options.outputDir),
        defines,
        captureTimings: options.verbose,
    };

    // Run batch
    const parallel = Math.max(1, parseInt(options.parallel));
    const results = await runBatchParallel(runner, pairs, baseConfig, parallel, options.verbose);

    // Generate summary
    const outputDir = resolve(WORKSPACE_ROOT, options.outputDir);
    if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

    const summaryPaths = await runner.generateBatchSummary(results, outputDir, 'batch_summary');
    console.log('\n📄 Summary reports:');
    for (const p of summaryPaths) {
        console.log(`   ${p}`);
    }

    // Print final summary
    const passed = results.filter(r => r.passed).length;
    const failed = results.filter(r => !r.passed).length;
    console.log('\n' + '='.repeat(50));
    console.log(`BATCH COMPLETE: ${passed} passed, ${failed} failed`);
    console.log('='.repeat(50));

    device.destroy();
    process.exit(failed > 0 ? 1 : 0);
}

async function runBatchParallel(
    runner,
    pairs,
    baseConfig,
    parallel,
    verbose
) {
    const results = [];
    const queue = [...pairs];
    const running = [];

    async function processNext() {
        if (queue.length === 0) return;
        
        const pair = queue.shift();
        const config = { ...baseConfig, pair };

        if (verbose) console.log(`\n▶ ${pair.category}/${pair.name}`);

        const promise = runner.run(config).then(result => {
            results.push(result);
            const status = result.passed ? '✅' : '❌';
            const time = `${result.hlslResult.totalTime.toFixed(1)}+${result.wgslResult.totalTime.toFixed(1)}ms`;
            console.log(`${status} ${pair.category}/${pair.name} (${time})`);
            if (!result.passed && verbose) {
                console.log(`   Max Error: ${result.comparison.aggregate.maxAbsError.toExponential(3)}`);
                console.log(`   Outliers:  ${result.comparison.aggregate.outlierCount}`);
            }
            if (result.errors.length > 0) {
                for (const err of result.errors) {
                    console.error(`   ❌ ${err}`);
                }
            }
        }).catch(err => {
            results.push({
                pair,
                passed: false,
                comparison: { passed: false, aggregate: { maxAbsError: 0, outlierCount: 0 } },
                hlslResult: { totalTime: 0 }, wgslResult: { totalTime: 0 },
                errors: [err.message],
            });
            console.error(`❌ ${pair.category}/${pair.name}: ${err.message}`);
        });

        running.push(promise);

        // Wait if we've hit parallel limit
        if (running.length >= parallel) {
            await Promise.race(running);
            // Clean up completed promises
            const idx = running.findIndex(p => p === promise);
            if (idx >= 0) running.splice(idx, 1);
            // Process next
            await processNext();
        }
    }

    // Start initial batch
    const initialCount = Math.min(parallel, queue.length);
    for (let i = 0; i < initialCount; i++) {
        await processNext();
    }

    // Wait for all to complete
    await Promise.all(running);

    return results;
}

function matchGlob(path, pattern) {
    // Simple glob matching: * matches any characters except /
    const regexPattern = pattern
        .replace(/\./g, '\\.')
        .replace(/\*/g, '[^/]*')
        .replace(/\?/g, '[^/]');
    return new RegExp(`^${regexPattern}$`).test(path);
}

main().catch(err => {
    console.error('❌ Fatal error:', err);
    process.exit(1);
});