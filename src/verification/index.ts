/**
 * WGFX Shader Verification Tool — Public API exports.
 * 
 * Provides a complete toolkit for comparing HLSL and WGSL shader outputs
 * pixel-by-pixel with detailed statistical analysis.
 */

// Core modules
export { EffectPairResolver, createResolver, type EffectPair, type ResolverOptions } from './EffectPairResolver';
export { HLSTranspiler, createTranspiler, type TranspileOptions, type TranspileResult, type TranspileDiagnostic, type GeneratedModule, type ModuleResources } from './HLSTranspiler';
export { GPUExecutor, createExecutor, type InputTexture, type ExecutionOptions, type ExecutionResult, type CompiledEffect } from './GPUExecutor';
export { PixelComparator, createComparator, type ComparisonOptions, type ComparisonResult, type ToleranceLevel } from './PixelComparator';
export { StatisticsEngine, createStatsEngine, type ChannelStats, type AggregateStats, type StatsOptions } from './StatisticsEngine';
export { DiffVisualizer, createVisualizer, type VisualizationMode, type VisualizationOptions, type DiffImageResult } from './DiffVisualizer';
export { ReportGenerator, createReportGenerator, type ReportFormat, type ReportOptions, type VerificationReport } from './ReportGenerator';

// High-level orchestration
export { VerificationRunner, createRunner, type VerificationConfig, type VerificationResult } from './VerificationRunner';

/**
 * Quick verification function for common use cases.
 */
import { EffectPairResolver } from './EffectPairResolver';
import { HLSTranspiler } from './HLSTranspiler';
import { GPUExecutor } from './GPUExecutor';
import { PixelComparator } from './PixelComparator';
import { ReportGenerator } from './ReportGenerator';
import { create, globals } from 'webgpu';

export interface QuickVerifyOptions {
    /** Effect name to verify (e.g., "Bilinear") */
    effect: string;
    /** Optional category to narrow search */
    category?: string;
    /** Input texture (generated if not provided) */
    input?: {
        width: number;
        height: number;
        data?: Uint8Array;
    };
    /** Tolerance level */
    tolerance?: 'strict' | 'standard' | 'lenient';
    /** Output directory for reports */
    outputDir?: string;
    /** Report formats to generate */
    formats?: ('json' | 'html' | 'markdown')[];
    /** Generate visualizations */
    visualize?: boolean;
}

/**
 * Run a quick verification of a single effect pair.
 * This is a convenience function that handles the full pipeline.
 */
export async function quickVerify(options: QuickVerifyOptions): Promise<{
    passed: boolean;
    reportPaths: string[];
    comparison: any;
}> {
    // Initialize WebGPU
    const nav = { gpu: create([]) };
    Object.assign(globalThis, globals);
    const adapter = await nav.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter found');
    const device = await adapter.requestDevice();

    // Resolve effect pair
    const resolver = new EffectPairResolver();
    const pair = await resolver.resolveByName(options.effect, options.category);
    if (!pair) {
        throw new Error(`Effect pair not found: ${options.effect}${options.category ? ` (category: ${options.category})` : ''}`);
    }

    // Read source files
    const fs = await import('fs');
    const hlslSource = fs.readFileSync(pair.hlslPath, 'utf-8');
    const wgslSource = fs.readFileSync(pair.wgslPath, 'utf-8');

    // Create input texture
    const width = options.input?.width || 64;
    const height = options.input?.height || 64;
    const inputData = options.input?.data || generateTestPattern(width, height);

    const inputTexture = {
        data: inputData,
        width,
        height,
        format: 'rgba8unorm' as const,
    };

    // Transpile HLSL
    const transpiler = new HLSTranspiler();
    const hlslResult = await transpiler.transpile(hlslSource);

    // Execute both — the transpiled modules are standalone compute shaders and
    // must be dispatched directly (runWGSL would parse them as an effect file
    // with zero passes).
    const executor = new GPUExecutor(device);

    const hlslExec = await executor.runModules(
        hlslResult,
        inputTexture,
        { outputWidth: width, outputHeight: height }
    );

    const wgslExec = await executor.runWGSL(
        wgslSource,
        inputTexture,
        { outputWidth: width, outputHeight: height }
    );

    // Compare
    const comparator = new PixelComparator();
    const comparison = comparator.compare(
        hlslExec.output,
        wgslExec.output,
        width,
        height,
        { tolerance: options.tolerance || 'standard' }
    );

    // Generate report
    const reportGen = new ReportGenerator();
    const outputDir = options.outputDir || './verification_output';
    const formats = options.formats || ['json', 'html'];

    const reportPaths: string[] = [];
    for (const format of formats) {
        const paths = await reportGen.generate({
            pair,
            comparison,
            timestamp: new Date().toISOString(),
            metadata: {
                hlslExecutionTime: hlslExec.totalTime,
                wgslExecutionTime: wgslExec.totalTime,
                totalTime: hlslExec.totalTime + wgslExec.totalTime,
                inputSize: [width, height],
                outputSize: [width, height],
            },
        }, { format, outputDir, baseName: `verify_${pair.name}` });
        reportPaths.push(...paths);
    }

    device.destroy();

    return {
        passed: comparison.passed,
        reportPaths,
        comparison,
    };
}

/**
 * Generate a synthetic test pattern for verification.
 */
function generateTestPattern(width: number, height: number): Uint8Array {
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