/**
 * VerificationRunner — High-level orchestration for shader verification.
 * 
 * Coordinates the full pipeline: pair resolution → transpilation → execution →
 * comparison → visualization → reporting.
 */

import { EffectPairResolver, EffectPair } from './EffectPairResolver';
import { HLSTranspiler, TranspileResult } from './HLSTranspiler';
import { GPUExecutor, ExecutionResult, InputTexture } from './GPUExecutor';
import { PixelComparator, ComparisonResult } from './PixelComparator';
import { DiffVisualizer, DiffImageResult } from './DiffVisualizer';
import { ReportGenerator, VerificationReport } from './ReportGenerator';
import { create, globals } from 'webgpu';

export interface VerificationConfig {
    /** Effect name or pair to verify */
    effect?: string;
    /** Category filter */
    category?: string;
    /** Specific pair (bypasses resolver) */
    pair?: EffectPair;
    /** Input texture configuration */
    input: {
        width: number;
        height: number;
        data?: Uint8Array;
        format?: 'rgba8unorm' | 'rgba16float';
    };
    /** Output size (defaults to input size) */
    outputSize?: [number, number];
    /** Tolerance level */
    tolerance?: 'strict' | 'standard' | 'lenient' | number;
    /** Generate visualizations */
    visualize?: boolean;
    /** Visualization modes */
    vizModes?: ('heatmap' | 'side-by-side' | 'difference' | 'overlay')[];
    /** Report formats */
    reportFormats?: ('json' | 'html' | 'markdown')[];
    /** Output directory */
    outputDir?: string;
    /** Base name for output files */
    baseName?: string;
    /** Capture GPU timings */
    captureTimings?: boolean;
    /** External resources for compilation */
    externalResources?: Record<string, any>;
    /** Macro defines */
    defines?: Record<string, number>;
}

export interface VerificationResult {
    /** The effect pair that was verified */
    pair: EffectPair;
    /** Whether verification passed */
    passed: boolean;
    /** Comparison results */
    comparison: ComparisonResult;
    /** HLSL execution result */
    hlslResult: ExecutionResult;
    /** WGSL execution result */
    wgslResult: ExecutionResult;
    /** Visualization results */
    visualizations?: Map<string, DiffImageResult>;
    /** Generated report file paths */
    reportPaths: string[];
    /** Full verification report */
    report: VerificationReport;
    /** Any errors during verification */
    errors: string[];
}

/**
 * VerificationRunner orchestrates the complete verification pipeline.
 */
export class VerificationRunner {
    private device: any; // GPUDevice
    private resolver: EffectPairResolver;
    private transpiler: HLSTranspiler;
    private executor: GPUExecutor;
    private comparator: PixelComparator;
    private visualizer: DiffVisualizer;
    private reportGenerator: ReportGenerator;

    constructor(device: any) {
        this.device = device;
        this.resolver = new EffectPairResolver();
        this.transpiler = new HLSTranspiler();
        this.executor = new GPUExecutor(device);
        this.comparator = new PixelComparator();
        this.visualizer = new DiffVisualizer();
        this.reportGenerator = new ReportGenerator();
    }

    /**
     * Run verification for a single effect pair.
     */
    async run(config: VerificationConfig): Promise<VerificationResult> {
        const errors: string[] = [];

        try {
            // Step 1: Resolve effect pair
            let pair: EffectPair | null = config.pair ?? null;
            if (!pair) {
                if (!config.effect) {
                    throw new Error('Either effect name or pair must be provided');
                }
                pair = await this.resolver.resolveByName(config.effect, config.category);
                if (!pair) {
                    throw new Error(`Effect pair not found: ${config.effect}${config.category ? ` (category: ${config.category})` : ''}`);
                }
            }

            // Step 2: Read source files
            const fs = await import('fs');
            const hlslSource = fs.readFileSync(pair.hlslPath, 'utf-8');
            const wgslSource = fs.readFileSync(pair.wgslPath, 'utf-8');

            // Step 3: Prepare input texture
            const inputWidth = config.input.width;
            const inputHeight = config.input.height;
            const inputData = config.input.data || this.generateTestPattern(inputWidth, inputHeight);
            const inputFormat = config.input.format || 'rgba8unorm';

            const inputTexture: InputTexture = {
                data: inputData,
                width: inputWidth,
                height: inputHeight,
                format: inputFormat,
            };

            const outputWidth = config.outputSize?.[0] || inputWidth;
            const outputHeight = config.outputSize?.[1] || inputHeight;

            // Prepare external resources with INPUT and OUTPUT textures
            const externalTextures = {
                INPUT: { size: [inputWidth, inputHeight], format: inputFormat, usage: 0x10 | 0x02 }, // TEXTURE_BINDING | COPY_DST
                OUTPUT: { size: [outputWidth, outputHeight], format: 'rgba16float', usage: 0x20 | 0x01 }, // STORAGE_BINDING | COPY_SRC
            };
            const mergedExternalResources = {
                ...config.externalResources,
                textures: {
                    ...externalTextures,
                    ...(config.externalResources?.textures || {}),
                },
            };

            const executionOptions = {
                outputWidth,
                outputHeight,
                outputFormat: 'rgba16float' as const,
                externalResources: mergedExternalResources,
                defines: config.defines,
                captureTimings: config.captureTimings,
            };

            // Step 4: Transpile HLSL
            const hlslTranspileResult = await this.transpiler.transpile(hlslSource, {
                defines: config.defines,
            });

            // Step 5: Execute both versions
            // 生成的模組須直接 dispatch（runWGSL 會把無 //!PASS 的生成碼 parse 成 0 個 pass）
            const [hlslExec, wgslExec] = await Promise.all([
                this.executor.runModules(hlslTranspileResult, inputTexture, executionOptions),
                this.executor.runWGSL(wgslSource, inputTexture, executionOptions),
            ]);

            // Step 6: Compare results
            const tolerance = config.tolerance || 'standard';
            const comparison = this.comparator.compare(
                hlslExec.output,
                wgslExec.output,
                outputWidth,
                outputHeight,
                { tolerance }
            );

            // Step 7: Generate visualizations
            let visualizations: Map<string, DiffImageResult> | undefined;
            if (config.visualize && comparison.errorMap) {
                const modes = config.vizModes || ['heatmap', 'side-by-side', 'difference'];
                visualizations = new Map();

                for (const mode of modes) {
                    let result: DiffImageResult;
                    switch (mode) {
                        case 'heatmap':
                            result = this.visualizer.generateHeatmap(comparison.errorMap, outputWidth, outputHeight);
                            break;
                        case 'side-by-side':
                            result = this.visualizer.generateSideBySide(
                                hlslExec.output,
                                wgslExec.output,
                                outputWidth,
                                outputHeight
                            );
                            break;
                        case 'difference':
                            result = this.visualizer.generateDifference(
                                hlslExec.output,
                                wgslExec.output,
                                outputWidth,
                                outputHeight
                            );
                            break;
                        case 'overlay':
                            result = this.visualizer.generateOverlay(
                                hlslExec.output,
                                wgslExec.output,
                                outputWidth,
                                outputHeight
                            );
                            break;
                        default:
                            continue;
                    }
                    visualizations.set(mode, result);
                }
            }

            // Step 8: Generate reports
            const reportPaths: string[] = [];
            const outputDir = config.outputDir || './verification_output';
            const baseName = config.baseName || `verify_${pair.name}`;
            const formats = config.reportFormats || ['json', 'html'];

            const report: VerificationReport = {
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

            for (const format of formats) {
                try {
                    const paths = await this.reportGenerator.generate(report, {
                        format,
                        outputDir,
                        baseName,
                        detailed: true,
                        embedImages: true,
                    });
                    reportPaths.push(...paths);
                } catch (e: any) {
                    errors.push(`Report generation (${format}) failed: ${e.message}`);
                }
            }

            return {
                pair,
                passed: comparison.passed,
                comparison,
                hlslResult: hlslExec,
                wgslResult: wgslExec,
                visualizations,
                reportPaths,
                report,
                errors,
            };

        } catch (error: any) {
            errors.push(`Verification failed: ${error.message}`);
            throw error;
        }
    }

    /**
     * Run verification for multiple effects (batch mode).
     */
    async runBatch(configs: VerificationConfig[]): Promise<VerificationResult[]> {
        const results: VerificationResult[] = [];

        for (const config of configs) {
            try {
                const result = await this.run(config);
                results.push(result);
            } catch (error: any) {
                // Create a failed result
                results.push({
                    pair: config.pair || { name: config.effect || 'unknown', category: config.category || 'unknown', hlslPath: '', wgslPath: '', hlslRelativePath: '', wgslRelativePath: '' },
                    passed: false,
                    comparison: {
                        passed: false,
                        perChannel: [],
                        aggregate: { maxAbsError: 0, mae: 0, rmse: 0, psnr: 0, ssim: 0, outlierCount: 0, outlierCountPerChannel: [], worstChannel: 0 },
                        width: 0,
                        height: 0,
                        pixelCount: 0,
                    },
                    hlslResult: { output: new Float32Array(), width: 0, height: 0, format: 'rgba16float', totalTime: 0, passCount: 0 },
                    wgslResult: { output: new Float32Array(), width: 0, height: 0, format: 'rgba16float', totalTime: 0, passCount: 0 },
                    reportPaths: [],
                    report: null as any,
                    errors: [error.message],
                });
            }
        }

        return results;
    }

    /**
     * Run verification for all effects in a category.
     */
    async runCategory(category: string, baseConfig: Omit<VerificationConfig, 'effect' | 'category' | 'pair'>): Promise<VerificationResult[]> {
        const pairs = await this.resolver.getByCategory(category);
        const configs: VerificationConfig[] = pairs.map(pair => ({
            ...baseConfig,
            pair,
        }));
        return this.runBatch(configs);
    }

    /**
     * Run verification for all discovered effects.
     */
    async runAll(baseConfig: Omit<VerificationConfig, 'effect' | 'category' | 'pair'>): Promise<VerificationResult[]> {
        const pairs = await this.resolver.discover();
        const configs: VerificationConfig[] = pairs.map(pair => ({
            ...baseConfig,
            pair,
        }));
        return this.runBatch(configs);
    }

    /**
     * Generate a batch summary report.
     */
    async generateBatchSummary(
        results: VerificationResult[],
        outputDir: string,
        baseName: string
    ): Promise<string[]> {
        const validResults = results.filter(r => r.report).map(r => ({
            pair: r.pair,
            report: r.report,
            passed: r.passed,
        }));

        return this.reportGenerator.generateBatchSummary(validResults, outputDir, baseName);
    }

    /**
     * Generate a synthetic test pattern.
     */
    private generateTestPattern(width: number, height: number): Uint8Array {
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

    /**
     * Get the underlying device.
     */
    getDevice(): any {
        return this.device;
    }
}

/**
 * Create a verification runner with a WebGPU device.
 */
export async function createRunner(device?: any): Promise<VerificationRunner> {
    if (!device) {
        const nav = { gpu: create([]) };
        Object.assign(globalThis, globals);
        const adapter = await nav.gpu.requestAdapter();
        if (!adapter) throw new Error('No WebGPU adapter found');
        device = await adapter.requestDevice();
    }
    return new VerificationRunner(device);
}

/**
 * Create a runner with default device initialization.
 */
export async function createDefaultRunner(): Promise<VerificationRunner> {
    return createRunner();
}