/**
 * PixelComparator — Pixel-by-pixel comparison of two Float32Array images.
 * 
 * Computes per-channel and aggregate error metrics for verification.
 */

import { StatisticsEngine, ChannelStats, AggregateStats } from './StatisticsEngine';

export type ToleranceLevel = 'strict' | 'standard' | 'lenient' | number;

export interface ComparisonOptions {
    /** Tolerance for outlier detection: 'strict'=0.5/65535, 'standard'=1e-3, 'lenient'=1e-2, or custom number */
    tolerance?: ToleranceLevel;
    /** Whether to compute SSIM (more expensive) */
    computeSSIM?: boolean;
    /** SSIM window size (default 11) */
    ssimWindowSize?: number;
    /** Histogram bins (default 64) */
    histogramBins?: number;
    /** Whether to generate error map */
    generateErrorMap?: boolean;
}

export interface ComparisonResult {
    /** Per-channel statistics (R, G, B, A) */
    perChannel: ChannelStats[];
    /** Aggregate statistics across all channels */
    aggregate: AggregateStats;
    /** Error map: max absolute error across channels per pixel */
    errorMap?: Float32Array;
    /** Outlier mask: 1 where any channel error > tolerance */
    outlierMask?: Uint8Array;
    /** Image dimensions */
    width: number;
    height: number;
    /** Total pixel count */
    pixelCount: number;
    /** Overall pass/fail based on tolerance */
    passed: boolean;
}

const DEFAULT_TOLERANCES: Record<ToleranceLevel, number> = {
    strict: 0.5 / 65535,   // ~0.5 ULP for fp16
    standard: 1e-3,
    lenient: 1e-2,
};

/**
 * PixelComparator compares two images pixel by pixel.
 */
export class PixelComparator {
    private statsEngine: StatisticsEngine;

    constructor() {
        this.statsEngine = new StatisticsEngine();
    }

    /**
     * Compare two Float32Array images (RGBA format).
     * 
     * @param reference - Reference image data [width * height * 4]
     * @param test - Test image data [width * height * 4]
     * @param width - Image width
     * @param height - Image height
     * @param options - Comparison options
     * @returns ComparisonResult with detailed metrics
     */
    compare(
        reference: Float32Array,
        test: Float32Array,
        width: number,
        height: number,
        options: ComparisonOptions = {}
    ): ComparisonResult {
        const {
            tolerance = 'standard',
            computeSSIM = true,
            ssimWindowSize = 11,
            histogramBins = 64,
            generateErrorMap = true,
        } = options;

        const tol = typeof tolerance === 'number' ? tolerance : DEFAULT_TOLERANCES[tolerance];
        const pixelCount = width * height;

        // Validate input sizes
        if (reference.length !== pixelCount * 4 || test.length !== pixelCount * 4) {
            throw new Error(
                `Size mismatch: expected ${pixelCount * 4} values, got ref=${reference.length} test=${test.length}`
            );
        }

        // Compute per-channel statistics
        const channels: ChannelStats[] = [];
        const channelNames = ['R', 'G', 'B', 'A'] as const;

        for (let c = 0; c < 4; c++) {
            const refChannel = this.extractChannel(reference, c, pixelCount);
            const testChannel = this.extractChannel(test, c, pixelCount);
            
            const stats = this.statsEngine.computeChannelStats(
                refChannel,
                testChannel,
                width,
                height,
                channelNames[c],
                {
                    tolerance: tol,
                    computeSSIM,
                    ssimWindowSize,
                    histogramBins,
                }
            );
            channels.push(stats);
        }

        // Compute aggregate statistics
        const aggregate = this.statsEngine.computeAggregateStats(channels, tol);

        // Generate error map and outlier mask
        let errorMap: Float32Array | undefined;
        let outlierMask: Uint8Array | undefined;

        if (generateErrorMap) {
            errorMap = new Float32Array(pixelCount);
            outlierMask = new Uint8Array(pixelCount);

            for (let i = 0; i < pixelCount; i++) {
                let maxError = 0;
                let isOutlier = 0;

                for (let c = 0; c < 4; c++) {
                    const refIdx = i * 4 + c;
                    const error = Math.abs(reference[refIdx] - test[refIdx]);
                    maxError = Math.max(maxError, error);
                    if (error > tol) isOutlier = 1;
                }

                errorMap[i] = maxError;
                outlierMask[i] = isOutlier;
            }
        }

        // Overall pass/fail
        const passed = aggregate.outlierCount === 0;

        return {
            perChannel: channels,
            aggregate,
            errorMap,
            outlierMask,
            width,
            height,
            pixelCount,
            passed,
        };
    }

    /**
     * Extract a single channel from RGBA Float32Array.
     */
    private extractChannel(data: Float32Array, channel: number, pixelCount: number): Float32Array {
        const out = new Float32Array(pixelCount);
        for (let i = 0; i < pixelCount; i++) {
            out[i] = data[i * 4 + channel];
        }
        return out;
    }

    /**
     * Quick comparison returning only pass/fail and max error.
     */
    quickCompare(
        reference: Float32Array,
        test: Float32Array,
        tolerance: ToleranceLevel = 'standard'
    ): { passed: boolean; maxError: number; outlierCount: number } {
        const tol = typeof tolerance === 'number' ? tolerance : DEFAULT_TOLERANCES[tolerance];
        const pixelCount = reference.length / 4;

        let maxError = 0;
        let outlierCount = 0;

        for (let i = 0; i < reference.length; i++) {
            const error = Math.abs(reference[i] - test[i]);
            if (error > maxError) maxError = error;
            if (error > tol) outlierCount++;
        }

        return {
            passed: outlierCount === 0,
            maxError,
            outlierCount,
        };
    }

    /**
     * Compare with per-pixel callback for custom analysis.
     */
    compareWithCallback(
        reference: Float32Array,
        test: Float32Array,
        width: number,
        height: number,
        callback: (x: number, y: number, ref: number[], test: number[], errors: number[]) => void
    ): void {
        const pixelCount = width * height;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                const ref = [
                    reference[idx],
                    reference[idx + 1],
                    reference[idx + 2],
                    reference[idx + 3],
                ];
                const tst = [
                    test[idx],
                    test[idx + 1],
                    test[idx + 2],
                    test[idx + 3],
                ];
                const errors = [
                    Math.abs(ref[0] - tst[0]),
                    Math.abs(ref[1] - tst[1]),
                    Math.abs(ref[2] - tst[2]),
                    Math.abs(ref[3] - tst[3]),
                ];
                callback(x, y, ref, tst, errors);
            }
        }
    }
}

/**
 * Create a pixel comparator instance.
 */
export function createComparator(): PixelComparator {
    return new PixelComparator();
}