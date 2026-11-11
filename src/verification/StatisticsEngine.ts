/**
 * StatisticsEngine — Statistical analysis for image comparison.
 * 
 * Computes RMSE, MAE, PSNR, SSIM, histograms, percentiles, and other metrics.
 */

export interface ChannelStats {
    channel: 'R' | 'G' | 'B' | 'A';
    /** Maximum absolute error */
    maxAbsError: number;
    /** Mean absolute error */
    mae: number;
    /** Root mean square error */
    rmse: number;
    /** Peak signal-to-noise ratio in dB */
    psnr: number;
    /** Structural similarity index (0-1, 1=identical) */
    ssim: number;
    /** Error histogram (log-spaced bins) */
    histogram: number[];
    /** Histogram bin edges */
    histogramBins: number[];
    /** Outlier count (error > tolerance) */
    outlierCount: number;
    /** Percentiles of absolute error */
    percentiles: {
        p50: number;
        p90: number;
        p95: number;
        p99: number;
        p999: number;
    };
    /** Mean of reference channel */
    refMean: number;
    /** Std dev of reference channel */
    refStdDev: number;
    /** Mean of test channel */
    testMean: number;
    /** Std dev of test channel */
    testStdDev: number;
}

export interface AggregateStats {
    /** Overall max absolute error across all channels */
    maxAbsError: number;
    /** Overall MAE across all channels */
    mae: number;
    /** Overall RMSE across all channels */
    rmse: number;
    /** Overall PSNR (worst channel) */
    psnr: number;
    /** Overall SSIM (average across channels) */
    ssim: number;
    /** Total outlier count across all channels */
    outlierCount: number;
    /** Per-channel outlier counts */
    outlierCountPerChannel: number[];
    /** Worst channel index (0=R, 1=G, 2=B, 3=A) */
    worstChannel: number;
}

export interface StatsOptions {
    /** Tolerance for outlier counting */
    tolerance: number;
    /** Whether to compute SSIM */
    computeSSIM: boolean;
    /** SSIM window size */
    ssimWindowSize: number;
    /** Number of histogram bins */
    histogramBins: number;
}

const DEFAULT_OPTIONS: StatsOptions = {
    tolerance: 1e-3,
    computeSSIM: true,
    ssimWindowSize: 11,
    histogramBins: 64,
};

/**
 * StatisticsEngine computes image quality metrics.
 */
export class StatisticsEngine {
    /**
     * Compute statistics for a single channel.
     */
    computeChannelStats(
        reference: Float32Array,
        test: Float32Array,
        width: number,
        height: number,
        channel: 'R' | 'G' | 'B' | 'A',
        options: Partial<StatsOptions> = {}
    ): ChannelStats {
        const opts = { ...DEFAULT_OPTIONS, ...options };
        const pixelCount = reference.length;

        if (test.length !== pixelCount) {
            throw new Error(`Channel length mismatch: ref=${pixelCount} test=${test.length}`);
        }

        // Compute absolute errors
        const absErrors = new Float32Array(pixelCount);
        let sumAbsError = 0;
        let sumSqError = 0;
        let maxAbsError = 0;
        let outlierCount = 0;

        let refSum = 0;
        let testSum = 0;
        let refSumSq = 0;
        let testSumSq = 0;
        let refTestSum = 0;

        for (let i = 0; i < pixelCount; i++) {
            const ref = reference[i];
            const tst = test[i];
            const error = Math.abs(ref - tst);

            absErrors[i] = error;
            sumAbsError += error;
            sumSqError += error * error;
            if (error > maxAbsError) maxAbsError = error;
            if (error > opts.tolerance) outlierCount++;

            refSum += ref;
            testSum += tst;
            refSumSq += ref * ref;
            testSumSq += tst * tst;
            refTestSum += ref * tst;
        }

        const mae = sumAbsError / pixelCount;
        const mse = sumSqError / pixelCount;
        const rmse = Math.sqrt(mse);

        // PSNR: 20 * log10(MAX / RMSE), where MAX=1.0 for normalized [0,1] data
        const psnr = mse > 0 ? 20 * Math.log10(1.0 / rmse) : Infinity;

        // Reference and test statistics
        const refMean = refSum / pixelCount;
        const testMean = testSum / pixelCount;
        const refVar = refSumSq / pixelCount - refMean * refMean;
        const testVar = testSumSq / pixelCount - testMean * testMean;
        const refStdDev = Math.sqrt(Math.max(0, refVar));
        const testStdDev = Math.sqrt(Math.max(0, testVar));

        // Covariance and correlation
        const covariance = refTestSum / pixelCount - refMean * testMean;
        const correlation = (refStdDev > 0 && testStdDev > 0)
            ? covariance / (refStdDev * testStdDev)
            : 1.0;

        // SSIM
        let ssim = 1.0;
        if (opts.computeSSIM) {
            ssim = this.computeSSIM(reference, test, width, height, opts.ssimWindowSize);
        }

        // Histogram (log-spaced bins from 1e-6 to 1.0)
        const { histogram, bins } = this.computeHistogram(absErrors, opts.histogramBins);

        // Percentiles
        const percentiles = this.computePercentiles(absErrors);

        return {
            channel,
            maxAbsError,
            mae,
            rmse,
            psnr,
            ssim,
            histogram,
            histogramBins: bins,
            outlierCount,
            percentiles,
            refMean,
            refStdDev,
            testMean,
            testStdDev,
        };
    }

    /**
     * Compute aggregate statistics across channels.
     */
    computeAggregateStats(channels: ChannelStats[], tolerance: number): AggregateStats {
        let maxAbsError = 0;
        let totalMAE = 0;
        let totalMSE = 0;
        let minPSNR = Infinity;
        let totalSSIM = 0;
        let totalOutliers = 0;
        const outlierCountPerChannel: number[] = [];

        for (const ch of channels) {
            if (ch.maxAbsError > maxAbsError) maxAbsError = ch.maxAbsError;
            totalMAE += ch.mae;
            totalMSE += ch.rmse * ch.rmse;
            if (ch.psnr < minPSNR) minPSNR = ch.psnr;
            totalSSIM += ch.ssim;
            totalOutliers += ch.outlierCount;
            outlierCountPerChannel.push(ch.outlierCount);
        }

        const channelCount = channels.length;
        const mae = totalMAE / channelCount;
        const rmse = Math.sqrt(totalMSE / channelCount);
        const ssim = totalSSIM / channelCount;

        // Find worst channel (by maxAbsError)
        let worstChannel = 0;
        let worstError = channels[0].maxAbsError;
        for (let i = 1; i < channelCount; i++) {
            if (channels[i].maxAbsError > worstError) {
                worstError = channels[i].maxAbsError;
                worstChannel = i;
            }
        }

        return {
            maxAbsError,
            mae,
            rmse,
            psnr: minPSNR,
            ssim,
            outlierCount: totalOutliers,
            outlierCountPerChannel,
            worstChannel,
        };
    }

    /**
     * Compute SSIM (Structural Similarity Index) between two images.
     * Uses 11x11 Gaussian window, k1=0.01, k2=0.03.
     */
    private computeSSIM(
        ref: Float32Array,
        test: Float32Array,
        width: number,
        height: number,
        windowSize: number
    ): number {
        const k1 = 0.01;
        const k2 = 0.03;
        const L = 1.0; // Dynamic range for normalized [0,1] data
        const c1 = (k1 * L) ** 2;
        const c2 = (k2 * L) ** 2;

        // Generate Gaussian window
        const radius = Math.floor(windowSize / 2);
        const window = this.generateGaussianWindow(windowSize, 1.5);
        const windowSum = window.reduce((a, b) => a + b, 0);
        const normalizedWindow = window.map(v => v / windowSum);

        let totalSSIM = 0;
        let validWindows = 0;

        // Slide window across image (with padding)
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let muX = 0, muY = 0;
                let sigmaX2 = 0, sigmaY2 = 0, sigmaXY = 0;

                // Apply window
                for (let wy = -radius; wy <= radius; wy++) {
                    const iy = Math.max(0, Math.min(height - 1, y + wy));
                    const wyIdx = wy + radius;

                    for (let wx = -radius; wx <= radius; wx++) {
                        const ix = Math.max(0, Math.min(width - 1, x + wx));
                        const wxIdx = wx + radius;
                        const w = normalizedWindow[wyIdx * windowSize + wxIdx];

                        const idx = (iy * width + ix);
                        const px = ref[idx];
                        const py = test[idx];

                        muX += w * px;
                        muY += w * py;
                    }
                }

                // Second pass for variances
                for (let wy = -radius; wy <= radius; wy++) {
                    const iy = Math.max(0, Math.min(height - 1, y + wy));
                    const wyIdx = wy + radius;

                    for (let wx = -radius; wx <= radius; wx++) {
                        const ix = Math.max(0, Math.min(width - 1, x + wx));
                        const wxIdx = wx + radius;
                        const w = normalizedWindow[wyIdx * windowSize + wxIdx];

                        const idx = (iy * width + ix);
                        const px = ref[idx];
                        const py = test[idx];

                        const dx = px - muX;
                        const dy = py - muY;

                        sigmaX2 += w * dx * dx;
                        sigmaY2 += w * dy * dy;
                        sigmaXY += w * dx * dy;
                    }
                }

                const sigmaX = Math.sqrt(Math.max(0, sigmaX2));
                const sigmaY = Math.sqrt(Math.max(0, sigmaY2));

                const numerator = (2 * muX * muY + c1) * (2 * sigmaXY + c2);
                const denominator = (muX * muX + muY * muY + c1) * (sigmaX2 + sigmaY2 + c2);

                if (denominator > 0) {
                    totalSSIM += numerator / denominator;
                    validWindows++;
                }
            }
        }

        return validWindows > 0 ? totalSSIM / validWindows : 1.0;
    }

    /**
     * Generate 1D Gaussian kernel.
     */
    private generateGaussianWindow(size: number, sigma: number): number[] {
        const radius = Math.floor(size / 2);
        const window = new Array(size);
        let sum = 0;

        for (let i = 0; i < size; i++) {
            const x = i - radius;
            const val = Math.exp(-(x * x) / (2 * sigma * sigma));
            window[i] = val;
            sum += val;
        }

        // Normalize
        for (let i = 0; i < size; i++) {
            window[i] /= sum;
        }

        return window;
    }

    /**
     * Compute log-spaced histogram of absolute errors.
     */
    private computeHistogram(
        absErrors: Float32Array,
        bins: number
    ): { histogram: number[]; bins: number[] } {
        const histogram = new Array(bins).fill(0);
        const binEdges = new Array(bins + 1);

        // Log-spaced bins from 1e-6 to 1.0
        const minVal = 1e-6;
        const maxVal = 1.0;
        const logMin = Math.log(minVal);
        const logMax = Math.log(maxVal);

        for (let i = 0; i <= bins; i++) {
            const t = i / bins;
            binEdges[i] = Math.exp(logMin + t * (logMax - logMin));
        }
        // Ensure first bin starts at 0
        binEdges[0] = 0;

        for (const error of absErrors) {
            if (error <= 0) {
                histogram[0]++;
                continue;
            }
            if (error >= maxVal) {
                histogram[bins - 1]++;
                continue;
            }

            // Find bin using log scale
            const logError = Math.log(error);
            const binIndex = Math.floor(((logError - logMin) / (logMax - logMin)) * bins);
            const clampedIndex = Math.max(0, Math.min(bins - 1, binIndex));
            histogram[clampedIndex]++;
        }

        return { histogram, bins: binEdges };
    }

    /**
     * Compute percentiles of absolute errors.
     */
    private computePercentiles(absErrors: Float32Array): {
        p50: number;
        p90: number;
        p95: number;
        p99: number;
        p999: number;
    } {
        const sorted = new Float32Array(absErrors);
        sorted.sort((a, b) => a - b);
        const n = sorted.length;

        const percentile = (p: number) => {
            const idx = Math.min(n - 1, Math.floor(p * n));
            return sorted[idx];
        };

        return {
            p50: percentile(0.50),
            p90: percentile(0.90),
            p95: percentile(0.95),
            p99: percentile(0.99),
            p999: percentile(0.999),
        };
    }

    /**
     * Compute basic statistics for an array.
     */
    static computeBasicStats(data: Float32Array): {
        mean: number;
        stdDev: number;
        min: number;
        max: number;
        sum: number;
    } {
        let sum = 0;
        let min = Infinity;
        let max = -Infinity;

        for (const v of data) {
            sum += v;
            if (v < min) min = v;
            if (v > max) max = v;
        }

        const mean = sum / data.length;

        let sumSq = 0;
        for (const v of data) {
            const d = v - mean;
            sumSq += d * d;
        }

        return {
            mean,
            stdDev: Math.sqrt(sumSq / data.length),
            min,
            max,
            sum,
        };
    }
}

/**
 * Create a statistics engine instance.
 */
export function createStatsEngine(): StatisticsEngine {
    return new StatisticsEngine();
}