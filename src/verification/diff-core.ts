/**
 * diff-core — Browser-safe, device-free pixel difference core.
 *
 * Pure TypeScript/ESM: no imports, no WebGPU device, no Node built-ins and no
 * third-party packages. It can therefore run unchanged in a browser worker,
 * a browser main thread or plain Node.
 *
 * CLI-equivalence notes (see `PixelComparator.ts` / `StatisticsEngine.ts`):
 *  - MAE uses the same definition as the CLI: `sum(|ref - test|) / sampleCount`.
 *    Because every channel has the same pixel count, the overall MAE is
 *    identical to the CLI aggregate MAE (the mean of the four per-channel MAEs).
 *  - PSNR uses the CLI formula `20 * log10(MAX / RMSE)`, but with the 8-bit
 *    full-scale value `MAX = 255` instead of the CLI's normalized `MAX = 1.0`.
 *    When MSE is exactly 0 the result is `Infinity` (same as the CLI).
 *  - Max abs diff is the maximum absolute difference over every channel/sample.
 *  - The heatmap is a pure `Uint8Array` RGBA image (no `pngjs`): the red
 *    channel carries the per-pixel maximum absolute difference (0..255) and
 *    the pixel is fully opaque.
 */

export type RGBAByteData = Uint8Array | Uint8ClampedArray;

/** Number of channels in RGBA8 data. */
export const DIFF_CHANNELS = 4;

/** Full-scale value used for 8-bit data (PSNR peak / heatmap range). */
export const DIFF_FULL_SCALE = 255;

/** Per-channel mean absolute error, in 8-bit units (0..255). */
export interface PerChannelMAE {
    R: number;
    G: number;
    B: number;
    A: number;
}

/** Per-channel maximum absolute difference, in 8-bit units (0..255). */
export interface PerChannelMaxAbsDiff {
    R: number;
    G: number;
    B: number;
    A: number;
}

/** Result of comparing two same-sized RGBA8 images. */
export interface DiffMetrics {
    /** Image width in pixels. */
    width: number;
    /** Image height in pixels. */
    height: number;
    /** Total pixel count (width * height). */
    pixelCount: number;
    /** Mean absolute error per channel (0..255). */
    perChannelMAE: PerChannelMAE;
    /** Maximum absolute difference per channel (0..255). */
    perChannelMaxAbsDiff: PerChannelMaxAbsDiff;
    /** Overall mean absolute error over every channel and pixel (0..255). */
    mae: number;
    /** Overall maximum absolute difference over every channel (0..255). */
    maxAbsDiff: number;
    /** Mean squared error over every channel sample. */
    mse: number;
    /** Root mean squared error over every channel sample. */
    rmse: number;
    /** PSNR in dB using a full-scale value of 255; `Infinity` when MSE is 0. */
    psnr: number;
    /** Number of pixels where at least one channel differs. */
    diffPixelCount: number;
    /** Fraction (0..1) of pixels where at least one channel differs. */
    diffPixelRatio: number;
    /** Percentage (0..100) of pixels where at least one channel differs. */
    diffPixelPercent: number;
}

function assertDimensions(width: number, height: number): void {
    if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
        throw new Error(`Invalid image dimensions: ${width}x${height}`);
    }
}

function assertBuffers(
    reference: RGBAByteData,
    test: RGBAByteData,
    expectedLength: number
): void {
    if (reference.length !== expectedLength || test.length !== expectedLength) {
        throw new Error(
            `Size mismatch: expected ${expectedLength} values, ` +
            `got ref=${reference.length} test=${test.length}`
        );
    }
}

/**
 * Compare two same-sized RGBA8 images and compute the difference metrics.
 *
 * @param reference - Reference pixels, RGBA order, `width * height * 4` bytes.
 * @param test - Test pixels, RGBA order, `width * height * 4` bytes.
 * @param width - Image width in pixels.
 * @param height - Image height in pixels.
 */
export function comparePixels(
    reference: RGBAByteData,
    test: RGBAByteData,
    width: number,
    height: number
): DiffMetrics {
    assertDimensions(width, height);

    const pixelCount = width * height;
    const expectedLength = pixelCount * DIFF_CHANNELS;
    assertBuffers(reference, test, expectedLength);

    const sumAbs = [0, 0, 0, 0];
    const maxAbs = [0, 0, 0, 0];
    let totalAbs = 0;
    let totalSquared = 0;
    let maxAbsDiff = 0;
    let diffPixelCount = 0;

    for (let pixel = 0; pixel < pixelCount; pixel++) {
        const base = pixel * DIFF_CHANNELS;
        let pixelMax = 0;

        for (let channel = 0; channel < DIFF_CHANNELS; channel++) {
            const index = base + channel;
            const diff = Math.abs(reference[index] - test[index]);

            sumAbs[channel] += diff;
            totalAbs += diff;
            totalSquared += diff * diff;

            if (diff > maxAbs[channel]) maxAbs[channel] = diff;
            if (diff > pixelMax) pixelMax = diff;
        }

        if (pixelMax > 0) diffPixelCount++;
        if (pixelMax > maxAbsDiff) maxAbsDiff = pixelMax;
    }

    const sampleCount = pixelCount * DIFF_CHANNELS;
    const mae = totalAbs / sampleCount;
    const mse = totalSquared / sampleCount;
    const rmse = Math.sqrt(mse);
    const psnr = mse > 0 ? 20 * Math.log10(DIFF_FULL_SCALE / rmse) : Infinity;

    return {
        width,
        height,
        pixelCount,
        perChannelMAE: {
            R: sumAbs[0] / pixelCount,
            G: sumAbs[1] / pixelCount,
            B: sumAbs[2] / pixelCount,
            A: sumAbs[3] / pixelCount,
        },
        perChannelMaxAbsDiff: {
            R: maxAbs[0],
            G: maxAbs[1],
            B: maxAbs[2],
            A: maxAbs[3],
        },
        mae,
        maxAbsDiff,
        mse,
        rmse,
        psnr,
        diffPixelCount,
        diffPixelRatio: diffPixelCount / pixelCount,
        diffPixelPercent: (diffPixelCount / pixelCount) * 100,
    };
}

/**
 * Build a diff heatmap for two same-sized RGBA8 images.
 *
 * The output is a raw RGBA `Uint8Array` with the same pixel dimensions as the
 * inputs: `R = max(|ref - test|)` over the four channels of the pixel,
 * `G = B = 0`, `A = 255`. No encoder (e.g. `pngjs`) is involved, so the result
 * can be uploaded directly to a browser canvas (`ImageData`) or a GPU texture.
 *
 * @param reference - Reference pixels, RGBA order, `width * height * 4` bytes.
 * @param test - Test pixels, RGBA order, `width * height * 4` bytes.
 * @param width - Image width in pixels.
 * @param height - Image height in pixels.
 * @returns RGBA heatmap bytes, `width * height * 4` long.
 */
export function generateDiffHeatmap(
    reference: RGBAByteData,
    test: RGBAByteData,
    width: number,
    height: number
): Uint8Array {
    assertDimensions(width, height);

    const pixelCount = width * height;
    const expectedLength = pixelCount * DIFF_CHANNELS;
    assertBuffers(reference, test, expectedLength);

    const heatmap = new Uint8Array(expectedLength);

    for (let pixel = 0; pixel < pixelCount; pixel++) {
        const base = pixel * DIFF_CHANNELS;
        let pixelMax = 0;

        for (let channel = 0; channel < DIFF_CHANNELS; channel++) {
            const diff = Math.abs(reference[base + channel] - test[base + channel]);
            if (diff > pixelMax) pixelMax = diff;
        }

        heatmap[base] = pixelMax;
        heatmap[base + 1] = 0;
        heatmap[base + 2] = 0;
        heatmap[base + 3] = 255;
    }

    return heatmap;
}
