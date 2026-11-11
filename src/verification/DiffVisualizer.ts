/**
 * DiffVisualizer — Generate visual difference images for shader verification.
 * 
 * Creates heatmaps, side-by-side comparisons, and difference images
 * using pure JavaScript with pngjs (no native dependencies).
 */

import { PNG } from 'pngjs';

export type VisualizationMode = 'heatmap' | 'side-by-side' | 'difference' | 'overlay' | 'all';

export interface VisualizationOptions {
    /** Gain factor for difference visualization (default: 10) */
    gain?: number;
    /** Color map for heatmap: 'viridis' | 'jet' | 'hot' | 'cool' (default: 'hot') */
    colormap?: 'viridis' | 'jet' | 'hot' | 'cool';
    /** Whether to include colorbar in heatmap */
    colorbar?: boolean;
    /** Label font size (not used in pngjs version) */
    fontSize?: number;
    /** Padding between panels in side-by-side */
    panelPadding?: number;
}

export interface DiffImageResult {
    /** PNG buffer */
    buffer: Buffer;
    /** Image width */
    width: number;
    /** Image height */
    height: number;
    /** Mode used */
    mode: VisualizationMode;
}

/**
 * DiffVisualizer creates visual representations of image differences.
 */
export class DiffVisualizer {
    /**
     * Generate a heatmap of the error map.
     */
    generateHeatmap(
        errorMap: Float32Array,
        width: number,
        height: number,
        options: VisualizationOptions = {}
    ): DiffImageResult {
        const {
            colormap = 'hot',
            gain = 1.0,
            colorbar = true,
        } = options;

        const canvasWidth = colorbar ? width + 60 : width;
        const png = new PNG({ width: canvasWidth, height, colorType: 6 }); // RGBA

        // Find min/max for normalization
        let minVal = Infinity;
        let maxVal = -Infinity;
        for (const v of errorMap) {
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }
        const range = maxVal - minVal || 1;

        // Apply colormap to main image area
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x);
                const normalized = Math.min(1, Math.max(0, (errorMap[idx] - minVal) / range * gain));
                const color = this.applyColormap(normalized, colormap);
                const pngIdx = (y * canvasWidth + x) * 4;
                png.data[pngIdx] = color[0];
                png.data[pngIdx + 1] = color[1];
                png.data[pngIdx + 2] = color[2];
                png.data[pngIdx + 3] = 255;
            }
        }

        // Draw colorbar
        if (colorbar) {
            const barWidth = 30;
            const barHeight = height - 40;
            const barX = width + 15;
            const barY = 20;

            for (let y = 0; y < barHeight; y++) {
                const t = 1 - y / barHeight; // 0 at bottom, 1 at top
                const color = this.applyColormap(t, colormap);
                for (let x = 0; x < barWidth; x++) {
                    const px = barX + x;
                    const py = barY + y;
                    const pngIdx = (py * canvasWidth + px) * 4;
                    png.data[pngIdx] = color[0];
                    png.data[pngIdx + 1] = color[1];
                    png.data[pngIdx + 2] = color[2];
                    png.data[pngIdx + 3] = 255;
                }
            }

            // Colorbar border
            for (let y = 0; y < barHeight; y++) {
                const py = barY + y;
                // Left border
                let pngIdx = (py * canvasWidth + barX) * 4;
                png.data[pngIdx] = 255; png.data[pngIdx + 1] = 255; png.data[pngIdx + 2] = 255; png.data[pngIdx + 3] = 255;
                // Right border
                pngIdx = (py * canvasWidth + barX + barWidth - 1) * 4;
                png.data[pngIdx] = 255; png.data[pngIdx + 1] = 255; png.data[pngIdx + 2] = 255; png.data[pngIdx + 3] = 255;
            }
            for (let x = 0; x < barWidth; x++) {
                const px = barX + x;
                // Top border
                let pngIdx = (barY * canvasWidth + px) * 4;
                png.data[pngIdx] = 255; png.data[pngIdx + 1] = 255; png.data[pngIdx + 2] = 255; png.data[pngIdx + 3] = 255;
                // Bottom border
                pngIdx = ((barY + barHeight - 1) * canvasWidth + px) * 4;
                png.data[pngIdx] = 255; png.data[pngIdx + 1] = 255; png.data[pngIdx + 2] = 255; png.data[pngIdx + 3] = 255;
            }
        }

        const buffer = PNG.sync.write(png);
        return { buffer, width: canvasWidth, height, mode: 'heatmap' };
    }

    /**
     * Generate side-by-side comparison: Reference | Test | Difference
     */
    generateSideBySide(
        reference: Float32Array,
        test: Float32Array,
        width: number,
        height: number,
        options: VisualizationOptions = {}
    ): DiffImageResult {
        const { gain = 10, panelPadding = 10 } = options;

        const panelWidth = width;
        const totalWidth = panelWidth * 3 + panelPadding * 2;
        const totalHeight = height + 30; // Space for labels
        const png = new PNG({ width: totalWidth, height: totalHeight, colorType: 6 });

        // Draw reference panel
        this.drawFloat32Image(png, reference, width, height, 0, 0);

        // Draw test panel
        this.drawFloat32Image(png, test, width, height, panelWidth + panelPadding, 0);

        // Draw difference panel
        const diffData = new Float32Array(width * height * 4);
        for (let i = 0; i < reference.length; i++) {
            diffData[i] = Math.abs(reference[i] - test[i]) * gain;
        }
        this.drawFloat32Image(png, diffData, width, height, 2 * (panelWidth + panelPadding), 0);

        // Draw simple text labels as pixel patterns (since pngjs doesn't support text)
        // We'll add colored bars at the top of each panel instead
        this.drawLabelBar(png, 0, 0, panelWidth, [200, 100, 100]);       // Red-ish for Reference
        this.drawLabelBar(png, panelWidth + panelPadding, 0, panelWidth, [100, 200, 100]); // Green-ish for Test
        this.drawLabelBar(png, 2 * (panelWidth + panelPadding), 0, panelWidth, [100, 100, 200]); // Blue-ish for Diff

        const buffer = PNG.sync.write(png);
        return { buffer, width: totalWidth, height: totalHeight, mode: 'side-by-side' };
    }

    /**
     * Generate difference image: |Reference - Test| * gain
     */
    generateDifference(
        reference: Float32Array,
        test: Float32Array,
        width: number,
        height: number,
        options: VisualizationOptions = {}
    ): DiffImageResult {
        const { gain = 10, colormap = 'hot' } = options;

        const diffData = new Float32Array(width * height);
        let maxDiff = 0;

        for (let i = 0; i < width * height; i++) {
            let maxChannelDiff = 0;
            for (let c = 0; c < 4; c++) {
                const diff = Math.abs(reference[i * 4 + c] - test[i * 4 + c]);
                if (diff > maxChannelDiff) maxChannelDiff = diff;
            }
            diffData[i] = maxChannelDiff * gain;
            if (diffData[i] > maxDiff) maxDiff = diffData[i];
        }

        const png = new PNG({ width, height, colorType: 6 });

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;
                const normalized = Math.min(1, diffData[idx] / (maxDiff || 1));
                const color = this.applyColormap(normalized, colormap);
                const pngIdx = idx * 4;
                png.data[pngIdx] = color[0];
                png.data[pngIdx + 1] = color[1];
                png.data[pngIdx + 2] = color[2];
                png.data[pngIdx + 3] = 255;
            }
        }

        const buffer = PNG.sync.write(png);
        return { buffer, width, height, mode: 'difference' };
    }

    /**
     * Generate overlay: reference with error contours
     */
    generateOverlay(
        reference: Float32Array,
        test: Float32Array,
        width: number,
        height: number,
        options: VisualizationOptions = {}
    ): DiffImageResult {
        const { gain = 10 } = options;

        const png = new PNG({ width, height, colorType: 6 });

        // Draw reference as base
        this.drawFloat32Image(png, reference, width, height, 0, 0);

        // Overlay error as semi-transparent red
        for (let i = 0; i < width * height; i++) {
            let maxError = 0;
            for (let c = 0; c < 4; c++) {
                const err = Math.abs(reference[i * 4 + c] - test[i * 4 + c]);
                if (err > maxError) maxError = err;
            }
            const intensity = Math.min(1, maxError * gain);
            const pngIdx = i * 4;
            // Alpha blend: red overlay on top of reference
            const srcR = png.data[pngIdx];
            const srcG = png.data[pngIdx + 1];
            const srcB = png.data[pngIdx + 2];
            const alpha = intensity * 0.7; // 70% max opacity
            png.data[pngIdx] = Math.round(srcR * (1 - alpha) + 255 * alpha);
            png.data[pngIdx + 1] = Math.round(srcG * (1 - alpha));
            png.data[pngIdx + 2] = Math.round(srcB * (1 - alpha));
            png.data[pngIdx + 3] = 255;
        }

        const buffer = PNG.sync.write(png);
        return { buffer, width, height, mode: 'overlay' };
    }

    /**
     * Generate all visualization modes at once.
     */
    generateAll(
        reference: Float32Array,
        test: Float32Array,
        errorMap: Float32Array,
        width: number,
        height: number,
        options: VisualizationOptions = {}
    ): Map<VisualizationMode, DiffImageResult> {
        const results = new Map<VisualizationMode, DiffImageResult>();

        results.set('heatmap', this.generateHeatmap(errorMap, width, height, options));
        results.set('side-by-side', this.generateSideBySide(reference, test, width, height, options));
        results.set('difference', this.generateDifference(reference, test, width, height, options));
        results.set('overlay', this.generateOverlay(reference, test, width, height, options));

        return results;
    }

    /**
     * Draw a Float32Array RGBA image to PNG at offset.
     */
    private drawFloat32Image(
        png: PNG,
        data: Float32Array,
        width: number,
        height: number,
        offsetX: number,
        offsetY: number
    ): void {
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const srcIdx = (y * width + x) * 4;
                const dstX = offsetX + x;
                const dstY = offsetY + y;
                const dstIdx = (dstY * png.width + dstX) * 4;
                png.data[dstIdx] = Math.round(Math.min(1, Math.max(0, data[srcIdx])) * 255);
                png.data[dstIdx + 1] = Math.round(Math.min(1, Math.max(0, data[srcIdx + 1])) * 255);
                png.data[dstIdx + 2] = Math.round(Math.min(1, Math.max(0, data[srcIdx + 2])) * 255);
                png.data[dstIdx + 3] = Math.round(Math.min(1, Math.max(0, data[srcIdx + 3])) * 255);
            }
        }
    }

    /**
     * Draw a colored label bar at the top of a panel.
     */
    private drawLabelBar(png: PNG, x: number, y: number, width: number, color: [number, number, number]): void {
        const barHeight = 20;
        for (let by = 0; by < barHeight; by++) {
            for (let bx = 0; bx < width; bx++) {
                const px = x + bx;
                const py = y + by;
                if (px < png.width && py < png.height) {
                    const idx = (py * png.width + px) * 4;
                    png.data[idx] = color[0];
                    png.data[idx + 1] = color[1];
                    png.data[idx + 2] = color[2];
                    png.data[idx + 3] = 255;
                }
            }
        }
    }

    /**
     * Apply colormap to normalized value [0,1].
     * Returns [R, G, B] each 0-255.
     */
    private applyColormap(value: number, colormap: string): [number, number, number] {
        const v = Math.min(1, Math.max(0, value));

        switch (colormap) {
            case 'hot':
                return this.colormapHot(v);
            case 'jet':
                return this.colormapJet(v);
            case 'viridis':
                return this.colormapViridis(v);
            case 'cool':
                return this.colormapCool(v);
            default:
                return this.colormapHot(v);
        }
    }

    private colormapHot(v: number): [number, number, number] {
        // Black -> Red -> Yellow -> White
        if (v < 1/3) {
            return [Math.round(v * 3 * 255), 0, 0];
        } else if (v < 2/3) {
            return [255, Math.round((v - 1/3) * 3 * 255), 0];
        } else {
            const t = (v - 2/3) * 3;
            return [255, 255, Math.round(t * 255)];
        }
    }

    private colormapJet(v: number): [number, number, number] {
        // Blue -> Cyan -> Yellow -> Red
        const fourV = 4 * v;
        let r, g, b;

        if (fourV < 1) {
            r = 0; g = fourV; b = 1;
        } else if (fourV < 2) {
            r = 0; g = 1; b = 2 - fourV;
        } else if (fourV < 3) {
            r = fourV - 2; g = 1; b = 0;
        } else {
            r = 1; g = 4 - fourV; b = 0;
        }

        return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
    }

    private colormapViridis(v: number): [number, number, number] {
        // Approximate viridis colormap
        const v2 = v * v;
        const v3 = v2 * v;
        const r = 0.2309 + 0.6708 * v - 0.6392 * v2 + 0.1894 * v3;
        const g = 0.0978 + 1.0682 * v - 0.9264 * v2 + 0.1297 * v3;
        const b = 0.4375 - 0.4671 * v + 0.3717 * v2 - 0.0934 * v3;
        return [
            Math.round(Math.min(1, Math.max(0, r)) * 255),
            Math.round(Math.min(1, Math.max(0, g)) * 255),
            Math.round(Math.min(1, Math.max(0, b)) * 255),
        ];
    }

    private colormapCool(v: number): [number, number, number] {
        // Cyan -> Magenta
        return [
            Math.round(v * 255),
            0,
            Math.round((1 - v) * 255),
        ];
    }

    /**
     * Save visualization to file.
     */
    async saveToFile(result: DiffImageResult, filePath: string): Promise<void> {
        const fs = await import('fs');
        fs.writeFileSync(filePath, result.buffer);
    }
}

/**
 * Create a diff visualizer instance.
 */
export function createVisualizer(): DiffVisualizer {
    return new DiffVisualizer();
}