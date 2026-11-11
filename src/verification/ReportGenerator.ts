/**
 * ReportGenerator — Generate verification reports in JSON, HTML, and Markdown formats.
 */

import { ComparisonResult } from './PixelComparator';
import { DiffImageResult } from './DiffVisualizer';
import { EffectPair } from './EffectPairResolver';

export type ReportFormat = 'json' | 'html' | 'markdown';

export interface ReportOptions {
    /** Output format */
    format: ReportFormat;
    /** Output directory */
    outputDir: string;
    /** Base filename (without extension) */
    baseName?: string;
    /** Include embedded images in HTML (base64) */
    embedImages?: boolean;
    /** Include detailed per-channel stats */
    detailed?: boolean;
    /** Include histogram data */
    includeHistograms?: boolean;
}

export interface VerificationReport {
    /** Effect pair information */
    pair: EffectPair;
    /** Comparison results */
    comparison: ComparisonResult;
    /** Timestamp */
    timestamp: string;
    /** Execution metadata */
    metadata: {
        hlslExecutionTime: number;
        wgslExecutionTime: number;
        totalTime: number;
        inputSize: [number, number];
        outputSize: [number, number];
    };
    /** Visualization results (if generated) */
    visualizations?: Map<string, DiffImageResult>;
}

/**
 * ReportGenerator creates verification reports in various formats.
 */
export class ReportGenerator {
    /**
     * Generate a report from verification results.
     */
    async generate(report: VerificationReport, options: ReportOptions): Promise<string[]> {
        const fs = await import('fs');
        const path = await import('path');

        const baseName = options.baseName || `verify_${report.pair.name}_${Date.now()}`;
        const outputDir = options.outputDir;

        // Ensure output directory exists
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }

        const generatedFiles: string[] = [];

        switch (options.format) {
            case 'json':
                generatedFiles.push(await this.generateJSON(report, outputDir, baseName, options));
                break;
            case 'html':
                generatedFiles.push(await this.generateHTML(report, outputDir, baseName, options));
                break;
            case 'markdown':
                generatedFiles.push(await this.generateMarkdown(report, outputDir, baseName, options));
                break;
        }

        return generatedFiles;
    }

    /**
     * Generate JSON report.
     */
    private async generateJSON(
        report: VerificationReport,
        outputDir: string,
        baseName: string,
        options: ReportOptions
    ): Promise<string> {
        const fs = await import('fs');
        const filePath = `${outputDir}/${baseName}.json`;

        const jsonData = {
            pair: {
                name: report.pair.name,
                category: report.pair.category,
                hlslPath: report.pair.hlslPath,
                wgslPath: report.pair.wgslPath,
            },
            timestamp: report.timestamp,
            metadata: report.metadata,
            comparison: {
                passed: report.comparison.passed,
                width: report.comparison.width,
                height: report.comparison.height,
                pixelCount: report.comparison.pixelCount,
                aggregate: report.comparison.aggregate,
                perChannel: options.detailed ? report.comparison.perChannel : undefined,
            },
        };

        fs.writeFileSync(filePath, JSON.stringify(jsonData, null, 2));
        return filePath;
    }

    /**
     * Generate HTML report with embedded visualizations.
     */
    private async generateHTML(
        report: VerificationReport,
        outputDir: string,
        baseName: string,
        options: ReportOptions
    ): Promise<string> {
        const fs = await import('fs');
        const filePath = `${outputDir}/${baseName}.html`;

        // Save visualization images if present
        const imagePaths: Map<string, string> = new Map();
        if (report.visualizations) {
            for (const [mode, result] of report.visualizations) {
                const imgName = `${baseName}_${mode}.png`;
                const imgPath = `${outputDir}/${imgName}`;
                fs.writeFileSync(imgPath, result.buffer);
                imagePaths.set(mode, imgName);
            }
        }

        const html = this.buildHTML(report, imagePaths, options);
        fs.writeFileSync(filePath, html);
        return filePath;
    }

    /**
     * Build HTML content.
     */
    private buildHTML(
        report: VerificationReport,
        imagePaths: Map<string, string>,
        options: ReportOptions
    ): string {
        const { comparison, pair, metadata, timestamp } = report;
        const passed = comparison.passed ? '✅ PASS' : '❌ FAIL';
        const passClass = comparison.passed ? 'pass' : 'fail';

        let html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shader Verification: ${pair.name}</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        .status { display: inline-block; padding: 8px 16px; border-radius: 4px; font-weight: bold; font-size: 1.2em; }
        .status.pass { background: #d4edda; color: #155724; }
        .status.fail { background: #f8d7da; color: #721c24; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .card { background: #f8f9fa; padding: 20px; border-radius: 6px; border: 1px solid #eee; }
        .card h3 { margin-top: 0; color: #495057; }
        .metric { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }
        .metric:last-child { border-bottom: none; }
        .metric-label { color: #6c757d; }
        .metric-value { font-weight: 500; font-family: monospace; }
        .visualizations { margin-top: 30px; }
        .viz-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .viz-item { text-align: center; }
        .viz-item img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
        .viz-item h4 { margin: 10px 0 5px; color: #495057; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; }
        .channel-R { border-left: 4px solid #dc3545; }
        .channel-G { border-left: 4px solid #28a745; }
        .channel-B { border-left: 4px solid #007bff; }
        .channel-A { border-left: 4px solid #6c757d; }
        .badge { padding: 2px 8px; border-radius: 3px; font-size: 0.85em; font-weight: bold; }
        .badge.ok { background: #d4edda; color: #155724; }
        .badge.warn { background: #fff3cd; color: #856404; }
        .badge.fail { background: #f8d7da; color: #721c24; }
        pre { background: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto; }
        code { font-family: 'SF Mono', Monaco, monospace; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Shader Verification Report: ${pair.name}</h1>
        <p><strong>Category:</strong> ${pair.category} | <strong>Timestamp:</strong> ${timestamp}</p>
        <span class="status ${passClass}">${passed}</span>

        <div class="grid">
            <div class="card">
                <h3>Effect Pair</h3>
                <div class="metric"><span class="metric-label">HLSL Source</span><span class="metric-value">${pair.hlslRelativePath || pair.hlslPath}</span></div>
                <div class="metric"><span class="metric-label">WGSL Source</span><span class="metric-value">${pair.wgslRelativePath || pair.wgslPath}</span></div>
            </div>
            <div class="card">
                <h3>Execution Metadata</h3>
                <div class="metric"><span class="metric-label">Input Size</span><span class="metric-value">${metadata.inputSize[0]}×${metadata.inputSize[1]}</span></div>
                <div class="metric"><span class="metric-label">Output Size</span><span class="metric-value">${metadata.outputSize[0]}×${metadata.outputSize[1]}</span></div>
                <div class="metric"><span class="metric-label">HLSL Time</span><span class="metric-value">${metadata.hlslExecutionTime.toFixed(2)} ms</span></div>
                <div class="metric"><span class="metric-label">WGSL Time</span><span class="metric-value">${metadata.wgslExecutionTime.toFixed(2)} ms</span></div>
                <div class="metric"><span class="metric-label">Total Time</span><span class="metric-value">${metadata.totalTime.toFixed(2)} ms</span></div>
            </div>
        </div>

        <div class="card">
            <h3>Aggregate Statistics</h3>
            <table>
                <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                <tbody>
                    <tr><td>Max Absolute Error</td><td><code>${comparison.aggregate.maxAbsError.toExponential(4)}</code></td></tr>
                    <tr><td>MAE (Mean Absolute Error)</td><td><code>${comparison.aggregate.mae.toExponential(4)}</code></td></tr>
                    <tr><td>RMSE (Root Mean Square Error)</td><td><code>${comparison.aggregate.rmse.toExponential(4)}</code></td></tr>
                    <tr><td>PSNR (Peak Signal-to-Noise Ratio)</td><td><code>${comparison.aggregate.psnr === Infinity ? '∞' : comparison.aggregate.psnr.toFixed(2) + ' dB'}</code></td></tr>
                    <tr><td>SSIM (Structural Similarity)</td><td><code>${comparison.aggregate.ssim.toFixed(6)}</code></td></tr>
                    <tr><td>Total Outliers</td><td><code>${comparison.aggregate.outlierCount}</code> / ${comparison.pixelCount * 4}</td></tr>
                    <tr><td>Worst Channel</td><td><code>${['R', 'G', 'B', 'A'][comparison.aggregate.worstChannel]}</code></td></tr>
                </tbody>
            </table>
        </div>

        ${options.detailed ? this.buildChannelTable(comparison.perChannel) : ''}

        ${imagePaths.size > 0 ? this.buildVisualizationsSection(imagePaths) : ''}

        <div class="card">
            <h3>Raw JSON Data</h3>
            <pre>${JSON.stringify({
                pair: { name: pair.name, category: pair.category },
                passed: comparison.passed,
                aggregate: comparison.aggregate,
            }, null, 2)}</pre>
        </div>
    </div>
</body>
</html>`;
        return html;
    }

    private buildChannelTable(channels: any[]): string {
        let html = `<div class="card"><h3>Per-Channel Statistics</h3><table>
            <thead><tr><th>Channel</th><th>Max Error</th><th>MAE</th><th>RMSE</th><th>PSNR (dB)</th><th>SSIM</th><th>Outliers</th><th>P50</th><th>P99</th></tr></thead><tbody>`;

        for (const ch of channels) {
            const p = ch.percentiles;
            html += `<tr class="channel-${ch.channel}">
                <td><strong>${ch.channel}</strong></td>
                <td><code>${ch.maxAbsError.toExponential(3)}</code></td>
                <td><code>${ch.mae.toExponential(3)}</code></td>
                <td><code>${ch.rmse.toExponential(3)}</code></td>
                <td><code>${ch.psnr === Infinity ? '∞' : ch.psnr.toFixed(2)}</code></td>
                <td><code>${ch.ssim.toFixed(6)}</code></td>
                <td><code>${ch.outlierCount}</code></td>
                <td><code>${p.p50.toExponential(3)}</code></td>
                <td><code>${p.p99.toExponential(3)}</code></td>
            </tr>`;
        }

        html += `</tbody></table></div>`;
        return html;
    }

    private buildVisualizationsSection(imagePaths: Map<string, string>): string {
        let html = `<div class="visualizations"><h3>Visualizations</h3><div class="viz-grid">`;
        for (const [mode, imgName] of imagePaths) {
            html += `<div class="viz-item">
                <img src="${imgName}" alt="${mode}">
                <h4>${mode}</h4>
            </div>`;
        }
        html += `</div></div>`;
        return html;
    }

    /**
     * Generate Markdown report.
     */
    private async generateMarkdown(
        report: VerificationReport,
        outputDir: string,
        baseName: string,
        options: ReportOptions
    ): Promise<string> {
        const fs = await import('fs');
        const filePath = `${outputDir}/${baseName}.md`;

        const { comparison, pair, metadata, timestamp } = report;
        const passed = comparison.passed ? '✅ PASS' : '❌ FAIL';

        let md = `# Shader Verification: ${pair.name}

**Category:** ${pair.category}  
**Timestamp:** ${timestamp}  
**Status:** ${passed}

## Effect Pair
- **HLSL:** \`${pair.hlslRelativePath || pair.hlslPath}\`
- **WGSL:** \`${pair.wgslRelativePath || pair.wgslPath}\`

## Execution Metadata
| Metric | Value |
|--------|-------|
| Input Size | ${metadata.inputSize[0]}×${metadata.inputSize[1]} |
| Output Size | ${metadata.outputSize[0]}×${metadata.outputSize[1]} |
| HLSL Time | ${metadata.hlslExecutionTime.toFixed(2)} ms |
| WGSL Time | ${metadata.wgslExecutionTime.toFixed(2)} ms |
| Total Time | ${metadata.totalTime.toFixed(2)} ms |

## Aggregate Statistics
| Metric | Value |
|--------|-------|
| Max Absolute Error | \`${comparison.aggregate.maxAbsError.toExponential(4)}\` |
| MAE | \`${comparison.aggregate.mae.toExponential(4)}\` |
| RMSE | \`${comparison.aggregate.rmse.toExponential(4)}\` |
| PSNR | \`${comparison.aggregate.psnr === Infinity ? '∞' : comparison.aggregate.psnr.toFixed(2) + ' dB'}\` |
| SSIM | \`${comparison.aggregate.ssim.toFixed(6)}\` |
| Total Outliers | ${comparison.aggregate.outlierCount} / ${comparison.pixelCount * 4} |
| Worst Channel | ${['R', 'G', 'B', 'A'][comparison.aggregate.worstChannel]} |

`;

        if (options.detailed) {
            md += `## Per-Channel Statistics\n\n`;
            md += `| Channel | Max Error | MAE | RMSE | PSNR (dB) | SSIM | Outliers | P50 | P99 |\n`;
            md += `|---------|-----------|-----|------|-----------|------|----------|-----|-----|\n`;
            for (const ch of comparison.perChannel) {
                const p = ch.percentiles;
                md += `| ${ch.channel} | \`${ch.maxAbsError.toExponential(3)}\` | \`${ch.mae.toExponential(3)}\` | \`${ch.rmse.toExponential(3)}\` | \`${ch.psnr === Infinity ? '∞' : ch.psnr.toFixed(2)}\` | \`${ch.ssim.toFixed(6)}\` | ${ch.outlierCount} | \`${p.p50.toExponential(3)}\` | \`${p.p99.toExponential(3)}\` |\n`;
            }
            md += `\n`;
        }

        if (report.visualizations) {
            md += `## Visualizations\n\n`;
            for (const [mode, result] of report.visualizations) {
                const imgName = `${baseName}_${mode}.png`;
                md += `### ${mode}\n![${mode}](${imgName})\n\n`;
            }
        }

        md += `---\n*Generated by WGFX Shader Verification Tool*`;

        fs.writeFileSync(filePath, md);
        return filePath;
    }

    /**
     * Generate a summary report for batch verification.
     */
    async generateBatchSummary(
        results: Array<{ pair: EffectPair; report: VerificationReport; passed: boolean }>,
        outputDir: string,
        baseName: string
    ): Promise<string[]> {
        const fs = await import('fs');
        const path = await import('path');

        const files: string[] = [];

        // JSON summary
        const jsonPath = `${outputDir}/${baseName}_summary.json`;
        const jsonData = {
            timestamp: new Date().toISOString(),
            total: results.length,
            passed: results.filter(r => r.passed).length,
            failed: results.filter(r => !r.passed).length,
            results: results.map(r => ({
                name: r.pair.name,
                category: r.pair.category,
                passed: r.passed,
                maxError: r.report.comparison.aggregate.maxAbsError,
                psnr: r.report.comparison.aggregate.psnr,
                ssim: r.report.comparison.aggregate.ssim,
                outliers: r.report.comparison.aggregate.outlierCount,
                hlslTime: r.report.metadata.hlslExecutionTime,
                wgslTime: r.report.metadata.wgslExecutionTime,
            })),
        };
        fs.writeFileSync(jsonPath, JSON.stringify(jsonData, null, 2));
        files.push(jsonPath);

        // Markdown summary
        const mdPath = `${outputDir}/${baseName}_summary.md`;
        let md = `# Batch Verification Summary\n\n`;
        md += `**Date:** ${new Date().toISOString()}  \n`;
        md += `**Total:** ${results.length} | **Passed:** ${jsonData.passed} | **Failed:** ${jsonData.failed}\n\n`;
        md += `| Effect | Category | Status | Max Error | PSNR (dB) | SSIM | Outliers | HLSL (ms) | WGSL (ms) |\n`;
        md += `|--------|----------|--------|-----------|-----------|------|----------|-----------|-----------|\n`;
        for (const r of results) {
            const agg = r.report.comparison.aggregate;
            const status = r.passed ? '✅' : '❌';
            md += `| ${r.pair.name} | ${r.pair.category} | ${status} | \`${agg.maxAbsError.toExponential(3)}\` | ${agg.psnr === Infinity ? '∞' : agg.psnr.toFixed(2)} | ${agg.ssim.toFixed(4)} | ${agg.outlierCount} | ${r.report.metadata.hlslExecutionTime.toFixed(1)} | ${r.report.metadata.wgslExecutionTime.toFixed(1)} |\n`;
        }
        fs.writeFileSync(mdPath, md);
        files.push(mdPath);

        // HTML summary
        const htmlPath = `${outputDir}/${baseName}_summary.html`;
        const html = this.buildBatchHTML(jsonData);
        fs.writeFileSync(htmlPath, html);
        files.push(htmlPath);

        return files;
    }

    private buildBatchHTML(data: any): string {
        const passRate = (data.passed / data.total * 100).toFixed(1);
        return `<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Batch Verification Summary</title>
<style>
body{font-family:sans-serif;margin:20px}.pass{color:green}.fail{color:red}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px;text-align:left}
th{background:#f5f5f5}.pass-row{background:#f0fff0}.fail-row{background:#fff0f0}
</style></head><body>
<h1>Batch Verification Summary</h1>
<p>Date: ${data.timestamp} | Total: ${data.total} | Passed: ${data.passed} | Failed: ${data.failed} (${passRate}%)</p>
<table><thead><tr><th>Effect</th><th>Category</th><th>Status</th><th>Max Error</th><th>PSNR</th><th>SSIM</th><th>Outliers</th><th>HLSL (ms)</th><th>WGSL (ms)</th></tr></thead><tbody>
${data.results.map((r: any) => `<tr class="${r.passed ? 'pass-row' : 'fail-row'}"><td>${r.name}</td><td>${r.category}</td><td class="${r.passed ? 'pass' : 'fail'}">${r.passed ? '✅' : '❌'}</td><td>${r.maxError.toExponential(3)}</td><td>${r.psnr === Infinity ? '∞' : r.psnr.toFixed(2)}</td><td>${r.ssim.toFixed(4)}</td><td>${r.outliers}</td><td>${r.hlslTime.toFixed(1)}</td><td>${r.wgslTime.toFixed(1)}</td></tr>`).join('')}
</tbody></table></body></html>`;
    }
}

/**
 * Create a report generator instance.
 */
export function createReportGenerator(): ReportGenerator {
    return new ReportGenerator();
}