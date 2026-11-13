/**
 * browser-entry — Browser-safe barrel for the WGFX verification toolkit.
 *
 * Import this module instead of `src/verification/index.ts`: the latter has a
 * top-level `import 'webgpu'` (the Node WebGPU polyfill) and therefore cannot
 * be loaded in a browser bundle.
 *
 * Safe to re-export (no Node built-ins, no Node-only packages):
 *  - `HLSTranspiler` — pure parsing/code generation. Its transitive deps
 *    (`runtime/ShaderParser`, `runtime/WGSLCodeGenerator`, `types`) are plain
 *    JS/TS. Note: only `transpile(source)` is browser-safe; the convenience
 *    method `transpileFile(path)` performs a dynamic `import('fs')` and is
 *    Node-only.
 *  - `PixelComparator` / `StatisticsEngine` — pure numeric code, no I/O.
 *  - `diff-core` — pure byte-level metrics + heatmap, no device required.
 *
 * Deliberately NOT re-exported:
 *  - `EffectPairResolver` — imports `glob`, `path` and `url` and uses
 *    `process.cwd()` to scan `effects_src/` and `effects/` (Node-only FS).
 *  - `DiffVisualizer` — imports `pngjs` and produces Node `Buffer` PNGs.
 *  - `ReportGenerator` — writes report files through a dynamic `import('fs')`
 *    and depends on the PNG/Buffer types of `DiffVisualizer`.
 *  - `VerificationRunner` — orchestrates resolver + visualizer + report
 *    generator and imports the `webgpu` polyfill (Node-only).
 *  - `GPUExecutor` — not Node-only, but it is device-bound: it requires a live
 *    `GPUDevice`; this barrel stays device-free on purpose.
 *  - `index.ts` — contains `import 'webgpu'`, so it is not browser-safe.
 */

export {
    HLSTranspiler,
    createTranspiler,
    type TranspileOptions,
    type TranspileResult,
    type TranspileDiagnostic,
    type GeneratedModule,
    type ModuleResources,
} from './HLSTranspiler';

export {
    PixelComparator,
    createComparator,
    type ComparisonOptions,
    type ComparisonResult,
    type ToleranceLevel,
} from './PixelComparator';

export {
    StatisticsEngine,
    createStatsEngine,
    type ChannelStats,
    type AggregateStats,
    type StatsOptions,
} from './StatisticsEngine';

export {
    comparePixels,
    generateDiffHeatmap,
    DIFF_CHANNELS,
    DIFF_FULL_SCALE,
    type RGBAByteData,
    type PerChannelMAE,
    type PerChannelMaxAbsDiff,
    type DiffMetrics,
} from './diff-core';
