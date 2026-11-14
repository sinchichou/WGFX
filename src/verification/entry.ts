/**
 * Verification toolkit entry point.
 * Exports all verification modules for external consumption.
 */

export {
    EffectPairResolver,
    HLSTranspiler,
    GPUExecutor,
    PixelComparator,
    StatisticsEngine,
    DiffVisualizer,
    ReportGenerator,
    VerificationRunner,
    quickVerify,
    createResolver,
    createTranspiler,
    createExecutor,
    createComparator,
    createStatsEngine,
    createVisualizer,
    createReportGenerator,
    createRunner,
} from './index.js';

// Browser-safe pure-function diff core（comparePixels/generateDiffHeatmap 不依賴
// device／fs，Node 22 無 type-stripping 時測試與工具自此 barrel 取用）。
export {
    comparePixels,
    generateDiffHeatmap,
} from './diff-core.js';