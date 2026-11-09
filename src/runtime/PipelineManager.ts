/// <reference types="@webgpu/types" />
import { ResourceManager } from './ResourceManager';
import { Logger, LogLevel } from '../utils/Logger';

/**
 * A precompiled `//! DISPATCH` expression evaluator.
 * ---
 * 已預先編譯的 `//! DISPATCH` 運算式求值器。
 * `keys` 為運算元名稱（defines 與 INPUT/OUTPUT 尺寸），於管線建立時固定，
 * 熱路徑只按相同順序傳值即可，無需每幀重新 `new Function(...)` 或組字串。
 */
export interface DispatchEvaluator {
    /** Original expression text (for error messages) / 原始運算式字串（錯誤訊息用） */
    expr: string;
    /** Operand names, bound as the compiled function's parameters / 運算元名稱（編譯後函式的參數） */
    keys: string[];
    /** Compiled evaluator / 編譯後的求值函式 */
    fn: (...operands: number[]) => unknown;
}

/**
 * Metadata for a stored WebGPU pipeline and its associated layouts.
 * ---
 * 儲存 WebGPU 管線及其關聯配置的元數據。
 *
 * @internal
 * @category Interfaces
 */
export interface StoredPipeline {
    /** The WebGPU compute pipeline / WebGPU 計算管線 */
    computePipeline: GPUComputePipeline;
    /** The bind group layout / 綁定組佈局 */
    bindGroupLayout: GPUBindGroupLayout;
    /** Resource requirements / 資源需求配置 */
    resources: any;
    /** Pass metadata / 通道元數據 */
    passInfo: any;
    /**
     * Cached bind group（快取的綁定組）。
     * 僅在 ResourceManager.revision 變動（紋理/緩衝區重建立）時重建，
     * 讓每幀 dispatch 不再重複配置 bind group。
     */
    bindGroup: GPUBindGroup | null;
    /** Resource revision the cached bind group was built at / 建立快取時的資源版本 */
    bindGroupRevision: number;
    /**
     * Precomputed dispatch grid divisors / 建立時預算的網格除數。
     * BLOCK_SIZE 優先於 NUM_THREADS（BLOCK_SIZE 描述每個工作群組涵蓋的像素範圍），
     * 於管線建立時解析一次，dispatch 熱路徑免每幀重新判斷。
     */
    gridBx: number;
    gridBy: number;
    /** Name of the output texture this pass writes / 此 pass 寫入的輸出紋理名稱 */
    outTexName: string;
    /**
     * Precompiled `//! DISPATCH` evaluators（每 pass 每條運算式僅編譯一次）。
     * 為 null 代表此 pass 沒有 explicit dispatch，或運算式編譯失敗
     * （後者由 {@link PipelineManager._evalDispatchExpr} 舊路徑處理，錯誤訊息不變）。
     */
    dispatchFns: DispatchEvaluator[] | null;
    /**
     * Explicit dispatch 求值快取（每幀命中；resize/資源重建才失效）。
     * 資源版本與 defines 物件身分皆未變 ⇒ 運算元值必然未變，直接重用結果，
     * 熱路徑完全不做字串組建或函式呼叫。
     */
    dispatchCacheRevision: number;
    dispatchCacheDefines: Record<string, number> | null;
    dispatchCacheResult: [number, number];
    /**
     * Texture-grid dispatch 快取（P4）。
     * 輸出紋理尺寸只在紋理（重）建立時改變（createTexture 會遞增資源版本），
     * 故網格除數以資源版本為鍵快取，resize 前每幀命中。
     */
    gridRevision: number;
    gridDx: number;
    gridDy: number;
}

/**
 * Manages the creation and execution of compute pipelines.
 * ---
 * 管理 WebGPU 計算管線的建立、編譯監控與執行分發。
 *
 * 綁定配置完全由 WGSLCodeGenerator 輸出的 resources 決定：
 * - binding 1: Uniform 參數緩衝區（有參數時）
 * - binding 4: SceneInfo 場景資訊緩衝區
 * - binding 5+: 自訂取樣器
 * - binding 10+: 紋理（sampled/storaged 成對別名綁定至同一邏輯紋理）
 *
 * @group Core
 */
export class PipelineManager {
    public device: GPUDevice;
    public resourceManager: ResourceManager;
    /** Map of pass index to pipeline / 通道索引與管線的映射 */
    public pipelines: Map<number, StoredPipeline>;

    constructor(device: GPUDevice, resourceManager: ResourceManager) {
        this.device = device;
        this.resourceManager = resourceManager;
        this.pipelines = new Map();
    }

    /**
     * Create pipelines for all shader modules.
     * ---
     * 為所有產生的 WGSL 模組建立計算管線。
     *
     * @group Pipelines
     * @param shaderInfo - Global shader metadata / 全域著色器元數據
     * @param generatedModules - Generated WGSL modules / 產生的 WGSL 模組
     */
    public async createPipelines(shaderInfo: any, generatedModules: any[]): Promise<void> {
        this.pipelines.clear();

        // O(1) pass lookup / 以索引建查找表，免線性搜尋
        const passByIndex = new Map<number, any>(
            shaderInfo.passes.map((p: any) => [p.index, p] as [number, any])
        );

        // Create all pipelines concurrently: WGSL→ISA compilation overlaps on the
        // driver side, so compile latency approaches the SLOWEST pass instead of
        // the SUM of all passes (matters most for multi-pass CNN effects).
        // 並行建立所有管線：WGSL 編譯在驅動端可重疊，總延遲趨近最慢的 pass
        // 而非各 pass 總和（對多 pass CNN 特效最有感）。
        await Promise.all(generatedModules.map(async (module) => {
            // Find corresponding pass info / 尋找對應的通道資訊
            const pass = passByIndex.get(module.passIndex);
            if (!pass) return;

            // Resolve dispatch grid divisor once per pipeline / 每管線僅解析一次網格除數
            // （BLOCK_SIZE 優先；存入 StoredPipeline 供 dispatch 熱路徑直接使用）
            const workgroupSize = pass.blockSize || pass.numThreads || [1, 1];

            // Create shader module from WGSL / 從 WGSL 代碼建立著色器模組
            const shaderModule = this.device.createShaderModule({
                code: module.wgslCode,
                label: `Pass ${module.passIndex} Shader`
            });

            // Monitor compilation status and log messages / 監控編譯狀態並記錄訊息
            shaderModule.getCompilationInfo().then(info => {
                info.messages.forEach(msg => {
                    const type = msg.type === 'error' ? 'ERROR' : (msg.type === 'warning' ? 'WARN' : 'INFO');
                    const logMsg = `Shader ${module.passIndex}: [${type}] ${msg.message} at line ${msg.lineNum}, col ${msg.linePos}`;
                    if (msg.type === 'error') Logger.error(logMsg);
                    else if (msg.type === 'warning') Logger.warn(logMsg);
                    else Logger.debug(logMsg);
                });
            });

            const bindingMap = new Map<number, GPUBindGroupLayoutEntry>();

            // Binding 1: Uniform buffer for parameters / 綁定 1：存放參數的 Uniform 緩衝區
            if (module.resources.parameters.length > 0) {
                bindingMap.set(1, {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' }
                });
            }

            // Binding 4 (or COMMON-declared slot): Scene info / 場景資訊（全域常量）
            // COMMON 自行宣告 scene 時使用 COMMON 選擇的槽位。
            const sceneBinding: number = module.resources.sceneBinding ?? 4;
            bindingMap.set(sceneBinding, {
                binding: sceneBinding,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'uniform' }
            });

            // Process textures (Sampled or Storage) / 處理紋理資源（取樣或儲存型）
            module.resources.textures.forEach((tex: any) => {
                if (tex.isStorage) {
                    bindingMap.set(tex.binding, {
                        binding: tex.binding,
                        visibility: GPUShaderStage.COMPUTE,
                        storageTexture: { format: tex.format as GPUTextureFormat, access: 'write-only' }
                    });
                } else {
                    bindingMap.set(tex.binding, {
                        binding: tex.binding,
                        visibility: GPUShaderStage.COMPUTE,
                        texture: { sampleType: 'float' }
                    });
                }
            });

            // Process custom samplers / 處理自定義取樣器
            module.resources.samplers.forEach((samp: any) => {
                bindingMap.set(samp.binding, {
                    binding: samp.binding,
                    visibility: GPUShaderStage.COMPUTE,
                    sampler: { type: 'filtering' }
                });
            });

            // Process storage buffers / 處理 storage buffer（CNN 權重與特徵圖）
            (module.resources.storageBuffers || []).forEach((buf: any) => {
                bindingMap.set(buf.binding, {
                    binding: buf.binding,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: buf.mode === 'read_write' ? 'storage' : 'read-only-storage' }
                });
            });

            // Create layouts and pipeline / 建立佈局與管線
            const entries = Array.from(bindingMap.values()).sort((a, b) => a.binding - b.binding);
            const bindGroupLayout = this.device.createBindGroupLayout({ entries });
            const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

            // Compile pipeline asynchronously / 非同步編譯管線
            const computePipeline = await this.device.createComputePipelineAsync({
                layout: pipelineLayout,
                compute: { module: shaderModule, entryPoint: 'main_cs' },
                label: `Pass ${module.passIndex} Pipeline`
            });

            this.pipelines.set(module.passIndex, {
                computePipeline, bindGroupLayout, resources: module.resources, passInfo: pass,
                bindGroup: null, bindGroupRevision: -1,
                // Precompute dispatch hot-path constants / 預先計算 dispatch 熱路徑常數
                // BLOCK_SIZE 優先於 NUM_THREADS：BLOCK_SIZE 描述每個工作群組涵蓋的像素範圍。
                gridBx: Math.max(1, (workgroupSize[0] || 1) | 0),
                gridBy: Math.max(1, (workgroupSize[1] || 1) | 0),
                outTexName: (pass.out && pass.out[0]) || '',
                // P1：//! DISPATCH 運算式在建管線時編譯一次（原本每幀每 pass 都 new Function）
                dispatchFns: this._compileDispatchExprs(pass),
                dispatchCacheRevision: -1,
                dispatchCacheDefines: null,
                dispatchCacheResult: [1, 1],
                gridRevision: -1,
                gridDx: 1,
                gridDy: 1
            });
        }));
    }

    /**
     * Dispatch a compute pass.
     * ---
     * 執行計算通道。
     *
     * @group Execution
     * @param passInfo - Pass metadata / 通道元數據
     * @param commandEncoder - GPU command encoder / 指令編碼器
     */
    public dispatchPass(passInfo: any, commandEncoder: GPUCommandEncoder): void {
        const pipeline = this.pipelines.get(passInfo.index);
        if (!pipeline) {
            throw new Error(`Pipeline for pass ${passInfo.index} not found`);
        }

        // Hot path: reuse the cached bind group unless the resource revision changed.
        // 熱路徑：資源版本未變時重用快取的 bind group，避免每幀重新配置。
        const revision = this.resourceManager.revision;
        if (pipeline.bindGroup === null || pipeline.bindGroupRevision !== revision) {
            pipeline.bindGroup = this._buildBindGroup(pipeline);
            pipeline.bindGroupRevision = revision;
        }

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline.computePipeline);
        passEncoder.setBindGroup(0, pipeline.bindGroup);

        // Explicit dispatch（//! DISPATCH）優先：buffer-only pass（如 OpenNR 的 1D conv
        // dispatch）沒有 OUT 紋理可推導網格，workgroup 數以運算式字串隨 passInfo 提供。
        if (passInfo.dispatch && passInfo.dispatch.length >= 1) {
            const [dx, dy] = this._resolveExplicitDispatch(pipeline, passInfo, revision);
            if (Logger.wouldLog(LogLevel.DEBUG)) {
                Logger.debug(`Dispatch Pass ${passInfo.index}: ${dx}x${dy} groups (explicit)`);
            }
            passEncoder.dispatchWorkgroups(Math.max(1, dx), Math.max(1, dy), 1);
            passEncoder.end();
            return;
        }

        // Grid divisors AND the resulting grid are cached per resource revision:
        // texture dimensions only change when a texture is (re)created, which bumps
        // the revision. 除數與網格皆以資源版本為鍵快取：紋理尺寸只在（重）建立時改變。
        if (pipeline.gridRevision !== revision) {
            const outTex = this.resourceManager.getTexture(pipeline.outTexName);
            if (!outTex) throw new Error(`Output texture "${pipeline.outTexName}" not found`);
            pipeline.gridDx = Math.ceil(outTex.width / pipeline.gridBx);
            pipeline.gridDy = Math.ceil(outTex.height / pipeline.gridBy);
            pipeline.gridRevision = revision;
        }
        const dx = pipeline.gridDx;
        const dy = pipeline.gridDy;

        // Hot-path log is gated to avoid per-frame string building
        // 熱路徑日誌加閘，避免每幀組字串
        if (Logger.wouldLog(LogLevel.DEBUG)) {
            Logger.debug(`Dispatch Pass ${passInfo.index}: ${dx}x${dy} groups`);
        }

        passEncoder.dispatchWorkgroups(dx, dy, 1);
        passEncoder.end();
    }

    /**
     * Build the evaluation context for `//! DISPATCH` expressions.
     * ---
     * 建立 `//! DISPATCH` 運算式的求值 context：
     * defines 為底，再以實際 INPUT/OUTPUT 紋理尺寸覆寫
     * （語意與歷史行為一致：宿主提供的紋理尺寸優先於同名 define）。
     */
    private _dispatchContext(): Record<string, number> {
        const ctx: Record<string, number> = {
            ...(this.resourceManager.defines || {}),
        };
        const outTex = this.resourceManager.getTexture('OUTPUT');
        if (outTex) {
            ctx.OUTPUT_WIDTH = outTex.width;
            ctx.OUTPUT_HEIGHT = outTex.height;
        }
        const inTex = this.resourceManager.getTexture('INPUT');
        if (inTex) {
            ctx.INPUT_WIDTH = inTex.width;
            ctx.INPUT_HEIGHT = inTex.height;
        }
        return ctx;
    }

    /**
     * Precompile a pass's `//! DISPATCH` expressions（管線建立時一次）。
     * ---
     * P1：原本 `_evalDispatchExpr` 對每個 pass、**每一幀**都執行 `new Function(...)`
     * （運算式重新解析 + 產生新函式），是每幀熱路徑最明確的浪費。
     * 這裡在建管線時就編譯好，並以當時的 context 鍵集固定參數名；dispatch 時只需
     * 在快取失效時按相同鍵序傳值。
     *
     * 編譯失敗（運算式語法錯誤）時回傳 null → 該 pass 走回 `_evalDispatchExpr`，
     * 讓錯誤時機與訊息文字與舊版完全相同。
     */
    private _compileDispatchExprs(pass: any): DispatchEvaluator[] | null {
        if (!pass || !pass.dispatch || pass.dispatch.length < 1) return null;
        const keys = Object.keys(this._dispatchContext());
        try {
            return pass.dispatch.slice(0, 2).map((expr: string) => ({
                expr,
                keys,
                // eslint-disable-next-line no-new-func
                fn: new Function(...keys, `return (${expr});`) as (...operands: number[]) => unknown,
            }));
        } catch {
            return null;
        }
    }

    /** 依預編譯時的鍵序，由當前運算元值取出一組實參。 */
    private _dispatchOperandValues(keys: string[]): number[] {
        const ctx = this._dispatchContext();
        const values = new Array<number>(keys.length);
        for (let i = 0; i < keys.length; i++) values[i] = ctx[keys[i]];
        return values;
    }

    /** 驗證並正規化求值結果（與舊版相同的整數/非負檢查）。 */
    private static _normalizeDispatchResult(v: unknown): number {
        const n = Math.floor(Number(v));
        if (!Number.isFinite(n) || n < 0) throw new Error('not a non-negative integer');
        return n;
    }

    /**
     * Evaluate a //! DISPATCH expression（數值運算式：defines/OUTPUT_WIDTH/HEIGHT + 四則運算）
     * ---
     * 舊路徑（未預編譯或預編譯失敗時使用）；語意與錯誤訊息與歷史版本一致。
     */
    private _evalDispatchExpr(expr: string, passInfo: any): number {
        const ctx = this._dispatchContext();
        try {
            // eslint-disable-next-line no-new-func
            const fn = new Function(...Object.keys(ctx), `return (${expr});`);
            return PipelineManager._normalizeDispatchResult(fn(...Object.values(ctx)));
        } catch (e: any) {
            throw new Error(`Pass ${passInfo.index}: DISPATCH expression "${expr}" failed: ${e?.message || e}`);
        }
    }

    /**
     * Resolve explicit `//! DISPATCH` workgroup counts, memoized per pass.
     * ---
     * 求值 `//! DISPATCH` 的 (dx, dy) workgroup 數，並以「資源版本 + defines 物件身分」
     * 記憶化：
     *   * INPUT/OUTPUT 尺寸只在 createTexture（重）建立時改變 → 會遞增資源版本；
     *   * defines 只在 ResourceManager.initialize 時以新物件指派（並伴隨資源建立）。
     * 兩者未變即代表運算元值必然未變，因此 resize 前每幀完全命中，熱路徑零配置、
     * 零函式呼叫、零字串組建。
     *
     * 直接就地改寫 `resourceManager.defines` 的欄位**不會**使快取失效
     * （這是刻意的效能取捨；請以 initialize()/重新 compile 變更 defines）。
     */
    private _resolveExplicitDispatch(
        pipeline: StoredPipeline,
        passInfo: any,
        revision: number,
    ): [number, number] {
        const defines = this.resourceManager.defines;
        if (pipeline.dispatchCacheRevision === revision && pipeline.dispatchCacheDefines === defines) {
            return pipeline.dispatchCacheResult;
        }

        let dx: number;
        let dy: number;
        const fns = pipeline.dispatchFns;
        if (fns) {
            const values = this._dispatchOperandValues(fns[0].keys);
            dx = this._evalCompiledDispatch(fns[0], values, passInfo);
            dy = fns.length >= 2 ? this._evalCompiledDispatch(fns[1], values, passInfo) : 1;
        } else {
            dx = this._evalDispatchExpr(passInfo.dispatch[0], passInfo);
            dy = passInfo.dispatch.length >= 2 ? this._evalDispatchExpr(passInfo.dispatch[1], passInfo) : 1;
        }

        pipeline.dispatchCacheRevision = revision;
        pipeline.dispatchCacheDefines = defines;
        pipeline.dispatchCacheResult = [dx, dy];
        return pipeline.dispatchCacheResult;
    }

    /** 以求值器求值（錯誤訊息格式與 {@link _evalDispatchExpr} 一致）。 */
    private _evalCompiledDispatch(
        evaluator: DispatchEvaluator,
        values: number[],
        passInfo: any,
    ): number {
        try {
            return PipelineManager._normalizeDispatchResult(evaluator.fn(...values));
        } catch (e: any) {
            throw new Error(
                `Pass ${passInfo.index}: DISPATCH expression "${evaluator.expr}" failed: ${e?.message || e}`,
            );
        }
    }

    /**
     * Build the bind group for a pipeline from current resources.
     * ---
     * 依目前資源為管線建立綁定組（僅在資源版本變動時呼叫）。
     */
    private _buildBindGroup(pipeline: StoredPipeline): GPUBindGroup {
        const { bindGroupLayout, resources, passInfo } = pipeline;
        const groupEntries: GPUBindGroupEntry[] = [];

        // 1. Bind uniform buffer if needed / 若有需要則綁定 Uniform 緩衝區
        if (resources.parameters.length > 0) {
            const ub = this.resourceManager.getUniformBuffer();
            if (!ub) throw new Error("Uniform buffer missing");
            groupEntries.push({ binding: 1, resource: { buffer: ub } });
        }

        // 2. Bind all required textures / 綁定所有需要的紋理
        //    sampled/storaged 成對別名會被 ResourceManager 解析到同一邏輯紋理的對應 view。
        resources.textures.forEach((tex: any) => {
            const mode = tex.isStorage ? 'storage' : 'sampled';
            const view = this.resourceManager.getTextureView(tex.name, mode);
            groupEntries.push({ binding: tex.binding, resource: view });
        });

        // 3. Bind custom samplers / 綁定自訂取樣器
        resources.samplers.forEach((samp: any) => {
            const s = this.resourceManager.getSampler(samp.name);
            if (s) {
                groupEntries.push({ binding: samp.binding, resource: s });
            } else {
                throw new Error(`Sampler "${samp.name}" not found for pass ${passInfo.index}`);
            }
        });

        // 4. Bind global scene constants / 綁定全域場景常量
        const sceneBuffer = this.resourceManager.getSceneBuffer();
        if (sceneBuffer) {
            const sceneBinding: number = resources.sceneBinding ?? 4;
            groupEntries.push({ binding: sceneBinding, resource: { buffer: sceneBuffer } });
        }

        // 5. Bind storage buffers / 綁定 storage buffer（CNN 權重與特徵圖）
        (resources.storageBuffers || []).forEach((buf: any) => {
            const b = this.resourceManager.getStorageBuffer(buf.name);
            if (!b) throw new Error(`Storage buffer "${buf.name}" not found for pass ${passInfo.index}`);
            groupEntries.push({ binding: buf.binding, resource: { buffer: b } });
        });

        return this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: groupEntries
        });
    }

    /**
     * Clear all cached pipelines.
     * ---
     * 清除所有管線緩存。
     *
     * @group Lifecycle
     */
    public dispose(): void {
        this.pipelines.clear();
    }
}
