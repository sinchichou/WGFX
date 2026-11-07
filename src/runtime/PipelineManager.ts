/// <reference types="@webgpu/types" />
import { ResourceManager } from './ResourceManager';
import { Logger, LogLevel } from '../utils/Logger';

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
                outTexName: pass.out[0]
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

        // Grid divisors are precomputed at pipeline creation; only the texture lookup
        // stays per-frame（texture 物件可能因 resize 而重建，故保留查詢）。
        // 網格除數已於管線建立時計算完成。
        const outTex = this.resourceManager.getTexture(pipeline.outTexName);
        if (!outTex) throw new Error(`Output texture "${pipeline.outTexName}" not found`);

        const dx = Math.ceil(outTex.width / pipeline.gridBx);
        const dy = Math.ceil(outTex.height / pipeline.gridBy);

        // Hot-path log is gated to avoid per-frame string building
        // 熱路徑日誌加閘，避免每幀組字串
        if (Logger.wouldLog(LogLevel.DEBUG)) {
            Logger.debug(`Dispatch Pass ${passInfo.index}: ${dx}x${dy} groups`);
        }

        passEncoder.dispatchWorkgroups(dx, dy, 1);
        passEncoder.end();
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
