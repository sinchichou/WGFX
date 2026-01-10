/// <reference types="@webgpu/types" />
import { ResourceManager } from './ResourceManager';
import { Logger } from '@/utils/Logger';

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
}

/**
 * Manages the creation and execution of compute pipelines.
 * ---
 * 管理 WebGPU 計算管線的建立、編譯監控與執行分發。
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

        for (const module of generatedModules) {
            // Find corresponding pass info / 尋找對應的通道資訊
            const pass = shaderInfo.passes.find((p: any) => p.index === module.passIndex);
            if (!pass) continue;

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

            // Setup Default Bindings / 設定預設綁定項目
            // Binding 0: Default sampler for texture sampling / 綁定 0：用於紋理取樣的預設取樣器
            bindingMap.set(0, {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                sampler: { type: 'filtering' }
            });

            // Binding 1: Uniform buffer for parameters / 綁定 1：存放參數的 Uniform 緩衝區
            if (module.resources.parameters.length > 0) {
                bindingMap.set(1, {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' }
                });
            }

            // Binding 4: Scene info (Global constants) / 綁定 4：場景資訊（全域常量）
            bindingMap.set(4, {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'uniform' }
            });

            // Process textures (Sampled or Storage) / 處理紋理資源（取樣或儲存型）
            module.resources.textures.forEach((tex: any) => {
                const format = 'rgba16float'; // Internal high-precision format / 內部高精度格式
                if (tex.isStorage) {
                    bindingMap.set(tex.binding, {
                        binding: tex.binding,
                        visibility: GPUShaderStage.COMPUTE,
                        storageTexture: { format: format as GPUTextureFormat, access: 'write-only' }
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
                if (samp.binding !== 0) {
                    bindingMap.set(samp.binding, {
                        binding: samp.binding,
                        visibility: GPUShaderStage.COMPUTE,
                        sampler: { type: 'filtering' }
                    });
                }
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
                computePipeline, bindGroupLayout, resources: module.resources, passInfo: pass
            });
        }
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

        const { computePipeline, bindGroupLayout, resources, passInfo: originalPassInfo } = pipeline;
        const groupEntries: GPUBindGroupEntry[] = [];

        // 1. Get or create default sampler / 獲取或建立預設取樣器
        let defaultSampler = this.resourceManager.getSampler('sam');
        if (!defaultSampler) {
            defaultSampler = this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
        }
        groupEntries.push({ binding: 0, resource: defaultSampler });

        // 2. Bind uniform buffer if needed / 若有需要則綁定 Uniform 緩衝區
        if (resources.parameters.length > 0) {
            const ub = this.resourceManager.getUniformBuffer();
            if (!ub) throw new Error("Uniform buffer missing");
            groupEntries.push({ binding: 1, resource: { buffer: ub } });
        }

        // 3. Bind all required textures / 綁定所有需要的紋理
        resources.textures.forEach((tex: any) => {
            const mode = tex.isStorage ? 'storage' : 'sampled';
            const view = this.resourceManager.getTextureView(tex.name, mode);
            groupEntries.push({ binding: tex.binding, resource: view });
        });

        // 4. Bind additional samplers / 綁定額外的取樣器
        resources.samplers.forEach((samp: any) => {
            if (samp.binding !== 0) {
                const s = this.resourceManager.getSampler(samp.name);
                if (s) groupEntries.push({ binding: samp.binding, resource: s });
            }
        });

        // 5. Bind global scene constants / 綁定全域場景常量
        const sceneBuffer = this.resourceManager.getSceneBuffer();
        if (sceneBuffer) {
            groupEntries.push({ binding: 4, resource: { buffer: sceneBuffer } });
        }

        // Initialize compute pass / 初始化計算通道
        const bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: groupEntries
        });

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);

        // Determine dispatch dimensions based on output size / 根據輸出大小決定分發維度
        const outName = originalPassInfo.out[0];
        const outTex = this.resourceManager.getTexture(outName);
        if (!outTex) throw new Error(`Output texture "${outName}" not found`);

        // Calculate grid size (Workgroups) / 計算網格大小
        const workgroupSize = originalPassInfo.blockSize || originalPassInfo.numThreads;
        const dx = Math.ceil(outTex.width / workgroupSize[0]);
        const dy = Math.ceil(outTex.height / workgroupSize[1]);

        Logger.debug(`Dispatch Pass ${passInfo.index}: ${dx}x${dy} groups`);

        passEncoder.dispatchWorkgroups(dx, dy, 1);
        passEncoder.end();
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