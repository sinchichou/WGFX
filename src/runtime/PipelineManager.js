// src/runtime/PipelineManager.js

/**
 * - EN: Manages the creation and scheduling of WebGPU compute pipelines.
 *   This class receives shader IR and WGSL code to build the required GPU state,
 *   including pipeline layouts, pipelines, and bind groups.
 * - TW: 管理 WebGPU 計算管線的建立和調度。
 *   此類別接收著色器 IR 和 WGSL 程式碼,以建立執行所需的 GPU 狀態,
 *   包括管線佈局、管線和綁定組。
 */
export class PipelineManager {
    /**
     * - EN: Constructs a new PipelineManager instance.
     * - TW: 建構一個新的 PipelineManager 實例。
     *
     * @param {GPUDevice} device
     * - EN: The active WebGPU device.
     * - TW: 作用中的 WebGPU 裝置。
     *
     * @param {import('./ResourceManager.js').ResourceManager} resourceManager
     * - EN: An instance of the resource manager.
     * - TW: 資源管理器的實例。
     */
    constructor(device, resourceManager) {
        this.device = device;
        this.resourceManager = resourceManager;

        /**
         * - EN: A map storing pipeline components and metadata for each pass, indexed by pass index.
         * - TW: 儲存每個通道的管線組件和中繼資料的映射表,以通道索引為鍵。
         *
         * @type {Map<number, {
         *   shaderModule: GPUShaderModule,
         *   bindGroupLayout: GPUBindGroupLayout,
         *   pipelineLayout: GPUPipelineLayout,
         *   computePipeline: GPUComputePipeline,
         *   resources: {
         *     textures: import('./Parser.js').WGFXTexture[],
         *     samplers: import('./Parser.js').WGFXSampler[],
         *     parameters: import('./Parser.js').WGFXParameter[]
         *   },
         *   passInfo: import('./Parser.js').WGFXPass
         * }>}
         */
        this.pipelines = new Map();
    }

    /**
     * - EN: Asynchronously creates compute pipelines for all passes defined in the shader IR.
     * - TW: 非同步建立著色器 IR 中定義的所有通道的計算管線。
     *
     * @param {import('./Parser.js').WGFXShaderInfo} shaderInfo
     * - EN: The parsed shader information (IR).
     * - TW: 解析後的著色器資訊 (IR)。
     *
     * @param {string} generatedModules
     * - EN: The complete, generated WGSL shader code.
     * - TW: 完整、生成的 WGSL 著色器程式碼。
     *
     * @returns {Promise<void>}
     * - EN: A promise that resolves when all pipelines are created.
     * - TW: 當所有管線建立完成時解析的 Promise。
     */
    async createPipelines(shaderInfo, generatedModules) {
        // EN: Clear any existing pipelines
        // TW: 清除任何現有的管線
        this.pipelines.clear();

        for (const module of generatedModules) {
            const pass = shaderInfo.passes.find(p => p.index === module.passIndex);
            if (!pass) {
                console.warn(`WGFXRuntime: Pass with index ${module.passIndex} not found in shaderInfo.`);
                continue;
            }

            // EN: 1. Create GPUShaderModule
            // TW: 1. 建立 GPUShaderModule
            const shaderModule = this.device.createShaderModule({
                code: module.wgslCode,
                label: `Pass ${module.passIndex} Shader Module`
            });

            // EN: 2. Create GPUBindGroupLayout
            // TW: 2. 建立 GPUBindGroupLayout
            const bindGroupLayoutEntries = [];

            // EN: Samplers
            // TW: 取樣器
            module.resources.samplers.forEach(samp => {
                bindGroupLayoutEntries.push({
                    binding: samp.binding,
                    visibility: GPUShaderStage.COMPUTE,
                    sampler: {type: 'filtering'}
                });
            });

            // EN: Uniforms (binding 1 if present)
            // TW: 統一變數 (如存在則使用綁定點 1)
            if (module.resources.parameters.length > 0) {
                bindGroupLayoutEntries.push({
                    binding: 1, // EN: Fixed binding for uniforms / TW: 統一變數的固定綁定點
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {type: 'uniform'}
                });
            }

            // EN: Textures
            // TW: 紋理
            module.resources.textures.forEach(tex => {
                bindGroupLayoutEntries.push({
                    binding: tex.binding,
                    visibility: GPUShaderStage.COMPUTE,
                    ...(tex.isStorage ? {
                        storageTexture: {
                            format: tex.format || 'rgba8unorm',
                            access: 'write-only'
                        }
                    } : {texture: {sampleType: 'float'}})
                });
            });

            const bindGroupLayout = this.device.createBindGroupLayout({
                entries: bindGroupLayoutEntries,
                label: `Pass ${module.passIndex} Bind Group Layout`
            });

            // EN: 3. Create GPUPipelineLayout
            // TW: 3. 建立 GPUPipelineLayout
            const pipelineLayout = this.device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout],
                label: `Pass ${module.passIndex} Pipeline Layout`
            });

            // EN: 4. Create GPUComputePipeline
            // TW: 4. 建立 GPUComputePipeline
            const computePipeline = await this.device.createComputePipelineAsync({
                layout: pipelineLayout,
                compute: {
                    module: shaderModule,
                    // EN: Assuming 'main_cs' is the entry point for all compute shaders
                    // TW: 假定 'main_cs' 是所有計算著色器的進入點
                    entryPoint: 'main_cs',
                },
                label: `Pass ${module.passIndex} Compute Pipeline`
            });

            // EN: Store the pipeline components
            // TW: 儲存管線組件
            this.pipelines.set(module.passIndex, {
                shaderModule,
                bindGroupLayout,
                pipelineLayout,
                computePipeline,
                resources: module.resources, // EN: Store resources for bind group creation later / TW: 儲存資源供後續建立綁定組使用
                passInfo: pass // EN: Store original pass info for dispatching / TW: 儲存原始通道資訊供調度使用
            });
        }
    }

    /**
     * - EN: Encodes commands to dispatch a specific compute pass.
     * - TW: 編碼調度特定計算通道的命令。
     *
     * @param {import('./Parser.js').WGFXPass} passInfo
     * - EN: The IR object of the pass to be dispatched.
     * - TW: 要調度的通道的 IR 物件。
     *
     * @param {GPUCommandEncoder} commandEncoder
     * - EN: The command encoder for the current frame.
     * - TW: 當前幀的命令編碼器。
     *
     * @throws {Error}
     * - EN: If the pipeline for the given pass is not found or if required resources are missing.
     * - TW: 如果找不到給定通道的管線或缺少必要資源時拋出錯誤。
     */
    dispatchPass(passInfo, commandEncoder) {
        const storedPipeline = this.pipelines.get(passInfo.index);
        if (!storedPipeline) {
            throw new Error(`找不到通道 ${passInfo.index} 的管線。是否已編譯?`);
        }

        const {computePipeline, bindGroupLayout, resources, passInfo: originalPassInfo} = storedPipeline;

        const bindGroupEntries = [];

        // EN: Samplers
        // TW: 取樣器
        resources.samplers.forEach(samp => {
            const sampler = this.resourceManager.getSampler(samp.name);
            if (!sampler) {
                // EN: If 'sam' is a default sampler, it might not be in resourceManager if not explicitly defined in the shader.
                // For now, assume it's always available or created by resourceManager.
                // TW: 如果 'sam' 是預設取樣器,若未在著色器中明確定義,可能不在 resourceManager 中。
                // 目前假定它總是可用或由 resourceManager 建立。
                if (samp.name === 'sam') {
                    // EN: Create a default linear sampler if 'sam' is not found
                    // TW: 如果找不到 'sam',建立預設的線性取樣器
                    const defaultSampler = this.device.createSampler({
                        magFilter: 'linear',
                        minFilter: 'linear',
                    });
                    bindGroupEntries.push({binding: samp.binding, resource: defaultSampler});
                } else {
                    throw new Error(`Sampler ${samp.name} not found in ResourceManager.`);
                }
            } else {
                bindGroupEntries.push({binding: samp.binding, resource: sampler});
            }
        });

        // EN: Uniforms (binding 1 if present)
        // TW: 統一變數 (如存在則使用綁定點 1)
        if (resources.parameters.length > 0) {
            const uniformBuffer = this.resourceManager.getUniformBuffer();
            if (!uniformBuffer) {
                throw new Error("Uniform buffer not found in ResourceManager.");
            }
            bindGroupEntries.push({binding: 1, resource: {buffer: uniformBuffer}}); // EN: Fixed binding for uniforms / TW: 統一變數的固定綁定點
        }

        // EN: Textures
        // TW: 紋理
        resources.textures.forEach(tex => {
            const texture = this.resourceManager.getTexture(tex.name);
            if (!texture) {
                throw new Error(`Texture ${tex.name} not found in ResourceManager.`);
            }
            bindGroupEntries.push({binding: tex.binding, resource: texture.createView()});
        });

        const bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: bindGroupEntries,
            label: `Pass ${passInfo.index} Bind Group`
        });

        // EN: Encode compute pass commands
        // TW: 編碼計算通道命令
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);

        // EN: Calculate the number of workgroups to dispatch.
        // This is based on the size of the primary output texture and the workgroup size.
        // TW: 計算要調度的工作組數量。
        // 這基於主要輸出紋理的大小和工作組大小。
        const outputTextureName = originalPassInfo.out[0]; // EN: Assume the first output is the primary output for sizing / TW: 假定第一個輸出是主要輸出,用於確定大小
        const outputTexture = this.resourceManager.getTexture(outputTextureName);
        const workgroupSize = originalPassInfo.numThreads; // EN: e.g., [8, 8, 1] / TW: 例如,[8, 8, 1]

        const dispatchX = Math.ceil(outputTexture.width / workgroupSize[0]);
        const dispatchY = Math.ceil(outputTexture.height / workgroupSize[1]);

        passEncoder.dispatchWorkgroups(dispatchX, dispatchY, 1);
        passEncoder.end();
    }

    /**
     * - EN: Clears all stored GPU objects. Does not destroy them as they are managed by the device.
     * - TW: 清除所有儲存的 GPU 物件。不銷毀它們,因為它們由裝置管理。
     *
     * @returns {void}
     */
    dispose() {
        this.pipelines.clear();
        console.log("PipelineManager: All stored pipeline states have been cleared.");
    }
}