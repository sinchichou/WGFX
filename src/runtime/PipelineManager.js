// src/runtime/PipelineManager.js

/**
 * @fileoverview 管理 WebGPU 管線的建立和計算通道的調度。
 * 此類別接收著色器 IR 和 WGSL 程式碼，以建立執行所需的 GPU 狀態，
 * 包括管線佈局、管線和綁定組。
 */

export class PipelineManager {
    /**
     * @param {GPUDevice} device - 作用中的 WebGPU 裝置。
     * @param {import('./ResourceManager.js').ResourceManager} resourceManager - 資源管理器的實例。
     */
    constructor(device, resourceManager) {
        this.device = device;
        this.resourceManager = resourceManager;

        /**
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
     * 非同步建立著色器 IR 中定義的所有通道的計算管線。
     * @param {import('./Parser.js').WGFXShaderInfo} shaderInfo - 解析後的著色器資訊 (IR)。
     * @param {string} generatedModules - 完整、生成的 WGSL 著色器程式碼。
     */
    async createPipelines(shaderInfo, generatedModules) {
        this.pipelines.clear(); // Clear any existing pipelines

        for (const module of generatedModules) {
            const pass = shaderInfo.passes.find(p => p.index === module.passIndex);
            if (!pass) {
                console.warn(`WGFXRuntime: Pass with index ${module.passIndex} not found in shaderInfo.`);
                continue;
            }

            // 1. Create GPUShaderModule
            const shaderModule = this.device.createShaderModule({
                code: module.wgslCode,
                label: `Pass ${module.passIndex} Shader Module`
            });

            // 2. Create GPUBindGroupLayout
            const bindGroupLayoutEntries = [];

            // Samplers
            module.resources.samplers.forEach(samp => {
                bindGroupLayoutEntries.push({
                    binding: samp.binding,
                    visibility: GPUShaderStage.COMPUTE,
                    sampler: {type: 'filtering'}
                });
            });

            // Uniforms (binding 1 if present)
            if (module.resources.parameters.length > 0) {
                bindGroupLayoutEntries.push({
                    binding: 1, // Fixed binding for uniforms
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {type: 'uniform'}
                });
            }

            // Textures
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

            // 3. Create GPUPipelineLayout
            const pipelineLayout = this.device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout],
                label: `Pass ${module.passIndex} Pipeline Layout`
            });

            // 4. Create GPUComputePipeline
            const computePipeline = await this.device.createComputePipelineAsync({
                layout: pipelineLayout,
                compute: {
                    module: shaderModule,
                    entryPoint: 'main_cs', // Assuming 'main_cs' is the entry point for all compute shaders
                },
                label: `Pass ${module.passIndex} Compute Pipeline`
            });

            // Store the pipeline components
            this.pipelines.set(module.passIndex, {
                shaderModule,
                bindGroupLayout,
                pipelineLayout,
                computePipeline,
                resources: module.resources, // Store resources for bind group creation later
                passInfo: pass // Store original pass info for dispatching
            });
        }
    }

    /**
     * 編碼調度特定計算通道的命令。
     * @param {import('./Parser.js').WGFXPass} passInfo - 要調度的通道的 IR 物件。
     * @param {GPUCommandEncoder} commandEncoder - 當前幀的命令編碼器。
     */
    dispatchPass(passInfo, commandEncoder) {
        const storedPipeline = this.pipelines.get(passInfo.index);
        if (!storedPipeline) {
            throw new Error(`找不到通道 ${passInfo.index} 的管線。是否已編譯？`);
        }

        const {computePipeline, bindGroupLayout, resources, passInfo: originalPassInfo} = storedPipeline;

        const bindGroupEntries = [];

        // Samplers
        resources.samplers.forEach(samp => {
            const sampler = this.resourceManager.getSampler(samp.name);
            if (!sampler) {
                // If 'sam' is a default sampler, it might not be in resourceManager if not explicitly defined in the shader.
                // For now, assume it's always available or created by resourceManager.
                if (samp.name === 'sam') {
                    // Create a default linear sampler if 'sam' is not found
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

        // Uniforms (binding 1 if present)
        if (resources.parameters.length > 0) {
            const uniformBuffer = this.resourceManager.getUniformBuffer();
            if (!uniformBuffer) {
                throw new Error("Uniform buffer not found in ResourceManager.");
            }
            bindGroupEntries.push({binding: 1, resource: {buffer: uniformBuffer}}); // Fixed binding for uniforms
        }

        // Textures
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

        // 編碼計算通道命令。
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);

        // 計算要調度的工作組數量。
        // 這基於主要輸出紋理的大小和工作組大小。
        const outputTextureName = originalPassInfo.out[0]; // 假定第一個輸出是主要輸出，用於確定大小。
        const outputTexture = this.resourceManager.getTexture(outputTextureName);
        const workgroupSize = originalPassInfo.numThreads; // 例如，[8, 8, 1]

        const dispatchX = Math.ceil(outputTexture.width / workgroupSize[0]);
        const dispatchY = Math.ceil(outputTexture.height / workgroupSize[1]);

        passEncoder.dispatchWorkgroups(dispatchX, dispatchY, 1);
        passEncoder.end();
    }
    /**
     * 清除所有儲存的 GPU 物件。不銷毀它們，因為它們由裝置管理。
     */
    dispose() {
        this.pipelines.clear();
        console.log("PipelineManager: 已清除所有儲存的管線狀態。");
    }
}