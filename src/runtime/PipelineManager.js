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

        /** @type {Map<number, GPUComputePipeline>} */
        this.pipelines = new Map();
        /** @type {Map<number, GPUBindGroup>} */
        this.bindGroups = new Map(); // 注意：目前在每次調度時重新建立。
        /** @type {Map<number, GPUBindGroupLayout>} */
        this.bindGroupLayouts = new Map();
    }

    /**
     * 非同步建立著色器 IR 中定義的所有通道的計算管線。
     * @param {import('./Parser.js').WGFXShaderInfo} shaderInfo - 解析後的著色器資訊 (IR)。
     * @param {string} compiledWGSL - 完整、生成的 WGSL 著色器程式碼。
     */
    async createPipelines(shaderInfo, compiledWGSL) {
        const shaderModule = this.device.createShaderModule({
            label: 'WGFX 著色器模組',
            code: compiledWGSL,
        });

        // 為簡化起見，此實作對所有通道使用單一、共享的綁定組佈局。
        // 它包含著色器中宣告的每個資源。更優化的方法可能會
        // 根據每個通道使用的特定資源建立不同的佈局。
        const layoutEntries = [];
        let bindingIndex = 0;

        // uniform 緩衝區的條目 (如果存在)。
        if (this.resourceManager.getUniformBuffer()) {
            layoutEntries.push({
                binding: bindingIndex++,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {type: 'uniform'},
            });
        }

        // 紋理的條目。
        // 警告：此簡化實作依賴於綁定的固定、排序順序。
        // 穩健的解決方案應在程式碼生成器和此管理器之間使用共享綁定映射。
        const sortedTextureNames = [...this.resourceManager.textures.keys()].sort();
        sortedTextureNames.forEach(name => {
            const texInfo = shaderInfo.textures.find(t => t.name === name) || {name};
            const isStorage = shaderInfo.passes.some(p => p.out.includes(name));
            layoutEntries.push({
                binding: bindingIndex++,
                visibility: GPUShaderStage.COMPUTE,
                ...(isStorage
                        ? {
                            storageTexture: {
                                access: 'write-only',
                                format: (texInfo.format || 'rgba8unorm').toLowerCase().replace(/_/g, '')
                            }
                        }
                        : {texture: {}}
                ),
            });
        });

        // 取樣器的條目。
        const sortedSamplerNames = [...this.resourceManager.samplers.keys()].sort();
        sortedSamplerNames.forEach(name => {
            layoutEntries.push({
                binding: bindingIndex++,
                visibility: GPUShaderStage.COMPUTE,
                sampler: {},
            });
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            label: 'WGFX 主綁定組佈局',
            entries: layoutEntries,
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
        });

        // 為 IR 中定義的每個通道建立計算管線。
        for (const pass of shaderInfo.passes) {
            const pipeline = await this.device.createComputePipelineAsync({
                label: `WGFX 通道 ${pass.index} 的管線`,
                layout: pipelineLayout,
                compute: {
                    module: shaderModule,
                    entryPoint: `pass_${pass.index}`, // 入口點名稱與 WGSLCodeGenerator 中的名稱匹配。
                },
            });
            this.pipelines.set(pass.index, pipeline);
            this.bindGroupLayouts.set(pass.index, bindGroupLayout); // 儲存佈局以供調度時使用。
        }
    }

    /**
     * 編碼調度特定計算通道的命令。
     * @param {import('./Parser.js').WGFXPass} passInfo - 要調度的通道的 IR 物件。
     * @param {GPUCommandEncoder} commandEncoder - 當前幀的命令編碼器。
     */
    dispatchPass(passInfo, commandEncoder) {
        const pipeline = this.pipelines.get(passInfo.index);
        if (!pipeline) {
            throw new Error(`找不到通道 ${passInfo.index} 的管線。是否已編譯？`);
        }

        // 為此調度即時建立 BindGroup。
        // 如果資源視圖保證在調度之間不變，則可以快取此操作。
        const bindGroupLayout = this.bindGroupLayouts.get(passInfo.index);
        const entries = [];
        let bindingIndex = 0;

        // 綁定 uniform 緩衝區 (如果存在)。
        const uniformBuffer = this.resourceManager.getUniformBuffer();
        if (uniformBuffer) {
            entries.push({binding: bindingIndex++, resource: {buffer: uniformBuffer}});
        }

        // 以用於佈局的相同排序順序綁定紋理和取樣器。
        const sortedTextureNames = [...this.resourceManager.textures.keys()].sort();
        sortedTextureNames.forEach(name => {
            entries.push({binding: bindingIndex++, resource: this.resourceManager.getTexture(name).createView()});
        });

        const sortedSamplerNames = [...this.resourceManager.samplers.keys()].sort();
        sortedSamplerNames.forEach(name => {
            entries.push({binding: bindingIndex++, resource: this.resourceManager.getSampler(name)});
        });

        const bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries,
        });

        // 編碼計算通道命令。
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);

        // 計算要調度的工作組數量。
        // 這基於主要輸出紋理的大小和工作組大小。
        const outputTextureName = passInfo.out[0]; // 假定第一個輸出是主要輸出，用於確定大小。
        const outputTexture = this.resourceManager.getTexture(outputTextureName);
        const workgroupSize = passInfo.numThreads; // 例如，[8, 8, 1]

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
        this.bindGroups.clear();
        this.bindGroupLayouts.clear();
        console.log("PipelineManager: 已清除所有儲存的管線狀態。");
    }
}