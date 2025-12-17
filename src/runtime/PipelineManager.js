// src/runtime/PipelineManager.js

/**
 * - EN: Manages the creation and scheduling of WebGPU compute pipelines.
 * - TW: 管理 WebGPU 計算管線的建立和調度。
 */
export class PipelineManager {
    constructor(device, resourceManager) {
        this.device = device;
        this.resourceManager = resourceManager;
        this.pipelines = new Map();

        // Setup uncaptured error handler for debugging
        this.device.addEventListener('uncapturederror', event => {
            console.error('WebGPU Uncaptured Error:', event.error.message);
        });
    }

    /**
     * - EN: Asynchronously creates compute pipelines for all passes.
     * - TW: 非同步建立所有通道的計算管線。
     */
    async createPipelines(shaderInfo, generatedModules) {
        this.pipelines.clear();

        for (const module of generatedModules) {
            const pass = shaderInfo.passes.find(p => p.index === module.passIndex);
            if (!pass) {
                console.warn(`PipelineManager: Pass ${module.passIndex} not found in shaderInfo.`);
                continue;
            }

            console.log(`建立 Pass ${module.passIndex} 的管線...`);
            console.log(`WGSL 程式碼長度: ${module.wgslCode.length} 字元`);

            try {
                // 1. Create GPUShaderModule with error scope
                this.device.pushErrorScope('validation');

                const shaderModule = this.device.createShaderModule({
                    code: module.wgslCode,
                    label: `Pass ${module.passIndex} Shader Module`
                });

                const shaderError = await this.device.popErrorScope();
                if (shaderError) {
                    console.error('Shader Module 建立錯誤:', shaderError.message);
                    throw shaderError;
                }

                // Check compilation info
                const compilationInfo = await shaderModule.getCompilationInfo();
                if (compilationInfo.messages.length > 0) {
                    console.group(`Shader 編譯訊息 (Pass ${module.passIndex}):`);
                    for (const msg of compilationInfo.messages) {
                        const level = msg.type === 'error' ? 'ERROR' : msg.type === 'warning' ? 'WARN' : 'INFO';
                        console.log(`${level} Line ${msg.lineNum}:${msg.linePos} - ${msg.message}`);

                        // Show context
                        if (msg.lineNum) {
                            const lines = module.wgslCode.split('\n');
                            const contextStart = Math.max(0, msg.lineNum - 2);
                            const contextEnd = Math.min(lines.length, msg.lineNum + 2);
                            console.log('Context:');
                            for (let i = contextStart; i < contextEnd; i++) {
                                const prefix = (i + 1) === msg.lineNum ? '>>> ' : '    ';
                                console.log(`${prefix}${i + 1}: ${lines[i]}`);
                            }
                        }
                    }
                    console.groupEnd();

                    // If there are errors, stop
                    const hasErrors = compilationInfo.messages.some(m => m.type === 'error');
                    if (hasErrors) {
                        throw new Error(`Shader compilation failed with ${compilationInfo.messages.filter(m => m.type === 'error').length} error(s)`);
                    }
                }

                console.log('Shader Module 建立成功');

                // 2. Create GPUBindGroupLayout
                const bindingMap = new Map();

                // Add Samplers
                module.resources.samplers.forEach(samp => {
                    console.log(`Sampler "${samp.name}" -> @binding(${samp.binding})`);
                    bindingMap.set(samp.binding, {
                        binding: samp.binding,
                        visibility: GPUShaderStage.COMPUTE,
                        sampler: {type: 'filtering'}
                    });
                });

                // Add Uniforms (binding 1 if present)
                if (module.resources.parameters.length > 0) {
                    console.log('Uniforms -> @binding(1)');
                    bindingMap.set(1, {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {type: 'uniform'}
                    });
                }

                // Add Textures
                module.resources.textures.forEach(tex => {
                    const texType = tex.isStorage ? 'storage' : 'sampled';
                    const normalizedFormat = (tex.format || 'rgba8unorm').toLowerCase().replace(/_/g, '');
                    console.log(`Texture "${tex.name}" (${texType}, ${normalizedFormat}) -> @binding(${tex.binding})`);

                    if (tex.isStorage) {
                        const validWriteStorageFormats = [
                            'r32float', 'r32sint', 'r32uint',
                            'rgba16float', 'rgba16sint', 'rgba16uint',
                            'rgba32float', 'rgba32sint', 'rgba32uint',
                            'rg32float', 'rg32sint', 'rg32uint'
                        ];

                        if (!validWriteStorageFormats.includes(normalizedFormat)) {
                            console.error(`Invalid write storage texture format: ${normalizedFormat}`);
                            throw new Error(
                                `Texture "${tex.name}" has invalid storage format "${normalizedFormat}". ` +
                                `Valid formats for write access: ${validWriteStorageFormats.join(', ')}`
                            );
                        }

                        bindingMap.set(tex.binding, {
                            binding: tex.binding,
                            visibility: GPUShaderStage.COMPUTE,
                            storageTexture: {
                                format: normalizedFormat,
                                access: 'write-only'
                            }
                        });
                    } else {
                        bindingMap.set(tex.binding, {
                            binding: tex.binding,
                            visibility: GPUShaderStage.COMPUTE,
                            texture: {sampleType: 'float'}
                        });
                    }
                });

                const bindGroupLayoutEntries = Array.from(bindingMap.values())
                    .sort((a, b) => a.binding - b.binding);

                console.log(`BindGroupLayout entries (${bindGroupLayoutEntries.length} total)`);

                this.device.pushErrorScope('validation');

                const bindGroupLayout = this.device.createBindGroupLayout({
                    entries: bindGroupLayoutEntries,
                    label: `Pass ${module.passIndex} Bind Group Layout`
                });

                const layoutError = await this.device.popErrorScope();
                if (layoutError) {
                    console.error('BindGroupLayout 建立錯誤:', layoutError.message);
                    throw layoutError;
                }

                console.log('BindGroupLayout 建立成功');

                // 3. Create GPUPipelineLayout
                this.device.pushErrorScope('validation');

                const pipelineLayout = this.device.createPipelineLayout({
                    bindGroupLayouts: [bindGroupLayout],
                    label: `Pass ${module.passIndex} Pipeline Layout`
                });

                const pipelineLayoutError = await this.device.popErrorScope();
                if (pipelineLayoutError) {
                    console.error('PipelineLayout 建立錯誤:', pipelineLayoutError.message);
                    throw pipelineLayoutError;
                }

                console.log('PipelineLayout 建立成功');

                // 4. Create GPUComputePipeline
                this.device.pushErrorScope('validation');

                const computePipeline = await this.device.createComputePipelineAsync({
                    layout: pipelineLayout,
                    compute: {
                        module: shaderModule,
                        entryPoint: 'main_cs',
                    },
                    label: `Pass ${module.passIndex} Compute Pipeline`
                });

                const pipelineError = await this.device.popErrorScope();
                if (pipelineError) {
                    console.error('ComputePipeline 建立錯誤:', pipelineError.message);
                    throw pipelineError;
                }

                await this.device.queue.submit([]);

                console.log(`Pass ${module.passIndex} 管線建立成功`);

                this.pipelines.set(module.passIndex, {
                    shaderModule,
                    bindGroupLayout,
                    pipelineLayout,
                    computePipeline,
                    resources: module.resources,
                    passInfo: pass
                });

            } catch (error) {
                console.error(`Pass ${module.passIndex} 管線建立失敗:`, error);

                console.group(`Pass ${module.passIndex} 完整 WGSL 程式碼:`);
                const lines = module.wgslCode.split('\n');
                lines.forEach((line, idx) => {
                    console.log(`${(idx + 1).toString().padStart(4)}: ${line}`);
                });
                console.groupEnd();

                throw error;
            }
        }

        console.log(`成功建立 ${this.pipelines.size} 個管線`);
    }

    /**
     * - EN: Encodes commands to dispatch a specific compute pass.
     * - TW: 編碼調度特定計算通道的命令。
     */
    dispatchPass(passInfo, commandEncoder) {
        const storedPipeline = this.pipelines.get(passInfo.index);
        if (!storedPipeline) {
            throw new Error(`找不到通道 ${passInfo.index} 的管線。是否已編譯?`);
        }

        const {computePipeline, bindGroupLayout, resources, passInfo: originalPassInfo} = storedPipeline;

        const bindingMap = new Map();

        resources.samplers.forEach(samp => {
            const sampler = this.resourceManager.getSampler(samp.name);
            if (!sampler) {
                if (samp.name === 'sam') {
                    const defaultSampler = this.device.createSampler({
                        magFilter: 'linear',
                        minFilter: 'linear',
                    });
                    bindingMap.set(samp.binding, {
                        binding: samp.binding,
                        resource: defaultSampler
                    });
                } else {
                    throw new Error(`Sampler ${samp.name} not found in ResourceManager.`);
                }
            } else {
                bindingMap.set(samp.binding, {
                    binding: samp.binding,
                    resource: sampler
                });
            }
        });

        if (resources.parameters.length > 0) {
            const uniformBuffer = this.resourceManager.getUniformBuffer();
            if (!uniformBuffer) {
                throw new Error('Uniform buffer not found in ResourceManager.');
            }
            bindingMap.set(1, {
                binding: 1,
                resource: {buffer: uniformBuffer}
            });
        }

        resources.textures.forEach(tex => {
            const texture = this.resourceManager.getTexture(tex.name);
            if (!texture) {
                throw new Error(`Texture ${tex.name} not found in ResourceManager.`);
            }
            bindingMap.set(tex.binding, {
                binding: tex.binding,
                resource: texture.createView()
            });
        });

        const bindGroupEntries = Array.from(bindingMap.values())
            .sort((a, b) => a.binding - b.binding);

        const bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: bindGroupEntries,
            label: `Pass ${passInfo.index} Bind Group`
        });

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);

        const outputTextureName = originalPassInfo.out[0];
        const outputTexture = this.resourceManager.getTexture(outputTextureName);
        const workgroupSize = originalPassInfo.numThreads;

        const dispatchX = Math.ceil(outputTexture.width / workgroupSize[0]);
        const dispatchY = Math.ceil(outputTexture.height / workgroupSize[1]);

        passEncoder.dispatchWorkgroups(dispatchX, dispatchY, 1);
        passEncoder.end();
    }

    dispose() {
        this.pipelines.clear();
        console.log('PipelineManager: All pipeline states cleared.');
    }
}
