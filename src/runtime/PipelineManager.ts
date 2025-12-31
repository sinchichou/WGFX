/// <reference types="@webgpu/types" />
import { ResourceManager } from './ResourceManager';

interface StoredPipeline {
    computePipeline: GPUComputePipeline;
    bindGroupLayout: GPUBindGroupLayout;
    resources: any;
    passInfo: any;
}

export class PipelineManager {
    public device: GPUDevice;
    public resourceManager: ResourceManager;
    public pipelines: Map<number, StoredPipeline>;

    constructor(device: GPUDevice, resourceManager: ResourceManager) {
        this.device = device;
        this.resourceManager = resourceManager;
        this.pipelines = new Map();
    }

    public async createPipelines(shaderInfo: any, generatedModules: any[]): Promise<void> {
        this.pipelines.clear();

        for (const module of generatedModules) {
            const pass = shaderInfo.passes.find((p: any) => p.index === module.passIndex);
            if (!pass) continue;

            const shaderModule = this.device.createShaderModule({
                code: module.wgslCode,
                label: `Pass ${module.passIndex} Shader`
            });

            // Retrieve compilation info for debugging
             shaderModule.getCompilationInfo().then(info => {
                 if (info.messages.length > 0) {
                     // Can implement logging here if needed
                 }
             });

            const bindingMap = new Map<number, GPUBindGroupLayoutEntry>();

            // Defaults: Sampler @ 0
            bindingMap.set(0, {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                sampler: { type: 'filtering' } 
            });

            // Uniforms @ 1
            if (module.resources.parameters.length > 0) {
                 bindingMap.set(1, {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: 'uniform' }
                });
            }

            // Scene Info @ 4 (User Specified)
            bindingMap.set(4, {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'uniform' }
            });

            // Textures
            module.resources.textures.forEach((tex: any) => {
                const format = 'rgba16float';
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

            // Extra samplers
            module.resources.samplers.forEach((samp: any) => {
                 if (samp.binding !== 0) {
                     bindingMap.set(samp.binding, {
                         binding: samp.binding,
                         visibility: GPUShaderStage.COMPUTE,
                         sampler: { type: 'filtering' }
                     });
                 }
            });

            const entries = Array.from(bindingMap.values()).sort((a, b) => a.binding - b.binding);
            const bindGroupLayout = this.device.createBindGroupLayout({ entries });
            const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
            
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

    public dispatchPass(passInfo: any, commandEncoder: GPUCommandEncoder): void {
        const pipeline = this.pipelines.get(passInfo.index);
        if (!pipeline) throw new Error(`Pipeline for pass ${passInfo.index} not found`);

        const { computePipeline, bindGroupLayout, resources, passInfo: originalPassInfo } = pipeline;
        const groupEntries: GPUBindGroupEntry[] = [];

        // 1. Defaut sampler
        let defaultSampler = this.resourceManager.getSampler('sam');
        if (!defaultSampler) {
             // Create one if mostly needed or throw? Assuming it might exist or create temp
             defaultSampler = this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
        }
        groupEntries.push({ binding: 0, resource: defaultSampler });

        // 2. Uniforms
        if (resources.parameters.length > 0) {
            const ub = this.resourceManager.getUniformBuffer();
            if (!ub) throw new Error("Uniform buffer missing");
            groupEntries.push({ binding: 1, resource: { buffer: ub } });
        }

        // 3. Textures
        resources.textures.forEach((tex: any) => {
            const mode = tex.isStorage ? 'storage' : 'sampled';
            const view = this.resourceManager.getTextureView(tex.name, mode);
            groupEntries.push({ binding: tex.binding, resource: view });
        });

        // 4. Other samplers
         resources.samplers.forEach((samp: any) => {
            if (samp.binding !== 0) {
                 const s = this.resourceManager.getSampler(samp.name);
                 if (s) groupEntries.push({ binding: samp.binding, resource: s });
            }
        });



        // 5. Scene Info (Binding 4)
        const sceneBuffer = this.resourceManager.getSceneBuffer();
        if (sceneBuffer) {
             groupEntries.push({ binding: 4, resource: { buffer: sceneBuffer } });
        }

        const bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: groupEntries
        });

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);

        const outName = originalPassInfo.out[0];
        const outTex = this.resourceManager.getTexture(outName);
        if (!outTex) throw new Error("Output texture not found");

        const workgroupSize = originalPassInfo.blockSize || originalPassInfo.numThreads;
        // Use blockSize[0] (width) and blockSize[1] (height). If 1D array provided for block_size, parser handles it?
        // Parser: data.blockSize.push(data.blockSize[0]) if length 1. So it's at least [w, h].
        // numThreads is [x, y, z].
        
        let blockW = workgroupSize[0];
        let blockH = workgroupSize[1];
        
        // Safety check if blockSize was 1D but parser didn't expand (it does).
        // If numThreads (64, 1, 1), blockH is 1. If it's a 1D compute shader mapping to 2D, we rely on having valid blockSize.
        if (!originalPassInfo.blockSize && originalPassInfo.numThreads[1] === 1 && outTex.height > 1) {
             // Heuristic/Warning: Using 1D workgroup size for 2D dispatch without explicit block_size?
             // Maybe default to square root if 64? No, too magic.
             // Just trust the values.
        }

        const dx = Math.ceil(outTex.width / blockW);
        const dy = Math.ceil(outTex.height / blockH);
        
        console.log(`Dispatch Pass ${passInfo.index}: outTex=${outTex.width}x${outTex.height}, block=${blockW}x${blockH}, dispatch=${dx}x${dy}`);

        passEncoder.dispatchWorkgroups(dx, dy, 1);
        passEncoder.end();
    }

    public dispose(): void {
        this.pipelines.clear();
    }
}
