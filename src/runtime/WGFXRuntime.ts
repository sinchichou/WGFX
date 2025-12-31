/// <reference types="@webgpu/types" />
import { parse } from './ShaderParser';
import { ResourceManager } from './ResourceManager';
import { PipelineManager } from './PipelineManager';
import { WGSLCodeGenerator } from './WGSLCodeGenerator';
import { UniformBinder } from './UniformBinder';
import { WGFXShaderInfo } from '../types/shader';

export class WGFXRuntime {
    public device: GPUDevice;
    public resourceManager: ResourceManager;
    public pipelineManager: PipelineManager;
    public wgslCodeGenerator: WGSLCodeGenerator;
    public uniformBinder: UniformBinder;
    public shaderInfo: WGFXShaderInfo | null;

    constructor(device: GPUDevice) {
        if (!device) {
            throw new Error("WGFXRuntime requires a valid WebGPU device.");
        }
        this.device = device;
        this.resourceManager = new ResourceManager(this.device);
        this.pipelineManager = new PipelineManager(this.device, this.resourceManager);
        this.wgslCodeGenerator = new WGSLCodeGenerator();
        this.uniformBinder = new UniformBinder(this.device, this.resourceManager);
        this.shaderInfo = null;
    }

    public async compile(effectCode: string, externalResources: any = {}): Promise<void> {
        console.log("WGFXRuntime: Starting effect compilation.");
        
        try {
            const shaderInfo = parse(effectCode);
            this.shaderInfo = shaderInfo;
            console.log("WGFXRuntime: ShaderInfo parsed");

            const generatedModules = this.wgslCodeGenerator.generate(shaderInfo);
            
            this.resourceManager.initialize(shaderInfo, externalResources);
            console.log("WGFXRuntime: Resources initialized.");

            await this.pipelineManager.createPipelines(shaderInfo, generatedModules);
            console.log("WGFXRuntime: Pipelines created.");
            console.log("WGFXRuntime: Compilation complete.");
        } catch (e) {
            console.error("WGFXRuntime: Parse/Compile error", e);
            throw e;
        }
    }

    public dispatchPass(passName: string, commandEncoder: GPUCommandEncoder): void {
        if (!this.shaderInfo) {
            throw new Error("Effect not compiled. Call compile() first.");
        }

        const passIndex = parseInt(passName.split('_')[1], 10);
        const passInfo = this.shaderInfo.passes.find(p => p.index === passIndex);
        
        if (!passInfo) {
             throw new Error(`Pass "${passName}" not found.`);
        }

        this.pipelineManager.dispatchPass(passInfo, commandEncoder);
    }

    public updateUniform(name: string, value: number | number[]): void {
        if (!this.shaderInfo) {
            throw new Error("Effect not compiled. Call compile() first.");
        }
        this.uniformBinder.updateUniform(name, value);
    }

    public getOutput(): GPUTextureView {
        const outputTexture = this.resourceManager.getTexture('OUTPUT');
        if (!outputTexture) {
             throw new Error("Output texture 'OUTPUT' not found.");
        }
        // Always return usage=TextureBinding/StorageBinding compatible view?
        // Usually output to screen, so sampled view might be enough, but check usage
        return outputTexture.createView();
    }

    public getResource(name: string): GPUTexture | GPUSampler | GPUBuffer | undefined {
        return this.resourceManager.getTexture(name)
            || this.resourceManager.getSampler(name)
            || this.resourceManager.getUniform(name)?.buffer;
    }
}
