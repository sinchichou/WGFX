import { WGFXShaderInfo, ParameterInfo } from './types/shader';
import { WGFXRuntime } from './runtime/WGFXRuntime';

export interface WGFXOptions {
    device: GPUDevice;
    effectCode: string;
    width: number;
    height: number;
    externalResources?: any;
}

export interface WGFXInfo {
    width: number;
    height: number;
    uniforms: {
        name: string;
        type: string;
        default: number;
        min: number;
        max: number;
        step: number;
    }[];
    passes: number;
}

export class WGFX {
    public runtime: WGFXRuntime;
    public initialized: boolean;
    public width: number;
    public height: number;
    public currentInputSource: any;

    constructor(runtime: WGFXRuntime) {
        this.runtime = runtime;
        this.initialized = false;
        this.currentInputSource = null;
        this.width = 0;
        this.height = 0;
    }

    public static async create({ device, effectCode, width, height }: WGFXOptions): Promise<WGFX> {
        if (!device) {
            throw new Error('Must provide a valid GPUDevice');
        }
        if (!effectCode || typeof effectCode !== 'string') {
            throw new Error('Must provide valid effectCode string');
        }
        if (!width || !height || width <= 0 || height <= 0) {
            throw new Error('Width and height must be positive numbers');
        }

        const runtime = new WGFXRuntime(device);

        // 2. Prepare external resources
        const externalResources = {
            defines: {
                INPUT_WIDTH: width,
                INPUT_HEIGHT: height
            },
            textures: {
                INPUT: {
                    size: [width, height],
                    format: 'rgba8unorm',
                    usage: GPUTextureUsage.TEXTURE_BINDING |
                        GPUTextureUsage.COPY_DST |
                        GPUTextureUsage.RENDER_ATTACHMENT
                },
                OUTPUT: {
                    size: [width, height],
                    format: 'rgba16float',
                    usage: GPUTextureUsage.STORAGE_BINDING |
                        GPUTextureUsage.COPY_SRC |
                        GPUTextureUsage.TEXTURE_BINDING
                }
            }
        };

        try {
            await runtime.compile(effectCode, externalResources);
        } catch (error: any) {
            console.error('WGFX compilation failed:', error);
            throw new Error(`Shader compilation error: ${error.message}`);
        }

        const instance = new WGFX(runtime);
        instance.width = width;
        instance.height = height;
        instance.initialized = true;

        console.log(`WGFX initialized: ${width}x${height}`);
        return instance;
    }

    public initialize(): WGFXInfo {
        if (!this.initialized) {
            throw new Error('Instance not initialized, use WGFX.create()');
        }

        const shaderInfo = this.runtime.shaderInfo;
        if (!shaderInfo) throw new Error("Shader info missing");

        return {
            width: this.width,
            height: this.height,
            uniforms: shaderInfo.parameters.map((p: ParameterInfo) => ({
                name: p.name,
                type: p.type || 'f32',
                default: (Array.isArray(p.default) ? p.default[0] : p.default) ?? 0.0,
                min: p.min ?? 0.0,
                max: p.max ?? 1.0,
                step: p.step ?? 0.01
            })),
            passes: shaderInfo.passes.length
        };
    }

    public updateUniforms(uniforms: Record<string, number | number[]>): void {
        if (!this.initialized) {
            throw new Error('Instance not initialized');
        }

        if (!uniforms || typeof uniforms !== 'object') {
            console.warn('updateUniforms: Invalid uniforms object');
            return;
        }

        for (const [name, value] of Object.entries(uniforms)) {
            try {
                this.runtime.updateUniform(name, value);
            } catch (error: any) {
                console.warn(`Failed to update uniform "${name}":`, error.message);
            }
        }
    }

    public async process(inputSource: ImageBitmap | VideoFrame | HTMLVideoElement | HTMLCanvasElement): Promise<GPUTexture> {
        if (!this.initialized) {
            throw new Error('Instance not initialized');
        }

        if (!inputSource) {
            throw new Error('Must provide input source');
        }

        let sourceWidth: number = 0;
        let sourceHeight: number = 0;

        if (inputSource instanceof HTMLVideoElement) {
            sourceWidth = inputSource.videoWidth;
            sourceHeight = inputSource.videoHeight;
        } else if (inputSource instanceof VideoFrame) {
            sourceWidth = inputSource.displayWidth;
            sourceHeight = inputSource.displayHeight;
        } else {
            sourceWidth = inputSource.width;
            sourceHeight = inputSource.height;
        }

        if (sourceWidth !== this.width || sourceHeight !== this.height) {
            throw new Error(
                `Input dimensions (${sourceWidth}x${sourceHeight}) ` +
                `do not match initialized dimensions (${this.width}x${this.height})`
            );
        }

        try {
            this.runtime.resourceManager.updateTextureFromImage('INPUT', inputSource);

            const commandEncoder = this.runtime.device.createCommandEncoder({
                label: 'WGFX Frame Processing'
            });

            if (this.runtime.shaderInfo) {
                for (const pass of this.runtime.shaderInfo.passes) {
                    this.runtime.dispatchPass(`PASS_${pass.index}`, commandEncoder);
                }
            }

            this.runtime.device.queue.submit([commandEncoder.finish()]);

            const output = this.runtime.resourceManager.getTexture('OUTPUT');
            if (!output) throw new Error("Output texture missing");
            return output;

        } catch (error) {
            console.error('Frame processing error:', error);
            throw error;
        }
    }

    public getOutputView(): GPUTextureView {
        return this.runtime.getOutput();
    }

    public dispose(): void {
        if (this.runtime) {
            this.runtime.resourceManager.dispose();
            this.runtime.pipelineManager.dispose();
        }
        this.initialized = false;
        console.log('WGFX resources disposed');
    }
}
export default WGFX;
