/// <reference types="@webgpu/types" />
import { WGFXShaderInfo, TextureInfo } from '../types/shader';

export class ResourceManager {
    public device: GPUDevice;
    public textures: Map<string, GPUTexture>;
    public views: Map<string, { sampled: GPUTextureView; storage: GPUTextureView }>;
    public samplers: Map<string, GPUSampler>;
    public uniforms: Map<string, { buffer: GPUBuffer; offset: number; size: number }>;
    public uniformBuffer: GPUBuffer | null;

    constructor(device: GPUDevice) {
        if (!device) {
            throw new Error('ResourceManager requires a valid GPUDevice');
        }
        this.device = device;
        this.textures = new Map();
        this.views = new Map();
        this.samplers = new Map();
        this.uniforms = new Map();
        this.uniformBuffer = null;
    }

    public initialize(shaderInfo: WGFXShaderInfo, externalResources: any = {}): void {
        const context: Record<string, number> = {};

        // 1. Load defines
        if (externalResources.defines) {
            Object.assign(context, externalResources.defines);
            console.log('Loaded external defines:', context);
        }

        // 2. Create external textures
        if (externalResources.textures) {
            for (const [name, descriptor] of Object.entries(externalResources.textures)) {
                console.log(`Creating external texture: ${name}`, descriptor);
                this.createTexture(name, descriptor as GPUTextureDescriptor);
            }
        }

        // Evaluate helper
        const evaluate = (expr: string | number, ctx: Record<string, number>): number => {
            if (typeof expr === 'number') return expr;
            
            // Remove whitespace
            expr = expr.replace(/\s+/g, '');
            
            // Simple recursive descent parser to handle basic arithmetic and variables
            let pos = 0;

            const parseNumber = (): number => {
                let startKey = pos;
                if (/[a-zA-Z_]/.test(expr[pos])) {
                    while (pos < expr.length && /[a-zA-Z0-9_]/.test(expr[pos])) pos++;
                    const key = expr.slice(startKey, pos);
                    if (key in ctx) return ctx[key];
                    throw new Error(`Undefined variable: ${key}`);
                } else if (/[0-9.]/.test(expr[pos])) {
                    while (pos < expr.length && /[0-9.]/.test(expr[pos])) pos++;
                    return parseFloat(expr.slice(startKey, pos));
                } else if (expr[pos] === '(') {
                    pos++;
                    const val = parseExpression();
                    if (expr[pos] !== ')') throw new Error("Expected ')'");
                    pos++;
                    return val;
                }
                throw new Error(`Unexpected character at ${pos}: ${expr[pos]}`);
            };

            const parseFactor = (): number => {
                let val = parseNumber();
                while (pos < expr.length && (expr[pos] === '*' || expr[pos] === '/')) {
                    const op = expr[pos++];
                    const right = parseNumber();
                    if (op === '*') val *= right;
                    else val /= right;
                }
                return val;
            };

            const parseExpression = (): number => {
                let val = parseFactor();
                while (pos < expr.length && (expr[pos] === '+' || expr[pos] === '-')) {
                    const op = expr[pos++];
                    const right = parseFactor();
                    if (op === '+') val += right;
                    else val -= right;
                }
                return val;
            };

            try {
                return parseExpression();
            } catch (e) {
                 console.warn(`Failed to evaluate expression: ${expr}`, e);
                 return 0;
            }
        };

        // 3. Create intermediate textures
        shaderInfo.textures.forEach((tex: TextureInfo) => {
            if (this.textures.has(tex.name)) {
                console.log(`Skipping existing texture: ${tex.name}`);
                return;
            }

            const width = evaluate(tex.width, context);
            const height = evaluate(tex.height, context);

            if (!width || !height) {
                throw new Error(`Failed to determine size for texture ${tex.name}`);
            }

            // Simplified: All internal textures use rgba16float
            const format: GPUTextureFormat = (tex.name === 'INPUT' ? 'rgba8unorm' : 'rgba16float');

            const descriptor: GPUTextureDescriptor = {
                size: [width, height],
                format: format,
                usage: GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.STORAGE_BINDING |
                    GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.COPY_SRC
            };

            this.createTexture(tex.name, descriptor);
            console.log(`Created texture: ${tex.name} (${width}x${height}, ${format})`);
        });

        // 5. Create samplers
        shaderInfo.samplers.forEach(samp => {
            const descriptor: GPUSamplerDescriptor = {
                magFilter: samp.filter === 'LINEAR' ? 'linear' : 'nearest',
                minFilter: samp.filter === 'LINEAR' ? 'linear' : 'nearest',
                addressModeU: samp.address === 'WRAP' ? 'repeat' : 'clamp-to-edge',
                addressModeV: samp.address === 'WRAP' ? 'repeat' : 'clamp-to-edge',
            };
            this.createSampler(samp.name, descriptor);
            console.log(`Created sampler: ${samp.name}`);
        });

        // 6. Create uniform buffer
        if (shaderInfo.parameters.length > 0) {
            this.createUniformBuffer(shaderInfo.parameters);
            console.log(`Created UniformBuffer for ${shaderInfo.parameters.length} parameters`);
        }

        // 7. Create Scene Buffer
        this.createSceneBuffer();
    }

    public createTexture(name: string, descriptor: GPUTextureDescriptor): GPUTexture {
        if (this.textures.has(name)) {
            this.textures.get(name)!.destroy();
        }

        const texture = this.device.createTexture(descriptor);
        texture.label = name;
        this.textures.set(name, texture);

        // Create dual views
        this.views.set(name, {
            sampled: texture.createView({ label: `${name}_sampled` }),
            storage: texture.createView({ label: `${name}_storage` })
        });

        return texture;
    }

    public getTextureView(name: string, type: 'sampled' | 'storage' = 'sampled'): GPUTextureView {
        const v = this.views.get(name);
        if (!v) throw new Error(`View "${name}" not found`);
        return v[type];
    }

    public getTexture(name: string): GPUTexture | undefined {
        return this.textures.get(name);
    }

    public createSampler(name: string, descriptor: GPUSamplerDescriptor): GPUSampler {
        const sampler = this.device.createSampler(descriptor);
        sampler.label = name;
        this.samplers.set(name, sampler);
        return sampler;
    }

    public getSampler(name: string): GPUSampler | undefined {
        return this.samplers.get(name);
    }

    public getUniform(name: string): { buffer: GPUBuffer; offset: number; size: number } | undefined {
        return this.uniforms.get(name);
    }

    public updateTextureFromImage(name: string, source: ImageBitmap | HTMLVideoElement | HTMLCanvasElement | VideoFrame | OffscreenCanvas): void {
        const texture = this.textures.get(name);
        if (!texture) {
            console.warn(`Texture ${name} not found`);
            return;
        }

        this.device.queue.copyExternalImageToTexture(
            { source: source },
            { texture: texture },
            [texture.width, texture.height]
        );

        // Auto-update scene info if input changes
        if(name === 'INPUT' && (source instanceof HTMLVideoElement || source instanceof ImageBitmap || source instanceof VideoFrame)) {
            const width = (source instanceof VideoFrame) ? source.displayWidth : source.width;
            const height = (source instanceof VideoFrame) ? source.displayHeight : source.height;
            this.updateSceneBuffer(width, height);
        }
    }

    private createUniformBuffer(parameters: any[]): void {
        // Simple implementation: aligned to 16 bytes for simplicity
        // In reality, should follow std140 layout rules strictly
        let offset = 0;
        parameters.forEach(p => {
            const size = 4; // float/int
            this.uniforms.set(p.name, { buffer: null as any, offset, size });
            offset += 16; // Naive alignment (vec4 aligned)
        });

        const bufferSize = Math.max(offset, 16);
        this.uniformBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'GlobalUniforms'
        });

        this.uniforms.forEach(u => u.buffer = this.uniformBuffer!);
    }

    public getUniformBuffer(): GPUBuffer | null {
        return this.uniformBuffer;
    }

    // --- Scene Buffer ---
    public sceneBuffer: GPUBuffer | null = null;
    
    public createSceneBuffer(): void {
        if (this.sceneBuffer) return;
        this.sceneBuffer = this.device.createBuffer({
            size: 16, // vec2<u32> + vec2<f32> = 4+4 + 4+4 = 16 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'SceneInfo'
        });
    }

    public updateSceneBuffer(width: number, height: number): void {
        if (!this.sceneBuffer) return;
        if (width === 0 || height === 0) return;

        const data = new ArrayBuffer(16);
        const u32 = new Uint32Array(data);
        const f32 = new Float32Array(data);

        u32[0] = width;
        u32[1] = height;
        f32[2] = 1.0 / width;
        f32[3] = 1.0 / height;

        this.device.queue.writeBuffer(this.sceneBuffer, 0, data);
        console.log(`Updated scene buffer with inputSize=${u32[0]}, ${u32[1]} and InputPt=${f32[2]}, ${f32[3]}`);
    }
    
    public getSceneBuffer(): GPUBuffer | null {
        return this.sceneBuffer;
    }

    public dispose(): void {
        this.textures.forEach(t => t.destroy());
        this.textures.clear();
        this.views.clear();
        this.samplers.clear(); // Samplers are not destroyable in WebGPU spec but we clear ref
        if (this.uniformBuffer) {
            this.uniformBuffer.destroy();
            this.uniformBuffer = null;
        }
        if (this.sceneBuffer) {
            this.sceneBuffer.destroy();
            this.sceneBuffer = null;
        }
        this.uniforms.clear();
    }
}
