/// <reference types="@webgpu/types" />

/**
 * GPUExecutor — Unified GPU execution wrapper for WGFX runtime.
 * 
 * Executes both transpiled HLSL and native WGSL effects on WebGPU,
 * providing consistent output format (Float32Array RGBA) for comparison.
 */

import { WGFXRuntime } from '../runtime/WGFXRuntime';
import type { TranspileResult } from './HLSTranspiler';

export interface InputTexture {
    data: Uint8Array | Float32Array | Uint16Array;
    width: number;
    height: number;
    format: GPUTextureFormat;
}

export interface ExecutionOptions {
    /** Output texture width (defaults to input width) */
    outputWidth?: number;
    /** Output texture height (defaults to input height) */
    outputHeight?: number;
    /** Output texture format (defaults to rgba16float) */
    outputFormat?: GPUTextureFormat;
    /** External resource definitions for compilation */
    externalResources?: Record<string, any>;
    /** Macro defines for compilation */
    defines?: Record<string, number>;
    /** Whether to capture per-pass timings */
    captureTimings?: boolean;
    /** Whether to honor declared output sizes from effect */
    honorDeclaredOutput?: boolean;
}

export interface ExecutionResult {
    /** Output pixel data as Float32Array [width * height * 4] in RGBA order */
    output: Float32Array;
    /** Output texture width */
    width: number;
    /** Output texture height */
    height: number;
    /** Output texture format */
    format: GPUTextureFormat;
    /** Per-pass execution times in milliseconds (if captureTimings enabled) */
    passTimings?: number[];
    /** Total execution time in milliseconds */
    totalTime: number;
    /** Number of passes executed */
    passCount: number;
}

export interface CompiledEffect {
    runtime: WGFXRuntime;
    shaderInfo: any;
    passes: any[];
}

/**
 * GPUExecutor provides a unified interface for executing effects on GPU.
 */
export class GPUExecutor {
    private device: GPUDevice;

    constructor(device: GPUDevice) {
        this.device = device;
    }

    /**
     * Compile and prepare a WGSL effect for execution.
     */
    async compileWGSL(
        wgslCode: string,
        options: ExecutionOptions = {}
    ): Promise<CompiledEffect> {
        const runtime = new WGFXRuntime(this.device);

        const externalResources = options.externalResources || {};
        const defines = options.defines || {};

        await runtime.compile(wgslCode, externalResources, {
            honorDeclaredOutput: options.honorDeclaredOutput,
        });

        return {
            runtime,
            shaderInfo: runtime.shaderInfo,
            passes: runtime.shaderInfo?.passes || [],
        };
    }

    /**
     * Compile and prepare an HLSL effect (via transpilation) for execution.
     * This requires the HLSTranspiler to be used first.
     */
    async compileHLSL(
        hlslCode: string,
        options: ExecutionOptions = {}
    ): Promise<CompiledEffect> {
        // For HLSL, we use the same compile path since WGFXRuntime.parse()
        // already handles Magpie-style HLSL via the PEG parser
        return this.compileWGSL(hlslCode, options);
    }

    /**
     * Execute a compiled effect with the given input texture.
     */
    async execute(
        compiled: CompiledEffect,
        input: InputTexture,
        options: ExecutionOptions = {}
    ): Promise<ExecutionResult> {
        const { runtime, passes } = compiled;
        const startTime = performance.now();

        const outputWidth = options.outputWidth || input.width;
        const outputHeight = options.outputHeight || input.height;
        const outputFormat = options.outputFormat || 'rgba16float';

        const resourceManager = runtime.resourceManager;

        // Update scene buffer with correct dimensions
        resourceManager.updateSceneBuffer(
            input.width,
            input.height,
            outputWidth,
            outputHeight
        );

        // Write input texture
        const inputTexture = resourceManager.getTexture('INPUT');
        if (!inputTexture) {
            throw new Error("Input texture 'INPUT' not found in effect resources");
        }

        const inputBytesPerRow = this.bytesPerRow(input.width, input.format);
        this.device.queue.writeTexture(
            { texture: inputTexture },
            this._toUploadView(input),
            { bytesPerRow: inputBytesPerRow, rowsPerImage: input.height },
            [input.width, input.height, 1]
        );

        const passTimings: number[] = [];
        const totalPasses = passes.length;

        // Execute each pass
        for (let i = 0; i < totalPasses; i++) {
            const passStart = options.captureTimings ? performance.now() : 0;

            const cmd = this.device.createCommandEncoder();
            // Use the actual pass index from shader info (e.g., PASS_1 for //!PASS 1)
            runtime.dispatchPass(`PASS_${passes[i].index}`, cmd);
            this.device.queue.submit([cmd.finish()]);

            if (options.captureTimings) {
                passTimings.push(performance.now() - passStart);
            }
        }

        // Read back output texture
        const outputTexture = resourceManager.getTexture('OUTPUT');
        if (!outputTexture) {
            throw new Error("Output texture 'OUTPUT' not found in effect resources");
        }

        const output = await this.readTextureAsFloat32(
            outputTexture,
            outputWidth,
            outputHeight,
            outputFormat
        );

        const totalTime = performance.now() - startTime;

        return {
            output,
            width: outputWidth,
            height: outputHeight,
            format: outputFormat,
            passTimings: options.captureTimings ? passTimings : undefined,
            totalTime,
            passCount: totalPasses,
        };
    }

    /**
     * Convenience: compile and execute in one call (WGSL).
     */
    async runWGSL(
        wgslCode: string,
        input: InputTexture,
        options: ExecutionOptions = {}
    ): Promise<ExecutionResult> {
        const compiled = await this.compileWGSL(wgslCode, options);
        return this.execute(compiled, input, options);
    }

    /**
     * Convenience: compile and execute in one call (HLSL).
     */
    async runHLSL(
        hlslCode: string,
        input: InputTexture,
        options: ExecutionOptions = {}
    ): Promise<ExecutionResult> {
        const compiled = await this.compileHLSL(hlslCode, options);
        return this.execute(compiled, input, options);
    }

    /**
     * Execute transpiled HLSL modules directly.
     * ---
     * 生成的 WGSL 模組是獨立的 compute shader（無 `//!PASS` 指令、無 `fn Pass<N>`），
     * 不能經 `WGFXRuntime.compile()`（那會把它當效果檔重新 parse 而得到 0 個 pass）。
     * 此處直接為每個模組建立 pipeline 與 bind group（場景 uniform、取樣器、
     * sampled/storage 紋理），再逐 pass dispatch 並讀回 OUTPUT。
     * 支援 Magpie PS 風格（每輸出像素一 thread，`@workgroup_size` 自生成碼反推）。
     */
    async runModules(
        result: TranspileResult,
        input: InputTexture,
        options: ExecutionOptions = {}
    ): Promise<ExecutionResult> {
        const startTime = performance.now();

        if (!result.modules || result.modules.length === 0) {
            throw new Error('No transpiled modules to execute');
        }

        const outputWidth = options.outputWidth || input.width;
        const outputHeight = options.outputHeight || input.height;
        const outputFormat = options.outputFormat || 'rgba16float';

        // --- 1. Collect unique logical textures across all modules ---
        const texSpecs = new Map<string, { format: string; isStorage: boolean }>();
        for (const m of result.modules) {
            for (const t of m.resources.textures) {
                const existing = texSpecs.get(t.logicalName);
                if (existing) {
                    existing.isStorage = existing.isStorage || t.isStorage;
                } else {
                    texSpecs.set(t.logicalName, { format: t.format, isStorage: t.isStorage });
                }
            }
        }

        // --- 2. Create GPU textures (INPUT is input-sized; others output-sized) ---
        const gpuTextures = new Map<string, GPUTexture>();
        for (const [name, spec] of texSpecs) {
            const size: [number, number] = name === 'INPUT'
                ? [input.width, input.height]
                : [outputWidth, outputHeight];
            let usage = GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING;
            if (spec.isStorage) usage |= GPUTextureUsage.STORAGE_BINDING;
            gpuTextures.set(name, this.device.createTexture({
                size: [size[0], size[1], 1],
                format: spec.format as GPUTextureFormat,
                usage,
            }));
        }

        // --- 3. Upload input texture ---
        const inputTexture = gpuTextures.get('INPUT');
        if (!inputTexture) throw new Error("Transpiled modules do not use an 'INPUT' texture");
        this.device.queue.writeTexture(
            { texture: inputTexture },
            this._toUploadView(input),
            { bytesPerRow: this.bytesPerRow(input.width, input.format), rowsPerImage: input.height },
            [input.width, input.height, 1]
        );

        // --- 4. Scene uniform buffer (layout must match ResourceManager.updateSceneBuffer) ---
        const sceneData = new ArrayBuffer(48);
        const su32 = new Uint32Array(sceneData);
        const sf32 = new Float32Array(sceneData);
        su32[0] = input.width;  su32[1] = input.height;        // inputSize
        sf32[2] = 1.0 / input.width; sf32[3] = 1.0 / input.height; // inputPt
        su32[4] = outputWidth; su32[5] = outputHeight;         // outputSize
        sf32[6] = 1.0 / outputWidth; sf32[7] = 1.0 / outputHeight; // outputPt
        sf32[8] = outputWidth / input.width; sf32[9] = outputHeight / input.height; // scale
        su32[10] = 0;                                          // frame
        sf32[11] = 0;                                          // _pad0
        const sceneBuffer = this.device.createBuffer({
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(sceneBuffer, 0, sceneData);

        // --- 5. Samplers (filter from parsed HLSL; Magpie default address = CLAMP) ---
        const samplers = new Map<string, GPUSampler>();
        for (const s of result.shaderInfo.samplers || []) {
            samplers.set(s.name, this.device.createSampler({
                magFilter: s.filter === 'LINEAR' ? 'linear' : 'nearest',
                minFilter: s.filter === 'LINEAR' ? 'linear' : 'nearest',
                addressModeU: (s as any).address === 'WRAP' ? 'repeat' : 'clamp-to-edge',
                addressModeV: (s as any).address === 'WRAP' ? 'repeat' : 'clamp-to-edge',
            }));
        }

        // --- 6. Execute each module pass ---
        const passTimings: number[] = [];
        for (const m of result.modules.slice().sort((a, b) => a.passIndex - b.passIndex)) {
            const passStart = options.captureTimings ? performance.now() : 0;

            const shaderModule = this.device.createShaderModule({ code: m.wgslCode });
            const ws = /@workgroup_size\s*\(\s*(\d+)\s*,\s*(\d+)/.exec(m.wgslCode);
            const bx = ws ? Number(ws[1]) : 1;
            const by = ws ? Number(ws[2]) : 1;
            const pipeline = this.device.createComputePipeline({
                layout: 'auto',
                compute: { module: shaderModule, entryPoint: 'main_cs' },
            });

            const entries: GPUBindGroupEntry[] = [];
            if (m.resources.hasScene) {
                entries.push({ binding: m.resources.sceneBinding ?? 4, resource: { buffer: sceneBuffer } });
            }
            for (const s of m.resources.samplers) {
                const sampler = samplers.get(s.name);
                if (!sampler) throw new Error(`Sampler '${s.name}' not found in parsed HLSL`);
                entries.push({ binding: s.binding, resource: sampler });
            }
            for (const t of m.resources.textures) {
                const tex = gpuTextures.get(t.logicalName);
                if (!tex) throw new Error(`Texture '${t.logicalName}' not allocated`);
                entries.push({ binding: t.binding, resource: tex.createView() });
            }
            const bindGroup = this.device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries,
            });

            const dx = Math.max(1, Math.ceil(outputWidth / bx));
            const dy = Math.max(1, Math.ceil(outputHeight / by));
            const cmd = this.device.createCommandEncoder();
            const pass = cmd.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(dx, dy, 1);
            pass.end();
            this.device.queue.submit([cmd.finish()]);

            if (options.captureTimings) passTimings.push(performance.now() - passStart);
        }

        // --- 7. Read back OUTPUT ---
        const outputTexture = gpuTextures.get('OUTPUT');
        if (!outputTexture) throw new Error("Transpiled modules do not write an 'OUTPUT' texture");
        const output = await this.readTextureAsFloat32(outputTexture, outputWidth, outputHeight, outputFormat);

        const totalTime = performance.now() - startTime;

        // --- 8. Cleanup GPU resources ---
        for (const tex of gpuTextures.values()) tex.destroy();
        sceneBuffer.destroy();

        return {
            output,
            width: outputWidth,
            height: outputHeight,
            format: outputFormat,
            passTimings: options.captureTimings ? passTimings : undefined,
            totalTime,
            passCount: result.modules.length,
        };
    }

    /**
     * Convert input pixel data to the ArrayBufferView expected by writeTexture.
     * ---
     * Uint8Array / Uint16Array 直接上傳；Float32Array 依格式轉換
     * （rgba8unorm → 0-255 位元組；float 格式 → f16 位元組序列）。
     */
    private _toUploadView(input: InputTexture): any {
        if (input.data instanceof Uint8Array || input.data instanceof Uint16Array) {
            return input.data;
        }
        // Float32Array - need to convert for texture upload
        if (input.format === 'rgba8unorm' || input.format === 'rgba8unorm-srgb') {
            const converted = new Uint8Array(input.data.length);
            for (let i = 0; i < input.data.length; i++) {
                converted[i] = Math.round(Math.min(1, Math.max(0, input.data[i])) * 255);
            }
            return converted;
        }
        // For float formats, create Uint16Array view (f16 bytes)
        const f32 = input.data as Float32Array;
        const out = new Uint16Array(f32.length);
        for (let i = 0; i < f32.length; i++) {
            out[i] = this.f32ToF16(f32[i]);
        }
        return out;
    }

    /**
     * Convert a float32 value to half-float bits (round-to-nearest-even not
     * required for verification; matches WGFX's truncation model tolerance).
     */
    private f32ToF16(v: number): number {
        const f32 = new Float32Array(1);
        const i32 = new Int32Array(f32.buffer);
        f32[0] = v;
        const x = i32[0];
        const sign = (x >> 16) & 0x8000;
        let exponent = (x >> 23) & 0xff;
        let mantissa = x & 0x7fffff;
        if (exponent === 0xff) {
            // Inf / NaN
            return sign | 0x7c00 | (mantissa ? 0x0200 : 0);
        }
        // Re-bias exponent: 127 -> 15
        exponent = exponent - 127 + 15;
        if (exponent >= 0x1f) return sign | 0x7c00; // overflow -> Inf
        if (exponent <= 0) {
            // Subnormal or zero
            if (exponent < -10) return sign;
            mantissa = (mantissa | 0x800000) >> (1 - exponent + 13);
            return sign | (mantissa & 0x3ff);
        }
        return sign | (exponent << 10) | (mantissa >> 13);
    }

    /**
     * Read a GPU texture back as Float32Array (RGBA, normalized 0-1).
     */
    private async readTextureAsFloat32(
        texture: GPUTexture,
        width: number,
        height: number,
        format: GPUTextureFormat
    ): Promise<Float32Array> {
        // Calculate buffer layout (256-byte aligned rows)
        const bytesPerPixel = this.formatByteSize(format);
        const bytesPerRow = Math.ceil((width * bytesPerPixel) / 256) * 256;
        const bufferSize = bytesPerRow * height;

        const stagingBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const cmd = this.device.createCommandEncoder();
        cmd.copyTextureToBuffer(
            { texture, origin: [0, 0, 0] },
            { buffer: stagingBuffer, bytesPerRow, rowsPerImage: height },
            [width, height, 1]
        );
        this.device.queue.submit([cmd.finish()]);

        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const buffer = stagingBuffer.getMappedRange();
        const rawData = new Uint8Array(buffer.slice(0));
        stagingBuffer.unmap();
        stagingBuffer.destroy();

        // Convert to Float32Array based on format
        return this.convertToFloat32(rawData, width, height, format, bytesPerRow);
    }

    /**
     * Convert raw texture data to normalized Float32Array RGBA.
     */
    private convertToFloat32(
        rawData: Uint8Array,
        width: number,
        height: number,
        format: GPUTextureFormat,
        bytesPerRow: number
    ): Float32Array {
        const output = new Float32Array(width * height * 4);
        const rowStride = bytesPerRow;

        switch (format) {
            case 'rgba16float': {
                const view = new Uint16Array(rawData.buffer, rawData.byteOffset, rawData.byteLength / 2);
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        const srcIdx = y * (rowStride / 2) + x * 4;
                        const dstIdx = (y * width + x) * 4;
                        output[dstIdx] = this.f16ToF32(view[srcIdx]);
                        output[dstIdx + 1] = this.f16ToF32(view[srcIdx + 1]);
                        output[dstIdx + 2] = this.f16ToF32(view[srcIdx + 2]);
                        output[dstIdx + 3] = this.f16ToF32(view[srcIdx + 3]);
                    }
                }
                break;
            }
            case 'rgba8unorm':
            case 'rgba8unorm-srgb': {
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        const srcIdx = y * rowStride + x * 4;
                        const dstIdx = (y * width + x) * 4;
                        output[dstIdx] = rawData[srcIdx] / 255;
                        output[dstIdx + 1] = rawData[srcIdx + 1] / 255;
                        output[dstIdx + 2] = rawData[srcIdx + 2] / 255;
                        output[dstIdx + 3] = rawData[srcIdx + 3] / 255;
                    }
                }
                break;
            }
            case 'rgba32float': {
                const view = new Float32Array(rawData.buffer, rawData.byteOffset, rawData.byteLength / 4);
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        const srcIdx = y * (rowStride / 4) + x * 4;
                        const dstIdx = (y * width + x) * 4;
                        output[dstIdx] = view[srcIdx];
                        output[dstIdx + 1] = view[srcIdx + 1];
                        output[dstIdx + 2] = view[srcIdx + 2];
                        output[dstIdx + 3] = view[srcIdx + 3];
                    }
                }
                break;
            }
            default:
                throw new Error(`Unsupported readback format: ${format}`);
        }

        return output;
    }

    /**
     * Get byte size per pixel for a texture format.
     */
    private formatByteSize(format: GPUTextureFormat): number {
        const sizes: Record<string, number> = {
            'r8unorm': 1, 'r8snorm': 1, 'r8uint': 1, 'r8sint': 1,
            'r16float': 2, 'r16uint': 2, 'r16sint': 2,
            'r32float': 4, 'r32uint': 4, 'r32sint': 4,
            'rg8unorm': 2, 'rg8snorm': 2, 'rg8uint': 2, 'rg8sint': 2,
            'rg16float': 4, 'rg16uint': 4, 'rg16sint': 4,
            'rg32float': 8, 'rg32uint': 8, 'rg32sint': 8,
            'rgba8unorm': 4, 'rgba8snorm': 4, 'rgba8uint': 4, 'rgba8sint': 4,
            'rgba8unorm-srgb': 4,
            'bgra8unorm': 4, 'bgra8unorm-srgb': 4,
            'rgba16float': 8, 'rgba16uint': 8, 'rgba16sint': 8,
            'rgba32float': 16, 'rgba32uint': 16, 'rgba32sint': 16,
            'rg11b10ufloat': 4,
            'rgb9e5ufloat': 4,
            'rgb10a2unorm': 4,
        };
        return sizes[format] || 4;
    }

    /**
     * Calculate bytes per row for texture upload (256-byte aligned).
     */
    private bytesPerRow(width: number, format: GPUTextureFormat): number {
        const bpp = this.formatByteSize(format);
        return Math.ceil((width * bpp) / 256) * 256;
    }

    /**
     * Convert half-precision (fp16) to single-precision (fp32).
     * Uses the same truncation logic as WGFX tests.
     */
    private f16ToF32(h: number): number {
        const s = (h & 0x8000) >> 15;
        const e = (h & 0x7c00) >> 10;
        const f = h & 0x03ff;

        if (e === 0) {
            return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
        }
        if (e === 0x1f) {
            return f ? NaN : (s ? -Infinity : Infinity);
        }
        return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    }

    /**
     * Dispose of compiled effect resources.
     */
    dispose(compiled: CompiledEffect): void {
        compiled.runtime.resourceManager.dispose();
    }
}

/**
 * Create a GPU executor for the given device.
 */
export function createExecutor(device: GPUDevice): GPUExecutor {
    return new GPUExecutor(device);
}