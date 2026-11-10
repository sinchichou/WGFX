/// <reference types="@webgpu/types" />
import { ResourceManager } from './ResourceManager';
import { PipelineManager } from './PipelineManager';
import { UniformBinder } from './UniformBinder';
import { readDDS } from './DDSReader';
import { Logger } from '../utils/Logger';

/**
 * WGFXPackage — `.wgx` 預編套件包的執行期載入器。
 *
 * `.wgx` 是 zip 容器，內含建置期預生成的 WGSL、外部紋理與 JSON manifest。
 * 本模組負責解壓、解析 manifest、準備資源，並在**不重複做 DSL 解析與程式碼生成**
 * 的前提下建立管線。
 *
 * MV3 相容性 / MV3 compatibility
 * ---
 * 套件包內只有**資料**（WGSL 文字、二進位紋理、JSON），沒有任何 JavaScript。
 * 本模組不使用 `eval` / `new Function`，因此可在 Chrome MV3 擴充套件的預設
 * CSP（`script-src 'self'; upgrade-insecure-requests;`）下運作。
 *
 * 與既有管線的關係 / Relationship to the normal pipeline
 * ---
 * 一般效果的 `WGFXRuntime.compile()` 依序做四件事：
 *   1. 解析 DSL → shaderInfo
 *   2. 程式碼生成 → 各 pass 的 WGSL
 *   3. 建立資源（紋理 / buffer / uniform）
 *   4. 建立管線
 *
 * `.wgx` 套件包把 **1 與 2 搬到建置期**，manifest 內攜帶精簡後的 shaderInfo 與
 * 預生成的 WGSL，執行期直接走 **3 與 4** —— 完全重用既有的 `ResourceManager` 與
 * `PipelineManager`，不另建一套執行環境。
 *
 * 尺寸彈性 / Resolution independence
 * ---
 * 打包時 `defines` 留空，因此產生的 WGSL 保留 `NW` / `NH` 等識別字，
 * 由本模組在載入時依實際輸入尺寸注入 → **同一套件包可處理任意解析度**。
 *
 * @group Core
 * @category Packages
 */

/** 套件包 manifest 中，單一 pass 的資源描述（對應 codegen 的 `module.resources`）。 */
export interface WgxPassResources {
    parameters?: any[];
    hasScene?: boolean;
    sceneBinding?: number;
    textures?: any[];
    samplers?: any[];
    storageBuffers?: any[];
}

/** 套件包 manifest（與 `tools/pack/manifest.mjs` 的 schema 對應）。 */
export interface WgxManifest {
    version: number;
    id: string;
    name: string;
    family?: string;
    description?: string;
    source?: string;
    tags?: string[];
    metadata?: Record<string, any>;
    parameters?: any[];
    /** 精簡後的 shaderInfo（不含 pass 的原始程式碼 —— 已被預生成的 WGSL 取代） */
    shaderInfo?: {
        textures?: any[];
        buffers?: any[];
        samplers?: any[];
        parameters?: any[];
        metadata?: Record<string, any>;
        passes?: any[];
    };
    passes: Array<{
        index: number;
        file: string;
        /** 預生成的 WGSL 對應的資源描述（直接餵給 PipelineManager） */
        resources: WgxPassResources;
        dispatch?: string[];
        workgroupSize?: number[];
        inputs?: string[];
        outputs?: string[];
    }>;
    resources?: Array<{
        kind: 'texture' | 'storageBuffer';
        name: string;
        file?: string;
        source?: string;
        size?: number;
        stride?: number;
        mode?: 'read' | 'read_write';
        runtimeSized?: boolean;
    }>;
    limits?: Record<string, number>;
    buildTool?: string;
    builtAt?: string;
}

/** 載入 `.wgx` 所需的選項。 */
export interface WGFXPackageOptions {
    /** WebGPU 裝置 */
    device: GPUDevice;
    /** `.wgx` 的位元組（zip 容器） */
    data: ArrayBuffer | Uint8Array;
    /** 輸入寬度 */
    width: number;
    /** 輸入高度 */
    height: number;
    /** 輸出寬度（省略時依效果宣告或等同 width） */
    outWidth?: number;
    /** 輸出高度 */
    outHeight?: number;
    /** 額外的外部資源（覆寫套件包內的同名資源） */
    externalResources?: any;
    /** 進度回呼 */
    onProgress?: (stage: string, detail?: string) => void;
}

/**
 * 最小 zip 讀取器（純 JS，瀏覽器與 Node 皆可）。
 *
 * 刻意不使用 Node 專用 API 或外部相依 —— 同一份程式碼必須能在 Chrome 擴充套件
 * 的 content script 中運作。
 *
 * 支援：local file header / central directory 解析、stored(0) 與 deflate-raw(8)、
 * ZIP64 之外的一般 zip（32 位元尺寸）。不支援：加密 zip、zip64、多磁碟 archive。
 */
async function inflateRaw(data: Uint8Array): Promise<Uint8Array> {
    // 瀏覽器內建（Chrome 103+）與 Node 18+ 皆有 DecompressionStream
    if (typeof DecompressionStream === 'function') {
        const ds = new DecompressionStream('deflate-raw');
        // 複製成 ArrayBuffer 專屬的 view —— BlobPart 不接受 SharedArrayBuffer 底層的 TypedArray。
        const copy = new Uint8Array(data.byteLength);
        copy.set(data);
        const stream = new Blob([copy.buffer]).stream().pipeThrough(ds);
        const buf = await new Response(stream).arrayBuffer();
        return new Uint8Array(buf);
    }
    throw new Error(
        '此環境不支援 DecompressionStream（需要 Chrome 103+ / Node 18+）；' +
        '請改用 WGFXPackage.fromBytes() 並預先解壓。',
    );
}

/**
 * 解析 zip 容器。
 * @param data zip bytes
 * @returns 檔名 → 解壓後內容
 */
export async function unzipPackage(data: ArrayBuffer | Uint8Array): Promise<Map<string, Uint8Array>> {
    const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

    // 從尾端找 End of Central Directory
    let eocd = -1;
    const scanStart = Math.max(0, bytes.length - 22 - 0xffff);
    for (let i = bytes.length - 22; i >= scanStart; i--) {
        if (view.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
    }
    if (eocd < 0) throw new Error('不是有效的 .wgx：找不到 zip 的 EOCD');

    const entryCount = view.getUint16(eocd + 10, true);
    let ptr = view.getUint32(eocd + 16, true);
    const decoder = new TextDecoder('utf-8');
    const files = new Map<string, Uint8Array>();

    for (let i = 0; i < entryCount; i++) {
        if (view.getUint32(ptr, true) !== 0x02014b50) {
            throw new Error(`zip central directory 損毀（entry ${i}）`);
        }
        const method = view.getUint16(ptr + 10, true);
        const compressedSize = view.getUint32(ptr + 20, true);
        const uncompressedSize = view.getUint32(ptr + 24, true);
        const nameLen = view.getUint16(ptr + 28, true);
        const extraLen = view.getUint16(ptr + 30, true);
        const commentLen = view.getUint16(ptr + 32, true);
        const localOffset = view.getUint32(ptr + 42, true);
        const name = decoder.decode(bytes.subarray(ptr + 46, ptr + 46 + nameLen));

        if (view.getUint32(localOffset, true) !== 0x04034b50) {
            throw new Error(`zip local header 損毀：${name}`);
        }
        const lNameLen = view.getUint16(localOffset + 26, true);
        const lExtraLen = view.getUint16(localOffset + 28, true);
        const dataStart = localOffset + 30 + lNameLen + lExtraLen;
        const raw = bytes.subarray(dataStart, dataStart + compressedSize);

        if (!name.endsWith('/')) {
            let content: Uint8Array;
            if (method === 0) content = raw;
            else if (method === 8) content = await inflateRaw(raw);
            else throw new Error(`不支援的 zip 壓縮方法 ${method}：${name}`);
            if (uncompressedSize > 0 && content.length !== uncompressedSize) {
                throw new Error(`zip entry 大小不符：${name}`);
            }
            files.set(name, content);
        }
        ptr += 46 + nameLen + extraLen + commentLen;
    }
    return files;
}

/** 讀取套件包內的檔案為文字。 */
function readText(files: Map<string, Uint8Array>, name: string): string {
    const data = files.get(name);
    if (!data) throw new Error(`套件包內找不到 ${name}`);
    return new TextDecoder('utf-8').decode(data);
}

/**
 * 已載入的 `.wgx` 套件包。
 *
 * 可直接以 {@link WGFXPackage.createInstance} 建立可執行的 WGFX 實例。
 */
export class WGFXPackage {
    /** 套件包 metadata */
    public readonly manifest: WgxManifest;
    /** 解壓後的檔案 */
    public readonly files: Map<string, Uint8Array>;

    private constructor(manifest: WgxManifest, files: Map<string, Uint8Array>) {
        this.manifest = manifest;
        this.files = files;
    }

    /**
     * 從位元組載入套件包。
     * @param data `.wgx` 的位元組
     */
    public static async fromBytes(data: ArrayBuffer | Uint8Array): Promise<WGFXPackage> {
        const files = await unzipPackage(data);
        const manifest = JSON.parse(readText(files, 'manifest.json')) as WgxManifest;
        if (!manifest || typeof manifest !== 'object') {
            throw new Error('套件包的 manifest.json 無效');
        }
        if (manifest.version !== 1) {
            throw new Error(`不支援的 .wgx manifest 版本: ${manifest.version}（期望 1）`);
        }
        return new WGFXPackage(manifest, files);
    }

    /**
     * 從 URL 載入套件包（瀏覽器擴充套件用 `chrome.runtime.getURL(...)`）。
     * @param url 套件包位置
     */
    public static async fromURL(url: string): Promise<WGFXPackage> {
        const resp = await fetch(url);
        if (!resp.ok) {
            throw new Error(`載入 .wgx 失敗: ${resp.status} ${resp.statusText}`);
        }
        return WGFXPackage.fromBytes(await resp.arrayBuffer());
    }

    /** 取得某個 pass 的預生成 WGSL。 */
    public getPassCode(passIndex: number): string {
        const pass = this.manifest.passes.find((p) => p.index === passIndex);
        if (!pass) throw new Error(`套件包內沒有 pass ${passIndex}`);
        return readText(this.files, pass.file);
    }

    /** 取得某個檔案的原始位元組（權重、外部紋理等）。 */
    public getFile(name: string): Uint8Array {
        const data = this.files.get(name);
        if (!data) throw new Error(`套件包內找不到 ${name}`);
        return data;
    }

    /**
     * 依套件包內容建立可執行的 WGFX 實例。
     *
     * **不做 DSL 解析、不做程式碼生成** —— 直接以套件包內預生成的 WGSL 建立管線。
     *
     * @param options 載入選項
     */
    public async createInstance(options: WGFXPackageOptions): Promise<WGFXPackageInstance> {
        const { device, width, height, outWidth, outHeight, onProgress } = options;

        // ---- 1. 外部紋理（套件包內的 DDS）----
        onProgress?.('preparing-textures');
        const outW = outWidth || width;
        const outH = outHeight || height;
        const textures: Record<string, any> = {
            INPUT: {
                size: [width, height],
                format: 'rgba8unorm' as GPUTextureFormat,
                usage: GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.RENDER_ATTACHMENT,
            },
            // OUTPUT 的尺寸：宿主未明確指定時，由效果自身的宣告決定（升頻效果
            // 宣告 `//! WIDTH INPUT_WIDTH * 2` 等）—— 與一般效果路徑的
            // honorDeclaredOutput 行為一致。此處先不帶 OUTPUT 外部資源，
            // 讓 ResourceManager 依宣告建立（下面的 honorDeclaredOutput 旗標
            // 會讓宣告優先）。
        };
        // 宿主明確指定輸出尺寸時，才提供外部 OUTPUT（其優先權最高）。
        if (outWidth !== undefined || outHeight !== undefined) {
            textures.OUTPUT = {
                size: [outW, outH],
                format: 'rgba16float' as GPUTextureFormat,
                usage: GPUTextureUsage.STORAGE_BINDING |
                    GPUTextureUsage.COPY_SRC |
                    GPUTextureUsage.TEXTURE_BINDING,
            };
        }
        for (const res of this.manifest.resources || []) {
            if (res.kind !== 'texture' || !res.file) continue;
            const tex = readDDS(this.getFile(res.file), res.file);
            textures[res.name] = {
                size: [tex.width, tex.height],
                format: tex.format,
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
                data: tex.data,
                bytesPerRow: tex.bytesPerRow,
            };
        }

        // ---- 2. 重建精簡的 shaderInfo（供 ResourceManager 使用）----
        onProgress?.('preparing-resources');
        const manifestInfo = this.manifest.shaderInfo || {};
        const shaderInfo: any = {
            textures: manifestInfo.textures || [],
            buffers: manifestInfo.buffers || [],
            samplers: manifestInfo.samplers || [],
            parameters: manifestInfo.parameters || [],
            metadata: manifestInfo.metadata || {},
            passes: (manifestInfo.passes || []).map((p: any) => ({
                index: p.index,
                in: p.in || [],
                out: p.out || [],
                blockSize: p.blockSize,
                numThreads: p.numThreads,
                dispatch: p.dispatch,
                desc: p.desc,
            })),
        };

        // ---- 3. 建立資源（重用既有的 ResourceManager）----
        const defines: Record<string, number> = {
            INPUT_WIDTH: width,
            INPUT_HEIGHT: height,
            OUTPUT_WIDTH: outW,
            OUTPUT_HEIGHT: outH,
        };
        const externalResources: any = {
            defines: { ...defines, ...(options.externalResources?.defines || {}) },
            textures: { ...textures, ...(options.externalResources?.textures || {}) },
            buffers: options.externalResources?.buffers,
        };
        const resourceManager = new ResourceManager(device);
        resourceManager.initialize(shaderInfo, externalResources, {
            // 宿主未明確指定輸出尺寸時，讓效果自身的宣告優先（升頻效果宣告 2x/3x/4x）
            honorDeclaredOutput: !(outWidth !== undefined || outHeight !== undefined),
        });
        // OUTPUT 實際尺寸（可能是宣告式放大後的尺寸）
        const actualOut = resourceManager.getTexture('OUTPUT');
        const finalOutW = actualOut ? actualOut.width : outW;
        const finalOutH = actualOut ? actualOut.height : outH;

        // ---- 4. 建立管線（重用既有的 PipelineManager，餵預生成的 WGSL）----
        onProgress?.('creating-pipelines');
        const generatedModules = this.manifest.passes.map((p) => ({
            wgslCode: this.getPassCode(p.index),
            passIndex: p.index,
            resources: p.resources || {},
        }));
        const pipelineManager = new PipelineManager(device, resourceManager);
        await pipelineManager.createPipelines(shaderInfo, generatedModules);

        const uniformBinder = new UniformBinder(device, resourceManager);
        resourceManager.updateSceneBuffer(width, height, finalOutW, finalOutH);

        onProgress?.('ready');
        return new WGFXPackageInstance({
            device,
            resourceManager,
            pipelineManager,
            uniformBinder,
            manifest: this.manifest,
            width,
            height,
            outWidth: finalOutW,
            outHeight: finalOutH,
        });
    }
}

/**
 * 由 `.wgx` 套件包建立的 WGFX 實例。
 *
 * API 與 {@link WGFX} 對齊（`process` / `updateUniforms` / `getOutputView` / `dispose`），
 * 因此消費端可把兩者互換使用。
 */
export class WGFXPackageInstance {
    public readonly device: GPUDevice;
    public readonly resourceManager: ResourceManager;
    public readonly pipelineManager: PipelineManager;
    public readonly uniformBinder: UniformBinder;
    public readonly manifest: WgxManifest;
    public readonly width: number;
    public readonly height: number;
    public readonly outWidth: number;
    public readonly outHeight: number;

    constructor(init: {
        device: GPUDevice;
        resourceManager: ResourceManager;
        pipelineManager: PipelineManager;
        uniformBinder: UniformBinder;
        manifest: WgxManifest;
        width: number;
        height: number;
        outWidth: number;
        outHeight: number;
    }) {
        this.device = init.device;
        this.resourceManager = init.resourceManager;
        this.pipelineManager = init.pipelineManager;
        this.uniformBinder = init.uniformBinder;
        this.manifest = init.manifest;
        this.width = init.width;
        this.height = init.height;
        this.outWidth = init.outWidth;
        this.outHeight = init.outHeight;
    }

    /** 效果的可調參數（沿用 WGFX 的 WGFXInfo 形狀）。 */
    public initialize(): {
        width: number;
        height: number;
        uniforms: any[];
        passes: number;
    } {
        return {
            width: this.width,
            height: this.height,
            uniforms: (this.manifest.parameters || []).map((p) => ({
                name: p.name,
                type: p.type || 'float',
                default: p.default ?? 0.0,
                min: p.min ?? 0.0,
                max: p.max ?? 1.0,
                step: p.step ?? 0.01,
                label: p.label,
            })),
            passes: this.manifest.passes.length,
        };
    }

    /** 更新 uniform 值。 */
    public updateUniforms(uniforms: Record<string, number | number[]>): void {
        for (const [name, value] of Object.entries(uniforms || {})) {
            try {
                this.uniformBinder.updateUniform(name, value);
            } catch (error: any) {
                Logger.warn(`Failed to update uniform "${name}": ${error.message}`);
            }
        }
    }

    /**
     * 處理一幀：上傳輸入 → dispatch 全部 pass → 回傳 OUTPUT 紋理。
     */
    public async process(
        inputSource: ImageBitmap | VideoFrame | HTMLVideoElement | HTMLCanvasElement,
        options: { outWidth?: number; outHeight?: number } = {},
    ): Promise<GPUTexture> {
        this.resourceManager.advanceFrame();
        this.resourceManager.updateTextureFromImage('INPUT', inputSource, options.outWidth, options.outHeight);

        const encoder = this.device.createCommandEncoder({ label: 'WGFX package frame' });
        for (const pass of this.manifest.passes) {
            // pass metadata 已在 manifest 的 shaderInfo.passes 中（index/in/out/dispatch…），
            // 這裡組出 PipelineManager.dispatchPass 所需的最小形狀。
            const meta = (this.manifest.shaderInfo?.passes || [])
                .find((p: any) => p.index === pass.index) || {};
            this.pipelineManager.dispatchPass({
                index: pass.index,
                in: pass.inputs || meta.in || [],
                out: pass.outputs || meta.out || [],
                blockSize: meta.blockSize,
                numThreads: meta.numThreads,
                dispatch: pass.dispatch || meta.dispatch,
            }, encoder);
        }
        this.device.queue.submit([encoder.finish()]);

        const output = this.resourceManager.getTexture('OUTPUT');
        if (!output) throw new Error("Output texture missing");
        return output;
    }

    /** 取得輸出紋理的 view。 */
    public getOutputView(): GPUTextureView {
        return this.resourceManager.getTexture('OUTPUT')!.createView();
    }

    /** 釋放所有資源。 */
    public dispose(): void {
        this.resourceManager.dispose();
        this.pipelineManager.dispose();
    }
}
