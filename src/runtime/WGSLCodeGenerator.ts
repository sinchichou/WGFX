import { WGFXShaderInfo, PassInfo, TextureInfo, ParameterInfo, SamplerInfo } from '../types';

/**
 * Resource binding index configuration strategy.
 * ---
 * 資源綁定索引配置策略。
 * 定義標準資源的固定綁定位置，確保著色器(Shader)與主機端(Host)之間的一致性。
 *
 * Binding index layout / 綁定索引配置：
 * - **保留**: $Index_{Reserved} \in \{0, 2, 3\}$
 * - **通用參數緩衝**: $Index_{Uniforms} = 1$
 * - **場景資訊**: $Index_{Scene} = 4$
 * - **採樣器範圍**: $Index_{Sampler} \in [5, 9]$
 * - **紋理資源範圍**: $Index_{Texture} \in [10, 29]$
 *
 * WGSLCodeGenerator 是這些綁定的唯一擁有者（Single Source of Truth）；
 * PipelineManager 依本模組輸出的 resources 建立對應的 BindGroupLayout。
 */
const BINDING_LAYOUT = {
    UNIFORMS: 1,        // Uniform buffer (adjustable parameters) / 可調整參數緩衝區
    SCENE_INFO: 4,      // Scene information (Resolution, etc.) / 場景資訊 (解析度等)
    SAMPLERS_START: 5,  // Sampler start index / 採樣器起始索引
    SAMPLERS_END: 9,    // Sampler end index (inclusive) / 採樣器結束索引
    TEXTURES_START: 10, // Texture start index / 紋理起始索引
} as const;

/**
 * Strip the sampled/storaged alias suffix to get the logical texture name.
 * 移除 sampled/storaged 別名後綴，取得邏輯紋理名稱。
 *
 * Magpie 風格的 `tex1_sampled` / `tex1_storaged` 成對宣告指的是**同一個**邏輯紋理：
 * 主機端只配置一塊 GPUTexture，並以 sampled/storage 兩種 view 分別綁定。
 */
export function logicalTextureName(name: string): string {
    return name.replace(/_(sampled|storaged)$/, '');
}

/** Scalar size (bytes) and WGSL type for a parameter type. */
const PARAM_TYPE_MAP: Record<string, { wgsl: string; size: number }> = {
    'float': { wgsl: 'f32', size: 4 },
    'f32':   { wgsl: 'f32', size: 4 },
    'int':   { wgsl: 'i32', size: 4 },
    'i32':   { wgsl: 'i32', size: 4 },
    'uint':  { wgsl: 'u32', size: 4 },
    'u32':   { wgsl: 'u32', size: 4 },
    'vec2':  { wgsl: 'vec2<f32>', size: 8 },
    'float2':{ wgsl: 'vec2<f32>', size: 8 },
    'int2':  { wgsl: 'vec2<i32>', size: 8 },
    'uint2': { wgsl: 'vec2<u32>', size: 8 },
    'vec3':  { wgsl: 'vec3<f32>', size: 12 },
    'float3':{ wgsl: 'vec3<f32>', size: 12 },
    'vec4':  { wgsl: 'vec4<f32>', size: 16 },
    'float4':{ wgsl: 'vec4<f32>', size: 16 },
    'int4':  { wgsl: 'vec4<i32>', size: 16 },
    'uint4': { wgsl: 'vec4<u32>', size: 16 },
};

/** Get the normalized {wgsl,size} for a parameter type (default: scalar f32). */
export function paramType(type: string | undefined): { wgsl: string; size: number } {
    return PARAM_TYPE_MAP[(type || 'float').toLowerCase()] || PARAM_TYPE_MAP['float'];
}

/**
 * Generator for WGSL shader code from parsed shader information.
 * ---
 * 從解析後的著色器資訊產生 WGSL 著色器代碼。
 * 負責注入標準資源並封裝計算著色器進入點。
 *
 * @group Core
 * @category Generators
 */
export class WGSLCodeGenerator {
    constructor() {
    }

    /**
     * Generate complete WGSL modules for each pass in the shader info.
     * ---
     * 為每個渲染通道產生完整的 WGSL 模組。
     * 自動處理資源綁定索引並注入必要的全域結構體。
     *
     * 每個 pass module 會宣告**全部**已宣告的紋理與取樣器（不只 in/out 用到的），
     * 因為 COMMON 區塊內的輔助函式可能引用任何資源。
     *
     * @group Generation
     * @param shaderInfo - The parsed shader metadata / 解析後的著色器元數據
     * @returns Array of generated modules / 產生的模組列表
     */
    public generate(shaderInfo: WGFXShaderInfo): { wgslCode: string; passIndex: number; resources: any }[] {
        const generatedModules: { wgslCode: string; passIndex: number; resources: any }[] = [];

        // Prepare common code sections / 準備共通代碼區段
        // COMMON 函式主體會依每個 pass 的實際呼叫進行**函式級裁剪**：
        // 未被呼叫的輔助函式不進入模組，避免其引用的資源在不需要的 pass
        // 中被迫宣告（導致 storage/sampled 同步範圍衝突或超出裝置上限）。
        const commonHeader = shaderInfo.commonCode
            ? `// --- COMMON CODE ---\n${this._stripCommonFunctions(shaderInfo.commonCode)}\n\n`
            : '';

        // Bindings already claimed by hand-written COMMON code (e.g. `@binding(4)`).
        // COMMON 手寫宣告已佔用的綁定槽位，產生器必須避開。
        const reservedBindings = new Set<number>();
        for (const m of shaderInfo.commonCode.matchAll(/@binding\((\d+)\)/g)) {
            reservedBindings.add(parseInt(m[1], 10));
        }

        // Textures/samplers referenced by COMMON code (helper functions may sample them
        // in every pass). Per-pass declarations add these on top of pass.in/out.
        // COMMON 代碼引用到的資源（輔助函式可能在每個 pass 取樣），需在每個 pass 宣告。
        const fnBodies = this._extractCommonFunctions(shaderInfo.commonCode);
        const texNameSet = new Set<string>(shaderInfo.textures.map((t: TextureInfo) => t.name));
        const samplerNameSet = new Set<string>(shaderInfo.samplers.map((s: SamplerInfo) => s.name));

        // Generate SceneInfo struct + uniform var / 產生場景資訊結構體與 uniform 變數
        const sceneInfoCode = this._generateSceneInfoStruct();

        // Generate Uniform buffer struct / 產生 Uniform 緩衝區結構體
        const uniformBufferCode = this._generateUniformStruct(shaderInfo.parameters);

        // Process each render pass / 處理每個渲染通道
        shaderInfo.passes.forEach(pass => {
            let wgsl = `// ========================================\n`;
            wgsl += `// Pass ${pass.index}\n`;
            wgsl += `// ========================================\n\n`;

            // Prepare resource tracking object / 準備資源追蹤物件
            // COMMON 自行宣告 scene 時（如舊版 Magpie 移植），主機端必須把 sceneBuffer
            // 綁定到 COMMON 選擇的槽位，而非預設的 SCENE_INFO。
            const commonSceneMatch = shaderInfo.commonCode.match(/@binding\((\d+)\)\s*var<uniform>\s+scene/);
            const sceneBinding = commonSceneMatch ? parseInt(commonSceneMatch[1], 10) : BINDING_LAYOUT.SCENE_INFO;

            const passResources = {
                textures: [] as any[],
                samplers: [] as any[],
                parameters: shaderInfo.parameters,
                hasScene: true,
                sceneBinding: sceneBinding
            };

            // Resolve which resources this pass needs by scanning the pass code and,
            // transitively, the COMMON helper functions it calls (a pass calling a
            // COMMON helper that samples a texture needs that texture declared).
            // 以 pass 代碼為起點，沿 COMMON 輔助函式呼叫圖遞迴解析資源引用。
            const referenced = this._collectReferencedResources(pass.code, fnBodies, texNameSet, samplerNameSet);

            // Emit COMMON header + only the helper functions this pass calls.
            // 輸出 COMMON 表頭 + 此 pass 實際呼叫的輔助函式（函式級裁剪）。
            wgsl += commonHeader;
            const calledFns = this._collectCalledFunctions(pass.code, fnBodies);
            for (const fnName of calledFns) {
                wgsl += fnBodies.get(fnName) + '\n';
            }
            wgsl += '\n';

            // Inject SceneInfo (Resolution, scaling etc.) unless COMMON declares its own.
            // 若 COMMON 已自行宣告 scene，則跳過注入（保持與舊版 Magpie 移植相容）。
            if (!shaderInfo.commonCode.includes('var<uniform> scene')) {
                wgsl += sceneInfoCode;
            }

            // Inject Uniforms (Parameters)
            if (uniformBufferCode && !shaderInfo.commonCode.includes('var<uniform> uniforms')) {
                wgsl += uniformBufferCode;
            }

            // Setup sampler bindings / 設定採樣器綁定
            // 僅為此 pass 實際引用（含 COMMON 間接引用）的 sampler 配置綁定。
            let samplerBinding = BINDING_LAYOUT.SAMPLERS_START;
            shaderInfo.samplers.forEach((sampler: SamplerInfo) => {
                // Skip if already hand-declared in COMMON / COMMON 已宣告者跳過
                if (new RegExp(`var\\s+${sampler.name}\\s*:`).test(shaderInfo.commonCode)) return;
                if (!referenced.has(sampler.name)) return;
                while (reservedBindings.has(samplerBinding) && samplerBinding <= BINDING_LAYOUT.SAMPLERS_END) {
                    samplerBinding++;
                }
                const currentBinding = samplerBinding++;
                wgsl += `@group(0) @binding(${currentBinding}) var ${sampler.name}: sampler;\n`;

                passResources.samplers.push({
                    ...sampler,
                    binding: currentBinding,
                    group: 0
                });
            });
            wgsl += '\n';

            // Setup texture bindings / 設定紋理綁定
            // 每個宣告（含 sampled/storaged 成對別名）都取得獨立綁定槽，
            // 但主機端將成對別名綁定到**同一個**邏輯紋理的對應 view。
            let textureBinding = BINDING_LAYOUT.TEXTURES_START;
            shaderInfo.textures.forEach((tex: TextureInfo) => {
                const logical = logicalTextureName(tex.name);
                const needed = referenced.has(tex.name) || referenced.has(logical);
                if (!needed) return;
                // Role comes from the declared name (Magpie convention: `X_storaged`
                // is the write alias, `X_sampled` is the read alias). Storage access
                // is granted only when the pass WRITES the texture (pass.out) — the
                // IN list is dependency bookkeeping only. Untouched twins are declared
                // as sampled so COMMON helper references always resolve.
                // 角色由宣告名稱決定：僅有出現在 pass.out（寫入目標）的 _storaged 紋理
                // 取得 storage 存取；IN 列表僅供依賴排序，其餘一律宣告為 sampled。
                let isStorage: boolean;
                if (tex.name.endsWith('_storaged')) {
                    isStorage = pass.out.includes(tex.name);
                } else if (tex.name.endsWith('_sampled')) {
                    isStorage = false;
                } else {
                    isStorage = pass.out.includes(tex.name) || pass.out.includes(logical) || tex.name === 'OUTPUT';
                }
                const format = (tex.format as string) || (logical === 'INPUT' ? 'rgba8unorm' : 'rgba16float');
                const textureType = isStorage
                    ? `texture_storage_2d<${format}, write>`
                    : `texture_2d<f32>`;

                while (reservedBindings.has(textureBinding)) {
                    textureBinding++;
                }
                const currentBinding = textureBinding++;
                wgsl += `@group(0) @binding(${currentBinding}) var ${tex.name}: ${textureType};\n`;

                passResources.textures.push({
                    ...tex,
                    logicalName: logical,
                    format: format,
                    binding: currentBinding,
                    group: 0,
                    isStorage: isStorage
                });
            });
            wgsl += '\n';

            // Transform pass logic to compute shader entry point / 轉換通道邏輯為計算著色器進入點
            let passCode = this._transformPassToComputeShader(pass);
            wgsl += passCode;

            generatedModules.push({
                wgslCode: wgsl,
                passIndex: pass.index,
                resources: passResources
            });
        });

        return generatedModules;
    }

    /**
     * Extract function bodies from COMMON code (line-based, brace-counting).
     * ---
     * 以行為單位（括號計數）從 COMMON 代碼中擷取函式主體，供引用解析使用。
     * 回傳函式主體映射與去除函式後的表頭代碼（struct/alias/const/var 等）。
     */
    private _extractCommonFunctions(commonCode: string): Map<string, string> {
        const fnBodies = new Map<string, string>();
        let curFn: string | null = null;
        let depth = 0;
        let started = false;
        let buf: string[] = [];
        for (const line of commonCode.split("\n")) {
            if (curFn === null) {
                const m = line.match(/\bfn\s+(\w+)\s*\(/);
                if (m) {
                    curFn = m[1];
                    depth = 0;
                    started = false;
                    buf = []; // Start a fresh body buffer / 開始新的函式主體緩衝
                } else {
                    continue;
                }
            }
            // Count braces on EVERY line including the fn signature line
            // (the opening `{` usually sits at the end of `fn X() ... {`).
            // 函式簽名行本身也要計算括號（`{` 通常在行尾）。
            for (const ch of line) {
                if (ch === "{") { depth++; started = true; }
                else if (ch === "}") { depth--; }
            }
            buf.push(line);
            if (started && depth <= 0) {
                fnBodies.set(curFn, buf.join("\n"));
                curFn = null;
            }
        }
        return fnBodies;
    }

    /**
     * Strip COMMON function bodies, keeping only the shared header
     * (structs, aliases, constants, uniform declarations).
     * ---
     * 移除 COMMON 中的函式主體，僅保留共用表頭（struct/alias/const/var 宣告）。
     */
    private _stripCommonFunctions(commonCode: string): string {
        if (!commonCode) return '';
        const kept: string[] = [];
        let curFn: string | null = null;
        let depth = 0;
        let started = false;
        for (const line of commonCode.split("\n")) {
            if (curFn === null) {
                const m = line.match(/\bfn\s+(\w+)\s*\(/);
                if (m) { curFn = m[1]; depth = 0; started = false; }
                else { kept.push(line); continue; }
            }
            // Count braces on the signature line too / 簽名行同樣計算括號
            for (const ch of line) {
                if (ch === "{") { depth++; started = true; }
                else if (ch === "}") { depth--; }
            }
            if (started && depth <= 0) {
                curFn = null; // end of function body / 函式主體結束
            }
        }
        return kept.join("\n");
    }

    /**
     * Collect COMMON helper functions reachable from a pass's code (transitive call closure).
     * ---
     * 收集 pass 代碼可達的 COMMON 輔助函式（遞迴呼叫閉包）。
     */
    private _collectCalledFunctions(passCode: string, fnBodies: Map<string, string>): string[] {
        const seen = new Set<string>();
        const called: string[] = [];
        const queue: string[] = passCode.match(/[A-Za-z_][A-Za-z0-9_]*/g) || [];
        while (queue.length > 0) {
            const w = queue.shift()!;
            if (seen.has(w)) continue;
            seen.add(w);
            const body = fnBodies.get(w);
            if (body !== undefined) {
                called.push(w);
                for (const w2 of body.match(/[A-Za-z_][A-Za-z0-9_]*/g) || []) {
                    if (!seen.has(w2)) queue.push(w2);
                }
            }
        }
        return called;
    }

    /**
     * Collect resource names referenced by a pass, transitively through COMMON helpers.
     * ---
     * 收集 pass 實際引用的資源名稱：以 pass 代碼中的識別字為起點，
     * 沿 COMMON 函式呼叫圖遞迴展開，比對紋理與取樣器名稱。
     */
    private _collectReferencedResources(
        passCode: string,
        fnBodies: Map<string, string>,
        texNames: Set<string>,
        samplerNames: Set<string>
    ): Set<string> {
        const referenced = new Set<string>();
        const seen = new Set<string>();
        const queue: string[] = passCode.match(/[A-Za-z_][A-Za-z0-9_]*/g) || [];
        while (queue.length > 0) {
            const w = queue.shift()!;
            if (seen.has(w)) continue;
            seen.add(w);
            if (texNames.has(w) || samplerNames.has(w)) {
                referenced.add(w);
            }
            const body = fnBodies.get(w);
            if (body) {
                for (const w2 of body.match(/[A-Za-z_][A-Za-z0-9_]*/g) || []) {
                    if (!seen.has(w2)) queue.push(w2);
                }
            }
        }
        return referenced;
    }

    /**
     * Generate SceneInfo struct definition with its uniform binding.
     * ---
     * 產生場景資訊結構體定義與對應的 uniform 綁定宣告。
     * 記憶體佈局與 ResourceManager.updateSceneBuffer 的寫入格式（40 bytes）一致：
     * inputSize(u32x2) @0, inputPt(f32x2) @8, outputSize(u32x2) @16, outputPt(f32x2) @24, scale(f32x2) @32。
     */
    private _generateSceneInfoStruct(): string {
        return `// Scene Info Struct / 場景資訊結構體
struct SceneInfo {
    inputSize: vec2<u32>,
    inputPt: vec2<f32>,
    outputSize: vec2<u32>,
    outputPt: vec2<f32>,
    scale: vec2<f32>,
};
@group(0) @binding(${BINDING_LAYOUT.SCENE_INFO}) var<uniform> scene: SceneInfo;
`;
    }

    /**
     * Generate Uniform buffer struct definition.
     * ---
     * 產生 Uniform 緩衝區結構體定義。
     * 每個參數佔用一個 16-byte slot（與 ResourceManager.createUniformBuffer 的
     * offset += 16 逐參數配置完全一致），不足部分以 pad 欄位填補。
     */
    private _generateUniformStruct(parameters: ParameterInfo[]): string {
        if (parameters.length === 0) return '';

        let code = `// Uniform Parameters (16-byte slot per parameter) / Uniform 參數（每參數 16-byte）\nstruct Uniforms {\n`;

        parameters.forEach(param => {
            const t = paramType(param.type);
            code += `    ${param.name}: ${t.wgsl},\n`;

            // Pad the slot to 16 bytes with f32 fields / 以 f32 欄位將 slot 補滿 16 bytes
            // f32 pad 永遠對齊安全（scalar align 4，且位於 16-byte slot 內）。
            let remaining = 16 - t.size;
            let padIndex = 0;
            while (remaining > 0) {
                code += `    ${param.name}_pad${padIndex++}: f32,\n`;
                remaining -= 4;
            }
        });

        code += `};\n`;
        code += `@group(0) @binding(${BINDING_LAYOUT.UNIFORMS}) var<uniform> uniforms: Uniforms;\n\n`;

        return code;
    }

    /**
     * Transform Pass function into a compute shader entry point.
     * ---
     * 將 Pass 函式轉換為計算著色器進入點。
     * 同時支援 `fn Pass<N>(...)` 與 `fn main(...)` 兩種命名，統一改名為 main_cs 並套用 @compute 屬性。
     */
    private _transformPassToComputeShader(pass: PassInfo): string {
        // Remove meta-command comments / 移除元指令註解
        let code = pass.code.replace(/\/\/!.*\n/g, '');

        // Rename PassX / main function to main_cs / 將 PassX 或 main 函式重新命名為 main_cs
        const entryRegex = new RegExp(`fn\\s+(?:Pass${pass.index}|main)\\s*\\(`);
        if (!entryRegex.test(code)) {
            // eslint-disable-next-line no-console
            console.warn(`[WGFX] Pass ${pass.index}: no "fn Pass${pass.index}(" or "fn main(" found; entry point rename skipped.`);
        }
        code = code.replace(entryRegex, `fn main_cs(`);

        // Apply @compute attribute and workgroup size / 套用 @compute 屬性與工作群組大小
        const numThreads = pass.numThreads || [1, 1, 1];
        code = code.replace(/@compute\s*@workgroup_size\([^)]+\)\s*/g, '');
        code = `@compute @workgroup_size(${numThreads[0]}, ${numThreads[1]}, ${numThreads[2]})\n${code}`;

        return code;
    }
}
