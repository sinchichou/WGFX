import {
    WGFXShaderInfo,
    PassInfo,
    TextureInfo,
    ParameterInfo,
    SamplerInfo,
} from "../types";

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
    UNIFORMS: 1, // Uniform buffer (adjustable parameters) / 可調整參數緩衝區
    SCENE_INFO: 4, // Scene information (Resolution, etc.) / 場景資訊 (解析度等)
    SAMPLERS_START: 5, // Sampler start index / 採樣器起始索引
    SAMPLERS_END: 9, // Sampler end index (inclusive) / 採樣器結束索引
    TEXTURES_START: 10, // Texture start index / 紋理起始索引
    BUFFERS_START: 40, // Storage buffer start index / storage buffer 起始索引（CNN 權重與特徵圖）
} as const;

/**
 * Strip the sampled/storaged alias suffix to get the logical texture name.
 * 移除 sampled/storaged 別名後綴，取得邏輯紋理名稱。
 *
 * Magpie 風格的 `tex1_sampled` / `tex1_storaged` 成對宣告指的是**同一個**邏輯紋理：
 * 主機端只配置一塊 GPUTexture，並以 sampled/storage 兩種 view 分別綁定。
 */
export function logicalTextureName(name: string): string {
    return name.replace(/_(sampled|storaged)$/, "");
}

/**
 * WebGPU core(本 Dawn/node-webgpu 建置)支援 write-only storage 存取的格式集合。
 * ---
 * Set of formats that support write-only storage texture access on this
 * Dawn/node-webgpu build. 實測矩陣見 tools/_storage_probe.mjs(2026-11 實測):
 *
 * 可 storage:rgba8unorm, rgba8snorm, rgba8uint, rgba8sint, rgba16float,
 *            rgba16uint, rgba16sint, r32float, rg32float, rgba32float,
 *            r32uint, rg32uint, rgba32uint, r32sint, rg32sint, rgba32sint
 * 不可 storage(常見):r8*、rg8*、r16float、rg16float、r16uint、rg16uint、
 *            r16sint、rg16sint、*16unorm(需 TextureFormatsTier1 feature)、
 *            *-srgb、bgra8unorm、rg11b10ufloat、rgb9e5ufloat、rgb10a2unorm
 */
const STORAGE_CAPABLE_FORMATS = new Set<string>([
    "rgba8unorm", "rgba8snorm", "rgba8uint", "rgba8sint",
    "rgba16float", "rgba16uint", "rgba16sint",
    "r32float", "rg32float", "rgba32float",
    "r32uint", "rg32uint", "rgba32uint",
    "r32sint", "rg32sint", "rgba32sint",
]);

/**
 * 不支援 storage 存取的格式 → 最接近的可 storage 替代格式。
 * 單/雙通道升為四通道(取樣端讀 .x/.xy 語意不變;textureStore 一律寫 vec4),
 * srgb → 對應 unorm,窄色域/特殊 packing → rgba16float。
 */
const STORAGE_FORMAT_SUBSTITUTE: Record<string, string> = {
    "r8unorm": "rgba8unorm",
    "r8snorm": "rgba8snorm",
    "r8uint": "rgba8uint",
    "r8sint": "rgba8sint",
    "rg8unorm": "rgba8unorm",
    "rg8snorm": "rgba8snorm",
    "rg8uint": "rgba8uint",
    "rg8sint": "rgba8sint",
    "rgba8unorm-srgb": "rgba8unorm",
    "bgra8unorm": "rgba8unorm",
    "bgra8unorm-srgb": "rgba8unorm",
    "r16float": "rgba16float",
    "rg16float": "rgba16float",
    "r16uint": "rgba16uint",
    "rg16uint": "rgba16uint",
    "r16sint": "rgba16sint",
    "rg16sint": "rgba16sint",
    "r16unorm": "rgba16float",
    "rg16unorm": "rgba16float",
    "rgba16unorm": "rgba16float",
    "rg11b10ufloat": "rgba16float",
    "rgb9e5ufloat": "rgba16float",
    "rgb10a2unorm": "rgba16float",
};

/**
 * Map a declared texture format to one that supports storage texture access.
 * ---
 * 將宣告的紋理格式映射為支援 storage 存取的格式(已是 storage-capable 時原樣返回)。
 * WGFX 的中間紋理一律以 STORAGE_BINDING 建立並以 `texture_storage_2d<fmt, write>`
 * 綁定;Dawn 拒絕如 r16float 的 storage 存取,故宣告端與建立端必須共用本映射,
 * 確保 WGSL 宣告、BindGroupLayout 與 GPUTexture 三者格式一致。
 */
export function storageCapableFormat(format: string): string {
    const f = (format || "").toLowerCase();
    if (STORAGE_CAPABLE_FORMATS.has(f)) return f;
    return STORAGE_FORMAT_SUBSTITUTE[f] || f;
}

/** Scalar size (bytes) and WGSL type for a parameter type. */
const PARAM_TYPE_MAP: Record<string, { wgsl: string; size: number }> = {
    float: { wgsl: "f32", size: 4 },
    f32: { wgsl: "f32", size: 4 },
    int: { wgsl: "i32", size: 4 },
    i32: { wgsl: "i32", size: 4 },
    uint: { wgsl: "u32", size: 4 },
    u32: { wgsl: "u32", size: 4 },
    vec2: { wgsl: "vec2<f32>", size: 8 },
    float2: { wgsl: "vec2<f32>", size: 8 },
    int2: { wgsl: "vec2<i32>", size: 8 },
    uint2: { wgsl: "vec2<u32>", size: 8 },
    vec3: { wgsl: "vec3<f32>", size: 12 },
    float3: { wgsl: "vec3<f32>", size: 12 },
    vec4: { wgsl: "vec4<f32>", size: 16 },
    float4: { wgsl: "vec4<f32>", size: 16 },
    int4: { wgsl: "vec4<i32>", size: 16 },
    uint4: { wgsl: "vec4<u32>", size: 16 },
};

/** Get the normalized {wgsl,size} for a parameter type (default: scalar f32). */
export function paramType(type: string | undefined): {
    wgsl: string;
    size: number;
} {
    return (
        PARAM_TYPE_MAP[(type || "float").toLowerCase()] || PARAM_TYPE_MAP["float"]
    );
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
    constructor() { }

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
     * @param defines - External macro definitions emitted as WGSL `const` /
     *   外部巨集定義，將以 WGSL `const` 形式注入每個 pass 模組開頭
     * @returns Array of generated modules / 產生的模組列表
     */
    public generate(
        shaderInfo: WGFXShaderInfo,
        defines: Record<string, number> = {},
    ): { wgslCode: string; passIndex: number; resources: any }[] {
        const generatedModules: {
            wgslCode: string;
            passIndex: number;
            resources: any;
        }[] = [];

        // Prepare common code sections / 準備共通代碼區段
        // COMMON 函式主體會依每個 pass 的實際呼叫進行**函式級裁剪**：
        // 未被呼叫的輔助函式不進入模組，避免其引用的資源在不需要的 pass
        // 中被迫宣告（導致 storage/sampled 同步範圍衝突或超出裝置上限）。
        // 單次行走同時產出：表頭（struct/alias/const/var）+ 函式主體映射。
        const commonSplit = this._splitCommonCode(shaderInfo.commonCode);
        const commonHeader = shaderInfo.commonCode
            ? `// --- COMMON CODE ---\n${commonSplit.header}\n\n`
            : "";

        // Bindings already claimed by hand-written COMMON code (e.g. `@binding(4)`).
        // COMMON 手寫宣告已佔用的綁定槽位，產生器必須避開。
        const reservedBindings = new Set<number>();
        for (const m of shaderInfo.commonCode.matchAll(/@binding\((\d+)\)/g)) {
            reservedBindings.add(parseInt(m[1], 10));
        }

        // Textures/samplers referenced by COMMON code (helper functions may sample them
        // in every pass). Per-pass declarations add these on top of pass.in/out.
        // COMMON 代碼引用到的資源（輔助函式可能在每個 pass 取樣），需在每個 pass 宣告。
        const fnBodies = commonSplit.fnBodies;
        const texNameSet = new Set<string>(
            shaderInfo.textures.map((t: TextureInfo) => t.name),
        );
        const samplerNameSet = new Set<string>(
            shaderInfo.samplers.map((s: SamplerInfo) => s.name),
        );
        // Storage buffer names are resource identifiers too（`//! BUFFER x`）。
        // 早期版本僅以 texture/sampler 名稱過濾 closure，導致 buffer 名稱永遠不進入
        // referenced 集合、storage buffer 從未被宣告（OpenNR spike 的 x/w/b/o 全部消失）。
        // Storage buffer 名稱同樣是資源識別字，必須一併納入過濾範圍。
        const bufferNameSet = new Set<string>(
            (shaderInfo.buffers || []).map((b: any) => b.name),
        );

        // Generate SceneInfo struct + uniform var / 產生場景資訊結構體與 uniform 變數
        const sceneInfoCode = this._generateSceneInfoStruct();

        // Generate external defines as WGSL constants / 將外部 defines 產生為 WGSL 常數
        // 每個 pass 模組開頭注入（COMMON 之後），讓 NW/NH/CIN/COUT 之類的識別字可解析。
        const definesCode = this._generateDefines(defines, shaderInfo.commonCode);

        // Generate Uniform buffer struct / 產生 Uniform 緩衝區結構體
        const uniformBufferCode = this._generateUniformStruct(
            shaderInfo.parameters,
        );

        // Process each render pass / 處理每個渲染通道
        shaderInfo.passes.forEach((pass) => {
            let wgsl = `// ========================================\n`;
            wgsl += `// Pass ${pass.index}\n`;
            wgsl += `// ========================================\n\n`;

            // Prepare resource tracking object / 準備資源追蹤物件
            // COMMON 自行宣告 scene 時（如舊版 Magpie 移植），主機端必須把 sceneBuffer
            // 綁定到 COMMON 選擇的槽位，而非預設的 SCENE_INFO。
            const commonSceneMatch = shaderInfo.commonCode.match(
                /@binding\((\d+)\)\s*var<uniform>\s+scene/,
            );
            const sceneBinding = commonSceneMatch
                ? parseInt(commonSceneMatch[1], 10)
                : BINDING_LAYOUT.SCENE_INFO;

            const passResources = {
                textures: [] as any[],
                samplers: [] as any[],
                storageBuffers: [] as any[],
                parameters: shaderInfo.parameters,
                hasScene: true,
                sceneBinding: sceneBinding,
            };

            // Resolve which resources this pass needs by scanning the pass code and,
            // transitively, the COMMON helper functions it calls (a pass calling a
            // COMMON helper that samples a texture needs that texture declared).
            // 以 pass 代碼為起點，沿 COMMON 輔助函式呼叫圖遞迴解析資源引用。
            const closure = this._resolveCommonClosure(pass.code, fnBodies);
            const referenced = new Set<string>();
            for (const w of closure.identifiers) {
                if (
                    texNameSet.has(w) ||
                    samplerNameSet.has(w) ||
                    bufferNameSet.has(w)
                ) {
                    referenced.add(w);
                }
            }

            // Emit COMMON header + only the helper functions this pass calls.
            // 輸出 COMMON 表頭 + 此 pass 實際呼叫的輔助函式（函式級裁剪）。
            wgsl += commonHeader;
            for (const fnName of closure.fns) {
                wgsl += fnBodies.get(fnName) + "\n";
            }
            wgsl += "\n";

            // Inject external defines as constants / 注入外部 defines 常數宣告
            wgsl += definesCode;

            // Inject SceneInfo (Resolution, scaling etc.) unless COMMON declares its own.
            // 若 COMMON 已自行宣告 scene，則跳過注入（保持與舊版 Magpie 移植相容）。
            if (!shaderInfo.commonCode.includes("var<uniform> scene")) {
                wgsl += sceneInfoCode;
            }

            // Inject Uniforms (Parameters)
            if (
                uniformBufferCode &&
                !shaderInfo.commonCode.includes("var<uniform> uniforms")
            ) {
                wgsl += uniformBufferCode;
            }

            // Setup sampler bindings / 設定採樣器綁定
            // 僅為此 pass 實際引用（含 COMMON 間接引用）的 sampler 配置綁定。
            let samplerBinding = BINDING_LAYOUT.SAMPLERS_START;
            let bufferBinding = BINDING_LAYOUT.BUFFERS_START;
            shaderInfo.samplers.forEach((sampler: SamplerInfo) => {
                // Skip if already hand-declared in COMMON / COMMON 已宣告者跳過
                if (
                    new RegExp(`var\\s+${sampler.name}\\s*:`).test(shaderInfo.commonCode)
                )
                    return;
                if (!referenced.has(sampler.name)) return;
                while (
                    reservedBindings.has(samplerBinding) &&
                    samplerBinding <= BINDING_LAYOUT.SAMPLERS_END
                ) {
                    samplerBinding++;
                }
                const currentBinding = samplerBinding++;
                wgsl += `@group(0) @binding(${currentBinding}) var ${sampler.name}: sampler;\n`;

                passResources.samplers.push({
                    ...sampler,
                    binding: currentBinding,
                    group: 0,
                });
            });
            wgsl += "\n";

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
                if (tex.name.endsWith("_storaged")) {
                    isStorage = pass.out.includes(tex.name);
                } else if (tex.name.endsWith("_sampled")) {
                    isStorage = false;
                } else {
                    isStorage =
                        pass.out.includes(tex.name) ||
                        pass.out.includes(logical) ||
                        tex.name === "OUTPUT";
                }
                const declaredFormat =
                    (tex.format as string) ||
                    (logical === "INPUT" ? "rgba8unorm" : "rgba16float");
                // storage 綁定必須使用 storage-capable 格式(r16float 等被 Dawn 拒絕);
                // 與 ResourceManager.createTexture 的建立端映射共用 storageCapableFormat,
                // 確保 WGSL 宣告 / BindGroupLayout / GPUTexture 三者一致。
                const format = isStorage
                    ? storageCapableFormat(declaredFormat)
                    : declaredFormat;
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
                    isStorage: isStorage,
                });
            });
            wgsl += "\n";

            // Setup storage buffer bindings / 設定 storage buffer 綁定
            // OpenNR/CNN 類特效的特徵圖與權重：`//! BUFFER x` + MODE read|read_write +
            // TYPE u32x2|f32x4|f32（元素型別決定 WGSL 陣列元素與宿主端 stride）。
            // 綁定槽續接紋理之後（同一遞增計數器），與 PipelineManager 一致。
            const BUF_ELEM: Record<string, string> = {
                u32x2: "vec2<u32>",
                f32x4: "vec4<f32>",
                f32: "f32",
            };
            passResources.storageBuffers = [];
            shaderInfo.buffers.forEach((buf: any) => {
                if (!referenced.has(buf.name)) return;
                const elem = BUF_ELEM[buf.type] || "vec2<u32>";
                const mode = buf.mode === "read_write" ? "read_write" : "read";
                while (reservedBindings.has(bufferBinding)) {
                    bufferBinding++;
                }
                const currentBinding = bufferBinding++;
                wgsl += `@group(0) @binding(${currentBinding}) var<storage, ${mode}> ${buf.name}: array<${elem}>;\n`;
                passResources.storageBuffers.push({
                    name: buf.name,
                    binding: currentBinding,
                    mode: mode,
                    type: buf.type,
                    group: 0,
                });
            });
            wgsl += "\n";

            // Transform pass logic to compute shader entry point / 轉換通道邏輯為計算著色器進入點
            let passCode = this._transformPassToComputeShader(pass);
            wgsl += passCode;

            // Module-level compat post-process / 模組層級相容後處理:
            // HLSL 風格的殘留語法(typed local declaration、mul()、scalar swizzle)與
            // 「引用 uniform 的 module-scope const」(WGSL 禁止)在此統一轉換。
            wgsl = this._postProcessModule(wgsl, pass.index);

            generatedModules.push({
                wgslCode: wgsl,
                passIndex: pass.index,
                resources: passResources,
            });
        });

        return generatedModules;
    }

    /**
     * Split COMMON code into the shared header and function bodies (single walk).
     * ---
     * 單次行走（括號計數）將 COMMON 代碼拆分為：
     * - header：函式以外的共用宣告（struct/alias/const/var 等）
     * - fnBodies：函式名稱 → 完整函式主體（含簽名行）
     */
    private _splitCommonCode(commonCode: string): {
        header: string;
        fnBodies: Map<string, string>;
    } {
        const header: string[] = [];
        const fnBodies = new Map<string, string>();
        if (!commonCode) return { header: "", fnBodies };

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
                    header.push(line);
                    continue;
                }
            }
            // Count braces on EVERY line including the fn signature line
            // (the opening `{` usually sits at the end of `fn X() ... {`).
            // 函式簽名行本身也要計算括號（`{` 通常在行尾）。
            for (const ch of line) {
                if (ch === "{") {
                    depth++;
                    started = true;
                } else if (ch === "}") {
                    depth--;
                }
            }
            buf.push(line);
            if (started && depth <= 0) {
                fnBodies.set(curFn, buf.join("\n"));
                curFn = null;
            }
        }
        return { header: header.join("\n"), fnBodies };
    }

    /**
     * Walk the transitive COMMON-helper closure reachable from a pass's code.
     * ---
     * 以 pass 代碼中的識別字為起點，沿 COMMON 函式呼叫圖遞迴展開，
     * 回傳可達函式（BFS 發現順序，供函式級裁剪）與全部出現過的識別字
     * （供資源引用判定），單一走訪同時服務兩種需求。
     */
    private _resolveCommonClosure(
        passCode: string,
        fnBodies: Map<string, string>,
    ): { fns: string[]; identifiers: Set<string> } {
        const identifiers = new Set<string>();
        const fns: string[] = [];
        const queue: string[] = passCode.match(/[A-Za-z_][A-Za-z0-9_]*/g) || [];
        while (queue.length > 0) {
            const w = queue.shift()!;
            if (identifiers.has(w)) continue;
            identifiers.add(w);
            const body = fnBodies.get(w);
            if (body !== undefined) {
                fns.push(w);
                for (const w2 of body.match(/[A-Za-z_][A-Za-z0-9_]*/g) || []) {
                    if (!identifiers.has(w2)) queue.push(w2);
                }
            }
        }
        return { fns, identifiers };
    }

    /**
     * Generate WGSL `const` declarations for external defines.
     * ---
     * 為外部 defines 產生 WGSL 常數宣告，注入每個 pass 模組的開頭。
     *
     * **型別決策（整數 → u32，非整數 → f32）**：
     * OpenNR/CNN 類特效把 defines 當成**維度與通道數**使用——`NW * NH`（像素數）、
     * `CIN / 4u`（u32 除法）、`i32(NW)`（顯式轉型），這些在 WGSL 中要求 u32 運算元；
     * 若產生 f32 則 `NW * NH` 會變成浮點乘法、`CIN / 4u` 直接型別不符而編譯失敗。
     * 既有特效與測試皆不使用 defines（實測 grep：defines 僅出現於 WGFX.ts 的
     * INPUT_WIDTH/HEIGHT 與資源管理層），因此無浮點相容性包袱。
     * 非整數值（如 0.5 之類的比例係數）則以 f32 宣告，維持數值語義。
     *
     * 值一律在產生期以字面量寫入（非 uniform），故不佔用綁定槽、無額外頻寬成本。
     *
     * @param defines - External defines to emit / 要輸出的外部 defines
     * @param commonCode - COMMON 原始碼，用於避免重複宣告同名常數
     * @returns WGSL constant declaration block（無 define 時為空字串）
     */
    private _generateDefines(
        defines: Record<string, number>,
        commonCode: string,
    ): string {
        const names = Object.keys(defines || {}).sort();
        if (names.length === 0) return "";

        const lines: string[] = [];
        for (const name of names) {
            // Only identifiers are emittable / 僅合法識別字可輸出
            if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
                // eslint-disable-next-line no-console
                console.warn(
                    `[WGFX] defines: skipping "${name}" (not a valid WGSL identifier).`,
                );
                continue;
            }
            const value = Number(defines[name]);
            if (!Number.isFinite(value)) {
                // eslint-disable-next-line no-console
                console.warn(
                    `[WGFX] defines: skipping "${name}" (non-finite value ${defines[name]}).`,
                );
                continue;
            }
            // COMMON 已自行宣告同名常數/變數時不得重複宣告（Naga 會報 duplicate）。
            // 以宣告式 regex 判定 const/let/var，避免誤判一般運算式中的同名識別字。
            const alreadyDeclared = new RegExp(
                `\\b(?:const|let|var)\\s+${name}\\b`,
            ).test(commonCode);
            if (alreadyDeclared) {
                continue;
            }

            if (Number.isInteger(value)) {
                lines.push(`const ${name}: u32 = ${value}u;`);
            } else {
                lines.push(`const ${name}: f32 = ${value};`);
            }
        }

        if (lines.length === 0) return "";
        return `// External defines / 外部定義常數\n${lines.join("\n")}\n\n`;
    }

    /**
     * Generate SceneInfo struct definition with its uniform binding.
     * ---
     * 產生場景資訊結構體定義與對應的 uniform 綁定宣告。
     * 記憶體佈局與 ResourceManager.updateSceneBuffer 的寫入格式（48 bytes）一致：
     * inputSize(u32x2) @0, inputPt(f32x2) @8, outputSize(u32x2) @16, outputPt(f32x2) @24,
     * scale(f32x2) @32, frame(u32) @40, _pad(f32) @44。
     *
     * frame 置於結構尾端：前 40 bytes 沿用既有佈局（向後相容），新增的 u32 之後補一個
     * f32 pad 將結構填滿 48 bytes。WGSL uniform 結構要求大小為 16 的倍數，
     * 40 + 4 = 44 不符；補至 48 同時滿足對齊與既有 16-byte 慣例。
     */
    private _generateSceneInfoStruct(): string {
        return `// Scene Info Struct / 場景資訊結構體
struct SceneInfo {
    inputSize: vec2<u32>,
    inputPt: vec2<f32>,
    outputSize: vec2<u32>,
    outputPt: vec2<f32>,
    scale: vec2<f32>,
    // 影格計數器（供時間性特效：TAA/降噪的抖動索引等）
    frame: u32,
    _pad0: f32,
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
        if (parameters.length === 0) return "";

        let code = `// Uniform Parameters (16-byte slot per parameter) / Uniform 參數（每參數 16-byte）\nstruct Uniforms {\n`;

        parameters.forEach((param) => {
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
        let code = pass.code.replace(/\/\/!.*\n/g, "");

        // Apply @compute attribute and workgroup size, anchored to the ENTRY POINT
        // function itself (not blindly prepended to the block) so that helper
        // functions may be defined before fn Pass<N> within the same pass block.
        // @compute 屬性直接掛在進入點函式上，允許 helper 定義在 fn Pass<N> 之前。
        const numThreads = pass.numThreads || [1, 1, 1];
        code = code.replace(/@compute\s*@workgroup_size\([^)]+\)\s*/g, "");

        const entryRegex = new RegExp(`fn\\s+(?:Pass${pass.index}|main)\\s*\\(`);
        if (entryRegex.test(code)) {
            code = code.replace(
                entryRegex,
                `@compute @workgroup_size(${numThreads[0]}, ${numThreads[1]}, ${numThreads[2]})\nfn main_cs(`,
            );
        } else {
            // eslint-disable-next-line no-console
            console.warn(
                `[WGFX] Pass ${pass.index}: no "fn Pass${pass.index}(" or "fn main(" found; prepending @compute to block start.`,
            );
            code = `@compute @workgroup_size(${numThreads[0]}, ${numThreads[1]}, ${numThreads[2]})\n${code}`;
        }

        return code;
    }

    // =====================================================================
    // Module-level compat post-processing / 模組層級相容後處理
    // =====================================================================

    /** WGSL 型別樣式（可用於 HLSL typed local declaration 的型別） */
    private static readonly WGSL_TYPE_SRC = "(?:f32|i32|u32|bool|vec[234]<[fiub]32>|mat[234]x[234]<f32>)";
    private static readonly WGSL_NAME_SRC = "[A-Za-z_][A-Za-z0-9_]*";

    /**
     * Apply all module-level compat transforms, in a fixed order.
     * ---
     * 固定順序套用全部模組層級相容轉換：
     * 1. HLSL typed local declaration → `var NAME: TYPE = EXPR;`
     * 2. HLSL `mul(a, b)` → `((a) * (b))`（任意巢狀位置）
     * 3. scalar swizzle 消除（`value.x`，value 為 f32/i32/u32/bool 時 → `value`）
     * 4. `vecN<f32>(…)` 參數內的 u32 場景尺寸運算式補 f32 轉換
     * 5. 引用 uniform 的 module-scope const → `var<private>` + entry point 開頭初始化
     */
    private _postProcessModule(wgsl: string, passIndex: number): string {
        let code = this._convertHlslTypedDecls(wgsl, passIndex);
        code = this._expandHlslMul(code, passIndex);
        code = this._eliminateScalarSwizzles(code);
        code = this._wrapU32ArgsInF32VectorCtors(code);
        code = this._promoteUniformConstsToPrivate(code, passIndex);
        return code;
    }

    /**
     * Wrap u32 scene-size expressions inside `vecN<f32>(…)` constructors with `f32(...)`.
     * ---
     * WGSL 的多參數向量建構子要求參數型別與分量型別完全一致（僅單參數有
     * conversion/splat 重載），因此 `vec2<f32>(scene.inputSize.x * 1, scene.inputSize.y * 2)`
     * 這種 HLSL 隱式轉換（float2(uint, uint) 合法）在 WGSL 不合法。
     * `scene.inputSize` / `scene.outputSize` 為 vecN<u32>，凡參數引用其一且
     * 參數不是「整個向量本身」時，補 `f32(arg)`。
     */
    private _wrapU32ArgsInF32VectorCtors(wgsl: string): string {
        const re = /\bvec[234]<f32>\s*\(/g;
        let out = "";
        let last = 0;
        let i = 0;
        while (i < wgsl.length) {
            re.lastIndex = i;
            const m = re.exec(wgsl);
            if (!m) break; // 無更多建構子;尾段由最終 return 統一拼接
            const open = m.index + m[0].length - 1;
            let depth = 0;
            let end = -1;
            for (let j = open; j < wgsl.length; j++) {
                if (wgsl[j] === "(") depth++;
                else if (wgsl[j] === ")") { depth--; if (depth === 0) { end = j; break; } }
            }
            if (end < 0) break; // 括號不平衡(不完整程式碼)
            const inner = wgsl.slice(open + 1, end);
            const args = WGSLCodeGenerator.splitTopLevelCommas(inner);
            let changed = false;
            const newArgs = args.map((arg) => {
                const t = arg.trim();
                if (t === "scene.inputSize" || t === "scene.outputSize") return arg; // 整向量轉換建構子,合法
                if (/\b(scene\.inputSize|scene\.outputSize)\b/.test(t)) {
                    changed = true;
                    return `f32(${t})`;
                }
                return arg;
            });
            if (changed) {
                out += wgsl.slice(last, open + 1) + newArgs.join(", ") + ")";
                last = end + 1;
            }
            i = end + 1;
        }
        if (last === 0) return wgsl;
        return out + wgsl.slice(last);
    }

    /** 依括號深度切分頂層逗號（不含字串/正規語法；WGSL 程式碼足夠安全） */
    private static splitTopLevelCommas(s: string): string[] {
        const out: string[] = [];
        let depth = 0;
        let cur = "";
        for (const ch of s) {
            if (ch === "(" || ch === "[") depth++;
            else if (ch === ")" || ch === "]") depth--;
            if (ch === "," && depth === 0) {
                if (cur.trim().length) out.push(cur.trim());
                cur = "";
                continue;
            }
            cur += ch;
        }
        if (cur.trim().length) out.push(cur.trim());
        return out;
    }

    /**
     * Convert HLSL-style typed local declarations to WGSL `var` declarations.
     * ---
     * 將 HLSL 式區域宣告轉為 WGSL：
     *   `vec2<f32> UV = ...;`        → `var UV: vec2<f32> = ...;`
     *   `f32 a = 0.0, b = 1.0;`      → `var a: f32 = 0.0; var b: f32 = 1.0;`
     *   `f32 acc;`                   → `var acc: f32;`
     *   `vec4<f32> samples[8];`      → `var samples: array<vec4<f32>, 8>;`
     * 值的擷取以括號平衡掃描（任意巢狀深度），僅在陳述式起點
     *（`{` / `;` / 行首之後）觸發，避免誤傷運算式中繼位置。
     */
    private _convertHlslTypedDecls(code: string, passIndex: number): string {
        // 前綴括號深度（僅計 `(`）：深度 > 0 的位置在參數列/呼叫引數內，一律跳過，
        // 避免把多行 fn 參數（`fn X(a: f32,\n f32 b)`）誤判為區域宣告。
        const parenDepth = new Int32Array(code.length + 1);
        {
            let d = 0;
            for (let i = 0; i < code.length; i++) {
                parenDepth[i] = d;
                if (code[i] === "(") d++;
                else if (code[i] === ")") d--;
                if (d < 0) d = 0;
            }
            parenDepth[code.length] = d;
        }

        const typeRe = new RegExp(
            `(?<=[{;\\n])([ \\t]*)(${WGSLCodeGenerator.WGSL_TYPE_SRC})[ \\t]+` +
            `(${WGSLCodeGenerator.WGSL_NAME_SRC})[ \\t]*`,
            "g",
        );
        let result = "";
        let last = 0;
        for (const m of code.matchAll(typeRe)) {
            if (parenDepth[m.index] !== 0) continue; // 參數列/呼叫引數內，非陳述式起點
            const ws = m[1];
            const type = m[2];
            const firstName = m[3];
            let i = m.index + m[0].length;

            // 陣列宣告：`TYPE NAME[N];`（HLSL 陣列式）
            if (code[i] === "[") {
                let depth = 0;
                let j = i;
                for (; j < code.length; j++) {
                    if (code[j] === "[") depth++;
                    else if (code[j] === "]") { depth--; if (depth === 0) { j++; break; } }
                }
                if (j >= code.length) continue;
                const dims = code.slice(i + 1, j - 1).trim();
                let k = j;
                while (k < code.length && /\s/.test(code[k])) k++;
                if (code[k] !== ";" || !/^\d+$/.test(dims)) continue;
                result += code.slice(last, m.index) + `${ws}var ${firstName}: array<${type}, ${dims}>;`;
                last = k + 1;
                continue;
            }

            // 純量/向量/矩陣宣告（可含多個宣告子與初始化值）
            // 掃描到頂層 `;`；頂層 `,` 分隔宣告子。
            let depth = 0;
            let j = i;
            for (; j < code.length; j++) {
                const ch = code[j];
                if (ch === "(" || ch === "[") depth++;
                else if (ch === ")" || ch === "]") depth--;
                else if (ch === ";" && depth === 0) break;
            }
            if (j >= code.length || depth !== 0) continue;
            const declBody = code.slice(i, j);
            const parts = WGSLCodeGenerator.splitTopLevelCommas(declBody.replace(/^[ \t]+/, ""));
            if (parts.length === 0) continue;
            // 每個宣告子：`NAME` 或 `NAME = VALUE`
            const decls: string[] = [];
            let bad = false;
            for (const p of parts) {
                const pm = p.match(new RegExp(`^(${WGSLCodeGenerator.WGSL_NAME_SRC})[ \\t]*([\\s\\S]*)$`));
                if (!pm) { bad = true; break; }
                const [, name, rhs] = pm;
                decls.push(rhs.trim().length
                    ? `var ${name}: ${type} = ${rhs.trim()};`
                    : `var ${name}: ${type};`);
            }
            if (bad) continue;
            result += code.slice(last, m.index) + ws + decls.join(" ");
            last = j + 1;
        }
        if (last === 0) return code;
        void passIndex;
        return result + code.slice(last);
    }

    /**
     * Expand HLSL `mul(a, b)` into `(a) * (b)` at arbitrary nesting depth.
     * ---
     * HLSL `mul(x, y)` 與 WGSL 的 `x * y` 一致（矩陣方向由建構時的欄/列安排決定，
     * WGFX 已在轉換器層以「欄 = HLSL 的欄」重建矩陣）。以括號平衡掃描處理，
     * 支援任意巢狀（mul 內再含 mul、函式呼叫等）。
     */
    private _expandHlslMul(code: string, passIndex: number): string {
        if (/\bfn\s+mul\s*\(/.test(code)) {
            // eslint-disable-next-line no-console
            console.warn(`[WGFX] Pass ${passIndex}: 使用者自訂了 fn mul()，跳過 mul() → * 轉換。`);
            return code;
        }
        const re = /\bmul\s*\(/;
        let s = code;
        for (let iter = 0; iter < 64; iter++) {
            const m = re.exec(s);
            if (!m) break;
            const open = m.index + m[0].length - 1;
            let depth = 0;
            let end = -1;
            for (let j = open; j < s.length; j++) {
                if (s[j] === "(") depth++;
                else if (s[j] === ")") { depth--; if (depth === 0) { end = j; break; } }
            }
            if (end < 0) break; // 括號不平衡（不完整程式碼），放棄
            const args = WGSLCodeGenerator.splitTopLevelCommas(s.slice(open + 1, end));
            let repl: string;
            if (args.length === 2) {
                repl = `((${args[0]}) * (${args[1]}))`;
            } else {
                // eslint-disable-next-line no-console
                console.warn(`[WGFX] Pass ${passIndex}: mul() 需要恰好 2 個參數，原樣保留。`);
                repl = s.slice(m.index, end + 1);
            }
            s = s.slice(0, m.index) + repl + s.slice(end + 1);
        }
        return s;
    }

    /**
     * Eliminate HLSL scalar swizzles on scalar-typed values (e.g. `value.x` → `value`).
     * ---
     * HLSL 允許對純量做 swizzle（`value.x` == value）；WGSL 不允許。
     * 逐函式收集「確定為純量」的識別字（參數與 var/let 宣告），
     * 僅改寫這些識別字上的單分量 swizzle，避免誤傷向量。
     *
     * **位置感知**：同名識別字在同一函式可能以不同型別重複宣告
     * （如 `let p = textureLoad(...)` 之後又有 `for (var p: u32 = ...)` 迴圈計數器）。
     * 僅在「該純量宣告結束位置之後」出現的 swizzle 才消除——WGSL 要求使用前宣告,
     * 因此宣告前出現的同名識別字必然是另一個（向量）值。
     */
    private _eliminateScalarSwizzles(code: string): string {
        // 找出所有 fn 的主體（含簽名），逐一處理
        const fnRe = /\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/g;
        let out = "";
        let last = 0;
        let match: RegExpExecArray | null;
        while ((match = fnRe.exec(code)) !== null) {
            const openParen = match.index + match[0].length - 1;
            // 簽名括號
            let depth = 0;
            let j = openParen;
            for (; j < code.length; j++) {
                if (code[j] === "(") depth++;
                else if (code[j] === ")") { depth--; if (depth === 0) { break; } }
            }
            if (j >= code.length) continue;
            const params = code.slice(openParen + 1, j);
            // 找函式主體 `{`（可能帶回傳型別 `) -> T {`）
            let k = j + 1;
            while (k < code.length && /\s/.test(code[k])) k++;
            if (code[k] === "-") { // 回傳型別 `-> T {`：跳至 `{`
                const brace = code.indexOf("{", k);
                if (brace < 0) continue;
                k = brace;
            }
            if (code[k] !== "{") continue;
            let bDepth = 0;
            let end = -1;
            for (let b = k; b < code.length; b++) {
                if (code[b] === "{") bDepth++;
                else if (code[b] === "}") { bDepth--; if (bDepth === 0) { end = b; break; } }
            }
            if (end < 0) continue;
            const body = code.slice(k + 1, end);

            // 收集純量識別字「及其宣告結束位置」：參數（位置=0）+ var/let 宣告
            const scalarDecls = new Map<string, number>(); // name → 宣告在 body 內的結束 offset
            for (const pm of params.matchAll(
                new RegExp(`\\b([A-Za-z_][A-Za-z0-9_]*)\\s*:\\s*(?:f32|i32|u32|bool)\\b`, "g"),
            )) scalarDecls.set(pm[1], 0); // 參數在整個函式可見
            for (const dm of body.matchAll(
                new RegExp(`\\b(?:var|let)\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*:\\s*(?:f32|i32|u32|bool)\\b`, "g"),
            )) {
                const endPos = (dm.index ?? 0) + dm[0].length;
                scalarDecls.set(dm[1], Math.max(scalarDecls.get(dm[1]) ?? 0, endPos));
            }
            if (scalarDecls.size === 0) continue;

            // 位置感知改寫單分量 swizzle：僅消除「宣告結束位置之後」出現者
            const nameAlt = [...scalarDecls.keys()].map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
            const swizzleRe = new RegExp(`\\b(${nameAlt})\\.(?:x|y|z|w|r|g|b|a)\\b`, "g");
            const newBody = body.replace(swizzleRe, (m0, name, off) => {
                const declEnd = scalarDecls.get(name) ?? 0;
                return off >= declEnd ? name : m0;
            });
            out += code.slice(last, k + 1) + newBody;
            last = end;
        }
        if (last === 0) return code;
        return out + code.slice(last);
    }

    /**
     * Promote module-scope `const` declarations that reference uniform variables
     * to `var<private>` initialized at the top of the entry point.
     * ---
     * WGSL 禁止在 module-scope const 的初始化式中引用 uniform var
     *（`'var scene' cannot be referenced at module-scope`）。
     * 本轉換將「直接或間接（鏈式引用其他被提升的 const）」依賴 uniform 的
     * module-scope const 改為 `var<private> NAME: TYPE;`，並在 entry point
     *（`fn main_cs`）開頭以原初始化式依宣告順序賦值，使所有 pass 與 COMMON
     * 輔助函式都能取得與 Magpie HLSL 相同的語意。
     *
     * 純常數的 const（如 `const N: u32 = 429u;`，可能作為 array 尺寸）不會被提升。
     */
    private _promoteUniformConstsToPrivate(wgsl: string, passIndex: number): string {
        // 收集 uniform 變數名（注入的 scene/uniforms 或 COMMON 自行宣告者）
        const uniformNames = new Set<string>();
        for (const m of wgsl.matchAll(/\bvar<uniform>\s+([A-Za-z_][A-Za-z0-9_]*)/g)) {
            uniformNames.add(m[1]);
        }
        if (uniformNames.size === 0) return wgsl;

        // 以「註解剝除」的同長度影子字串做掃描，避免註解內的括號干擾深度計算；
        // 位置索引在原字串上仍然有效（置換字元數相同）。
        const shadow = wgsl
            .replace(/\/\*[\s\S]*?\*\//g, (m) => " ".repeat(m.length))
            .replace(/\/\/[^\n]*/g, (m) => " ".repeat(m.length));

        // 找出 module-scope（所有括號深度為 0）的 const 宣告
        interface ConstDecl { name: string; type: string | null; expr: string; start: number; end: number; }
        const consts: ConstDecl[] = [];
        let depth = 0;
        let i = 0;
        while (i < shadow.length) {
            const ch = shadow[i];
            if (ch === "{" || ch === "(" || ch === "[") { depth++; i++; continue; }
            if (ch === "}" || ch === ")" || ch === "]") { depth--; i++; continue; }
            if (depth === 0 && /^const\b/.test(shadow.slice(i, i + 6))) {
                // 掃描至頂層 `;`
                let d = 0;
                let j = i;
                for (; j < shadow.length; j++) {
                    const c = shadow[j];
                    if (c === "{" || c === "(" || c === "[") d++;
                    else if (c === "}" || c === ")" || c === "]") d--;
                    else if (c === ";" && d === 0) break;
                }
                if (j >= shadow.length) break;
                const text = wgsl.slice(i, j + 1);
                const cm = text.match(
                    new RegExp(`^const\\s+(${WGSLCodeGenerator.WGSL_NAME_SRC})\\s*(?::\\s*([^=]+?))?\\s*=\\s*([\\s\\S]+);\\s*$`),
                );
                if (cm) {
                    consts.push({ name: cm[1], type: cm[2] ? cm[2].trim() : null, expr: cm[3].trim(), start: i, end: j + 1 });
                }
                i = j + 1;
                continue;
            }
            i++;
        }
        if (consts.length === 0) return wgsl;

        // 不動點提升：引用 uniform，或引用其他被提升 const 的，一律提升
        const promoted = new Set<string>();
        let changed = true;
        while (changed) {
            changed = false;
            for (const c of consts) {
                if (promoted.has(c.name)) continue;
                const refsUniform = [...uniformNames].some((u) =>
                    new RegExp(`\\b${u}\\b`).test(c.expr));
                const refsPromoted = [...promoted].some((p) =>
                    new RegExp(`\\b${p}\\b`).test(c.expr));
                if (refsUniform || refsPromoted) {
                    promoted.add(c.name);
                    changed = true;
                }
            }
        }
        if (promoted.size === 0) return wgsl;

        // 改寫宣告（**由後往前**,避免前面改寫造成的索引位移）,
        // 並依原宣告順序收集初始化陳述式（鏈式引用 temp_pt ← temp_size 依此有序）。
        const inits: string[] = [];
        const promotedAll = consts.filter((c) => promoted.has(c.name)).sort((a, b) => a.start - b.start);
        for (const c of promotedAll) {
            if (c.type === null) {
                // 無型別標註的 const 無法推出 var<private> 的型別——保留原樣（Tint 會報
                // 較清晰的錯誤）；實務上轉換器產出的宣告都帶型別。
                // eslint-disable-next-line no-console
                console.warn(
                    `[WGFX] Pass ${passIndex}: const ${c.name} 引用 uniform 但無型別標註，無法提升為 var<private>;請補型別。`,
                );
                continue;
            }
            inits.push(`${c.name} = ${c.expr};`);
        }
        if (inits.length === 0) return wgsl;
        const rewritable = promotedAll.filter((c) => c.type !== null);
        for (let k = rewritable.length - 1; k >= 0; k--) {
            const c = rewritable[k];
            wgsl = wgsl.slice(0, c.start) + `var<private> ${c.name}: ${c.type};` + wgsl.slice(c.end);
        }

        // 在 entry point（fn main_cs）主體開頭插入初始化（依宣告順序）
        const entryIdx = wgsl.search(/\bfn\s+main_cs\s*\(/);
        if (entryIdx < 0) {
            // eslint-disable-next-line no-console
            console.warn(
                `[WGFX] Pass ${passIndex}: 找不到 fn main_cs，已提升 ${inits.length} 個 const 為 var<private> 但無法插入初始化。`,
            );
            return wgsl;
        }
        const braceIdx = wgsl.indexOf("{", entryIdx);
        if (braceIdx < 0) return wgsl;
        const initBlock =
            "\n    // module-scope const(引用 uniform)→ var<private>,於進入點開頭初始化\n    " +
            inits.join("\n    ") + "\n";
        return wgsl.slice(0, braceIdx + 1) + initBlock + wgsl.slice(braceIdx + 1);
    }
}
