import {WGFXShaderInfo, PassInfo, TextureInfo, ParameterInfo} from '@/types';

/**
 * Resource binding index configuration strategy.
 * ---
 * 資源綁定索引配置策略。
 * 定義標準資源的固定綁定位置，確保著色器(Shader)與主機端(Host)之間的一致性。

 * Binding index calculation / 索引計算方式：
 * - **場景資訊**: $Index_{Scene} = 0$
 * - **通用緩衝**: $Index_{Uniforms} = 1$
 * - **採樣器範圍**: $Index_{Sampler} \in [2, 9]$
 * - **紋理資源範圍**: $Index_{Texture} \in [10, \infty)$
 *
 */
const BINDING_LAYOUT = {
    SCENE_INFO: 0,      // Scene information (Resolution, etc.) / 場景資訊 (解析度等)
    UNIFORMS: 1,        // Uniform buffer / Uniform 緩衝區
    SAMPLERS_START: 2,  // Sampler start index / 採樣器起始索引
    TEXTURES_START: 10, // Texture start index / 紋理起始索引
} as const;

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
     * @group Generation
     * @param shaderInfo - The parsed shader metadata / 解析後的著色器元數據
     * @returns Array of generated modules / 產生的模組列表
     */
    public generate(shaderInfo: WGFXShaderInfo): { wgslCode: string; passIndex: number; resources: any }[] {
        const generatedModules: { wgslCode: string; passIndex: number; resources: any }[] = [];

        // Prepare common code sections / 準備共通代碼區段
        const commonCode = shaderInfo.commonCode
            ? `// --- COMMON CODE ---\n${shaderInfo.commonCode}\n\n`
            : '';

        // Generate SceneInfo struct / 產生場景資訊結構體
        const sceneInfoCode = this._generateSceneInfoStruct();

        // Generate Uniform buffer struct / 產生 Uniform 緩衝區結構體
        const uniformBufferCode = this._generateUniformStruct(shaderInfo.parameters);

        // Process each render pass / 處理每個渲染通道
        shaderInfo.passes.forEach(pass => {
            let wgsl = `// ========================================\n`;
            wgsl += `// Pass ${pass.index}\n`;
            wgsl += `// ========================================\n\n`;

            // Inject common code / 注入共通代碼
            wgsl += commonCode;

            // Inject scene info / 注入場景資訊
            if (!commonCode.includes('SceneInfo')) {
                wgsl += sceneInfoCode;
            }

            // Inject Uniforms / 注入 Uniforms
            if (uniformBufferCode) {
                wgsl += uniformBufferCode;
            }

            // Prepare resource tracking object / 準備資源追蹤物件
            const passResources = {
                textures: [] as any[],
                samplers: [] as any[],
                parameters: shaderInfo.parameters,
                hasScene: true
            };

            // Setup sampler bindings / 設定採樣器綁定
            let samplerBinding = BINDING_LAYOUT.SAMPLERS_START;
            shaderInfo.samplers.forEach(sampler => {
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
            const usedTextureNames = new Set([...pass.in, ...pass.out]);
            let textureBinding = BINDING_LAYOUT.TEXTURES_START;

            shaderInfo.textures.forEach(tex => {
                const isUsed = [...usedTextureNames].some(usedName =>
                    tex.name === usedName || tex.name.startsWith(usedName + "_")
                );
                if (!isUsed) return;

                const isOutputInThisPass = pass.out.includes(tex.name);
                const isStorage = (tex.name === 'OUTPUT' || isOutputInThisPass);

                const format = tex.format || 'rgba16float';
                const textureType = isStorage
                    ? `texture_storage_2d<${format}, write>`
                    : `texture_2d<f32>`;

                const currentBinding = textureBinding++;
                wgsl += `@group(0) @binding(${currentBinding}) var ${tex.name}: ${textureType};\n`;

                passResources.textures.push({
                    ...tex,
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
     * Generate SceneInfo struct definition.
     * ---
     * 產生場景資訊結構體定義。
     * 包含輸入輸出解析度、像素大小等場景相關資訊。
     */
    private _generateSceneInfoStruct(): string {
        return `// Scene Info Struct / 場景資訊結構體
struct SceneInfo {
    inputSize: vec2<u32>,
    inputPt: vec2<f32>,
    outputSize: vec2<u32>,
    outputPt: vec2<f32>,
    scale: vec2<f32>,
}
@group(0) @binding(${BINDING_LAYOUT.SCENE_INFO}) var<uniform> scene: SceneInfo;\n\n`;
    }

    /**
     * Generate Uniform buffer struct definition.
     * ---
     * 產生 Uniform 緩衝區結構體定義。
     * 根據參數列表產生對應的 WGSL 結構體。
     * 注意：不再自動加入填充，由呼叫方負責記憶體對齊。
     */
    private _generateUniformStruct(parameters: ParameterInfo[]): string {
        if (parameters.length === 0) return '';

        let code = `// Uniform Parameters / Uniform 參數\nstruct Uniforms {\n`;

        parameters.forEach(param => {
            const wgslType = param.type === 'int' ? 'i32' : 'f32';
            code += `    ${param.name}: ${wgslType},\n`;
        });

        code += `};\n`;
        code += `@group(0) @binding(${BINDING_LAYOUT.UNIFORMS}) var<uniform> uniforms: Uniforms;\n\n`;

        return code;
    }

    /**
     * Transform Pass function into a compute shader entry point.
     * ---
     * 將 Pass 函式轉換為計算著色器進入點。
     * 重新命名函式為 main_cs 並套用 @compute 屬性。
     */
    private _transformPassToComputeShader(pass: PassInfo): string {
        // Remove meta-command comments / 移除元指令註解
        let code = pass.code.replace(/\/\/!.*\n/g, '');

        // Rename PassX function to main_cs / 將 PassX 函式重新命名為 main_cs
        code = code.replace(
            new RegExp(`fn Pass${pass.index}\\s*\\(`),
            `fn main_cs(`
        );

        // Apply @compute attribute and workgroup size / 套用 @compute 屬性與工作群組大小
        const numThreads = pass.numThreads || [1, 1, 1];
        code = code.replace(/@compute\s*@workgroup_size\([^)]+\)\s*/g, '');
        code = `@compute @workgroup_size(${numThreads[0]}, ${numThreads[1]}, ${numThreads[2]})\n${code}`;

        return code;
    }
}