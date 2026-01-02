import {WGFXShaderInfo, PassInfo, TextureInfo} from '@/types';

/**
 * Generator for WGSL shader code from parsed shader information.
 * ---
 * 從解析後的著色器資訊產生 WGSL 著色器代碼。
 * 負責注入標準資源、修正語法相容性並封裝計算著色器進入點。
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
     * 自動處理資源綁定索引（Binding Index）並注入全域 Uniforms。
     *
     * @group Generation
     * @param shaderInfo - The parsed shader metadata / 解析後的著色器元數據
     * @returns Array of generated modules / 產生的模組列表
     */
    public generate(shaderInfo: WGFXShaderInfo): { wgslCode: string; passIndex: number; resources: any }[] {
        const generatedModules: { wgslCode: string; passIndex: number; resources: any }[] = [];

        // 1. Prepare Common Code section / 準備共通代碼區段
        const commonCode = shaderInfo.commonCode
            ? `// --- COMMON ---\n${shaderInfo.commonCode}\n\n`
            : '';

        // 2. Generate Uniform Buffer Struct (Handled with 16-byte padding for std140)
        // 產生 Uniform 結構體（手動加入填充以符合 std140 對齊規範）
        let uniformBufferCode = '';
        if (shaderInfo.parameters.length > 0) {
            uniformBufferCode += `struct Uniforms {\n`;
            shaderInfo.parameters.forEach(p => {
                const type = p.type === 'int' ? 'i32' : 'f32';
                uniformBufferCode += `    ${p.name}: ${type},\n`;
                // Padding ensures that each parameter starts at a 16-byte boundary
                // 填充確保每個參數都從 16 位元組邊界開始，簡化內存佈局
                uniformBufferCode += `    _pad_${p.name}_1: f32,\n`;
                uniformBufferCode += `    _pad_${p.name}_2: f32,\n`;
                uniformBufferCode += `    _pad_${p.name}_3: f32,\n`;
            });
            uniformBufferCode += `};\n`;
            uniformBufferCode += `@group(0) @binding(1) var<uniform> uniforms: Uniforms;\n\n`;
        }

        // 3. Generate SceneInfo struct (Resolution data) / 產生場景資訊結構體（解析度數據）
        const sceneInfoCode = `struct SceneInfo {
        inputSize: vec2<u32>,
        inputPt: vec2<f32>,
        }
        @group(0) @binding(4) var<uniform> scene: SceneInfo;\n\n`;

        // 4. Process individual passes / 遍歷並處理各個通道
        shaderInfo.passes.forEach(pass => {
            let wgsl = `// Pass ${pass.index}\n\n`;
            wgsl += commonCode;

            // Inject default sampler if not defined in COMMON / 若共通代碼未定義則注入預設取樣器
            if (!commonCode.includes('var sam:') && !commonCode.includes('var sam :')) {
                wgsl += `@group(0) @binding(0) var sam: sampler;\n\n`;
            }

            wgsl += uniformBufferCode;

            // Inject SceneInfo if not defined in COMMON / 若共通代碼未定義則注入場景資訊
            if (!commonCode.includes('SceneInfo')) {
                wgsl += sceneInfoCode;
            }

            const passResources = {
                textures: [] as any[],
                samplers: [] as any[],
                parameters: shaderInfo.parameters,
                hasScene: true
            };

            // Reserved bindings: 0 (Sampler), 1 (Uniforms), 4 (SceneInfo)
            // Start dynamic bindings from index 6
            // 預留綁定：0 (取樣器), 1 (Uniforms), 4 (場景資訊)，其餘資源從索引 6 開始
            let bindingIndex = 6;

            const usedTextureNames = new Set([...pass.in, ...pass.out]);

            // Setup Texture Bindings / 設定紋理繫結
            shaderInfo.textures.forEach(tex => {
                const isUsed = [...usedTextureNames].some(usedName =>
                    tex.name === usedName || tex.name.startsWith(usedName + "_")
                );
                if (!isUsed) return;

                const isOutputInThisPass = pass.out.includes(tex.name);
                let isStorage = (tex.name === 'OUTPUT' || isOutputInThisPass);

                const format = 'rgba16float';
                const textureType = isStorage
                    ? `texture_storage_2d<${format}, write>`
                    : `texture_2d<f32>`;

                const currentBinding = bindingIndex++;
                wgsl += `@group(0) @binding(${currentBinding}) var ${tex.name}: ${textureType};\n`;

                passResources.textures.push({
                    ...tex,
                    format: format,
                    binding: currentBinding,
                    group: 0,
                    isStorage: isStorage
                });
            });

            // Setup Additional Sampler Bindings / 設定額外的取樣器繫結
            shaderInfo.samplers.forEach(samp => {
                if (samp.name !== 'sam') {
                    const currentBinding = bindingIndex++;
                    wgsl += `@group(0) @binding(${currentBinding}) var ${samp.name}: sampler;\n`;
                    passResources.samplers.push({...samp, binding: currentBinding, group: 0});
                }
            });
            wgsl += '\n';

            // 5. Transform Pass Logic into main_cs / 將通道邏輯轉換為進入點函數 main_cs
            let processedPassCode = pass.code.replace(/\/\/!.*\n/g, ''); // Remove meta directives / 移除元指令
            processedPassCode = this._preprocessPassCode(processedPassCode, pass.index);

            // Rename PassX function to main_cs / 將函式 PassX 重新命名為 main_cs
            processedPassCode = processedPassCode.replace(
                new RegExp(`fn Pass${pass.index}`),
                `fn main_cs `
            );

            // Inject 'blockStart' for Anime4K shaders if needed
            // 針對 Anime4K 著色器注入 'blockStart' 變數定義
            if (processedPassCode.includes('blockStart') && !processedPassCode.includes('let blockStart')) {
                processedPassCode = processedPassCode.replace(/\{/, '{\n    let blockStart = workgroup_id.xy * 8u;');
            }

            // Apply @compute attribute with workgroup size / 套用 @compute 屬性與工作群組大小
            const numThreads = pass.numThreads || [1, 1, 1];
            processedPassCode = processedPassCode.replace(/@compute\s*@workgroup_size\([^)]+\)\s*/g, '');
            processedPassCode = `@compute @workgroup_size(${numThreads[0]}, ${numThreads[1]}, ${numThreads[2]}) ${processedPassCode}`;

            wgsl += processedPassCode;

            generatedModules.push({
                wgslCode: wgsl,
                passIndex: pass.index,
                resources: passResources
            });
        });

        return generatedModules;
    }

    /**
     * Preprocess pass-specific code to fix common compatibility issues.
     * ---
     * 預處理特定通道的代碼，修正常見的 WGSL 相容性問題。
     *
     * @internal
     * @param passCode - Original WGSL code / 原始 WGSL 代碼
     * @param passIndex - Index of the current pass / 目前通道的索引
     * @returns Processed WGSL code / 處理後的 WGSL 代碼
     */
    private _preprocessPassCode(passCode: string, passIndex: number): string {
        let processedCode = passCode;

        // Fix WGSL type error: vec2<u32> + f32 -> Requires MF2 (vec2<f32>) cast
        // 修正 WGSL 型別錯誤：無法讓 vec2<u32> 直接加 f32，需透過 MF2 轉換
        processedCode = processedCode.replace(/\(gxy \+ 0\.5f\)/g, '(MF2(gxy) + 0.5f)');

        // Map common aliases to WebGPU built-ins / 將常見別名映射至 WebGPU 內建變數
        processedCode = processedCode.replace(/threadId\.x/g, 'local_id.x');

        // Legacy fixes for specific algorithm compatibility (e.g., Anime4K)
        // 針對特定演算法（如 Anime4K）的舊版相容性修正
        if (passIndex >= 1) {
            // Fix vector shift: (vec2<u32> << u32) -> (vec2<u32> << vec2<u32>(u32))
            // WGSL requires both sides of bitwise shift to be the same vector type
            // 修正向量位移運算：WGSL 要求位移運算子兩側必須是相同的向量型別
            processedCode = processedCode.replace(
                /(Rmp8x8\(local_id\.x\)\s*<<\s*)(\d+u)/g,
                '$1vec2<u32>($2)'
            );
        }

        // Fix variable redeclarations: change 'var x = max(x, ...)' to assignment
        // 修正變數重複宣告：將含有重複 var 的語句改為純賦值操作
        processedCode = processedCode.replace(
            /var\s+([a-z0-9]+)\s+=\s+max\(\1,\s+MF4\(0\.0\)\);/g,
            '$1 = max($1, MF4(0.0));'
        );

        return processedCode;
    }
}