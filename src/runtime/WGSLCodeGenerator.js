// src/runtime/WGSLCodeGenerator.js
/**
 * - EN: Assembles a complete WGSL shader module from WGFX IR objects.
 * This class does not translate HLSL to WGSL; based on user feedback,
 * it assumes each pass in the IR already contains valid WGSL code.
 * Responsibilities:
 *   1. Generate resource bindings (@group, @binding).
 *   2. Create a uniform buffer structure for parameters.
 *   3. Inject common code blocks.
 *   4. Wrap each pass's WGSL code into a properly attributed entry-point function.
 *
 * - TW: 從 WGFX IR 物件組裝完整的 WGSL 著色器模組。
 * 此類別不負責將 HLSL 轉換為 WGSL；根據使用者回饋，
 * 假設 IR 中每個 pass 的程式碼本身已為有效 WGSL。
 * 職責包括：
 *   1. 生成資源綁定（@group、@binding）。
 *   2. 建立參數用的 uniform buffer 結構。
 *   3. 注入通用程式碼區塊。
 *   4. 將每個 pass 的 WGSL 程式碼包裝成具正確屬性的入口函式。
 */

export class WGSLCodeGenerator {
    /**
     * - EN: Constructs a new WGSLCodeGenerator instance.
     * - TW: 建構一個新的 WGSLCodeGenerator 實例。
     */
    constructor() {
        /**
         * - EN: No complex translation logic needed.
         * - TW: 不需要複雜的轉譯邏輯。
         */
    }

    /**
     * - EN: Preprocesses pass code to fix WGSL type errors and syntax issues.
     * Applies pass-specific transformations based on pass index.
     * - TW: 根據通道索引套用特定的轉換。
     * 預處理通道程式碼以修正 WGSL 型別錯誤和語法問題。
     *
     * @param {string} passCode
     * - EN: The raw WGSL code for the pass.
     * - TW: 通道的原始 WGSL 程式碼。
     * @param {number} passIndex
     * - EN: The index of the pass being processed.
     * - TW: 正在處理的通道索引。
     * @returns {string}
     * - EN: The processed WGSL code with fixes applied.
     * - TW: 套用修正後的處理過的 WGSL 程式碼。
     * @private
     */
    _preprocessPassCode(passCode, passIndex) {
        let processedCode = passCode;

        /**
         * - EN: Fix WGSL type error: vec2<u32> + f32
         * - TW: 修正 WGSL 型別錯誤: vec2<u32> + f32
         */
        processedCode = processedCode.replace(/\(gxy \+ 0\.5f\)/g, '(MF2(gxy) + 0.5f)');

        /**
         * - EN: Replace threadId.x with local_id.x
         * - TW: 將 threadId.x 替換為 local_id.x
         */
        processedCode = processedCode.replace(/threadId\.x/g, 'local_id.x');

        /**
         * - EN: Apply gxy fixes based on passIndex
         * - TW: 根據 passIndex 套用 gxy 修正
         */
        if (passIndex === 1) {
            /**
             * - EN: Pass1: Change bitshift to multiplication for WGSL compatibility
             * - TW: Pass1: 將位元移位改為乘法以符合 WGSL 相容性
             */
            processedCode = processedCode.replace(
                /let gxy = \(Rmp8x8\(local_id\.x\) << 1u\) \+ workgroup_id\.xy;/,
                'let gxy = (Rmp8x8(local_id.x) * 2u) + workgroup_id.xy;'
            );
        } else if (passIndex >= 2 && passIndex <= 7) {
            /**
             * - EN: Pass2-Pass7: Replace blockStart with workgroup_id calculation
             * - TW: Pass2-Pass7: 將 blockStart 替換為 workgroup_id 計算
             */
            processedCode = processedCode.replace(
                /let gxy: uint2\s*=\s*Rmp8x8\(local_id\.x\) \+ blockStart;/,
                'let gxy: uint2 = Rmp8x8(local_id.x) + workgroup_id.xy * 8u;'
            );
        } else if (passIndex === 8) {
            /**
             * - EN: Pass8: Adjust workgroup_id calculation for proper indexing
             * - TW: Pass8: 調整 workgroup_id 計算以正確索引
             */
            processedCode = processedCode.replace(
                /let gxy: uint2 = Rmp8x8\(local_id\.x\) \+ workgroup_id\.xy;/,
                'let gxy: uint2 = Rmp8x8(local_id.x) + workgroup_id.xy * 8u;'
            );
        }

        /**
         * - EN: Fix variable redeclarations within a pass for specific patterns.
         * This targets lines like 'var a1 = max(a1, MF4(0.0));' where the variable is already declared.
         * - TW: 這針對像 'var a1 = max(a1, MF4(0.0));' 這樣的行,其中變數已經被宣告
         * 修正通道內特定模式的變數重複宣告。
         */
        processedCode = processedCode.replace(
            /var\s+(a1|b1|c1|d1|e1|f1|g1|h1|i1|na1|nb1|nc1|nd1|ne1|nf1|ng1|nh1|ni1|na2|nb2|nc2|nd2|ne2|nf2|ng2|nh2|ni2|na3|nb3|nc3|nd3|ne3|nf3|ng3|nh3|ni3)\s+=\s+max\(\1,\s+MF4\(0\.0\)\);/g,
            '$1 = max($1, MF4(0.0));'
        );

        return processedCode;
    }

    /**
     * - EN: Assembles a final WGSL shader module from parsed shader information (IR).
     * Generates separate shader modules for each pass with appropriate resource bindings.
     * - TW: 為每個通道生成獨立的著色器模組,包含適當的資源綁定。
     * 從解析後的著色器資訊 (IR) 組裝一個最終的 WGSL 著色器模組。
     *
     * @param {import('./Parser.js').WGFXShaderInfo} shaderInfo
     * - EN: The parsed shader information (IR).
     * - TW: 解析後的著色器資訊 (IR)。
     * @returns {Array<{wgslCode: string, passIndex: number, resources: {textures: import('./Parser.js').WGFXTexture[], samplers: import('./Parser.js').WGFXSampler[], parameters: import('./Parser.js').WGFXParameter[]}}>}
     * - EN: Array of generated WGSL shader modules, one per pass, with associated resource metadata.
     * - TW: 生成的 WGSL 著色器模組陣列,每個通道一個模組,包含相關的資源中繼資料。
     */
    generate(shaderInfo) {
        const generatedModules = [];

        /**
         * - EN: Inject common code block. Replace 'type' with 'alias' for WGSL compatibility if present.
         * - TW: 注入通用程式碼區塊。如果存在,將 'type' 替換為 'alias' 才能符合 WGSL 相容性。
         */
        const commonCode = shaderInfo.commonCode
            ? `// --- 通用程式碼 ---\n${shaderInfo.commonCode}\n\n`
            : '';

        /**
         * - EN: Generate uniform buffer structure and binding (if any parameters exist)
         * - TW: 生成統一緩衝區結構和綁定 (如果有參數的話)
         */
        let uniformBufferCode = '';
        if (shaderInfo.parameters.length > 0) {
            uniformBufferCode += `struct Uniforms {\n`;
            shaderInfo.parameters.forEach(p => {
                const type = p.type === 'int' ? 'i32' : 'f32';
                uniformBufferCode += `    ${p.name}: ${type},\n`;
            });
            uniformBufferCode += `};\n`;
            /**
             * - EN: Uniforms always bound to group 0, binding 1 (binding 0 is reserved for 'sam' sampler)
             * - TW: Uniforms 始終綁定到 group 0, binding 1 (binding 0 保留給 'sam' 取樣器)
             */
            uniformBufferCode += `@group(0) @binding(1) var<uniform> uniforms: Uniforms;\n\n`;
        }

        /**
         * - EN: Process each pass to generate individual shader modules
         * - TW: 處理每個通道以生成個別的著色器模組
         */
        shaderInfo.passes.forEach(pass => {
            /**
             * - EN: Add generation timestamp and pass identifier
             * - TW: 加入生成時間戳記和通道識別碼
             */
            let wgsl = `// 由 WGFX 組裝器生成 - 獨特註解: ${Date.now()} - Pass ${pass.index}\n\n`;
            wgsl += commonCode;

            /**
             * - EN: Declare default sampler 'sam' (always bound to binding 0)
             * - TW: 宣告預設取樣器 'sam' (始終綁定到 binding 0)
             */
            wgsl += `@group(0) @binding(0) var sam: sampler;\n\n`;

            wgsl += uniformBufferCode;

            /**
             * - EN: Initialize resource tracking for this pass
             * - TW: 初始化此通道的資源追蹤
             */
            const passResources = {
                textures: [],
                samplers: [],
                parameters: shaderInfo.parameters
            };

            /**
             * - EN: Add default sampler 'sam' to resource list
             * - TW: 將預設取樣器 'sam' 加入資源列表
             */
            passResources.samplers.push({ name: 'sam', binding: 0, group: 0, filter: 'LINEAR' });

            /**
             * - EN: Collect textures and samplers used by this pass
             * - TW: 收集此通道使用的紋理和取樣器
             */
            const usedTextureNames = new Set([...pass.in, ...pass.out]);
            const usedSamplerNames = new Set(pass.in.filter(name => shaderInfo.samplers.some(s => s.name === name)));

            /**
             * - EN: Automatically add globally declared samplers to every pass
             * - TW: 自動將全域宣告的取樣器加入每個通道
             */
            shaderInfo.samplers.forEach(resource => {
                /**
                 * - EN: Avoid duplicates and skip 'sam' as it's already added
                 * - TW: 避免重複並跳過 'sam',因為它已經被加入
                 */
                if (resource.name !== 'sam' && !passResources.samplers.some(s => s.name === resource.name)) {
                    passResources.samplers.push(resource);
                }
            });

            /**
             * - EN: Allocate bindings for textures and samplers in this pass.
             * Start from binding 2 if uniforms exist (sam:0, uniforms:1), otherwise binding 1 (sam:0).
             * - TW: 如果有 uniforms,從 binding 2 開始 (sam:0, uniforms:1),否則從 binding 1 開始 (sam:0)。
             * 為此通道的紋理和取樣器分配綁定。
             */
            let bindingIndex = (shaderInfo.parameters.length > 0) ? 2 : 1;

            /**
             * - EN: Declare textures used by this pass
             * - TW: 宣告此通道使用的紋理
             */
            shaderInfo.textures.forEach(tex => {
                const isUsed = [...usedTextureNames].some(usedName => tex.name === usedName || tex.name.startsWith(usedName + "_"));
                if (isUsed) {
                    /**
                     * - EN: Determine if texture is storage or sampled based on naming conventions
                     * - TW: 根據命名慣例判斷紋理是儲存紋理還是取樣紋理
                     */
                    const isStorage = tex.name.endsWith('_storaged') || tex.name === 'OUTPUT';
                    const format = (tex.format || 'rgba8unorm').toLowerCase().replace(/_/g, '');
                    let textureType;

                    if (tex.name.endsWith('_sampled')) {
                        /**
                         * - EN: Heuristic: sampled textures are texture_2d<f32>
                         * - TW: 啟發式: 取樣紋理是 texture_2d<f32>
                         */
                        textureType = 'texture_2d<f32>';
                    } else if (tex.name.endsWith('_storaged')) {
                        /**
                         * - EN: Heuristic: storaged textures are texture_storage_2d
                         * - TW: 啟發式: 儲存紋理是 texture_storage_2d
                         */
                        textureType = `texture_storage_2d<${format}, write>`;
                    } else if (tex.name === 'INPUT') {
                        textureType = 'texture_2d<f32>';
                    } else if (tex.name === 'OUTPUT') {
                        textureType = `texture_storage_2d<${format}, write>`;
                    } else {
                        /**
                         * - EN: Fallback: use isStorage flag from pass usage if no clear naming convention
                         * - TW: 後備方案: 如果沒有明確的命名慣例,使用通道使用的 isStorage 標誌
                         */
                        textureType = isStorage ? `texture_storage_2d<${format}, write>` : `texture_2d<f32>`;
                    }

                    const currentBinding = bindingIndex++;
                    wgsl += `@group(0) @binding(${currentBinding}) var ${tex.name}: ${textureType};\n`;
                    passResources.textures.push({ ...tex, binding: currentBinding, group: 0, isStorage: isStorage });
                }
            });

            /**
             * - EN: Declare samplers used by this pass (excluding `sam` as it's already handled)
             * - TW: 宣告此通道使用的取樣器 (除了 `sam`,因為它已經被處理了)
             */
            shaderInfo.samplers.forEach(samp => {
                if (samp.name !== 'sam' && usedSamplerNames.has(samp.name)) {
                    const currentBinding = bindingIndex++;
                    wgsl += `@group(0) @binding(${currentBinding}) var ${samp.name}: sampler;\n`;
                    passResources.samplers.push({ ...samp, binding: currentBinding, group: 0 });
                }
            });
            wgsl += '\n';

            /**
             * - EN: Append the pass's WGSL code to the module
             * - TW: 將通道的 WGSL 程式碼附加到模組中
             */
            wgsl += `// --- 通道 ${pass.index} ---\n`;

            /**
             * - EN: Remove any remaining //! directives
             * - TW: 移除任何剩餘的 //! 指令
             */
            let processedPassCode = pass.code.replace(/\/!.*\n/g, '');
            processedPassCode = this._preprocessPassCode(processedPassCode, pass.index);

            /**
             * - EN: Default workgroup size in case not specified
             * - TW: 預設工作組大小以防未指定
             */
            const numThreads = pass.numThreads || [1, 1, 1];

            /**
             * - EN: Remove any existing @compute or @workgroup_size attributes from the passCode
             * - TW: 移除 passCode 中任何現有的 @compute 或 @workgroup_size 屬性
             */
            processedPassCode = processedPassCode.replace(/@compute\s*@workgroup_size\([^)]+\)\s*/g, '');

            /**
             * - EN: Replace fn PassX with fn main_cs (standardized entry point)
             * - TW: 將 fn PassX 替換為 fn main_cs (標準化進入點)
             */
            const passFunctionSignatureRegex = new RegExp(`fn Pass${pass.index}`);
            processedPassCode = processedPassCode.replace(
                passFunctionSignatureRegex,
                `fn main_cs `
            );

            /**
             * - EN: Prepend the @compute @workgroup_size attributes
             * - TW: 在前面加上 @compute @workgroup_size 屬性
             */
            processedPassCode = `@compute @workgroup_size(${numThreads[0]}, ${numThreads[1]}, ${numThreads[2]}) ${processedPassCode}`;

            wgsl += processedPassCode;

            /**
             * - EN: Store the generated module with metadata
             * - TW: 儲存生成的模組及其中繼資料
             */
            generatedModules.push({
                wgslCode: wgsl.replace(/\r\n/g, '\n'),
                passIndex: pass.index,
                resources: passResources
            });
        });

        return generatedModules;
    }
}