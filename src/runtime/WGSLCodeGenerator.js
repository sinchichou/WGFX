// src/runtime/WGSLCodeGenerator.js

/**
 * @fileoverview 從 WGFX IR 物件組裝一個完整的 WGSL 著色器模組。
 *
 * 根據使用者回饋，此類別不將 HLSL 翻譯為 WGSL。
 * 它假設 IR 中每個通道的程式碼已經是有效的 WGSL。
 * 其主要職責是：
 * 1. 生成資源綁定 (`@group` 和 `@binding`)。
 * 2. 為參數建立一個統一緩衝區結構。
 * 3. 注入通用程式碼區塊。
 * 4. 將每個通道的 WGSL 程式碼包裝在一個正確屬性的入口點函數中。
 */

export class WGSLCodeGenerator {
    constructor() {
        // 不需要複雜的轉譯邏輯。
    }

    /**
     * 從解析後的著色器資訊 (IR) 組裝一個最終的 WGSL 著色器模組。
     * @param {import('./Parser.js').WGFXShaderInfo} shaderInfo - 解析後的著色器資訊。
     * @returns {string} 生成的 WGSL 著色器程式碼作為單一字串。
     */
    generate(shaderInfo) {
        let wgsl = `// 由 WGFX 組裝器生成\n\n`;
        const resourceBindings = new Map();
        let bindingIndex = 0;

        /**
         * 為命名資源獲取或建立一個唯一的、連續的綁定索引。
         * @param {string} name - 資源的名稱 (例如，紋理或 "uniforms")。
         * @returns {number} 綁定索引。
         */
        const getBinding = (name) => {
            if (!resourceBindings.has(name)) {
                resourceBindings.set(name, {index: bindingIndex++});
            }
            return resourceBindings.get(name).index;
        };

        // 1. 生成統一緩衝區結構和綁定
        // 所有參數都打包到綁定 0 的單個統一緩衝區中。
        if (shaderInfo.parameters.length > 0) {
            const uniformBinding = getBinding('uniforms'); // 應該是 0
            wgsl += `struct Uniforms {
`;
            shaderInfo.parameters.forEach(p => {
                const type = p.type === 'int' ? 'i32' : 'f32';
                wgsl += `    ${p.name}: ${type},\n`;
            });
            wgsl += `};
`;
            wgsl += `@group(0) @binding(${uniformBinding}) var<uniform> uniforms: Uniforms;\n\n`;
        }

        // 2. 預先宣告所有紋理和取樣器以建立其綁定索引。
        // 這確保了整個著色器模組中綁定佈局的一致性。
        shaderInfo.textures.forEach(tex => getBinding(tex.name));
        shaderInfo.samplers.forEach(samp => getBinding(samp.name));

        // 3. 為所有紋理和取樣器寫入 WGSL 宣告及其分配的綁定。
        shaderInfo.textures.forEach(tex => {
            const binding = getBinding(tex.name);
            // 如果紋理曾被用作輸出，則將其視為「儲存」紋理。
            const isStorage = shaderInfo.passes.some(p => p.out.includes(tex.name));

            if (isStorage) {
                // 將 IR 中的格式映射到有效的 WGSL 儲存紋理格式。
                const format = (tex.format || 'rgba8unorm').toLowerCase().replace(/_/g, '');
                wgsl += `@group(0) @binding(${binding}) var ${tex.name}: texture_storage_2d<${format}, write>;\n`;
            } else {
                // 否則，它是採樣紋理。
                wgsl += `@group(0) @binding(${binding}) var ${tex.name}: texture_2d<f32>;\n`;
            }
        });

        shaderInfo.samplers.forEach(samp => {
            const binding = getBinding(samp.name);
            wgsl += `@group(0) @binding(${binding}) var ${samp.name}: sampler;\n`;
        });
        wgsl += '\n';

        // 4. 注入通用程式碼區塊。
        if (shaderInfo.commonCode) {
            wgsl += `// --- 通用程式碼 ---\n`;
            wgsl += `${shaderInfo.commonCode}\n\n`;
        }

        // 5. 為每個通道生成入口點函數。
        shaderInfo.passes.forEach(pass => {
            wgsl += `// --- 通道 ${pass.index} ---\n`;
            const workgroupSize = pass.numThreads || [1, 1, 1];
            wgsl += `@compute @workgroup_size(${workgroupSize.join(', ')})\n`;
            wgsl += `fn pass_${pass.index}(@builtin(global_invocation_id) global_id: vec3<u32>) {\n`;

            // 通道程式碼假定為 WGSL。我們只刪除任何剩餘的 `//!` 指令。
            const passCode = pass.code.replace(/\/!.*\n/g, '');

            // 縮排並將使用者的通道程式碼添加到函數主體中。
            wgsl += `    ${passCode.split('\n').join('\n    ')}\n`;
            wgsl += `}\n\n`;
        });

        return wgsl;
    }
}
