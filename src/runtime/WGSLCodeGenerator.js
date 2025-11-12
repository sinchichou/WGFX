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

    _preprocessPassCode(passCode, passIndex) {
        let processedCode = passCode;

        // Replace threadId.x with local_id.x
        processedCode = processedCode.replace(/threadId\.x/g, 'local_id.x');

        // Apply gxy fixes based on passIndex
        if (passIndex === 1) {
            // Pass1: let gxy = (Rmp8x8(local_id.x) << 1u) + workgroup_id.xy;
            // Change to: let gxy = (Rmp8x8(local_id.x) * 2u) + workgroup_id.xy;
            processedCode = processedCode.replace(
                /let gxy = \(Rmp8x8\(local_id\.x\) << 1u\) \+ workgroup_id\.xy;/,
                'let gxy = (Rmp8x8(local_id.x) * 2u) + workgroup_id.xy;'
            );
        } else if (passIndex >= 2 && passIndex <= 7) {
            // Pass2-Pass7: let gxy: uint2 = Rmp8x8(local_id.x) + blockStart;
            // Change to: let gxy: uint2 = Rmp8x8(local_id.x) + workgroup_id.xy * 8u;
            processedCode = processedCode.replace(
                /let gxy: uint2 = Rmp8x8\(local_id\.x\) \+ blockStart;/,
                'let gxy: uint2 = Rmp8x8(local_id.x) + workgroup_id.xy * 8u;'
            );
        } else if (passIndex === 8) {
            // Pass8: let gxy: uint2 = Rmp8x8(local_id.x) + workgroup_id.xy;
            // Change to: let gxy: uint2 = Rmp8x8(local_id.x) + workgroup_id.xy * 8u;
            processedCode = processedCode.replace(
                /let gxy: uint2 = Rmp8x8\(local_id\.x\) \+ workgroup_id\.xy;/,
                'let gxy: uint2 = Rmp8x8(local_id.x) + workgroup_id.xy * 8u;'
            );
        }

        // Fix variable redeclarations within a pass for specific patterns
        // This targets lines like 'var a1 = max(a1, MF4(0.0));' where the variable is already declared.
        processedCode = processedCode.replace(
            /var\s+(a1|b1|c1|d1|e1|f1|g1|h1|i1|na1|nb1|nc1|nd1|ne1|nf1|ng1|nh1|ni1|na2|nb2|nc2|nd2|ne2|nf2|ng2|nh2|ni2|na3|nb3|nc3|nd3|ne3|nf3|ng3|nh3|ni3)\s+=\s+max\(\1,\s+MF4\(0\.0\)\);/g,
            '$1 = max($1, MF4(0.0));'
        );

        return processedCode;
    }

    /**
     * 從解析後的著色器資訊 (IR) 組裝一個最終的 WGSL 著色器模組。
     * @param {import('./Parser.js').WGFXShaderInfo} shaderInfo - 解析後的著色器資訊。
     * @returns {Array<{ wgslCode: string, passIndex: number, resources: { textures: import('./Parser.js').WGFXTexture[], samplers: import('./Parser.js').WGFXSampler[], parameters: import('./Parser.js').WGFXParameter[] } }>} 生成的 WGSL 著色器程式碼作為單一字串。
     */
    generate(shaderInfo) {
        const generatedModules = [];

        // 注入通用程式碼區塊。
        const commonCode = shaderInfo.commonCode ? `// --- 通用程式碼 ---\n${shaderInfo.commonCode.replace(/type\s+/g, 'alias ')}\n\n` : '';

        // 生成統一緩衝區結構和綁定 (如果有的話)
        let uniformBufferCode = '';
        if (shaderInfo.parameters.length > 0) {
            uniformBufferCode += `struct Uniforms {\n`;
            shaderInfo.parameters.forEach(p => {
                const type = p.type === 'int' ? 'i32' : 'f32';
                uniformBufferCode += `    ${p.name}: ${type},\n`;
            });
            uniformBufferCode += `};
`;
            // Uniforms 始終綁定到 group 0, binding 0
            uniformBufferCode += `@group(0) @binding(0) var<uniform> uniforms: Uniforms;\n\n`;
        }

        shaderInfo.passes.forEach(pass => {
            let wgsl = `// 由 WGFX 組裝器生成 - 獨特註解: ${Date.now()} - Pass ${pass.index}\n\n`;
            wgsl += commonCode;
            wgsl += uniformBufferCode;

            const passResources = {
                textures: [],
                samplers: [],
                parameters: shaderInfo.parameters // Parameters are global for now
            };

            // 為此通道的紋理和取樣器分配綁定
            let bindingIndex = (shaderInfo.parameters.length > 0) ? 1 : 0; // 如果有 uniforms，從 binding 1 開始

            // 收集此通道使用的紋理和取樣器
            const usedTextureNames = new Set([...pass.in, ...pass.out]);
            const usedSamplerNames = new Set(pass.in.filter(name => shaderInfo.samplers.some(s => s.name === name))); // Samplers are only 'in'

            // 宣告此通道使用的紋理
            shaderInfo.textures.forEach(tex => {
                if (usedTextureNames.has(tex.name)) {
                    const isStorage = pass.out.includes(tex.name) || shaderInfo.passes.some(p => p.out.includes(tex.name) && p.index < pass.index && usedTextureNames.has(tex.name));
                    const format = (tex.format || 'rgba8unorm').toLowerCase().replace(/_/g, '');

                    if (isStorage) {
                        wgsl += `@group(0) @binding(${bindingIndex}) var ${tex.name}: texture_storage_2d<${format}, write>;\n`;
                    } else {
                        wgsl += `@group(0) @binding(${bindingIndex}) var ${tex.name}: texture_2d<f32>;\n`;
                    }
                    passResources.textures.push({ ...tex, binding: bindingIndex, group: 0, isStorage: isStorage });
                    bindingIndex++;
                }
            });

            // 宣告此通道使用的取樣器
            shaderInfo.samplers.forEach(samp => {
                if (usedSamplerNames.has(samp.name)) {
                    wgsl += `@group(0) @binding(${bindingIndex}) var ${samp.name}: sampler;\n`;
                    passResources.samplers.push({ ...samp, binding: bindingIndex, group: 0 });
                    bindingIndex++;
                }
            });
            wgsl += '\n';

            // 將通道的 WGSL 程式碼附加到模組中
            wgsl += `// --- 通道 ${pass.index} ---\n`;
            const passCode = pass.code.replace(/\/!.*\n/g, ''); // 移除任何剩餘的 //! 指令
            wgsl += `${this._preprocessPassCode(passCode, pass.index)}\n\n`;

            generatedModules.push({
                wgslCode: wgsl.replace(/\r\n/g, '\n'),
                passIndex: pass.index,
                resources: passResources
            });
        });

        return generatedModules;
    }
}
