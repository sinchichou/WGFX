// src/runtime/WGSLCodeGenerator.js
/**
 * - EN: Assembles a complete WGSL shader module from WGFX IR objects.
 * - TW: 從 WGFX IR 物件組裝完整的 WGSL 著色器模組。
 */

export class WGSLCodeGenerator {
    constructor() {
    }

    /**
     * - EN: Determines if a texture format is valid for Storage Texture write access
     * - TW: 判斷紋理格式是否有效可用於 Storage Texture 寫入存取
     */
    _isValidStorageFormat(format) {
        const validStorageFormats = [
            'r32float', 'r32sint', 'r32uint',
            'rgba16float', 'rgba16sint', 'rgba16uint',
            'rgba32float', 'rgba32sint', 'rgba32uint',
            'rg32float', 'rg32sint', 'rg32uint'
        ];
        return validStorageFormats.includes(format.toLowerCase().replace(/_/g, ''));
    }

    /**
     * - EN: Upgrades texture format to a storage-compatible format
     * - TW: 將紋理格式升級為相容於 Storage 的格式
     */
    _upgradeToStorageFormat(format) {
        const normalizedFormat = format.toLowerCase().replace(/_/g, '');

        // 8-bit formats must be upgraded to rgba16float for storage use
        if (normalizedFormat === 'rgba8unorm' || normalizedFormat === 'bgra8unorm') {
            return 'rgba16float';
        }

        // 16-bit formats are already valid
        if (normalizedFormat.includes('rgba16') || normalizedFormat.includes('r16')) {
            return normalizedFormat;
        }

        // 32-bit formats are already valid
        if (normalizedFormat.includes('32')) {
            return normalizedFormat;
        }

        // Default fallback: upgrade to rgba16float
        console.warn(`Unknown format "${format}" upgraded to rgba16float for storage texture compatibility`);
        return 'rgba16float';
    }

    /**
     * - EN: Preprocesses pass code to fix WGSL type errors and syntax issues.
     * - TW: 預處理通道程式碼以修正 WGSL 型別錯誤和語法問題。
     */
    _preprocessPassCode(passCode, passIndex) {
        let processedCode = passCode;

        // Fix WGSL type error: vec2<u32> + f32
        processedCode = processedCode.replace(/\(gxy \+ 0\.5f\)/g, '(MF2(gxy) + 0.5f)');

        // Replace threadId.x with local_id.x
        processedCode = processedCode.replace(/threadId\.x/g, 'local_id.x');

        // Apply gxy fixes based on passIndex
        if (passIndex === 1) {
            processedCode = processedCode.replace(
                /let gxy = \(Rmp8x8\(local_id\.x\) << 1u\) \+ workgroup_id\.xy;/,
                'let gxy = (Rmp8x8(local_id.x) * 2u) + workgroup_id.xy;'
            );
        } else if (passIndex >= 2 && passIndex <= 7) {
            processedCode = processedCode.replace(
                /let gxy: uint2\s*=\s*Rmp8x8\(local_id\.x\) \+ blockStart;/,
                'let gxy: uint2 = Rmp8x8(local_id.x) + workgroup_id.xy * 8u;'
            );
        } else if (passIndex === 8) {
            processedCode = processedCode.replace(
                /let gxy: uint2 = Rmp8x8\(local_id\.x\) \+ workgroup_id\.xy;/,
                'let gxy: uint2 = Rmp8x8(local_id.x) + workgroup_id.xy * 8u;'
            );
        }

        // Fix variable redeclarations
        processedCode = processedCode.replace(
            /var\s+(a1|b1|c1|d1|e1|f1|g1|h1|i1|na1|nb1|nc1|nd1|ne1|nf1|ng1|nh1|ni1|na2|nb2|nc2|nd2|ne2|nf2|ng2|nh2|ni2|na3|nb3|nc3|nd3|ne3|nf3|ng3|nh3|ni3)\s+=\s+max\(\1,\s+MF4\(0\.0\)\);/g,
            '$1 = max($1, MF4(0.0));'
        );

        return processedCode;
    }

    /**
     * - EN: Assembles final WGSL shader modules from parsed shader information (IR).
     * - TW: 從解析後的著色器資訊 (IR) 組裝最終的 WGSL 著色器模組。
     */
    generate(shaderInfo) {
        const generatedModules = [];

        // Inject common code block
        const commonCode = shaderInfo.commonCode
            ? `// --- 通用程式碼 ---\n${shaderInfo.commonCode}\n\n`
            : '';

        // Generate uniform buffer structure if parameters exist
        let uniformBufferCode = '';
        if (shaderInfo.parameters.length > 0) {
            uniformBufferCode += `struct Uniforms {\n`;
            shaderInfo.parameters.forEach(p => {
                const type = p.type === 'int' ? 'i32' : 'f32';
                uniformBufferCode += `    ${p.name}: ${type},\n`;
            });
            uniformBufferCode += `};\n`;
            uniformBufferCode += `@group(0) @binding(1) var<uniform> uniforms: Uniforms;\n\n`;
        }

        // Process each pass
        shaderInfo.passes.forEach(pass => {
            let wgsl = `// 由 WGFX 組裝器生成 - 時間戳記: ${Date.now()} - Pass ${pass.index}\n\n`;
            wgsl += commonCode;

            // Declare default sampler
            wgsl += `@group(0) @binding(0) var sam: sampler;\n\n`;
            wgsl += uniformBufferCode;

            // Initialize resource tracking
            const passResources = {
                textures: [],
                samplers: [],
                parameters: shaderInfo.parameters
            };

            // Add default sampler
            passResources.samplers.push({
                name: 'sam',
                binding: 0,
                group: 0,
                filter: 'LINEAR'
            });

            // Collect textures and samplers used by this pass
            const usedTextureNames = new Set([...pass.in, ...pass.out]);

            // Add globally declared samplers
            shaderInfo.samplers.forEach(resource => {
                if (resource.name !== 'sam' && !passResources.samplers.some(s => s.name === resource.name)) {
                    passResources.samplers.push(resource);
                }
            });

            // Allocate bindings (sam:0, uniforms:1 if exists, then textures/samplers from 2 or 1)
            let bindingIndex = (shaderInfo.parameters.length > 0) ? 2 : 1;

            // Process textures
            shaderInfo.textures.forEach(tex => {
                // Check if this texture is used by this pass
                const isUsed = [...usedTextureNames].some(usedName =>
                    tex.name === usedName || tex.name.startsWith(usedName + "_")
                );

                if (!isUsed) return;

                // Determine texture usage (Sampled vs Storage)
                const isOutputInThisPass = pass.out.includes(tex.name);
                let isStorage = false;

                if (tex.name === 'OUTPUT') {
                    isStorage = true;
                } else if (tex.name === 'INPUT') {
                    isStorage = false; // INPUT is always read-only (Sampled)
                } else if (isOutputInThisPass) {
                    isStorage = true; // Output textures must be Storage
                } else {
                    isStorage = false; // Other cases (inputs) are Sampled
                }

                // Process format and compatibility
                let format = (tex.format || 'rgba8unorm').toLowerCase().replace(/_/g, '');

                // CRITICAL FIX: Upgrade format for storage textures
                if (isStorage) {
                    const originalFormat = format;
                    format = this._upgradeToStorageFormat(format);

                    if (originalFormat !== format) {
                        console.log(`Pass ${pass.index}: Upgraded texture "${tex.name}" format from ${originalFormat} to ${format} for storage compatibility`);
                    }

                    // Double-check format validity
                    if (!this._isValidStorageFormat(format)) {
                        console.error(`Pass ${pass.index}: Invalid storage format "${format}" for texture "${tex.name}"`);
                        throw new Error(`Texture "${tex.name}" has invalid storage format "${format}". Valid formats: r32float, rgba16float, rgba32float, etc.`);
                    }
                }

                // Generate WGSL texture declaration
                let textureType;
                if (isStorage) {
                    textureType = `texture_storage_2d<${format}, write>`;
                } else {
                    textureType = `texture_2d<f32>`;
                }

                const currentBinding = bindingIndex++;
                wgsl += `@group(0) @binding(${currentBinding}) var ${tex.name}: ${textureType};\n`;

                // Store resource info with corrected format
                passResources.textures.push({
                    ...tex,
                    format: format, // Use corrected format
                    binding: currentBinding,
                    group: 0,
                    isStorage: isStorage
                });
            });

            // Declare additional samplers (excluding 'sam')
            shaderInfo.samplers.forEach(samp => {
                if (samp.name !== 'sam') {
                    const currentBinding = bindingIndex++;
                    wgsl += `@group(0) @binding(${currentBinding}) var ${samp.name}: sampler;\n`;
                    passResources.samplers.push({
                        ...samp,
                        binding: currentBinding,
                        group: 0
                    });
                }
            });
            wgsl += '\n';

            // Append pass code
            wgsl += `// --- 通道 ${pass.index} ---\n`;

            // Remove //! directives and preprocess
            let processedPassCode = pass.code.replace(/\/\/!.*\n/g, '');
            processedPassCode = this._preprocessPassCode(processedPassCode, pass.index);

            // Set workgroup size
            const numThreads = pass.numThreads || [1, 1, 1];

            // Remove existing @compute or @workgroup_size attributes
            processedPassCode = processedPassCode.replace(/@compute\s*@workgroup_size\([^)]+\)\s*/g, '');

            // Standardize entry point name
            const passFunctionSignatureRegex = new RegExp(`fn Pass${pass.index}`);
            processedPassCode = processedPassCode.replace(
                passFunctionSignatureRegex,
                `fn main_cs `
            );

            // Prepend attributes
            processedPassCode = `@compute @workgroup_size(${numThreads[0]}, ${numThreads[1]}, ${numThreads[2]}) ${processedPassCode}`;

            wgsl += processedPassCode;

            // Store generated module
            generatedModules.push({
                wgslCode: wgsl.replace(/\r\n/g, '\n'),
                passIndex: pass.index,
                resources: passResources
            });
        });

        return generatedModules;
    }
}