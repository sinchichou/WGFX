import { WGFXShaderInfo, PassInfo, TextureInfo } from '../types/shader';

export class WGSLCodeGenerator {
    constructor() {}

    private _preprocessPassCode(passCode: string, passIndex: number): string {
        let processedCode = passCode;

        // Fix WGSL type error: vec2<u32> + f32
        processedCode = processedCode.replace(/\(gxy \+ 0\.5f\)/g, '(MF2(gxy) + 0.5f)');
        processedCode = processedCode.replace(/threadId\.x/g, 'local_id.x');

        // Fix undefined 'blockStart'
        // This is a common variable in Anime4K shaders, usually derived from workgroup_id
        // We handle injection in generate() for robustness.
        
        // Better strategy: Replace usage if it's a simple addition
        // Or ensure we inject the let definition after the function opening brace.
        
        // Legacy fixes (simplified for readability, keeping core logic)
        if (passIndex >= 1) {
             // Fix vector shift: (vec2<u32> << u32) -> (vec2<u32> << vec2<u32>(u32))
             // Matches: (Rmp8x8(local_id.x) << 1u)
             processedCode = processedCode.replace(
                /(Rmp8x8\(local_id\.x\)\s*<<\s*)(\d+u)/g,
                '$1vec2<u32>($2)'
             );
        }

        // Fix variable redeclarations
        processedCode = processedCode.replace(
            /var\s+([a-z0-9]+)\s+=\s+max\(\1,\s+MF4\(0\.0\)\);/g,
            '$1 = max($1, MF4(0.0));'
        );

        return processedCode;
    }

    public generate(shaderInfo: WGFXShaderInfo): { wgslCode: string; passIndex: number; resources: any }[] {
        const generatedModules: { wgslCode: string; passIndex: number; resources: any }[] = [];

        const commonCode = shaderInfo.commonCode
            ? `// --- COMMON ---\n${shaderInfo.commonCode}\n\n`
            : '';

        let uniformBufferCode = '';
        if (shaderInfo.parameters.length > 0) {
            uniformBufferCode += `struct Uniforms {\n`;
            shaderInfo.parameters.forEach(p => {
                const type = p.type === 'int' ? 'i32' : 'f32';
                uniformBufferCode += `    ${p.name}: ${type},\n`;
                // Add padding if necessary for std140 (simplified behavior here)
                uniformBufferCode += `    _pad_${p.name}_1: f32,\n`;
                uniformBufferCode += `    _pad_${p.name}_2: f32,\n`;
                uniformBufferCode += `    _pad_${p.name}_3: f32,\n`;
            });
            uniformBufferCode += `};\n`;
            uniformBufferCode += `@group(0) @binding(1) var<uniform> uniforms: Uniforms;\n\n`;
        }
        // SceneInfo struct and binding (binding 4)
        const sceneInfoCode = `struct SceneInfo {
    inputSize: vec2<u32>,
    inputPt: vec2<f32>,
}
@group(0) @binding(4) var<uniform> scene: SceneInfo;

// `;

        shaderInfo.passes.forEach(pass => {
            let wgsl = `// Pass ${pass.index}\n\n`;
            wgsl += commonCode;
            
            // Only inject default sampler if not already defined in COMMON
            if (!commonCode.includes('var sam:') && !commonCode.includes('var sam :')) {
                wgsl += `@group(0) @binding(0) var sam: sampler;\n\n`;
            }
            
            wgsl += uniformBufferCode;
            
            // Only inject SceneInfo if not already defined in COMMON
            if (!commonCode.includes('SceneInfo')) {
                wgsl += sceneInfoCode;
            }

            const passResources = {
                textures: [] as any[],
                samplers: [] as any[],
                parameters: shaderInfo.parameters,
                hasScene: true
            };

            passResources.samplers.push({ name: 'sam', binding: 0, group: 0 });

            // Add other samplers
             shaderInfo.samplers.forEach(resource => {
                if (resource.name !== 'sam' && !passResources.samplers.some(s => s.name === resource.name)) {
                    passResources.samplers.push(resource);
                }
            });

            // Start bindings at 6 to avoid collision with SceneInfo (Binding 4)
            // 0: Sampler, 1: Uniforms, 4: SceneInfo
            // Reserved: 2, 3, 5 (Future use?)
            let bindingIndex = 6; 

            const usedTextureNames = new Set([...pass.in, ...pass.out]);

            shaderInfo.textures.forEach(tex => {
                const isUsed = [...usedTextureNames].some(usedName =>
                    tex.name === usedName || tex.name.startsWith(usedName + "_")
                );
                if (!isUsed) return;

                const isOutputInThisPass = pass.out.includes(tex.name);
                let isStorage = false;

                if (tex.name === 'OUTPUT') isStorage = true;
                else if (tex.name === 'INPUT') isStorage = false;
                else if (isOutputInThisPass) isStorage = true;
                else isStorage = false;

                // Unified format
                const format = 'rgba16float';
                
                let textureType;
                if (isStorage) {
                    textureType = `texture_storage_2d<${format}, write>`;
                } else {
                    textureType = `texture_2d<f32>`;
                }

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

             // Samplers bindings
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

            // Pass code
            let processedPassCode = pass.code.replace(/\/\/!.*\n/g, '');
            processedPassCode = this._preprocessPassCode(processedPassCode, pass.index);
            
            const numThreads = pass.numThreads || [1, 1, 1];
            processedPassCode = processedPassCode.replace(/@compute\s*@workgroup_size\([^)]+\)\s*/g, '');
            processedPassCode = processedPassCode.replace(
                new RegExp(`fn Pass${pass.index}`),
                `fn main_cs `
            );
            
            // Inject 'blockStart' definition if used
            if (processedPassCode.includes('blockStart') && !processedPassCode.includes('let blockStart')) {
                 // Try to inject execution logic at start of function
                 processedPassCode = processedPassCode.replace(/\{/, '{\n    let blockStart = workgroup_id.xy * 8u;');
            }

            processedPassCode = `@compute @workgroup_size(${numThreads[0]}, ${numThreads[1]}, ${numThreads[2]}) ${processedPassCode}`;
            
            wgsl += processedPassCode;

            generatedModules.push({
                wgslCode: wgsl,
                passIndex: pass.index,
                resources: passResources
            });
            console.log(`--- Generated WGSL for Pass ${pass.index} ---`);
            console.log(wgsl);
            console.log(`-------------------------------------------`);
        });

        return generatedModules;
    }
}
