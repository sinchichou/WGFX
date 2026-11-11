/**
 * HLSTranspiler — Transpile HLSL (Magpie-style) to WGSL using WGFX parser/generator.
 * 
 * Reuses the existing ShaderParser (PEG.js) and WGSLCodeGenerator to convert
 * Magpie v4 HLSL effects into executable WGSL compute shader modules.
 */

import { parse } from '../runtime/ShaderParser';
import { WGSLCodeGenerator } from '../runtime/WGSLCodeGenerator';
import { WGFXShaderInfo } from '../types';

export interface TranspileOptions {
    /** External macro definitions (e.g., INPUT_WIDTH, INPUT_HEIGHT) */
    defines?: Record<string, number>;
    /** Whether to honor declared output sizes from the effect */
    honorDeclaredOutput?: boolean;
}

export interface TranspileResult {
    /** Original parsed shader info */
    shaderInfo: WGFXShaderInfo;
    /** Generated WGSL modules per pass */
    modules: GeneratedModule[];
    /** Any diagnostics/warnings during transpilation */
    diagnostics: TranspileDiagnostic[];
    /** The raw HLSL source for reference */
    source: string;
}

export interface TranspileDiagnostic {
    severity: 'info' | 'warning' | 'error';
    message: string;
    passIndex?: number;
    line?: number;
    column?: number;
}

export interface GeneratedModule {
    wgslCode: string;
    passIndex: number;
    resources: ModuleResources;
}

export interface ModuleResources {
    textures: Array<{
        name: string;
        logicalName: string;
        format: string;
        binding: number;
        group: number;
        isStorage: boolean;
    }>;
    samplers: Array<{
        name: string;
        binding: number;
        group: number;
    }>;
    storageBuffers: Array<{
        name: string;
        binding: number;
        group: number;
        mode: string;
        type: string;
    }>;
    parameters: any[];
    hasScene: boolean;
    sceneBinding: number;
}

/**
 * HLSTranspiler converts Magpie-style HLSL to WGSL compute shaders.
 */
export class HLSTranspiler {
    private codeGenerator: WGSLCodeGenerator;

    constructor() {
        this.codeGenerator = new WGSLCodeGenerator();
    }

    /**
     * Transpile HLSL source code to WGSL modules.
     * 
     * @param hlslSource - The HLSL effect source code
     * @param options - Transpilation options
     * @returns TranspileResult with generated modules and diagnostics
     */
    async transpile(hlslSource: string, options: TranspileOptions = {}): Promise<TranspileResult> {
        const diagnostics: TranspileDiagnostic[] = [];

        try {
            // Step 1: Parse HLSL using WGFX's ShaderParser
            // The PEG parser handles both WGSL and HLSL/Magpie syntax
            const shaderInfo = parse(hlslSource);

            // Validate version
            if (shaderInfo.metadata.version !== 4) {
                diagnostics.push({
                    severity: 'warning',
                    message: `Effect version ${shaderInfo.metadata.version} != 4; Magpie v4 expected`,
                });
            }

            // Step 2: Generate WGSL modules using WGSLCodeGenerator
            const defines = options.defines || {};
            const generatedModules = this.codeGenerator.generate(shaderInfo, defines);

            // Convert to our output format
            const modules: GeneratedModule[] = generatedModules.map(m => ({
                wgslCode: m.wgslCode,
                passIndex: m.passIndex,
                resources: this.extractResources(m.resources),
            }));

            // Check for common issues
            this.validateGeneratedCode(modules, diagnostics);

            return {
                shaderInfo,
                modules,
                diagnostics,
                source: hlslSource,
            };

        } catch (error: any) {
            diagnostics.push({
                severity: 'error',
                message: `Transpilation failed: ${error.message}`,
            });

            // Return partial result with error
            return {
                shaderInfo: {
                    metadata: { use: {}, capability: {} },
                    parameters: [],
                    textures: [],
                    samplers: [],
                    buffers: [],
                    passes: [],
                    commonCode: '',
                },
                modules: [],
                diagnostics,
                source: hlslSource,
            };
        }
    }

    /**
     * Transpile from file.
     */
    async transpileFile(filePath: string, options: TranspileOptions = {}): Promise<TranspileResult> {
        const fs = await import('fs');
        const source = fs.readFileSync(filePath, 'utf-8');
        return this.transpile(source, options);
    }

    /**
     * Extract resource information from generator output.
     */
    private extractResources(resources: any): ModuleResources {
        return {
            textures: (resources.textures || []).map((t: any) => ({
                name: t.name,
                logicalName: t.logicalName,
                format: t.format,
                binding: t.binding,
                group: t.group,
                isStorage: t.isStorage,
            })),
            samplers: (resources.samplers || []).map((s: any) => ({
                name: s.name,
                binding: s.binding,
                group: s.group,
            })),
            storageBuffers: (resources.storageBuffers || []).map((b: any) => ({
                name: b.name,
                binding: b.binding,
                group: b.group,
                mode: b.mode,
                type: b.type,
            })),
            parameters: resources.parameters || [],
            hasScene: resources.hasScene ?? true,
            sceneBinding: resources.sceneBinding ?? 4,
        };
    }

    /**
     * Validate generated WGSL code for common issues.
     */
    private validateGeneratedCode(modules: GeneratedModule[], diagnostics: TranspileDiagnostic[]): void {
        for (const module of modules) {
            const code = module.wgslCode;

            // Check for unresolved HLSL constructs
            if (code.includes('mul(')) {
                diagnostics.push({
                    severity: 'warning',
                    message: `Pass ${module.passIndex}: 'mul()' calls may not have been fully converted`,
                    passIndex: module.passIndex,
                });
            }

            // Check for scalar swizzles
            if (/\b[a-zA-Z_][a-zA-Z0-9_]*\.(x|y|z|w|r|g|b|a)\b/.test(code)) {
                // This is a rough check; the post-processor should handle it
                diagnostics.push({
                    severity: 'info',
                    message: `Pass ${module.passIndex}: Possible scalar swizzles detected (post-processor should fix)`,
                    passIndex: module.passIndex,
                });
            }

            // Check for module-scope const referencing uniforms
            if (code.includes('var<uniform>') && code.includes('const ') && code.includes('scene.')) {
                diagnostics.push({
                    severity: 'info',
                    message: `Pass ${module.passIndex}: Module-scope const referencing uniform (post-processor promotes to var<private>)`,
                    passIndex: module.passIndex,
                });
            }

            // Warn if no passes generated
            if (modules.length === 0) {
                diagnostics.push({
                    severity: 'warning',
                    message: 'No passes generated — effect may have no valid PASS blocks',
                });
            }
        }
    }

    /**
     * Get the WGSL code for a specific pass index.
     */
    getPassCode(result: TranspileResult, passIndex: number): string | null {
        const module = result.modules.find(m => m.passIndex === passIndex);
        return module?.wgslCode || null;
    }

    /**
     * Get all WGSL code concatenated (for debugging).
     */
    getAllCode(result: TranspileResult): string {
        return result.modules
            .sort((a, b) => a.passIndex - b.passIndex)
            .map(m => `// ===== PASS ${m.passIndex} =====\n${m.wgslCode}`)
            .join('\n\n');
    }
}

/**
 * Create a transpiler instance.
 */
export function createTranspiler(): HLSTranspiler {
    return new HLSTranspiler();
}