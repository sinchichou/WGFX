// src/cli/PipelineMetadataGenerator.js

/**
 * @fileoverview 從著色器 IR 生成結構化元資料。
 * 此類別由 CLI 使用，以建立描述管線結構和使用者可見參數的 JSON 檔案，
 * 這些檔案可由運行時或 UI 消耗。
 */

export class PipelineMetadataGenerator {
    constructor() {
        // 此類別是無狀態的。
    }

    /**
     * 從解析後的著色器 IR 生成管線和一般元資料 JSON。
     * @param {import('../runtime/Parser.js').WGFXShaderInfo} shaderInfo - 來自 Parser.js 的解析後著色器資訊。
     * @returns {{pipeline: object, metadata: object}} 包含兩個元資料結構的物件：
     * - `pipeline`: 描述通道序列及其屬性。
     * - `metadata`: 描述使用者可見的詳細資訊，例如用於 UI 生成的參數。
     */
    generate(shaderInfo) {
        // 'pipeline.json' 的內容。
        // 這描述了渲染過程的結構。
        const pipeline = {
            passes: shaderInfo.passes.map(pass => ({
                index: pass.index,
                in: pass.in,
                out: pass.out,
                workgroup_size: pass.numThreads,
                style: pass.style || 'CS',
                description: pass.desc || '',
            })),
            // TODO: 更完整的實作將在此處序列化綁定組佈局，
            // 以便運行時無需重新推導它。
        };

        // 'metadata.json' 的內容。
        // 這主要用於需要顯示效果控制項的 UI。
        const generalMetadata = {
            version: shaderInfo.metadata.version || '1.0',
            name: shaderInfo.metadata.sortName || '未命名效果',
            description: 'WGFX 效果套件',
            parameters: shaderInfo.parameters.map(p => ({
                name: p.name,
                label: p.label || p.name,
                type: p.type,
                default: p.default,
                min: p.min,
                max: p.max,
                step: p.step,
            })),
        };

        return {
            pipeline,
            metadata: generalMetadata,
        };
    }
}