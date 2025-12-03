// src/utils/ResourceFinder.js

/**
 * @fileoverview
 * - EN: Utility functions for finding resources in WGFX shader information.
 * - TW: 用於在 WGFX 著色器資訊中尋找資源的實用函數。
 */

/**
 * - EN: Finds a resource by name in the shader info.
 * - TW: 在著色器資訊中按名稱尋找資源。
 * @param {import('../cli/StaticParser.js').WGFXShaderInfo} shaderInfo
 * - EN: The shader information object.
 * - TW: 著色器資訊物件。
 * @param {string} name
 * - EN: The name of the resource to find.
 * - TW: 要尋找的資源名稱。
 * @returns {{type: string, [key: string]: any} | null}
 * - EN: The found resource with its type, or null if not found.
 * - TW: 找到的資源及其類型，如果找不到則為 null。
 */
export function findResource(shaderInfo, name) {
    /**
     * - EN: Check textures
     */
    for (const tex of shaderInfo.textures) {
        if (tex.name === name || (tex.sampledInfo && tex.sampledInfo.name === name) || (tex.storagedInfo && tex.storagedInfo.name === name)) {
            /**
             * - EN: Return the base texture info, marking it as a texture type
             *   Determine if it's a sampled or storaged variant
             */
            let resourceType = 'texture';
            if (tex.sampledInfo && tex.sampledInfo.name === name) {
                resourceType = 'sampledTexture';
            } else if (tex.storagedInfo && tex.storagedInfo.name === name) {
                resourceType = 'storagedTexture';
            }
            return {...tex, type: resourceType, baseName: tex.name};
        }
    }

    /**
     * - EN: Check samplers
     */
    const sampler = shaderInfo.samplers.find(s => s.name === name);
    if (sampler) {
        return {...sampler, type: 'sampler'};
    }

    /**
     * - EN: Check if it's a uniform parameter (assuming parameters are part of a 'uniforms' buffer)
     */
    const isUniformParam = shaderInfo.parameters.some(p => p.name === name);
    if (isUniformParam) {
        return {type: 'uniformParam', name: name}; // Specific type for individual uniform parameters
    }

    /**
     * - EN: Check for the 'uniforms' buffer itself
     */
    if (name === 'uniforms') {
        return {type: 'uniformBuffer', name: 'uniforms'};
    }

    /**
     * - EN: Check if the name matches any of the special texture or sampler names directly referenced in WGSL
     *   For cases where tex.name might be 'tex1', but WGSL directly uses 'tex1_sampled' or 'tex1_storaged'
     */
    for (const tex of shaderInfo.textures) {
        if (tex.sampledInfo && tex.sampledInfo.name === name) {
            return {...tex, type: 'sampledTexture', baseName: tex.name};
        }
        if (tex.storagedInfo && tex.storagedInfo.name === name) {
            return {...tex, type: 'storagedTexture', baseName: tex.name};
        }
    }


    return null;
}
