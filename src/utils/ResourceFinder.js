// src/utils/ResourceFinder.js

/**
 * @fileoverview Utility functions for finding resources in WGFX shader information.
 */

/**
 * Finds a resource by name in the shader info.
 * @param {import('../cli/StaticParser.js').WGFXShaderInfo} shaderInfo
 * @param {string} name
 * @returns {{type: string, [key: string]: any} | null}
 */
export function findResource(shaderInfo, name) {
    // Check textures
    for (const tex of shaderInfo.textures) {
        if (tex.name === name || (tex.sampledInfo && tex.sampledInfo.name === name) || (tex.storagedInfo && tex.storagedInfo.name === name)) {
            // Return the base texture info, marking it as a texture type
            // Determine if it's a sampled or storaged variant
            let resourceType = 'texture';
            if (tex.sampledInfo && tex.sampledInfo.name === name) {
                resourceType = 'sampledTexture';
            } else if (tex.storagedInfo && tex.storagedInfo.name === name) {
                resourceType = 'storagedTexture';
            }
            return {...tex, type: resourceType, baseName: tex.name};
        }
    }

    // Check samplers
    const sampler = shaderInfo.samplers.find(s => s.name === name);
    if (sampler) {
        return {...sampler, type: 'sampler'};
    }

    // Check if it's a uniform parameter (assuming parameters are part of a 'uniforms' buffer)
    const isUniformParam = shaderInfo.parameters.some(p => p.name === name);
    if (isUniformParam) {
        return {type: 'uniformParam', name: name}; // Specific type for individual uniform parameters
    }

    // Check for the 'uniforms' buffer itself
    if (name === 'uniforms') {
        return {type: 'uniformBuffer', name: 'uniforms'};
    }

    // Check if the name matches any of the special texture or sampler names directly referenced in WGSL
    // For cases where tex.name might be 'tex1', but WGSL directly uses 'tex1_sampled' or 'tex1_storaged'
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
