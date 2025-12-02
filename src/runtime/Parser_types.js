// src/runtime/Parser_types.js

/**
 * @fileoverview This file contains the type definitions (JSDoc typedefs)
 * for the structured intermediate representation (IR) of a parsed WGFX
 * shader file. Separating these types improves readability of the main
 * parser logic.
 */

/**
 * Describes a shader's uniform parameter.
 * @typedef {Object} WGFXParameter
 * @property {string} name - The variable name in the shader code (e.g., "Strength").
 * @property {string} id - The identifier for the parameter block (e.g., "Strength").
 * @property {string} type - The data type ('float' or 'int').
 * @property {string} [label] - The UI label for this parameter (e.g., "Effect Strength").
 * @property {number} default - The default value.
 * @property {number} min - The minimum allowed value.
 * @property {number} max - The maximum allowed value.
 * @property {number} step - The slider step value.
 */

/**
 * Describes a texture resource.
 * @typedef {Object} WGFXTexture
 * @property {string} name - The base name for the texture resource (e.g., "InputTexture", "tex1").
 * @property {string} id - The identifier for the texture block (e.g., "InputTexture").
 * @property {string} [source] - The source of the texture if file-backed.
 * @property {string} [format] - The texture format (e.g., "rgba8unorm").
 * @property {string} [width] - The width expression (e.g., "INPUT_WIDTH").
 * @property {string} [height] - The height expression (e.g., "INPUT_HEIGHT").
 * @property {Object} [sampledInfo] - Information for the sampled texture.
 * @property {string} [sampledInfo.name] - The actual variable name for the sampled texture (e.g., "tex1_sampled").
 * @property {number} [sampledInfo.group] - The bind group for the sampled texture.
 * @property {number} [sampledInfo.binding] - The binding point for the sampled texture.
 * @property {Object} [storagedInfo] - Information for the storage texture.
 * @property {string} [storagedInfo.name] - The actual variable name for the storage texture (e.g., "tex1_storaged").
 * @property {number} [storagedInfo.group] - The bind group for the storage texture.
 * @property {number} [storagedInfo.binding] - The binding point for the storage texture.
 */

/**
 * Describes a sampler resource.
 * @typedef {Object} WGFXSampler
 * @property {string} name - The name of the sampler resource (e.g., "LinearSampler").
 * @property {string} id - The identifier for the sampler block (e.g., "LinearSampler").
 * @property {'LINEAR' | 'POINT'} filter - The filtering mode.
 * @property {'CLAMP' | 'WRAP'} [address] - The address mode.
 */

/**
 * Describes a single compute pass.
 * @typedef {Object} WGFXPass
 * @property {number} index - The pass index, starting from 1.
 * @property {string[]} in - A list of resource names to be used as inputs.
 * @property {string[]} out - A list of resource names to be used as outputs.
 * @property {[number, number]} [blockSize] - The block size [w, h].
 * @property {[number, number, number]} [numThreads] - The number of threads [x, y, z].
 * @property {'PS' | 'CS'} [style] - The shader style. 'CS' for compute, 'PS' for pixel-shader style.
 * @property {string} [desc] - A description of the pass.
 * @property {string} code - The WGSL code for this pass.
 */

/**
 * The Intermediate Representation (IR) of a parsed WGFX shader file.
 * @typedef {Object} WGFXShaderInfo
 * @property {Object.<string, any>} metadata - General metadata (version, sortName, use, capability).
 * @property {WGFXParameter[]} parameters - A list of uniform parameters.
 * @property {WGFXTexture[]} textures - A list of texture resources.
 * @property {WGFXSampler[]} samplers - A list of sampler resources.
 * @property {WGFXPass[]} passes - A list of rendering passes.
 * @property {string} commonCode - A block of common WGSL code to be prepended.
 */

// This file is purely for type definitions and does not export any runtime code.
// To use these types in JSDoc, use `import('./Parser_types.js').WGFXShaderInfo`
export {};
