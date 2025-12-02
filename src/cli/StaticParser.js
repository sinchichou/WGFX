/**
 * @fileoverview A static parser for WGFX shader files.
 * This parser is used at build time (by the CLI) to create a static
 * representation of a shader's structure.
 * It uses the Peggy-generated parser.
 */
import {createRequire} from 'module';

const require = createRequire(import.meta.url);
const shaderParser = require('../runtime/ShaderParser.cjs');

/**
 * The static parser for WGFX files.
 */
export class StaticParser {
    /**
     * @param {string} shaderCode The shader code to parse.
     * @returns {any} The parsed shader information object (IR).
     */
    parse(shaderCode) {
        // Peggy parser might throw its own error type
        try {
            return shaderParser.parse(shaderCode);
        } catch (e) {
            // Re-throw with a more generic error or format it if needed
            throw new Error(`Parsing failed: ${e.message}`);
        }
    }
}