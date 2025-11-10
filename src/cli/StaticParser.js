// src/cli/StaticParser.js

/**
 * @fileoverview A parser for the CLI context.
 * This parser is intended for static, compile-time analysis of FX files.
 * It inherits from the main ShaderParser but could be extended with
 * CLI-specific validation or logic.
 */

import {ShaderParser} from '../runtime/Parser.js';

/**
 * Represents the parser used in the command-line interface.
 * For now, it is identical to the runtime ShaderParser. It could be
 * extended to enforce stricter rules, such as disallowing dynamic
 * expressions or ensuring all parameters have constant default values.
 */
export class StaticParser extends ShaderParser {
    /**
     * @param {boolean} [debug=false] - If true, logs parsing steps to the console.
     */
    constructor(debug = false) {
        super(debug);
        if (debug) {
            console.log("StaticParser initialized. This parser is used for static analysis and may have stricter rules than the runtime parser.");
        }
    }

    // No overrides are necessary for now, as the parsing logic is shared.
    // Future CLI-specific validation could be added here.
    // For example, a static parser might throw an error if a `//! PARAMETER`
    // is missing a `//! DEFAULT` value, whereas a runtime parser might allow it
    // if the value is expected to be provided dynamically.
}