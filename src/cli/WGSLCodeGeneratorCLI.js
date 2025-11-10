// src/cli/WGSLCodeGeneratorCLI.js

/**
 * @fileoverview A WGSL code generator for the CLI context.
 * This generator is intended for static, compile-time generation of WGSL modules.
 * It inherits from the main WGSLCodeGenerator but could be extended with
 * CLI-specific optimizations.
 */

import {WGSLCodeGenerator} from '../runtime/WGSLCodeGenerator.js';

/**
 * Represents the WGSL code generator used in the command-line interface.
 * For now, it is identical to the runtime WGSLCodeGenerator. It could be
 * extended to perform optimizations specific to static compilation, such as
 * inlining uniform parameters as constants if they are not marked as dynamic.
 */
export class WGSLCodeGeneratorCLI extends WGSLCodeGenerator {
    constructor() {
        super();
    }

    // No overrides are necessary for now, as the code generation logic is shared.
    // A potential future optimization:
    // If a parameter is not marked with `//! USE _DYNAMIC`, its `//! DEFAULT`
    // value could be inlined directly into the shader as a `const`, removing
    // the need for a uniform buffer for that parameter.
    //
    // generate(shaderInfo) {
    //     // ... custom logic ...
    //     const baseWgsl = super.generate(shaderInfo);
    //     // ... further optimizations ...
    //     return optimizedWgsl;
    // }
}