export { WGFX } from './WGFX';
export { WGFXRuntime } from './runtime/WGFXRuntime';
export { ResourceManager } from './runtime/ResourceManager';
export { PipelineManager } from './runtime/PipelineManager';
export { WGSLCodeGenerator } from './runtime/WGSLCodeGenerator';
export {
    DispatchExpression,
    DispatchExprError,
    parseDispatchExpression,
    evaluateDispatchExpression,
} from './runtime/DispatchExpr';
export { Logger, LogLevel } from './utils/Logger';

// Default export so consumers can write either form:
//   import WGFX from 'wgfx';            // default
//   import { WGFX } from 'wgfx';        // named
// 同時提供預設匯出，讓消費端兩種寫法皆可。
export { default } from './WGFX';
