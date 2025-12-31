import { WGFXShaderInfo } from '../types/shader';

export function parse(input: string, options?: any): WGFXShaderInfo;

export class SyntaxError extends Error {
    location: any;
    expected: any;
    found: any;
}
