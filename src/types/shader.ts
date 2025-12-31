/// <reference types="@webgpu/types" />

export interface TextureInfo {
  name: string;
  width: string | number; // String expression or number
  height: string | number;
  format?: GPUTextureFormat; // Optional, defaults to rgba16float internally often
}

export interface SamplerInfo {
  name: string;
  filter: string; // 'LINEAR' | 'NEAREST'
  address: string; // 'WRAP' | 'CLAMP'
}

export interface ParameterInfo {
  name: string;
  type: string; // 'float' | 'int'
  default: number | number[];
  min?: number;
  max?: number;
  step?: number;
}

export interface PassInfo {
  index: number;
  in: string[];
  out: string[];
  code: string;
  numThreads: [number, number, number]; // [x, y, z]
  blockSize?: number[];
}

export interface WGFXShaderInfo {
  textures: TextureInfo[];
  samplers: SamplerInfo[];
  parameters: ParameterInfo[];
  passes: PassInfo[];
  commonCode: string;
}
