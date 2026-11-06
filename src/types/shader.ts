/// <reference types="@webgpu/types" />

/**
 * Information about a texture resource used in the shader
 * 著色器中使用的紋理資源資訊
 */
export interface TextureInfo {
  /** Texture identifier used in code / 程式碼中使用的紋理識別符 */
  name: string;
  /** Width can be a numeric literal or expression (required except INPUT) / 寬度運算式（INPUT 除外必填） */
  width?: string | number;
  /** Height can be a numeric literal or expression (required except INPUT) / 高度運算式（INPUT 除外必填） */
  height?: string | number;
  /** GPU texture format / GPU 紋理格式 */
  format?: GPUTextureFormat;
}

/**
 * Information about a sampler resource used in the shader
 * 著色器中使用的取樣器資源資訊
 */
export interface SamplerInfo {
  /** Sampler identifier used in code / 程式碼中使用的取樣器識別符 */
  name: string;
  /** Filter mode: 'LINEAR' or 'NEAREST' / 過濾模式 */
  filter: string;
  /** Address mode: 'WRAP' or 'CLAMP' / 定址模式 */
  address: string;
}

/**
 * Information about an adjustable parameter (uniform)
 * 可調整的參數 (Uniform) 資訊
 */
export interface ParameterInfo {
  /** Parameter identifier / 參數識別符 */
  name: string;
  /**
   * Data type / 資料類型
   * 'float' | 'vec2' | 'vec3' | 'vec4' | 'int' | 'uint' | 'int2' | 'uint2' ...
   */
  type: string;
  /** Default value or array of values / 預設值或數值陣列 */
  default: number | number[];
  /** Minimum value allowed / 允許的最小值 */
  min?: number;
  /** Maximum value allowed / 允許的最大值 */
  max?: number;
  /** Step value for UI adjustment / UI 調整的步進值 */
  step?: number;
  /** UI display label / UI 顯示標籤 */
  label?: string;
}

/**
 * Information about a rendering/compute pass
 * 渲染/計算通道資訊
 */
export interface PassInfo {
  /** 1-based index of the pass / 通道索引 (從 1 開始) */
  index: number;
  /** Names of input textures / 輸入紋理名稱列表 */
  in: string[];
  /** Names of output textures / 輸出紋理名稱列表 */
  out: string[];
  /** WGSL/HLSL code snippet for this pass / 此通道的 WGSL/HLSL 代碼片段 */
  code: string;
  /** Workgroup size [x, y, z] / 工作群組尺寸 [x, y, z] */
  numThreads: [number, number, number];
  /** Optional block size for coordinate calculation / 用於座標計算的選用區塊大小 */
  blockSize?: number[];
}

/**
 * Parsed effect-file metadata directives
 * 特效檔元數據指令的解析結果
 */
export interface ShaderMetadata {
  /** Required feature flags (//! USE) / 必要功能旗標 */
  use: Record<string, boolean>;
  /** Required capability flags (//! CAPABILITY) / 必要硬體能力旗標 */
  capability: Record<string, boolean>;
  /** Effect format version (//! VERSION) / 特效格式版本 */
  version?: number;
  /** Sort name for UI listing (//! SORT_NAME) / UI 排序名稱 */
  sortName?: string;
}

/**
 * Structured information of a complete WGFX shader
 * 完整 WGFX 著色器的結構化資訊
 */
export interface WGFXShaderInfo {
  /** List of all textures (sampled/storaged pairs share one logical texture) / 所有紋理列表 */
  textures: TextureInfo[];
  /** List of all samplers / 所有取樣器的列表 */
  samplers: SamplerInfo[];
  /** List of all adjustable parameters / 所有可調整參數的列表 */
  parameters: ParameterInfo[];
  /** List of all execution passes / 所有執行通道的列表 */
  passes: PassInfo[];
  /** Common code shared across all passes / 所有通道共享的通用代碼 */
  commonCode: string;
  /** Metadata directives / 元數據指令 */
  metadata: ShaderMetadata;
}
