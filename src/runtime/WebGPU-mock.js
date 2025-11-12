// src/runtime/WebGPU-mock.js

/**
 * @fileoverview
 * Provides mock objects for the WebGPU API for use in a Node.js environment.
 * This allows testing and development of WebGPU-dependent code without a browser
 * or a real GPU. The mocks simulate the object structure and method signatures
 * but do not perform any actual GPU operations.
 */

console.warn("警告：未找到 WebGPU 環境。正在使用模擬的 GPU 資源。");

class MockGPUObject {
    constructor(descriptor) {
        this.label = descriptor?.label || 'mock-object';
    }
    destroy() {
        // console.log(`Mock Destroy: ${this.label}`);
    }
}

export const GPUTexture = class extends MockGPUObject {
    constructor(descriptor) {
        super(descriptor);
        this.width = descriptor?.size[0] || 1;
        this.height = descriptor?.size[1] || 1;
    }
    createView() { return new MockGPUObject({ label: `${this.label}-view` }); }
};
export const GPUSampler = class extends MockGPUObject {};
export const GPUBuffer = class extends MockGPUObject {};
export const GPUShaderModule = class extends MockGPUObject {};
export const GPUBindGroupLayout = class extends MockGPUObject {};
export const GPUPipelineLayout = class extends MockGPUObject {};
export const GPUComputePipeline = class extends MockGPUObject {};

export const GPUComputePassEncoder = class extends MockGPUObject {
    setPipeline(pipeline) {}
    setBindGroup(index, bindGroup) {}
    dispatchWorkgroups(x, y, z) {}
    end() {}
}

export const GPUCommandEncoder = class extends MockGPUObject {
    beginComputePass() {
        return new GPUComputePassEncoder({ label: 'mock-compute-pass-encoder' });
    }
    finish() {
        return new MockGPUObject({ label: 'mock-command-buffer' });
    }
}

export const GPUDevice = class extends MockGPUObject {
    constructor() {
        super({ label: 'mock-device' });
        this.queue = {
            submit: (commandBuffers) => {},
            writeBuffer: (buffer, bufferOffset, data, dataOffset, size) => {}
        };
    }
    createTexture(descriptor) { return new GPUTexture(descriptor); }
    createSampler(descriptor) { return new GPUSampler(descriptor); }
    createBuffer(descriptor) { return new GPUBuffer(descriptor); }
    createShaderModule(descriptor) { return new GPUShaderModule(descriptor); }
    createBindGroupLayout(descriptor) { return new GPUBindGroupLayout(descriptor); }
    createPipelineLayout(descriptor) { return new GPUPipelineLayout(descriptor); }
    async createComputePipelineAsync(descriptor) { return new GPUComputePipeline(descriptor); }
    createBindGroup(descriptor) { return new MockGPUObject(descriptor); }
    createCommandEncoder(descriptor) { return new GPUCommandEncoder(descriptor); }
};

export const GPUTextureUsage = {
    TEXTURE_BINDING: 1,
    COPY_DST: 2,
    STORAGE_BINDING: 4,
    COPY_SRC: 8,
};

export const GPUBufferUsage = {
    UNIFORM: 1,
    COPY_DST: 2,
};

export const GPUShaderStage = {
    COMPUTE: 1,
};
