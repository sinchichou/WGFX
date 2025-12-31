import { ResourceManager } from './ResourceManager';

export class UniformBinder {
    public device: GPUDevice;
    public resourceManager: ResourceManager;

    constructor(device: GPUDevice, resourceManager: ResourceManager) {
        this.device = device;
        this.resourceManager = resourceManager;
    }

    public updateUniform(name: string, value: number | number[]): void {
        const metadata = this.resourceManager.getUniform(name);
        if (!metadata) {
            console.warn(`Uniform ${name} not found`);
            return;
        }

        const { buffer, offset, size } = metadata;
        let data: Float32Array | Int32Array;

         // Naive type check assuming float for now, ideally strictly typed in metadata
        if (Array.isArray(value)) {
            data = new Float32Array(value);
        } else {
            data = new Float32Array([value]);
        }

        // Write to buffer
        this.device.queue.writeBuffer(
            buffer,
            offset,
            data.buffer,
            data.byteOffset,
            data.byteLength
        );
    }
}
