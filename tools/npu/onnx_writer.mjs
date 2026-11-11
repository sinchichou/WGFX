/**
 * onnx_writer.mjs — Minimal ONNX ModelProto writer (hand-rolled protobuf).
 *
 * Avoids a heavyweight modelling dependency: ONNX's wire format is stable and
 * the subset needed here (Conv/Relu/Sub/Concat/QDQ/Tanh/Resize/Slice/Add/Clip/
 * DepthToSpace + float initializers) maps to a handful of protobuf messages.
 *
 * Implementation note: weight tensors reach hundreds of KiB, so nothing here
 * may use spread/apply on data-sized arrays — everything streams through a
 * chunk-list builder (Buf) and is concatenated once at the end.
 *
 * Encoding reference (onnx/onnx.in.proto, protobuf wire format):
 *   varint = base-128; tag = (field << 3) | wiretype;
 *   wiretype 0 = varint, 2 = length-delimited, 5 = 32-bit.
 */

const te = new TextEncoder();

class Buf {
    constructor() {
        this.chunks = [];
        this.len = 0;
    }
    /** append an array-like of bytes without copying/spreading */
    raw(arr) {
        if (arr.length > 0) {
            this.chunks.push(arr instanceof Uint8Array ? arr : Uint8Array.from(arr));
            this.len += arr.length;
        }
        return this;
    }
    varint(value) {
        let v = typeof value === 'bigint' ? value : BigInt(Math.trunc(value));
        if (v < 0n) v += 1n << 64n;
        const out = [];
        do {
            let byte = Number(v & 0x7fn);
            v >>= 7n;
            if (v > 0n) byte |= 0x80;
            out.push(byte);
        } while (v > 0n);
        return this.raw(out);
    }
    tag(field, wiretype) {
        return this.varint((field << 3) | wiretype);
    }
    bytes(field, arr) {
        this.tag(field, 2).varint(arr.length);
        return this.raw(arr);
    }
    str(field, s) {
        return this.bytes(field, te.encode(s));
    }
    int64(field, value) {
        this.tag(field, 0);
        return this.varint(value);
    }
    f32(field, value) {
        const b = new Uint8Array(4);
        new DataView(b.buffer).setFloat32(0, value, true);
        this.tag(field, 5);
        return this.raw(b);
    }
    msg(field, fn) {
        const inner = new Buf();
        fn(inner);
        this.tag(field, 2).varint(inner.len);
        for (const c of inner.chunks) this.chunks.push(c);
        this.len += inner.len;
        return this;
    }
    toUint8Array() {
        const out = new Uint8Array(this.len);
        let o = 0;
        for (const c of this.chunks) {
            out.set(c, o);
            o += c.length;
        }
        return out;
    }
}

/** f32 values → Uint8Array (little-endian) */
function f32Bytes(values) {
    const b = new Uint8Array(values.length * 4);
    const dv = new DataView(b.buffer);
    for (let i = 0; i < values.length; i++) dv.setFloat32(i * 4, values[i], true);
    return b;
}

/** TensorProto (float data via raw_data). dims: number[], data: number[] */
export function tensor(name, dims, data) {
    return (buf) => {
        buf.bytes(1, packVarints(dims)); // dims, packed int64
        buf.int64(2, 1); // data_type = FLOAT
        buf.bytes(9, f32Bytes(data));
        buf.str(8, name);
    };
}

const ATTR_TYPE = { FLOAT: 1, INT: 2, STRING: 3, TENSOR: 4, INTS: 7, FLOATS: 8 };

function attribute(buf, name, kv) {
    buf.str(1, name);
    if ('f' in kv) {
        buf.f32(2, kv.f);
        buf.int64(20, ATTR_TYPE.FLOAT);
    } else if ('i' in kv) {
        buf.int64(3, kv.i);
        buf.int64(20, ATTR_TYPE.INT);
    } else if ('s' in kv) {
        buf.bytes(4, te.encode(kv.s));
        buf.int64(20, ATTR_TYPE.STRING);
    } else if ('t' in kv) {
        buf.msg(5, kv.t);
        buf.int64(20, ATTR_TYPE.TENSOR);
    } else if ('ints' in kv) {
        buf.bytes(8, packVarints(kv.ints));
        buf.int64(20, ATTR_TYPE.INTS);
    } else if ('floats' in kv) {
        buf.bytes(7, f32Bytes(kv.floats));
        buf.int64(20, ATTR_TYPE.FLOATS);
    }
}

function packVarints(values) {
    const out = [];
    for (const v of values) {
        let x = BigInt(Math.trunc(v));
        do { let b = Number(x & 0x7fn); x >>= 7n; if (x > 0n) b |= 0x80; out.push(b); } while (x > 0n);
    }
    return out;
}

/** NodeProto */
export function node(opType, inputs, outputs, attrs = {}) {
    return (buf) => {
        for (const i of inputs) buf.str(1, i);
        for (const o of outputs) buf.str(2, o);
        buf.str(3, outputs[0] ?? opType);
        buf.str(4, opType);
        for (const [k, v] of Object.entries(attrs)) buf.msg(5, (ab) => attribute(ab, k, v));
    };
}

/** ValueInfoProto: name + TypeProto{ tensor_type(1) = Tensor{ elem_type(1), shape(2) = { dim(1) } } } */
export function valueInfo(name, dims) {
    return (buf) => {
        buf.str(1, name);
        buf.msg(2, (tp) => {
            tp.msg(1, (tt) => {
                tt.int64(1, 1); // elem_type FLOAT
                tt.msg(2, (sb) => {
                    for (const d of dims) sb.msg(1, (db) => db.int64(1, d));
                });
            });
        });
    };
}

/**
 * Build a complete ONNX model.
 * @param {object} m { nodes, initializers, inputs, outputs, name }
 *   nodes: byte-array-producing fns (from node()); initializers: fns (tensor()/int8Tensor());
 *   inputs/outputs: {name, dims}.
 */
export function buildModel({ name, nodes, initializers, inputs, outputs }) {
    const buf = new Buf();
    buf.int64(1, 8); // ir_version 8
    buf.str(2, 'wgfx-npu-experiment');
    buf.msg(7, (g) => {
        for (const n of nodes) g.msg(1, n);
        g.str(2, name);
        for (const t of initializers) g.msg(5, t);
        for (const i of inputs) g.msg(11, valueInfo(i.name, i.dims));
        for (const o of outputs) g.msg(12, valueInfo(o.name, o.dims));
    });
    buf.msg(8, (op) => {
        op.str(1, '');
        op.int64(2, 17);
    });
    return buf.toUint8Array();
}

/** float initializer */
export function initializer(name, dims, data) {
    return tensor(name, dims, data);
}

/** int64 TensorProto (raw_data, little-endian) — Slice starts/ends/axes etc. */
export function int64Tensor(name, dims, values) {
    return (buf) => {
        buf.bytes(1, packVarints(dims));
        buf.int64(2, 7); // data_type = INT64
        const b = new Uint8Array(values.length * 8);
        const dv = new DataView(b.buffer);
        values.forEach((v, i) => dv.setBigInt64(i * 8, BigInt(Math.trunc(v)), true));
        buf.bytes(9, b);
        buf.str(8, name);
    };
}

/** int8 TensorProto (raw_data) — for QuantizeLinear zero points (1-element shape: ORT rejects 0-d int8) */
export function int8Tensor(name, values) {
    return (buf) => {
        buf.bytes(1, packVarints([1])); // dims [1]
        buf.int64(2, 3); // data_type = INT8
        buf.bytes(9, Uint8Array.from(values, (v) => v & 0xff));
        buf.str(8, name);
    };
}
