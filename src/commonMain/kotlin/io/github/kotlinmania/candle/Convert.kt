// port-lint: source convert.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.anyhow.Ok
import io.github.kotlinmania.anyhow.Result

/**
 * Implement conversion traits for tensors
 */
public interface Error

public fun Tensor.tryFrom(): List<Float> =
    this.toVec1()

public fun Tensor.tryFrom(dim2: Unit): List<List<Float>> =
    this.toVec2().also { _ = dim2 }

public fun Tensor.tryFrom(dim2: Unit, dim3: Unit): List<List<List<Float>>> =
    this.toVec3().also {
        _ = dim2
        _ = dim3
    }

public fun tryFrom(tensor: Tensor): List<Float> =
    tensor.tryFrom()

public fun tryFrom(tensor: Tensor, dim2: Unit): List<List<Float>> =
    tensor.tryFrom(dim2)

public fun tryFrom(tensor: Tensor, dim2: Unit, dim3: Unit): List<List<List<Float>>> =
    tensor.tryFrom(dim2, dim3)

public fun Tensor.Companion.tryFrom(v: FloatArray): Tensor =
    Tensor.fromSlice(v, Shape(v.size), Device.Cpu)

public fun Tensor.Companion.tryFrom(v: List<Float>): Tensor {
    val len = v.size
    return Tensor.fromVec(v, Shape(len), Device.Cpu)
}

public fun Tensor.writeBytes(f: NpyByteWriter): Result<Unit> {
    val vs = this.flattenAll()
    when (this.dtype()) {
        DType.BF16 -> {
            val vec = vs.toVec1Bf16()
            for (v in vec) {
                f.writeU16(v.toBits()).unwrap()
            }
        }
        DType.F16 -> {
            val vec = vs.toVec1F16()
            for (v in vec) {
                f.writeU16(v.toBits()).unwrap()
            }
        }
        DType.F32 -> {
            for (v in vs.toVec1()) {
                f.writeF32(v).unwrap()
            }
        }
        DType.F64 -> {
            for (v in vs.toVec1F64()) {
                f.writeF64(v).unwrap()
            }
        }
        DType.U32 -> {
            for (v in vs.toVec1U32()) {
                f.writeU32(v.toUInt()).unwrap()
            }
        }
        DType.I16 -> {
            for (v in vs.toVec1I16()) {
                f.writeI16(v).unwrap()
            }
        }
        DType.I32 -> {
            for (v in vs.toVec1I32()) {
                f.writeI32(v).unwrap()
            }
        }
        DType.I64 -> {
            for (v in vs.toVec1I64()) {
                f.writeI64(v).unwrap()
            }
        }
        DType.U8 -> {
            val vec = vs.toVec1U8()
            f.writeAll(ByteArray(vec.size) { vec[it].toByte() }).unwrap()
        }
        DType.F8E4M3 -> {
            val vec = vs.toVec1F8E4M3()
            for (v in vec) {
                f.writeU8(v.toBits()).unwrap()
            }
        }
        DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
            return Result.failure(CandleError.UnsupportedDTypeForOp(this.dtype(), "write_bytes").bt())
        }
    }
    return Ok(Unit)
}
