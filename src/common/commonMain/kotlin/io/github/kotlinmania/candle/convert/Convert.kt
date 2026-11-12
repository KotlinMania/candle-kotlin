// port-lint: source convert.rs
package io.github.kotlinmania.candle.convert

import io.github.kotlinmania.byteorder.LittleEndian
import io.github.kotlinmania.candle.device.Device
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.float8.F8e4m3
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16

// Implement conversion traits for tensors

public fun <T : WithDType> tryFromVecT(tensor: Tensor): Result<List<T>> = tensor.toVec1<T>()

public fun <T : WithDType> tryFromVecVecT(tensor: Tensor): Result<List<List<T>>> = tensor.toVec2<T>()

public fun <T : WithDType> tryFromVecVecVecT(tensor: Tensor): Result<List<List<List<T>>>> = tensor.toVec3<T>()

public fun <T : WithDType> tryFromTensor(v: List<T>): Result<Tensor> = Tensor.fromSlice(v, v.size, Device.Cpu)

public fun tryFromDouble(tensor: Tensor): Result<Double> = tensor.toScalar<Double>()

public fun tryFromTensor(v: Double): Result<Tensor> = Tensor.new(v, Device.Cpu)

public fun tryFromFloat(tensor: Tensor): Result<Float> = tensor.toScalar<Float>()

public fun tryFromTensor(v: Float): Result<Tensor> = Tensor.new(v, Device.Cpu)

public fun tryFromF16(tensor: Tensor): Result<F16> = tensor.toScalar<F16>()

public fun tryFromTensor(v: F16): Result<Tensor> = Tensor.new(v, Device.Cpu)

public fun tryFromBf16(tensor: Tensor): Result<Bf16> = tensor.toScalar<Bf16>()

public fun tryFromTensor(v: Bf16): Result<Tensor> = Tensor.new(v, Device.Cpu)

public fun tryFromLong(tensor: Tensor): Result<Long> = tensor.toScalar<Long>()

public fun tryFromTensor(v: Long): Result<Tensor> = Tensor.new(v, Device.Cpu)

public fun tryFromInt(tensor: Tensor): Result<Int> = tensor.toScalar<Int>()

public fun tryFromTensor(v: Int): Result<Tensor> = Tensor.new(v, Device.Cpu)

public fun tryFromShort(tensor: Tensor): Result<Short> = tensor.toScalar<Short>()

public fun tryFromTensor(v: Short): Result<Tensor> = Tensor.new(v, Device.Cpu)

public fun tryFromUInt(tensor: Tensor): Result<UInt> = tensor.toScalar<UInt>()

public fun tryFromTensor(v: UInt): Result<Tensor> = Tensor.new(v, Device.Cpu)

public fun tryFromUByte(tensor: Tensor): Result<UByte> = tensor.toScalar<UByte>()

public fun tryFromTensor(v: UByte): Result<Tensor> = Tensor.new(v, Device.Cpu)

public fun <W : Write> Tensor.writeBytes(f: W): Result<Unit> {
    val vs = this.flattenAll().getOrElse { return Result.failure(it) }
    when (this.dtype()) {
        DType.bf16 -> {
            val vs1 = vs.toVec1<Bf16>().getOrElse { return Result.failure(it) }
            for (v in vs1.reinterpretCast()) {
                f.writeU16<LittleEndian>(v).getOrElse { return Result.failure(it) }
            }
        }
        DType.f16 -> {
            val vs2 = vs.toVec1<F16>().getOrElse { return Result.failure(it) }
            for (v1 in vs2.reinterpretCast()) {
                f.writeU16<LittleEndian>(v1).getOrElse { return Result.failure(it) }
            }
        }
        DType.f32 -> {
            // Note: Avoid using a buffer when data is already on the CPU.
            for (v2 in vs.toVec1<Float>().getOrElse { return Result.failure(it) }) {
                f.writeF32<LittleEndian>(v2).getOrElse { return Result.failure(it) }
            }
        }
        DType.f64 -> {
            for (v3 in vs.toVec1<Double>().getOrElse { return Result.failure(it) }) {
                f.writeF64<LittleEndian>(v3).getOrElse { return Result.failure(it) }
            }
        }
        DType.u32 -> {
            for (v4 in vs.toVec1<UInt>().getOrElse { return Result.failure(it) }) {
                f.writeU32<LittleEndian>(v4).getOrElse { return Result.failure(it) }
            }
        }
        DType.i16 -> {
            for (v5 in vs.toVec1<Short>().getOrElse { return Result.failure(it) }) {
                f.writeI16<LittleEndian>(v5).getOrElse { return Result.failure(it) }
            }
        }
        DType.i32 -> {
            for (v6 in vs.toVec1<Int>().getOrElse { return Result.failure(it) }) {
                f.writeI32<LittleEndian>(v6).getOrElse { return Result.failure(it) }
            }
        }
        DType.i64 -> {
            for (v7 in vs.toVec1<Long>().getOrElse { return Result.failure(it) }) {
                f.writeI64<LittleEndian>(v7).getOrElse { return Result.failure(it) }
            }
        }
        DType.u8 -> {
            val vs3 = vs.toVec1<UByte>().getOrElse { return Result.failure(it) }
            f.writeAll(vs3).getOrElse { return Result.failure(it) }
        }
        DType.f8e4m3 -> {
            val vs4 = vs.toVec1<F8e4m3>().getOrElse { return Result.failure(it) }
            for (v8 in vs4) {
                f.writeU8(v8.toRawBits().toULong()).getOrElse { return Result.failure(it) }
            }
        }
        DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
            return Result.failure(Error.UnsupportedDTypeForOp(this.dtype(), "write_bytes").bt())
        }
    }
    return Result.success(Unit)
}
