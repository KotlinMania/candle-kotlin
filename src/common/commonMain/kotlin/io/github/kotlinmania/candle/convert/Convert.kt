// port-lint: source convert.rs
package io.github.kotlinmania.candle.convert

import io.github.kotlinmania.byteorder.LittleEndian
import io.github.kotlinmania.candle.device.Device
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.float8.F8e4m3
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.rustlang.core.result.Result
import io.github.kotlinmania.rustlang.core.result.fromResidual

// Implement conversion traits for tensors

public fun <T : WithDType> tryFromVecT(tensor: Tensor): Result<List<T>, Error> = tensor.toVec1<T>()

public fun <T : WithDType> tryFromVecVecT(tensor: Tensor): Result<List<List<T>>, Error> = tensor.toVec2<T>()

public fun <T : WithDType> tryFromVecVecVecT(tensor: Tensor): Result<List<List<List<T>>>, Error> = tensor.toVec3<T>()

public fun <T : WithDType> tryFromTensor(v: List<T>): Result<Tensor, Error> = Tensor.fromSlice(v, v.size, Device.Cpu)

public fun tryFromDouble(tensor: Tensor): Result<Double, Error> = tensor.toScalar<Double>()

public fun tryFromTensor(v: Double): Result<Tensor, Error> = Tensor.new(v, Device.Cpu)

public fun tryFromFloat(tensor: Tensor): Result<Float, Error> = tensor.toScalar<Float>()

public fun tryFromTensor(v: Float): Result<Tensor, Error> = Tensor.new(v, Device.Cpu)

public fun tryFromF16(tensor: Tensor): Result<F16, Error> = tensor.toScalar<F16>()

public fun tryFromTensor(v: F16): Result<Tensor, Error> = Tensor.new(v, Device.Cpu)

public fun tryFromBf16(tensor: Tensor): Result<Bf16, Error> = tensor.toScalar<Bf16>()

public fun tryFromTensor(v: Bf16): Result<Tensor, Error> = Tensor.new(v, Device.Cpu)

public fun tryFromLong(tensor: Tensor): Result<Long, Error> = tensor.toScalar<Long>()

public fun tryFromTensor(v: Long): Result<Tensor, Error> = Tensor.new(v, Device.Cpu)

public fun tryFromInt(tensor: Tensor): Result<Int, Error> = tensor.toScalar<Int>()

public fun tryFromTensor(v: Int): Result<Tensor, Error> = Tensor.new(v, Device.Cpu)

public fun tryFromShort(tensor: Tensor): Result<Short, Error> = tensor.toScalar<Short>()

public fun tryFromTensor(v: Short): Result<Tensor, Error> = Tensor.new(v, Device.Cpu)

public fun tryFromUInt(tensor: Tensor): Result<UInt, Error> = tensor.toScalar<UInt>()

public fun tryFromTensor(v: UInt): Result<Tensor, Error> = Tensor.new(v, Device.Cpu)

public fun tryFromUByte(tensor: Tensor): Result<UByte, Error> = tensor.toScalar<UByte>()

public fun tryFromTensor(v: UByte): Result<Tensor, Error> = Tensor.new(v, Device.Cpu)

public fun <W : Write> Tensor.writeBytes(f: W): Result<Unit> {
    val vs =
        when (val tryResult = this.flattenAll()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    when (this.dtype()) {
        DType.bf16 -> {
            val vs1 =
                when (val tryResult = vs.toVec1<Bf16>()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            for (v in vs1.reinterpretCast()) {
                when (val tryResult = f.writeU16<LittleEndian>(v)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        }
        DType.f16 -> {
            val vs2 =
                when (val tryResult = vs.toVec1<F16>()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            for (v1 in vs2.reinterpretCast()) {
                when (val tryResult = f.writeU16<LittleEndian>(v1)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        }
        DType.f32 -> {
            // Note: Avoid using a buffer when data is already on the CPU.
            for (v2 in when (val tryResult = vs.toVec1<Float>()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }) {
                when (val tryResult = f.writeF32<LittleEndian>(v2)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        }
        DType.f64 -> {
            for (v3 in when (val tryResult = vs.toVec1<Double>()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }) {
                when (val tryResult = f.writeF64<LittleEndian>(v3)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        }
        DType.u32 -> {
            for (v4 in when (val tryResult = vs.toVec1<UInt>()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }) {
                when (val tryResult = f.writeU32<LittleEndian>(v4)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        }
        DType.i16 -> {
            for (v5 in when (val tryResult = vs.toVec1<Short>()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }) {
                when (val tryResult = f.writeI16<LittleEndian>(v5)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        }
        DType.i32 -> {
            for (v6 in when (val tryResult = vs.toVec1<Int>()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }) {
                when (val tryResult = f.writeI32<LittleEndian>(v6)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        }
        DType.i64 -> {
            for (v7 in when (val tryResult = vs.toVec1<Long>()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }) {
                when (val tryResult = f.writeI64<LittleEndian>(v7)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        }
        DType.u8 -> {
            val vs3 =
                when (val tryResult = vs.toVec1<UByte>()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            when (val tryResult = f.writeAll(vs3)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        DType.f8e4m3 -> {
            val vs4 =
                when (val tryResult = vs.toVec1<F8e4m3>()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            for (v8 in vs4) {
                when (val tryResult = f.writeU8(v8.toRawBits().toULong())) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        }
        DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
            return Err(Error.UnsupportedDTypeForOp(this.dtype(), "write_bytes").bt())
        }
    }
    return Ok(Unit)
}
