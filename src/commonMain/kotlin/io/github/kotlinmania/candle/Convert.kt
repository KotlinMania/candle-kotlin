// port-lint: source convert.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.CandleResult.Err
import io.github.kotlinmania.candle.CandleResult.Ok

/**
 * Implement conversion traits for tensors
 */
public object TryFromTensorRefVec1 {
    public inline fun <reified T : WithDType<*>> tryFrom(tensor: Tensor): CandleResult<List<T>> {
        return Ok(tensor.toVec1<T>())
    }
}

public object TryFromTensorRefVec2 {
    public inline fun <reified T : WithDType<*>> tryFrom(tensor: Tensor): CandleResult<List<List<T>>> {
        return Ok(tensor.toVec2<T>())
    }
}

public object TryFromTensorRefVec3 {
    public inline fun <reified T : WithDType<*>> tryFrom(tensor: Tensor): CandleResult<List<List<List<T>>>> {
        return Ok(tensor.toVec3<T>())
    }
}

public object TryFromTensorVec1 {
    public inline fun <reified T : WithDType<*>> tryFrom(tensor: Tensor): CandleResult<List<T>> {
        return TryFromTensorRefVec1.tryFrom(tensor)
    }
}

public object TryFromTensorVec2 {
    public inline fun <reified T : WithDType<*>> tryFrom(tensor: Tensor): CandleResult<List<List<T>>> {
        return TryFromTensorRefVec2.tryFrom(tensor)
    }
}

public object TryFromTensorVec3 {
    public inline fun <reified T : WithDType<*>> tryFrom(tensor: Tensor): CandleResult<List<List<List<T>>>> {
        return TryFromTensorRefVec3.tryFrom(tensor)
    }
}

public object TryFromSliceTensor {
    public fun <T : WithDType<*>> tryFrom(v: FloatArray): CandleResult<Tensor> {
        return Ok(Tensor.fromSlice(v, Shape(v.size), Device.Cpu))
    }
}

public object TryFromVecTensor {
    public fun <T : WithDType<*>> tryFrom(v: List<Float>): CandleResult<Tensor> {
        val len = v.size
        return Ok(Tensor.fromVec(v, Shape(len), Device.Cpu))
    }
}

internal fun Tensor.toVec1Bf16(): List<BF16> {
    val cont = contiguous()
    val cpu = (cont.storage() as Storage.Cpu).storage as CpuStorage.BF16
    return List(elemCount()) { BF16.fromBits(cpu.data[it].toUShort()) }
}

internal fun Tensor.toVec1F16(): List<F16> {
    val cont = contiguous()
    val cpu = (cont.storage() as Storage.Cpu).storage as CpuStorage.F16
    return List(elemCount()) { F16.fromBits(cpu.data[it].toUShort()) }
}

internal fun Tensor.toVec1U32(): UIntArray {
    val cont = contiguous()
    val cpu = (cont.storage() as Storage.Cpu).storage as CpuStorage.U32
    return cpu.data.copyOf()
}

internal fun Tensor.toVec1I16(): ShortArray {
    val cont = contiguous()
    val cpu = (cont.storage() as Storage.Cpu).storage as CpuStorage.I16
    return cpu.data.copyOf()
}

internal fun Tensor.toVec1I32(): IntArray {
    val cont = contiguous()
    val cpu = (cont.storage() as Storage.Cpu).storage as CpuStorage.I32
    return cpu.data.copyOf()
}

internal fun Tensor.toVec1U8(): UByteArray {
    val cont = contiguous()
    val cpu = (cont.storage() as Storage.Cpu).storage as CpuStorage.U8
    return cpu.data.copyOf()
}

internal fun Tensor.toVec1F8E4M3(): List<F8E4M3> {
    val cont = contiguous()
    val cpu = (cont.storage() as Storage.Cpu).storage as CpuStorage.F8E4M3
    return List(elemCount()) { F8E4M3.fromBits(cpu.data[it].toUByte()) }
}

public fun Tensor.writeBytes(f: NpyByteWriter, self: Tensor = this): CandleResult<Unit> {
    val vs = self.flattenAll()
    when (self.dtype()) {
        DType.BF16 -> {
            val vsVec = vs.toVec1Bf16()
            for (v in vsVec) {
                f.writeU16<io.github.kotlinmania.byteorder.LittleEndian>(v.toBits()).unwrap()
            }
        }
        DType.F16 -> {
            val vsVec = vs.toVec1F16()
            for (v in vsVec) {
                f.writeU16<io.github.kotlinmania.byteorder.LittleEndian>(v.toBits()).unwrap()
            }
        }
        DType.F32 -> {
            for (v in vs.toVec1F32()) {
                f.writeF32<io.github.kotlinmania.byteorder.LittleEndian>(v).unwrap()
            }
        }
        DType.F64 -> {
            for (v in vs.toVec1F64()) {
                f.writeF64<io.github.kotlinmania.byteorder.LittleEndian>(v).unwrap()
            }
        }
        DType.U32 -> {
            for (v in vs.toVec1U32()) {
                f.writeU32<io.github.kotlinmania.byteorder.LittleEndian>(v).unwrap()
            }
        }
        DType.I16 -> {
            for (v in vs.toVec1I16()) {
                f.writeI16<io.github.kotlinmania.byteorder.LittleEndian>(v).unwrap()
            }
        }
        DType.I32 -> {
            for (v in vs.toVec1I32()) {
                f.writeI32<io.github.kotlinmania.byteorder.LittleEndian>(v).unwrap()
            }
        }
        DType.I64 -> {
            for (v in vs.toVec1I64()) {
                f.writeI64<io.github.kotlinmania.byteorder.LittleEndian>(v).unwrap()
            }
        }
        DType.U8 -> {
            val vsVec = vs.toVec1U8()
            f.writeAll(ByteArray(vsVec.size) { vsVec[it].toByte() }).unwrap()
        }
        DType.F8E4M3 -> {
            val vsVec = vs.toVec1F8E4M3()
            for (v in vsVec) {
                f.writeU8(v.toBits()).unwrap()
            }
        }
        DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
            return Err(CandleError.UnsupportedDTypeForOp(self.dtype(), "write_bytes").bt())
        }
    }
    return Ok(Unit)
}
