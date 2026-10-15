// port-lint: source convert.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.anyhow.Ok
import io.github.kotlinmania.anyhow.Result

/**
 * Implement conversion traits for tensors
 */
public interface Error

private object Vec {
    val tryFrom1: (Tensor) -> Result<List<Any?>> = { tensor ->
        Ok(tensor.toVec1())
    }
    val tryFrom2: (Tensor) -> Result<List<List<Any?>>> = { tensor ->
        Ok(tensor.toVec2())
    }
    val tryFrom3: (Tensor) -> Result<List<List<List<Any?>>>> = { tensor ->
        Ok(tensor.toVec3())
    }
}

public object TryFromTensorRefVec1 {
    public fun <T : WithDType> tryFrom(tensor: Tensor): Result<List<T>> {
        return Ok(tensor.toVec1() as List<T>)
    }
}

public object TryFromTensorRefVec2 {
    public fun <T : WithDType> tryFrom(tensor: Tensor): Result<List<List<T>>> {
        return Ok(tensor.toVec2() as List<List<T>>)
    }
}

public object TryFromTensorRefVec3 {
    public fun <T : WithDType> tryFrom(tensor: Tensor): Result<List<List<List<T>>>> {
        return Ok(tensor.toVec3() as List<List<List<T>>>)
    }
}

public object TryFromTensorVec1 {
    public fun <T : WithDType> tryFrom(tensor: Tensor): Result<List<T>> {
        return Vec.tryFrom1(tensor) as Result<List<T>>
    }
}

public object TryFromTensorVec2 {
    public fun <T : WithDType> tryFrom(tensor: Tensor): Result<List<List<T>>> {
        return Vec.tryFrom2(tensor) as Result<List<List<T>>>
    }
}

public object TryFromTensorVec3 {
    public fun <T : WithDType> tryFrom(tensor: Tensor): Result<List<List<List<T>>>> {
        return Vec.tryFrom3(tensor) as Result<List<List<List<T>>>>
    }
}

public object TryFromSliceTensor {
    public fun <T : WithDType> tryFrom(v: FloatArray): Result<Tensor> {
        return Ok(Tensor.fromSlice(v, Shape(v.size), Device.Cpu))
    }
}

public object TryFromVecTensor {
    public fun <T : WithDType> tryFrom(v: List<Float>): Result<Tensor> {
        val len = v.size
        return Ok(Tensor.fromVec(v, Shape(len), Device.Cpu))
    }
}

public fun <W : Write> Tensor.writeBytes(f: W, self: Tensor = this): Result<Unit> {
    val vs = self.flattenAll()
    when (self.dtype()) {
        DType.BF16 -> {
            val vsVec = vs.toVec1Bf16()
            for (v in vsVec) {
                f.writeU16(v.toBits()).unwrap()
            }
        }
        DType.F16 -> {
            val vsVec = vs.toVec1F16()
            for (v in vsVec) {
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
            return Result.failure(CandleError.UnsupportedDTypeForOp(self.dtype(), "write_bytes").bt())
        }
    }
    return Ok(Unit)
}
