// port-lint: source test_utils.rs
package io.github.kotlinmania.candle.testutils

import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.tensor.Tensor

public fun assertTensorEq(t1: Tensor, t2: Tensor): Result<Unit> {
    check(t1.shape() == t2.shape())
    // Default U8 may not be large enough to hold the sum (`t.sumAll` defaults to the dtype of `t`)
    val eqTensor = t1.eq(t2).getOrElse { return Result.failure(it) }.toDtype(DType.U32).getOrElse { return Result.failure(it) }
    val allEqual = eqTensor.sumAll().getOrElse { return Result.failure(it) }
    check(allEqual.toScalar<UInt>().getOrElse { return Result.failure(it) } == eqTensor.elemCount().toUInt())
    return Result.success(Unit)
}

public fun toVec0Round(t: Tensor, digits: Int): Result<Float> {
    val b = 10.0f.powi(digits)
    val t1 = t.toVec0<Float>().getOrElse { return Result.failure(it) }
    return Result.success(Float.round(t1 * b) / b)
}

public fun toVec1Round(t: Tensor, digits: Int): Result<List<Float>> {
    val b = 10.0f.powi(digits)
    val t1 = t.toVec1<Float>().getOrElse { return Result.failure(it) }
    val t2 = t1.iter().map({ t2 -> Float.round(t2 * b) / b }).collect()
    return Result.success(t2)
}

public fun toVec2Round(t: Tensor, digits: Int): Result<List<List<Float>>> {
    val b = 10.0f.powi(digits)
    val t1 = t.toVec2<Float>().getOrElse { return Result.failure(it) }
    val t2 = t1.iter().map({ t2 -> t2.iter().map({ t3 -> Float.round(t3 * b) / b }).collect() }).collect()
    return Result.success(t2)
}

public fun toVec3Round(t: Tensor, digits: Int): Result<List<List<List<Float>>>> {
    val b = 10.0f.powi(digits)
    val t1 = t.toVec3<Float>().getOrElse { return Result.failure(it) }
    val t2 =
        t1.iter().map({ t2 ->
            t2.iter().map({ t3 -> t3.iter().map({ t4 -> Float.round(t4 * b) / b }).collect() }).collect()
        }).collect()
    return Result.success(t2)
}

private const val TEST_DEVICE_TEMPLATE: String = "fun \$testCpu()-> Result<()> {\$fnName(& Device.Cpu)} fun \$testCuda()-> Result<()> {\$fnName(& Device.newCuda(0)?)} fun \$testMetal()-> Result<()> {\$fnName(& Device.newMetal(0)?)}"

public fun testDevice(fnName: String = "", testCpu: String = "", testCuda: String = "", testMetal: String = ""): String =
    arrayOf("\$fnName" to fnName, "\$testCpu" to testCpu, "\$testCuda" to testCuda, "\$testMetal" to testMetal).fold(TEST_DEVICE_TEMPLATE) { acc0, (k0, v0) -> acc0.replace(k0, v0) }
