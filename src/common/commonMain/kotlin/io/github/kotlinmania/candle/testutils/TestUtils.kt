// port-lint: source test_utils.rs
package io.github.kotlinmania.candle.testutils

import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.tensor.Tensor

public fun assertTensorEq(t1: Tensor, t2: Tensor): Result<Unit> {
    check(t1.shape() == t2.shape())
    // Default U8 may not be large enough to hold the sum (`t.sumAll` defaults to the dtype of `t`)
    val eqTensor =
        when (
            val tryResult =
                when (val tryResult = t1.eq(t2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toDtype(DType.U32)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val allEqual =
        when (val tryResult = eqTensor.sumAll()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = allEqual.toScalar<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == eqTensor.elemCount().toUInt(),
    )
    return Result.Ok(Unit)
}

public fun toVec0Round(t: Tensor, digits: Int): Result<Float> {
    val b = 10.0f.powi(digits)
    val t1 =
        when (val tryResult = t.toVec0<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    return Result.Ok(Float.round(t1 * b) / b)
}

public fun toVec1Round(t: Tensor, digits: Int): Result<List<Float>> {
    val b = 10.0f.powi(digits)
    val t1 =
        when (val tryResult = t.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t2 = t1.iter().map({ t2 -> Float.round(t2 * b) / b }).collect()
    return Result.Ok(t2)
}

public fun toVec2Round(t: Tensor, digits: Int): Result<List<List<Float>>> {
    val b = 10.0f.powi(digits)
    val t1 =
        when (val tryResult = t.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t2 = t1.iter().map({ t2 -> t2.iter().map({ t3 -> Float.round(t3 * b) / b }).collect() }).collect()
    return Result.Ok(t2)
}

public fun toVec3Round(t: Tensor, digits: Int): Result<List<List<List<Float>>>> {
    val b = 10.0f.powi(digits)
    val t1 =
        when (val tryResult = t.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t2 =
        t1
            .iter()
            .map({ t2 ->
                t2.iter().map({ t3 -> t3.iter().map({ t4 -> Float.round(t4 * b) / b }).collect() }).collect()
            })
            .collect()
    return Result.Ok(t2)
}

private const val TEST_DEVICE_TEMPLATE: String = "fun \$testCpu()-> Result<()> {\$fnName(& Device.Cpu)} fun \$testCuda()-> Result<()> {\$fnName(& Device.newCuda(0)?)} fun \$testMetal()-> Result<()> {\$fnName(& Device.newMetal(0)?)}"

public fun testDevice(fnName: String = "", testCpu: String = "", testCuda: String = "", testMetal: String = ""): String =
    arrayOf("\$fnName" to fnName, "\$testCpu" to testCpu, "\$testCuda" to testCuda, "\$testMetal" to testMetal).fold(TEST_DEVICE_TEMPLATE) { acc0, (k0, v0) -> acc0.replace(k0, v0) }
