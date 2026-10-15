package io.github.kotlinmania.candle

import kotlin.math.pow
import kotlin.math.round

/**
 * Tensor testing utilities and rounding comparison helpers for CPU, CUDA, and Metal backends.
 *
 * Provides element-wise tensor equality assertions and rounded vector extraction helpers
 * for 0D, 1D, 2D, and 3D floating-point tensors.
 */
// port-lint: source test_utils.rs

private val assertEq: (Any?, Any?) -> Unit = { a, b ->
    if (a != b) bail("assertion failed: $a != $b")
}
private val powi: (Float, Int) -> Float = { base, exp -> base.pow(exp) }
private val f32Round: (Float) -> Float = { round(it) }
private val Ok: (Unit) -> Unit = { }

public fun assertTensorEq(t1: Tensor, t2: Tensor) {
    assertEq(t1.shape(), t2.shape())
    // Default U8 may not be large enough to hold the sum (t.sumAll defaults to the dtype of t)
    val eqTensor = t1.eq(t2).toDtype(DType.U32)
    val allEqual = eqTensor.sumAll()
    assertEq(allEqual.toScalar<UInt>(), eqTensor.elemCount().toUInt())
    Ok(Unit)
}

public fun toVec0Round(t: Tensor, digits: Int): Float {
    val b = powi(10.0f, digits)
    val t = t.toVec0<Float>()
    return f32Round(t * b) / b
}

public fun toVec1Round(t: Tensor, digits: Int): List<Float> {
    val b = powi(10.0f, digits)
    val t = t.toVec1<Float>()
    val res = t.map { t -> f32Round(t * b) / b }
    return res
}

public fun toVec2Round(t: Tensor, digits: Int): List<List<Float>> {
    val b = powi(10.0f, digits)
    val t = t.toVec2<Float>()
    val res = t
        .asSequence()
        .map { t -> t.map { t -> f32Round(t * b) / b } }
        .toList()
    return res
}

public fun toVec3Round(t: Tensor, digits: Int): List<List<List<Float>>> {
    val b = powi(10.0f, digits)
    val t = t.toVec3<Float>()
    val res = t
        .asSequence()
        .map { t ->
            t.asSequence()
                .map { t -> t.map { t -> f32Round(t * b) / b } }
                .toList()
        }
        .toList()
    return res
}
