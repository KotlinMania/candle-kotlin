// port-lint: source candle-core/src/test_utils.rs
package io.github.kotlinmania.candle

import kotlin.math.pow
import kotlin.math.round

public fun assertTensorEq(t1: Tensor, t2: Tensor) {
    if (t1.shape() != t2.shape()) {
        bail("assertion failed: shape mismatch ${t1.shape()} != ${t2.shape()}")
    }
    val eqTensor = t1.eq(t2).toDtype(DType.U32)
    val allEqual = eqTensor.sumAll()
    val count = allEqual.toScalarF32().toLong()
    if (count != eqTensor.elemCount().toLong()) {
        bail("assertion failed: tensors are not equal")
    }
}

public fun toVec0Round(t: Tensor, digits: Int): Float {
    val b = 10.0f.pow(digits)
    val v = t.toScalarF32()
    return round(v * b) / b
}

public fun toVec1Round(t: Tensor, digits: Int): List<Float> {
    val b = 10.0f.pow(digits)
    val vec = t.toVec1F32()
    return vec.map { round(it * b) / b }
}

public fun toVec2Round(t: Tensor, digits: Int): List<List<Float>> {
    val b = 10.0f.pow(digits)
    val dims = t.shape().dims2()
    val vec = t.toVec1F32()
    val dim1 = dims.d0
    val dim2 = dims.d1
    val result = ArrayList<List<Float>>(dim1)
    for (i in 0 until dim1) {
        val row = ArrayList<Float>(dim2)
        for (j in 0 until dim2) {
            val v = vec[i * dim2 + j]
            row.add(round(v * b) / b)
        }
        result.add(row)
    }
    return result
}

public fun toVec3Round(t: Tensor, digits: Int): List<List<List<Float>>> {
    val b = 10.0f.pow(digits)
    val dims = t.shape().dims3()
    val vec = t.toVec1F32()
    val dim1 = dims.d0
    val dim2 = dims.d1
    val dim3 = dims.d2
    val result = ArrayList<List<List<Float>>>(dim1)
    for (i in 0 until dim1) {
        val plane = ArrayList<List<Float>>(dim2)
        for (j in 0 until dim2) {
            val row = ArrayList<Float>(dim3)
            for (k in 0 until dim3) {
                val v = vec[(i * dim2 + j) * dim3 + k]
                row.add(round(v * b) / b)
            }
            plane.add(row)
        }
        result.add(plane)
    }
    return result
}
