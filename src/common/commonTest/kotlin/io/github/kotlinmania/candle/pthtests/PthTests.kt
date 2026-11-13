package io.github.kotlinmania.candle.pthtests

import io.github.kotlinmania.candlecore.pickle.PthTensors

/**
 * Regression test for pth files not loading on Windows.
 */
@kotlin.test.Test
public fun testPth() {
    val tensors = PthTensors.new("tests/test.pt", null).unwrap()
    tensors.get("test")!!.unwrap()
}

@kotlin.test.Test
public fun testPthWithKey() {
    val tensors = PthTensors.new("tests/test_with_key.pt", "model_state_dict").unwrap()
    tensors.get("test")!!.unwrap()
}

@kotlin.test.Test
public fun testPthFortranContiguous() {
    val tensors = PthTensors.new("tests/fortran_tensor_3d.pth", null).unwrap()
    val tensor = tensors.get("tensor_fortran")!!.unwrap()
    check(tensor.dims3().unwrap() == Triple(2, 3, 4))
    check(tensor.toVec3<Long>().unwrap() == listOf(listOf(listOf(1, 2, 3, 4), listOf(5, 6, 7, 8), listOf(9, 10, 11, 12)), listOf(listOf(13, 14, 15, 16), listOf(17, 18, 19, 20), listOf(21, 22, 23, 24))))
}
