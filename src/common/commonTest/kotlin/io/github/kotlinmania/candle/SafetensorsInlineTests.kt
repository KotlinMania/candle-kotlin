// port-lint: tests safetensors.rs
package io.github.kotlinmania.candle.safetensors

import io.github.kotlinmania.candle.device.Device
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.tensor.Tensor

// Module to load `safetensor` files into CPU/GPU memory.
//
// There are multiple ways to load tensors from safetensor files:
// - `load` function for loading directly into memory and returning a HashMap of tensors
// - `MmapedSafetensors` for memory mapping files and avoiding full allocation
// - `SliceSafetensors` for working with in-memory buffers
// - `BufferedSafetensors` for owning a buffer of data
//
// Tensors can also be serialized to safetensor format using the `save` function or
// `Tensor.saveSafetensors` method.

@kotlin.test.Test
public fun saveSingleTensor() {
    val t = Tensor.zeros(Pair(2, 2), DType.F32, Device.Cpu).unwrap()
    t.saveSafetensors("t", "t.safetensors").unwrap()
    val bytes = read("t.safetensors").unwrap()
    check(bytes == "@\u0000\u0000\u0000\u0000\u0000\u0000\u0000{\"t\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}       \u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000".encodeToByteArray())
    removeFile("t.safetensors").unwrap()
}

@kotlin.test.Test
public fun saveLoadMultipleTensors() {
    val t = Tensor.zeros(Pair(2, 2), DType.F32, Device.Cpu).unwrap()
    val u = Tensor.zeros(Pair(1, 2), DType.F32, Device.Cpu).unwrap()
    val map: HashMap = listOf(Pair("t", t), Pair("u", u)).intoIter().collect()
    save(map, "multi.safetensors").unwrap()
    val weights = load("multi.safetensors", Device.Cpu).unwrap()
    check(weights.get("t")!!.dims() == listOf(2, 2))
    check(weights.get("u")!!.dims() == listOf(1, 2))
    val bytes = read("multi.safetensors").unwrap()
    check(bytes == "x\u0000\u0000\u0000\u0000\u0000\u0000\u0000{\"t\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"u\":{\"dtype\":\"F32\",\"shape\":[1,2],\"data_offsets\":[16,24]}}      \u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000".encodeToByteArray())
    removeFile("multi.safetensors").unwrap()
}

@kotlin.test.Test
public fun loadU8() {
    val bytes = "8\u0000\u0000\u0000\u0000\u0000\u0000\u0000{\"x\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}   \u0001\u0003".encodeToByteArray()
    write("test_u8.safetensors", bytes).unwrap()
    val weights = load("test_u8.safetensors", Device.Cpu).unwrap()
    val tensor = weights.get("x")!!
    check(tensor.dims() == listOf(2))
    check(tensor.dtype() == DType.U8)
    val data: List<UByte> = tensor.toVec1().unwrap()
    check(data == mutableListOf(1, 3))
    removeFile("test_u8.safetensors").unwrap()
}
