// port-lint: source convert.rs
package io.github.kotlinmania.candle

/**
 * Implement conversion traits for tensors
 */
public interface ByteWriter {
    public interface Error

    public fun write(bytes: ByteArray)
}

public fun Tensor.Companion.tryFrom(v: FloatArray): Tensor = fromSlice(v, Shape(v.size))

public fun Tensor.Companion.tryFrom(v: DoubleArray): Tensor = fromSlice(v, Shape(v.size))

public fun Tensor.Companion.tryFrom(v: IntArray): Tensor = fromSlice(v, Shape(v.size))

public fun Tensor.Companion.tryFrom(v: LongArray): Tensor = fromSlice(v, Shape(v.size))

public fun Tensor.Companion.tryFrom(v: List<Float>): Tensor = v.toTensor1()

public fun Tensor.Companion.tryFrom(v: Float): Tensor = v.toTensor()

public fun Tensor.Companion.tryFrom(v: Double): Tensor = v.toTensor()

public fun Tensor.Companion.tryFrom(v: Int): Tensor = v.toTensor()

public fun Tensor.Companion.tryFrom(v: Long): Tensor = v.toTensor()

public fun Tensor.writeBytes(): ByteArray {
    val vs = flattenAll()
    val cont = vs.contiguous()
    val cpu = (cont.storage() as Storage.Cpu).storage
    val count = cont.elemCount()
    return when (dtype()) {
        DType.U8 -> {
            val arr = (cpu as CpuStorage.U8).data
            ByteArray(count) { arr[it].toByte() }
        }
        DType.U32 -> {
            val arr = (cpu as CpuStorage.U32).data
            val res = ByteArray(count * 4)
            for (i in 0 until count) {
                val v = arr[i].toLong()
                res[i * 4] = (v and 0xFF).toByte()
                res[i * 4 + 1] = ((v shr 8) and 0xFF).toByte()
                res[i * 4 + 2] = ((v shr 16) and 0xFF).toByte()
                res[i * 4 + 3] = ((v shr 24) and 0xFF).toByte()
            }
            res
        }
        DType.I16 -> {
            val arr = (cpu as CpuStorage.I16).data
            val res = ByteArray(count * 2)
            for (i in 0 until count) {
                val v = arr[i].toInt()
                res[i * 2] = (v and 0xFF).toByte()
                res[i * 2 + 1] = ((v shr 8) and 0xFF).toByte()
            }
            res
        }
        DType.I32 -> {
            val arr = (cpu as CpuStorage.I32).data
            val res = ByteArray(count * 4)
            for (i in 0 until count) {
                val v = arr[i]
                res[i * 4] = (v and 0xFF).toByte()
                res[i * 4 + 1] = ((v shr 8) and 0xFF).toByte()
                res[i * 4 + 2] = ((v shr 16) and 0xFF).toByte()
                res[i * 4 + 3] = ((v shr 24) and 0xFF).toByte()
            }
            res
        }
        DType.I64 -> {
            val arr = (cpu as CpuStorage.I64).data
            val res = ByteArray(count * 8)
            for (i in 0 until count) {
                val v = arr[i]
                for (b in 0 until 8) {
                    res[i * 8 + b] = ((v shr (b * 8)) and 0xFF).toByte()
                }
            }
            res
        }
        DType.BF16 -> {
            val arr = (cpu as CpuStorage.BF16).data
            val res = ByteArray(count * 2)
            for (i in 0 until count) {
                val v = arr[i].toInt()
                res[i * 2] = (v and 0xFF).toByte()
                res[i * 2 + 1] = ((v shr 8) and 0xFF).toByte()
            }
            res
        }
        DType.F16 -> {
            val arr = (cpu as CpuStorage.F16).data
            val res = ByteArray(count * 2)
            for (i in 0 until count) {
                val v = arr[i].toInt()
                res[i * 2] = (v and 0xFF).toByte()
                res[i * 2 + 1] = ((v shr 8) and 0xFF).toByte()
            }
            res
        }
        DType.F32 -> {
            val arr = (cpu as CpuStorage.F32).data
            val res = ByteArray(count * 4)
            for (i in 0 until count) {
                val v = arr[i].toBits()
                res[i * 4] = (v and 0xFF).toByte()
                res[i * 4 + 1] = ((v shr 8) and 0xFF).toByte()
                res[i * 4 + 2] = ((v shr 16) and 0xFF).toByte()
                res[i * 4 + 3] = ((v shr 24) and 0xFF).toByte()
            }
            res
        }
        DType.F64 -> {
            val arr = (cpu as CpuStorage.F64).data
            val res = ByteArray(count * 8)
            for (i in 0 until count) {
                val v = arr[i].toBits()
                for (b in 0 until 8) {
                    res[i * 8 + b] = ((v shr (b * 8)) and 0xFF).toByte()
                }
            }
            res
        }
        DType.F8E4M3 -> {
            val arr = (cpu as CpuStorage.F8E4M3).data
            arr.copyOf()
        }
        DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
            throw CandleException.UnsupportedDTypeForOp(dtype(), "write_bytes")
        }
    }
}

public fun Tensor.writeBytes(writer: ByteWriter) {
    writer.write(writeBytes())
}

public fun Float.toTensor(device: Device = Device.Cpu): Tensor =
    Tensor.fromSlice(floatArrayOf(this), Shape(1), device)

public fun Double.toTensor(device: Device = Device.Cpu): Tensor =
    Tensor.fromSlice(doubleArrayOf(this), Shape(1), device)

public fun Int.toTensor(device: Device = Device.Cpu): Tensor =
    Tensor.fromSlice(intArrayOf(this), Shape(1), device)

public fun Long.toTensor(device: Device = Device.Cpu): Tensor =
    Tensor.fromSlice(longArrayOf(this), Shape(1), device)

public fun FloatArray.toTensor(device: Device = Device.Cpu): Tensor =
    Tensor.fromSlice(this, Shape(size), device)

public fun DoubleArray.toTensor(device: Device = Device.Cpu): Tensor =
    Tensor.fromSlice(this, Shape(size), device)

public fun IntArray.toTensor(device: Device = Device.Cpu): Tensor =
    Tensor.fromSlice(this, Shape(size), device)

public fun LongArray.toTensor(device: Device = Device.Cpu): Tensor =
    Tensor.fromSlice(this, Shape(size), device)

public fun UIntArray.toTensor(device: Device = Device.Cpu): Tensor =
    Tensor.fromSlice(this, Shape(size), device)

public fun UByteArray.toTensor(device: Device = Device.Cpu): Tensor =
    Tensor.fromSlice(this, Shape(size), device)

public fun List<Float>.toTensor1(device: Device = Device.Cpu): Tensor =
    Tensor.fromSlice(toFloatArray(), Shape(size), device)

public fun List<List<Float>>.toTensor2(device: Device = Device.Cpu): Tensor {
    val d0 = size
    val d1 = if (d0 > 0) this[0].size else 0
    val flat = FloatArray(d0 * d1)
    var idx = 0
    for (row in this) {
        for (v in row) {
            flat[idx++] = v
        }
    }
    return Tensor.fromSlice(flat, Shape(d0, d1), device)
}

public fun List<List<List<Float>>>.toTensor3(device: Device = Device.Cpu): Tensor {
    val d0 = size
    val d1 = if (d0 > 0) this[0].size else 0
    val d2 = if (d1 > 0) this[0][0].size else 0
    val flat = FloatArray(d0 * d1 * d2)
    var idx = 0
    for (plane in this) {
        for (row in plane) {
            for (v in row) {
                flat[idx++] = v
            }
        }
    }
    return Tensor.fromSlice(flat, Shape(d0, d1, d2), device)
}
