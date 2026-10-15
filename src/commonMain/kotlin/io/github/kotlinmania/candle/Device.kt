// port-lint: source device.rs
package io.github.kotlinmania.candle

/**
 * A `DeviceLocation` represents a physical device whereas multiple `Device`
 * can live on the same location (typically for cuda devices).
 */
public sealed class DeviceLocation {
    public object Cpu : DeviceLocation() {
        override fun toString(): String = "cpu"
    }

    public data class Cuda(
        val gpuId: Int,
    ) : DeviceLocation() {
        override fun toString(): String = "cuda:$gpuId"
    }

    public data class Metal(
        val gpuId: Int,
    ) : DeviceLocation() {
        override fun toString(): String = "metal:$gpuId"
    }
}

public interface NdArray {
    public fun shape(): Shape

    public fun toCpuStorage(): CpuStorage

    public companion object {
        private fun scalarCpuStorage(value: Any?): CpuStorage =
            when (value) {
                is UByte -> CpuStorage.U8(ubyteArrayOf(value))
                is UInt -> CpuStorage.U32(uintArrayOf(value))
                is Short -> CpuStorage.I16(shortArrayOf(value))
                is Int -> CpuStorage.I32(intArrayOf(value))
                is Long -> CpuStorage.I64(longArrayOf(value))
                is BFloat16 -> CpuStorage.BF16(shortArrayOf(value.bits))
                is Float16 -> CpuStorage.F16(shortArrayOf(value.bits))
                is Float -> CpuStorage.F32(floatArrayOf(value))
                is Double -> CpuStorage.F64(doubleArrayOf(value))
                is F8E4M3 -> CpuStorage.F8E4M3(byteArrayOf(value.bits))
                else -> throw CandleException.Msg("unsupported scalar type")
            }

        private fun listCpuStorage(values: List<*>): CpuStorage {
            if (values.isEmpty()) {
                return CpuStorage.F32(FloatArray(0))
            }
            return when (values.first()) {
                is UByte -> CpuStorage.U8(UByteArray(values.size) { values[it] as UByte })
                is UInt -> CpuStorage.U32(UIntArray(values.size) { values[it] as UInt })
                is Short -> CpuStorage.I16(ShortArray(values.size) { values[it] as Short })
                is Int -> CpuStorage.I32(IntArray(values.size) { values[it] as Int })
                is Long -> CpuStorage.I64(LongArray(values.size) { values[it] as Long })
                is BFloat16 -> CpuStorage.BF16(ShortArray(values.size) { (values[it] as BFloat16).bits })
                is Float16 -> CpuStorage.F16(ShortArray(values.size) { (values[it] as Float16).bits })
                is Float -> CpuStorage.F32(FloatArray(values.size) { values[it] as Float })
                is Double -> CpuStorage.F64(DoubleArray(values.size) { values[it] as Double })
                is F8E4M3 -> CpuStorage.F8E4M3(ByteArray(values.size) { (values[it] as F8E4M3).bits })
                else -> throw CandleException.Msg("unsupported element type")
            }
        }
    }

    public class Scalar<S>(
        public val value: S,
    ) : NdArray {
        override fun shape(): Shape = Shape()

        override fun toCpuStorage(): CpuStorage = scalarCpuStorage(value)
    }

    public class Array1<S>(
        public val values: Array<S>,
    ) : NdArray {
        override fun shape(): Shape = Shape(values.size)

        override fun toCpuStorage(): CpuStorage = listCpuStorage(values.asList())
    }

    public class Slice1<S>(
        public val values: List<S>,
    ) : NdArray {
        override fun shape(): Shape = Shape(values.size)

        override fun toCpuStorage(): CpuStorage = listCpuStorage(values)
    }

    public class Array2<S>(
        public val values: Array<Array<S>>,
    ) : NdArray {
        override fun shape(): Shape {
            val m = values.size
            val n = if (m == 0) 0 else values[0].size
            return Shape(m, n)
        }

        override fun toCpuStorage(): CpuStorage =
            listCpuStorage(values.flatMap { it.asList() })
    }

    public class Array3<S>(
        public val values: Array<Array<Array<S>>>,
    ) : NdArray {
        override fun shape(): Shape {
            val n1 = values.size
            val n2 = if (n1 == 0) 0 else values[0].size
            val n3 = if (n2 == 0) 0 else values[0][0].size
            return Shape(n1, n2, n3)
        }

        override fun toCpuStorage(): CpuStorage {
            val dst = ArrayList<S>()
            for (v1 in values) {
                for (v2 in v1) {
                    dst.addAll(v2.asList())
                }
            }
            return listCpuStorage(dst)
        }
    }

    public class Array4<S>(
        public val values: Array<Array<Array<Array<S>>>>,
    ) : NdArray {
        override fun shape(): Shape {
            val n1 = values.size
            val n2 = if (n1 == 0) 0 else values[0].size
            val n3 = if (n2 == 0) 0 else values[0][0].size
            val n4 = if (n3 == 0) 0 else values[0][0][0].size
            return Shape(n1, n2, n3, n4)
        }

        override fun toCpuStorage(): CpuStorage {
            val dst = ArrayList<S>()
            for (v1 in values) {
                for (v2 in v1) {
                    for (v3 in v2) {
                        dst.addAll(v3.asList())
                    }
                }
            }
            return listCpuStorage(dst)
        }
    }

    public class Vec1<S>(
        public val values: List<S>,
    ) : NdArray {
        override fun shape(): Shape = Shape(values.size)

        override fun toCpuStorage(): CpuStorage = listCpuStorage(values)
    }

    public class VecSlice<S>(
        public val values: List<List<S>>,
    ) : NdArray {
        override fun shape(): Shape {
            if (values.isEmpty()) {
                throw CandleException.Msg("empty array")
            }
            val n = values.size
            val m = values[0].size
            for (v in values) {
                if (v.size != m) {
                    throw CandleException.Msg("two elements have different len $m ${v.size}")
                }
            }
            return Shape(n, m)
        }

        override fun toCpuStorage(): CpuStorage =
            listCpuStorage(values.flatten())
    }

    public class Vec2<S>(
        public val values: List<List<S>>,
    ) : NdArray {
        override fun shape(): Shape {
            if (values.isEmpty()) {
                throw CandleException.Msg("empty array")
            }
            val n = values.size
            val m = values[0].size
            for (v in values) {
                if (v.size != m) {
                    throw CandleException.Msg("two elements have different len $m ${v.size}")
                }
            }
            return Shape(n, m)
        }

        override fun toCpuStorage(): CpuStorage {
            val len = values.sumOf { it.size }
            val dst = ArrayList<S>(len)
            for (v in values) {
                dst.addAll(v)
            }
            return listCpuStorage(dst)
        }
    }

    public class Vec3<S>(
        public val values: List<List<List<S>>>,
    ) : NdArray {
        override fun shape(): Shape {
            if (values.isEmpty()) {
                throw CandleException.Msg("empty array")
            }
            val shape0 = Vec2(values[0]).shape()
            val n = values.size
            for (v in values) {
                val shape = Vec2(v).shape()
                if (shape != shape0) {
                    throw CandleException.Msg("two elements have different shapes $shape $shape0")
                }
            }
            return Shape(listOf(n) + shape0.dims())
        }

        override fun toCpuStorage(): CpuStorage {
            if (values.isEmpty()) {
                return listCpuStorage(emptyList<S>())
            }
            val len = values.sumOf { v -> v.sumOf { it.size } }
            val dst = ArrayList<S>(len)
            for (v1 in values) {
                for (v2 in v1) {
                    dst.addAll(v2)
                }
            }
            return listCpuStorage(dst)
        }
    }

    public class Vec4<S>(
        public val values: List<List<List<List<S>>>>,
    ) : NdArray {
        override fun shape(): Shape {
            if (values.isEmpty()) {
                throw CandleException.Msg("empty array")
            }
            val shape0 = Vec3(values[0]).shape()
            val n = values.size
            for (v in values) {
                val shape = Vec3(v).shape()
                if (shape != shape0) {
                    throw CandleException.Msg("two elements have different shapes $shape $shape0")
                }
            }
            return Shape(listOf(n) + shape0.dims())
        }

        override fun toCpuStorage(): CpuStorage {
            val len = values.sumOf { v -> v.sumOf { v2 -> v2.sumOf { it.size } } }
            val dst = ArrayList<S>(len)
            for (v1 in values) {
                for (v2 in v1) {
                    for (v3 in v2) {
                        dst.addAll(v3)
                    }
                }
            }
            return listCpuStorage(dst)
        }
    }
}

/**
 * Cpu, Cuda, or Metal
 */
public sealed class Device {
    public object Cpu : Device() {
        override fun toString(): String = "Device::Cpu"
    }

    public data class Cuda(
        val device: CudaDevice,
    ) : Device() {
        override fun toString(): String = "Device::Cuda(${device.gpuId})"
    }

    public data class Metal(
        val device: MetalDevice,
    ) : Device() {
        override fun toString(): String = "Device::Metal(${device.gpuId})"
    }

    public fun asCudaDevice(): CudaDevice =
        when (this) {
            is Cuda -> device
            is Cpu -> throw CandleException.Msg("expected a cuda device, got cpu")
            is Metal -> throw CandleException.Msg("expected a cuda device, got Metal")
        }

    public fun asMetalDevice(): MetalDevice =
        when (this) {
            is Cuda -> throw CandleException.Msg("expected a metal device, got cuda")
            is Cpu -> throw CandleException.Msg("expected a metal device, got cpu")
            is Metal -> device
        }

    /**
     * Run `f` with device specific context.
     *
     * On CPU this installs candle's private rayon thread pool for the
     * duration of `f`, keeping worker threads warm across the many short
     * parallel sections in a model forward pass. Currently noop for other backends.
     */
    public fun <R> withContext(f: () -> R): R =
        when (this) {
            is Cpu -> withThreadpool(f)
            else -> f()
        }

    public fun setSeed(seed: Long) {
        when (this) {
            is Cpu -> CpuDevice.setSeed(seed)
            is Cuda -> device.setSeed(seed)
            is Metal -> device.setSeed(seed)
        }
    }

    public fun getCurrentSeed(): Long =
        when (this) {
            is Cpu -> CpuDevice.getCurrentSeed()
            is Cuda -> device.getCurrentSeed()
            is Metal -> device.getCurrentSeed()
        }

    public fun sameDevice(rhs: Device): Boolean =
        when {
            this is Cpu && rhs is Cpu -> true
            this is Cuda && rhs is Cuda -> device.gpuId == rhs.device.gpuId
            this is Metal && rhs is Metal -> device.gpuId == rhs.device.gpuId
            else -> false
        }

    public fun location(): DeviceLocation =
        when (this) {
            is Cpu -> DeviceLocation.Cpu
            is Cuda -> DeviceLocation.Cuda(device.gpuId)
            is Metal -> DeviceLocation.Metal(device.gpuId)
        }

    public fun isCpu(): Boolean = this is Cpu

    public fun isCuda(): Boolean = this is Cuda

    public fun isMetal(): Boolean = this is Metal

    public fun supportsBf16(): Boolean =
        when (this) {
            is Cuda, is Metal -> true
            is Cpu -> false
        }

    /**
     * Return `BF16` for devices that support it, otherwise default to `F32`.
     */
    public fun bf16DefaultToF32(): DType =
        if (supportsBf16()) {
            DType.BF16
        } else {
            DType.F32
        }

    public fun randUniformF64(
        lo: Double,
        up: Double,
        shape: Shape,
        dtype: DType,
    ): Storage =
        when (this) {
            is Cpu -> {
                val storage = CpuDevice.randUniform(shape, dtype, lo, up)
                Storage.Cpu(storage)
            }
            is Cuda -> {
                if (dtype == DType.F16 || dtype == DType.BF16) {
                    val storage = device.randUniform(shape, DType.F32, lo, up) as CudaStorage
                    Storage.Cuda(storage).toDtype(Layout.contiguous(shape), dtype)
                } else {
                    val storage = device.randUniform(shape, dtype, lo, up) as CudaStorage
                    Storage.Cuda(storage)
                }
            }
            is Metal -> {
                val storage = device.randUniform(shape, dtype, lo, up) as MetalStorage
                Storage.Metal(storage)
            }
        }

    public fun randUniform(
        lo: Double,
        up: Double,
        shape: Shape,
        dtype: DType = DType.F32,
    ): Storage = randUniformF64(lo, up, shape, dtype)

    public fun randNormalF64(
        mean: Double,
        std: Double,
        shape: Shape,
        dtype: DType,
    ): Storage =
        when (this) {
            is Cpu -> {
                val storage = CpuDevice.randNormal(shape, dtype, mean, std)
                Storage.Cpu(storage)
            }
            is Cuda -> {
                if (dtype == DType.F16 || dtype == DType.BF16) {
                    val storage = device.randNormal(shape, DType.F32, mean, std) as CudaStorage
                    Storage.Cuda(storage).toDtype(Layout.contiguous(shape), dtype)
                } else {
                    val storage = device.randNormal(shape, dtype, mean, std) as CudaStorage
                    Storage.Cuda(storage)
                }
            }
            is Metal -> {
                val storage = device.randNormal(shape, dtype, mean, std) as MetalStorage
                Storage.Metal(storage)
            }
        }

    public fun randNormal(
        mean: Double,
        std: Double,
        shape: Shape,
        dtype: DType = DType.F32,
    ): Storage = randNormalF64(mean, std, shape, dtype)

    public fun zeros(shape: Shape, dtype: DType): Storage =
        when (this) {
            is Cpu -> Storage.Cpu(CpuDevice.zerosImpl(shape, dtype))
            is Cuda -> Storage.Cuda(device.zerosImpl(shape, dtype) as CudaStorage)
            is Metal -> Storage.Metal(device.zerosImpl(shape, dtype) as MetalStorage)
        }

    public fun allocUninit(shape: Shape, dtype: DType): Storage =
        when (this) {
            is Cpu -> Storage.Cpu(CpuDevice.allocUninit(shape, dtype))
            is Cuda -> Storage.Cuda(device.allocUninit(shape, dtype) as CudaStorage)
            is Metal -> Storage.Metal(device.allocUninit(shape, dtype) as MetalStorage)
        }

    public fun <D> storageFromSlice(data: List<D>): Storage =
        when (this) {
            is Cpu -> Storage.Cpu(NdArray.Slice1(data).toCpuStorage())
            is Cuda -> Storage.Cuda(device.storageFromSlice(data))
            is Metal -> Storage.Metal(device.storageFromSlice(data))
        }

    public fun storage(array: NdArray): Storage =
        when (this) {
            is Cpu -> Storage.Cpu(array.toCpuStorage())
            is Cuda -> {
                val s = array.toCpuStorage()
                Storage.Cuda(device.storageFromCpuStorageOwned(s) as CudaStorage)
            }
            is Metal -> {
                val s = array.toCpuStorage()
                Storage.Metal(device.storageFromCpuStorageOwned(s) as MetalStorage)
            }
        }

    public fun <S> storageOwned(data: List<S>): Storage =
        when (this) {
            is Cpu -> Storage.Cpu(NdArray.Vec1(data).toCpuStorage())
            is Cuda -> {
                val s = NdArray.Vec1(data).toCpuStorage()
                Storage.Cuda(device.storageFromCpuStorageOwned(s) as CudaStorage)
            }
            is Metal -> {
                val s = NdArray.Vec1(data).toCpuStorage()
                Storage.Metal(device.storageFromCpuStorageOwned(s) as MetalStorage)
            }
        }

    public fun synchronize() {
        when (this) {
            is Cpu -> {}
            is Cuda -> device.synchronize()
            is Metal -> device.synchronize()
        }
    }

    public companion object {
        public fun newCuda(ordinal: Int): Device = Cuda(CudaDevice(ordinal))

        public fun newCudaWithStream(ordinal: Int): Device = Cuda(CudaDevice.newWithStream(ordinal))

        public fun newMetal(ordinal: Int): Device = Metal(MetalDevice(ordinal))

        public fun cudaIfAvailable(ordinal: Int): Device =
            if (cudaIsAvailable()) {
                newCuda(ordinal)
            } else {
                Cpu
            }

        public fun metalIfAvailable(ordinal: Int): Device =
            if (metalIsAvailable()) {
                newMetal(ordinal)
            } else {
                Cpu
            }
    }
}
