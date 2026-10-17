// port-lint: source device.rs
package io.github.kotlinmania.candle

/**
 * A `DeviceLocation` represents a physical device whereas multiple `Device`
 * can live on the same location (typically for cuda devices).
 */
public sealed class DeviceLocation {
    public object Cpu : DeviceLocation()

    public data class Cuda(
        val gpuId: Int,
    ) : DeviceLocation()

    public data class Metal(
        val gpuId: Int,
    ) : DeviceLocation()
}

private fun <T> Array<T>.len(): Int = this.size

private fun <T> Array<T>.asSlice(): List<T> = this.asList()

private fun <T> Array<Array<T>>.concat(): List<T> = this.flatMap { it.asList() }

private fun <T> List<T>.copied(): List<T> = this

private fun <T> List<T>.toCpuStorage(): CpuStorage = NdArray.S.toCpuStorage(this)

private fun <T> MutableList<T>.extend(arr: Array<T>) {
    this.addAll(arr.asList())
}

private fun <T> MutableList<T>.extendList(list: List<T>) {
    this.addAll(list)
}

private val IntArray.asSlice: () -> List<Int>
    get() = { this.toList() }

private val List<List<Int>>.concat: () -> List<Int>
    get() = { this.flatten() }

private fun List<List<*>>.shape(): Shape {
    if (this.isNotEmpty() && this[0].isNotEmpty() && this[0][0] is List<*>) {
        val list3 = this.map { row -> row.map { cell -> cell as List<*> } }
        return NdArray.Vec3(list3).shape()
    }
    return NdArray.Vec2(this).shape()
}

public interface NdArray {
    public fun shape(): Shape

    public fun toCpuStorage(): CpuStorage

    public object S {
        public val toCpuStorage: (Any?) -> CpuStorage = { value ->
            when (value) {
                is List<*> -> listCpuStorage(value)
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
        }

        public val toCpuStorageOwned: (List<*>) -> CpuStorage = { values ->
            listCpuStorage(values)
        }

        private val listCpuStorage: (List<*>) -> CpuStorage = listLambda@{ values ->
            if (values.isEmpty()) {
                return@listLambda CpuStorage.F32(FloatArray(0))
            }
            when (values.first()) {
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

    public object Vec {
        internal fun <T> withCapacity(cap: Int): MutableList<T> = ArrayList(cap)
    }

    public class Scalar<T>(
        public val value: T,
    ) : NdArray {
        override fun shape(): Shape = Shape.from()

        override fun toCpuStorage(): CpuStorage = S.toCpuStorage(listOf(value))
    }

    public class Array1<T>(
        public val value: Array<T>,
    ) : NdArray {
        override fun shape(): Shape = Shape.from(value.len())

        override fun toCpuStorage(): CpuStorage = S.toCpuStorage(value.asSlice())
    }

    public class Slice1<T>(
        public val value: List<T>,
    ) : NdArray {
        override fun shape(): Shape = Shape.from(value.len())

        override fun toCpuStorage(): CpuStorage = S.toCpuStorage(value)
    }

    public class Array2<T> internal constructor(
        public val value: Array<Array<T>>,
        public val m: Int = value.size,
        public val n: Int = if (value.isEmpty()) 0 else value[0].size,
    ) : NdArray {
        override fun shape(): Shape = Shape.from(m, n)

        override fun toCpuStorage(): CpuStorage = S.toCpuStorageOwned(value.concat())
    }

    public class Array3<T> internal constructor(
        public val value: Array<Array<Array<T>>>,
        public val n1: Int = value.size,
        public val n2: Int = if (value.isEmpty()) 0 else value[0].size,
        public val n3: Int = if (value.isEmpty() || value[0].isEmpty()) 0 else value[0][0].size,
    ) : NdArray {
        override fun shape(): Shape = Shape.from(n1, n2, n3)

        override fun toCpuStorage(): CpuStorage {
            val vec = Vec.withCapacity<T>(n1 * n2 * n3)
            for (i1 in 0 until n1) {
                for (i2 in 0 until n2) {
                    vec.extend(value[i1][i2])
                }
            }
            return S.toCpuStorageOwned(vec)
        }
    }

    public class Array4<T> internal constructor(
        public val value: Array<Array<Array<Array<T>>>>,
        public val n1: Int = value.size,
        public val n2: Int = if (value.isEmpty()) 0 else value[0].size,
        public val n3: Int = if (value.isEmpty() || value[0].isEmpty()) 0 else value[0][0].size,
        public val n4: Int = if (value.isEmpty() || value[0].isEmpty() || value[0][0].isEmpty()) 0 else value[0][0][0].size,
    ) : NdArray {
        override fun shape(): Shape = Shape.from(n1, n2, n3, n4)

        override fun toCpuStorage(): CpuStorage {
            val vec = Vec.withCapacity<T>(n1 * n2 * n3 * n4)
            for (i1 in 0 until n1) {
                for (i2 in 0 until n2) {
                    for (i3 in 0 until n3) {
                        vec.extend(value[i1][i2][i3])
                    }
                }
            }
            return S.toCpuStorageOwned(vec)
        }
    }

    public class Vec1<T>(
        public val value: List<T>,
    ) : NdArray {
        override fun shape(): Shape = Shape.from(value.len())

        override fun toCpuStorage(): CpuStorage = S.toCpuStorage(value.asSlice())
    }

    public class VecSlice<T>(
        public val value: List<List<T>>,
    ) : NdArray {
        override fun shape(): Shape {
            if (value.isEmpty()) {
                bail("empty array")
            }
            val n = value.len()
            val m = value[0].len()
            for (v in value.iter()) {
                if (v.len() != m) {
                    bail("two elements have different len $m ${v.len()}")
                }
            }
            return Shape.from(n, m)
        }

        override fun toCpuStorage(): CpuStorage {
            val data =
                value
                    .iter()
                    .copied()
                    .flatten()
                    .copied()
                    .collect()
            return S.toCpuStorageOwned(data)
        }
    }

    public class Vec2<T>(
        public val value: List<List<T>>,
    ) : NdArray {
        override fun shape(): Shape {
            if (value.isEmpty()) {
                bail("empty array")
            }
            val n = value.len()
            val m = value[0].len()
            for (v in value.iter()) {
                if (v.len() != m) {
                    bail("two elements have different len $m ${v.len()}")
                }
            }
            return Shape.from(n, m)
        }

        override fun toCpuStorage(): CpuStorage {
            val len = value.iter().map { v -> v.len() }.sum()
            val dst = Vec.withCapacity<T>(len)
            for (v in value.iter()) {
                dst.extendList(v.iter().copied())
            }
            return S.toCpuStorageOwned(dst)
        }
    }

    public class Vec3<T>(
        public val value: List<List<List<T>>>,
    ) : NdArray {
        override fun shape(): Shape {
            if (value.isEmpty()) {
                bail("empty array")
            }
            val shape0 = value[0].shape()
            val n = value.len()
            for (v in value.iter()) {
                val shape = v.shape()
                if (shape != shape0) {
                    bail("two elements have different shapes $shape $shape0")
                }
            }
            return Shape.from(listOf(intArrayOf(n).asSlice(), shape0.dims()).concat())
        }

        override fun toCpuStorage(): CpuStorage {
            if (value.isEmpty()) {
                return S.toCpuStorageOwned(emptyList<T>())
            }
            val len =
                value
                    .iter()
                    .map { v -> v.iter().map { v2 -> v2.len() }.sum() }
                    .sum()
            val dst = Vec.withCapacity<T>(len)
            for (v1 in value.iter()) {
                for (v2 in v1.iter()) {
                    dst.extendList(v2.iter().copied())
                }
            }
            return S.toCpuStorageOwned(dst)
        }
    }

    public class Vec4<T>(
        public val value: List<List<List<List<T>>>>,
    ) : NdArray {
        override fun shape(): Shape {
            if (value.isEmpty()) {
                bail("empty array")
            }
            val shape0 = value[0].shape()
            val n = value.len()
            for (v in value.iter()) {
                val shape = v.shape()
                if (shape != shape0) {
                    bail("two elements have different shapes $shape $shape0")
                }
            }
            return Shape.from(listOf(intArrayOf(n).asSlice(), shape0.dims()).concat())
        }

        override fun toCpuStorage(): CpuStorage {
            val len =
                value
                    .iter()
                    .map { v ->
                        v
                            .iter()
                            .map { v2 -> v2.iter().map { v3 -> v3.len() }.sum() }
                            .sum()
                    }.sum()
            val dst = Vec.withCapacity<T>(len)
            for (v1 in value.iter()) {
                for (v2 in v1.iter()) {
                    for (v3 in v2.iter()) {
                        dst.extendList(v3.iter().copied())
                    }
                }
            }
            return S.toCpuStorageOwned(dst)
        }
    }
}

/**
 * Cpu, Cuda, or Metal
 */
public sealed class Device {
    private val self: Device = this

    public object Cpu : Device()

    public data class Cuda(
        val device: CudaDevice,
    ) : Device()

    public data class Metal(
        val device: MetalDevice,
    ) : Device()

    public fun asCudaDevice(): CudaDevice =
        when (self) {
            is Cuda -> self.device
            is Cpu -> bail("expected a cuda device, got cpu")
            is Metal -> bail("expected a cuda device, got Metal")
        }

    public fun asMetalDevice(): MetalDevice =
        when (self) {
            is Cuda -> bail("expected a metal device, got cuda")
            is Cpu -> bail("expected a metal device, got cpu")
            is Metal -> self.device
        }

    /**
     * Run `f` with device specific context.
     *
     * On CPU this installs candle's private rayon thread pool for the
     * duration of `f`, keeping worker threads warm across the many short
     * parallel sections in a model forward pass. Currently noop for other backends.
     */
    public fun <R> withContext(f: () -> R): R =
        when (self) {
            is Cpu -> {
                withThreadpool(f)
            }

            else -> {
                f()
            }
        }

    public fun setSeed(seed: Long) {
        when (self) {
            is Cpu -> CpuDevice.setSeed(seed)
            is Cuda -> self.device.setSeed(seed)
            is Metal -> self.device.setSeed(seed)
        }
    }

    public fun getCurrentSeed(): Long =
        when (self) {
            is Cpu -> CpuDevice.getCurrentSeed()
            is Cuda -> self.device.getCurrentSeed()
            is Metal -> self.device.getCurrentSeed()
        }

    public fun sameDevice(rhs: Device): Boolean =
        when {
            self is Cpu && rhs is Cpu -> true
            self is Cuda && rhs is Cuda -> self.device.sameDevice(rhs.device)
            self is Metal && rhs is Metal -> self.device.sameDevice(rhs.device)
            else -> false
        }

    public fun location(): DeviceLocation =
        when (self) {
            is Cpu -> DeviceLocation.Cpu
            is Cuda -> self.device.location()
            is Metal -> self.device.location()
        }

    public fun isCpu(): Boolean = matches(self, Cpu::class)

    public fun isCuda(): Boolean = matches(self, Cuda::class)

    public fun isMetal(): Boolean = matches(self, Metal::class)

    public fun supportsBf16(): Boolean =
        when (self) {
            is Cuda, is Metal -> true
            is Cpu -> false
        }

    /**
     * Return `BF16` for devices that support it, otherwise default to `F32`.
     */
    public fun bf16DefaultToF32(): DType =
        if (self.supportsBf16()) {
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
        when (self) {
            is Cpu -> {
                val storage = CpuDevice.randUniform(shape, dtype, lo, up)
                Storage.Cpu(storage)
            }

            is Cuda -> {
                // Convert generated f32 samples
                // when generating f16 or bf16 on CUDA.
                if (dtype == DType.F16 || dtype == DType.BF16) {
                    val storage = self.device.randUniform(shape, DType.F32, lo, up) as CudaStorage
                    Storage.Cuda(storage).toDtype(Layout.contiguous(shape), dtype)
                } else {
                    val storage = self.device.randUniform(shape, dtype, lo, up) as CudaStorage
                    Storage.Cuda(storage)
                }
            }

            is Metal -> {
                val storage = self.device.randUniform(shape, dtype, lo, up) as MetalStorage
                Storage.Metal(storage)
            }
        }

    public fun randUniform(
        lo: Double,
        up: Double,
        shape: Shape,
        dtype: DType = DType.F32,
    ): Storage = self.randUniformF64(lo, up, shape, dtype)

    public fun randNormalF64(
        mean: Double,
        std: Double,
        shape: Shape,
        dtype: DType,
    ): Storage =
        when (self) {
            is Cpu -> {
                val storage = CpuDevice.randNormal(shape, dtype, mean, std)
                Storage.Cpu(storage)
            }

            is Cuda -> {
                // Convert generated f32 samples
                // when generating f16 or bf16 on CUDA.
                if (dtype == DType.F16 || dtype == DType.BF16) {
                    val storage = self.device.randNormal(shape, DType.F32, mean, std) as CudaStorage
                    Storage.Cuda(storage).toDtype(Layout.contiguous(shape), dtype)
                } else {
                    val storage = self.device.randNormal(shape, dtype, mean, std) as CudaStorage
                    Storage.Cuda(storage)
                }
            }

            is Metal -> {
                val storage = self.device.randNormal(shape, dtype, mean, std) as MetalStorage
                Storage.Metal(storage)
            }
        }

    public fun randNormal(
        mean: Double,
        std: Double,
        shape: Shape,
        dtype: DType = DType.F32,
    ): Storage = self.randNormalF64(mean, std, shape, dtype)

    public fun zeros(
        shape: Shape,
        dtype: DType,
    ): Storage =
        when (self) {
            is Cpu -> {
                val storage = CpuDevice.zerosImpl(shape, dtype)
                Storage.Cpu(storage)
            }

            is Cuda -> {
                val storage = self.device.zerosImpl(shape, dtype) as CudaStorage
                Storage.Cuda(storage)
            }

            is Metal -> {
                val storage = self.device.zerosImpl(shape, dtype) as MetalStorage
                Storage.Metal(storage)
            }
        }

    public fun allocUninit(
        shape: Shape,
        dtype: DType,
    ): Storage =
        when (self) {
            is Cpu -> {
                val storage = CpuDevice.allocUninit(shape, dtype)
                Storage.Cpu(storage)
            }

            is Cuda -> {
                val storage = self.device.allocUninit(shape, dtype) as CudaStorage
                Storage.Cuda(storage)
            }

            is Metal -> {
                val storage = self.device.allocUninit(shape, dtype) as MetalStorage
                Storage.Metal(storage)
            }
        }

    public fun <D> storageFromSlice(data: List<D>): Storage =
        when (self) {
            is Cpu -> {
                Storage.Cpu(data.toCpuStorage())
            }

            is Cuda -> {
                val storage = self.device.storageFromSlice(data)
                Storage.Cuda(storage)
            }

            is Metal -> {
                val storage = self.device.storageFromSlice(data)
                Storage.Metal(storage)
            }
        }

    public fun storage(array: NdArray): Storage =
        when (self) {
            is Cpu -> {
                Storage.Cpu(array.toCpuStorage())
            }

            is Cuda -> {
                val storage = array.toCpuStorage()
                val cudaStorage = self.device.storageFromCpuStorageOwned(storage) as CudaStorage
                Storage.Cuda(cudaStorage)
            }

            is Metal -> {
                val storage = array.toCpuStorage()
                val metalStorage = self.device.storageFromCpuStorageOwned(storage) as MetalStorage
                Storage.Metal(metalStorage)
            }
        }

    public fun <S> storageOwned(data: List<S>): Storage =
        when (self) {
            is Cpu -> {
                Storage.Cpu(NdArray.S.toCpuStorageOwned(data))
            }

            is Cuda -> {
                val storage = NdArray.S.toCpuStorageOwned(data)
                val cudaStorage = self.device.storageFromCpuStorageOwned(storage) as CudaStorage
                Storage.Cuda(cudaStorage)
            }

            is Metal -> {
                val storage = NdArray.S.toCpuStorageOwned(data)
                val metalStorage = self.device.storageFromCpuStorageOwned(storage) as MetalStorage
                Storage.Metal(metalStorage)
            }
        }

    public fun synchronize() {
        when (self) {
            is Cpu -> Unit
            is Cuda -> self.device.synchronize()
            is Metal -> self.device.synchronize()
        }
    }

    public companion object {
        public val matches: (Device, kotlin.reflect.KClass<out Device>) -> Boolean = { d, k -> k.isInstance(d) }

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
