package io.github.kotlinmania.candle

import io.github.kotlinmania.envie.getenv
import io.github.kotlinmania.threadlocal.ThreadLocal

private val ByteArray.len: () -> Int get() = { this.size }
private val ByteArray.asPtr: () -> Int get() = { this.size }
private inline fun Tensor.unwrapOrElse(f: (CandleException) -> Tensor): Tensor = this
private inline fun Int.unwrapOrElse(f: (CandleException) -> Int): Int = this

internal fun asTSlice(data: ByteArray, companion: Any? = null): List<Any> {
    val size = QuantizedHelper.sizeOf(companion)
    check(
        data.len() % size == 0,
    ) {
        "Data length must be a multiple of element size"
    }
    val ptr = data.asPtr()
    check(
        ptr % QuantizedHelper.alignOf(companion) == 0,
    ) {
        "Data pointer must be aligned to element alignment"
    }
    return QuantizedHelper.fromRawParts(ptr, data.len() / size, data, companion)
}

internal fun Device.qzeros(elemCount: Int, dtype: GgmlDType): CandleResult<QStorage> {
    return when (val self = this) {
        is Device.Cpu -> {
            val storage = dtype.cpuZeros(elemCount)
            CandleResult.Ok(QStorage.Cpu(storage))
        }
        is Device.Metal -> {
            val storage = QMetalStorage.zeros(self.device, elemCount, dtype).unwrapOrElse { return CandleResult.Err(it) }
            CandleResult.Ok(QStorage.Metal(storage))
        }
        is Device.Cuda -> {
            val storage = QCudaStorage.zeros(self.device, elemCount, dtype).unwrapOrElse { return CandleResult.Err(it) }
            CandleResult.Ok(QStorage.Cuda(storage))
        }
    }
}

sealed class QStorage {
    private val self: QStorage get() = this

    data class Cpu(val storage: QuantizedType) : QStorage()
    data class Metal(val storage: QMetalStorage) : QStorage()
    data class Cuda(val storage: QCudaStorage) : QStorage()

    fun blockSize(): Int {
        return when (self) {
            is Cpu -> (this as Cpu).storage.blockSize()
            is Metal -> (this as Metal).storage.dtype().blockSize()
            is Cuda -> (this as Cuda).storage.dtype().blockSize()
        }
    }

    fun dtype(): GgmlDType {
        return when (self) {
            is Cpu -> (this as Cpu).storage.dtype()
            is Metal -> (this as Metal).storage.dtype()
            is Cuda -> (this as Cuda).storage.dtype()
        }
    }

    fun device(): Device {
        return when (self) {
            is Cpu -> Device.Cpu
            is Metal -> Device.Metal((this as Metal).storage.device())
            is Cuda -> Device.Cuda((this as Cuda).storage.device())
        }
    }

    fun sizeInBytes(): Int {
        return when (self) {
            is Cpu -> (this as Cpu).storage.storageSizeInBytes()
            is Metal -> (this as Metal).storage.storageSizeInBytes()
            is Cuda -> (this as Cuda).storage.storageSizeInBytes()
        }
    }

    fun quantize(src: Storage): CandleResult<Unit> {
        when {
            self is Cpu && src is Storage.Cpu -> {
                (this as Cpu).storage.fromFloat(QuantizedHelper.asSliceF32(src.storage).unwrapOrElse { return CandleResult.Err(it) })
            }
            self is Metal && src is Storage.Metal -> (this as Metal).storage.quantize(src.storage).unwrapOrElse { return CandleResult.Err(it) }
            self is Cuda && src is Storage.Cuda -> (this as Cuda).storage.quantize(src.storage).unwrapOrElse { return CandleResult.Err(it) }
            else -> return CandleResult.Err(CandleError.Msg("Invalid quantize storage locations do not match").bt())
        }
        return CandleResult.Ok(Unit)
    }

    fun quantizeImatrix(
        src: Storage,
        imatrixWeights: FloatArray,
        nPerRow: Int,
    ): CandleResult<Unit> {
        when {
            self is Cpu && src is Storage.Cpu -> {
                (this as Cpu).storage.fromFloatImatrix(
                    QuantizedHelper.asSliceF32(src.storage).unwrapOrElse { return CandleResult.Err(it) },
                    imatrixWeights,
                    nPerRow,
                )
            }
            self is Metal && src is Storage.Metal ->
                (this as Metal).storage.quantizeImatrix(src.storage, imatrixWeights, nPerRow).unwrapOrElse { return CandleResult.Err(it) }
            self is Cuda && src is Storage.Cuda -> (this as Cuda).storage.quantizeImatrix(src.storage, imatrixWeights, nPerRow).unwrapOrElse { return CandleResult.Err(it) }
            else -> return CandleResult.Err(CandleError.Msg("Invalid quantize storage locations do not match").bt())
        }
        return CandleResult.Ok(Unit)
    }

    fun quantizeOnto(src: Storage): CandleResult<Unit> {
        when {
            self is Cpu && src is Storage.Cpu -> {
                (this as Cpu).storage.fromFloat(QuantizedHelper.asSliceF32(src.storage).unwrapOrElse { return CandleResult.Err(it) })
            }
            self is Metal && src is Storage.Cpu -> (this as Metal).storage.quantizeOnto(src.storage).unwrapOrElse { return CandleResult.Err(it) }
            self is Cuda && src is Storage.Cpu -> (this as Cuda).storage.quantizeOnto(src.storage).unwrapOrElse { return CandleResult.Err(it) }
            else -> return CandleResult.Err(CandleError.Msg("Invalid quantize source storage locations: not on cpu").bt())
        }
        return CandleResult.Ok(Unit)
    }

    fun quantizeImatrixOnto(
        src: Storage,
        imatrixWeights: FloatArray,
        nPerRow: Int,
    ): CandleResult<Unit> {
        when {
            self is Cpu && src is Storage.Cpu -> {
                (this as Cpu).storage.fromFloatImatrix(
                    QuantizedHelper.asSliceF32(src.storage).unwrapOrElse { return CandleResult.Err(it) },
                    imatrixWeights,
                    nPerRow,
                )
            }
            self is Metal && src is Storage.Cpu ->
                (this as Metal).storage.quantizeImatrixOnto(src.storage, imatrixWeights, nPerRow).unwrapOrElse { return CandleResult.Err(it) }
            self is Cuda && src is Storage.Cpu -> (this as Cuda).storage.quantizeImatrixOnto(src.storage, imatrixWeights, nPerRow).unwrapOrElse { return CandleResult.Err(it) }
            else -> return CandleResult.Err(CandleError.Msg("Invalid quantize storage locations do not match").bt())
        }
        return CandleResult.Ok(Unit)
    }

    fun dequantize(elemCount: Int): CandleResult<Storage> {
        return when (self) {
            is Cpu -> CandleResult.Ok(Storage.Cpu((this as Cpu).storage.dequantize(elemCount).unwrapOrElse { return CandleResult.Err(it) }))
            is Metal -> CandleResult.Ok(Storage.Metal((this as Metal).storage.dequantize(elemCount).unwrapOrElse { return CandleResult.Err(it) }))
            is Cuda -> CandleResult.Ok(Storage.Cuda((this as Cuda).storage.dequantize(elemCount).unwrapOrElse { return CandleResult.Err(it) }))
        }
    }

    fun data(): CandleResult<ByteArray> {
        return when (self) {
            is Cpu -> {
                val dataPtr = (this as Cpu).storage.asPtr()
                val sizeInBytes = (this as Cpu).storage.storageSizeInBytes()
                val data = QuantizedHelper.cowFromParts(dataPtr, sizeInBytes)
                CandleResult.Ok(QuantizedHelper.cowFrom(data))
            }
            is Cuda -> CandleResult.Ok(QuantizedHelper.cowFrom((this as Cuda).storage.data().unwrapOrElse { return CandleResult.Err(it) }))
            is Metal -> CandleResult.Ok(QuantizedHelper.cowFrom((this as Metal).storage.data().unwrapOrElse { return CandleResult.Err(it) }))
        }
    }

    fun devicePtr(): CandleResult<Long> {
        return when (self) {
            is Cuda -> (this as Cuda).storage.devicePtr()
            is Metal, is Cpu -> {
                CandleResult.Err(CandleError.Msg("not implemented").bt())
            }
        }
    }

    fun devicePtrWithGuard(
        stream: Long,
    ): CandleResult<
        Pair<Long,
            Long
        >
    > {
        return when (self) {
            is Cuda -> QuantizedHelper.devicePtrWithGuard((this as Cuda).storage, stream)
            is Metal, is Cpu -> {
                CandleResult.Err(CandleError.Msg("not implemented").bt())
            }
        }
    }

    companion object {
        fun zeros(device: Device, elemCount: Int, dtype: GgmlDType): CandleResult<QStorage> =
            device.qzeros(elemCount, dtype)

        fun fromData(data: ByteArray, device: Device, dtype: GgmlDType): CandleResult<QStorage> {
            val rawData = data
            return when (device) {
                is Device.Cpu -> CandleResult.Ok(Cpu(dtype.fromData(QuantizedHelper.cowBorrowed(rawData))))
                is Device.Metal -> when (dtype) {
                    GgmlDType.F32 -> loadQuantized(device.device, asTSlice(rawData, F32GgmlType))
                    GgmlDType.F16 -> loadQuantized(device.device, asTSlice(rawData, F16))
                    GgmlDType.Q4_0 -> loadQuantized(device.device, asTSlice(rawData, BlockQ40))
                    GgmlDType.Q4_1 -> loadQuantized(device.device, asTSlice(rawData, BlockQ41))
                    GgmlDType.Q5_0 -> loadQuantized(device.device, asTSlice(rawData, BlockQ50))
                    GgmlDType.Q5_1 -> loadQuantized(device.device, asTSlice(rawData, BlockQ51))
                    GgmlDType.Q8_0 -> loadQuantized(device.device, asTSlice(rawData, BlockQ80))
                    GgmlDType.Q8_1 -> loadQuantized(device.device, asTSlice(rawData, BlockQ81))
                    GgmlDType.Q2K -> loadQuantized(device.device, asTSlice(rawData, BlockQ2K))
                    GgmlDType.Q3K -> loadQuantized(device.device, asTSlice(rawData, BlockQ3K))
                    GgmlDType.Q4K -> loadQuantized(device.device, asTSlice(rawData, BlockQ4K))
                    GgmlDType.Q5K -> loadQuantized(device.device, asTSlice(rawData, BlockQ5K))
                    GgmlDType.Q6K -> loadQuantized(device.device, asTSlice(rawData, BlockQ6K))
                    GgmlDType.Q8K -> loadQuantized(device.device, asTSlice(rawData, BlockQ8K))
                    GgmlDType.BF16 -> loadQuantized(device.device, asTSlice(rawData, BF16))
                }
                is Device.Cuda -> when (dtype) {
                    GgmlDType.F32 -> loadQuantized(device.device, asTSlice(rawData, F32GgmlType))
                    GgmlDType.F16 -> loadQuantized(device.device, asTSlice(rawData, F16))
                    GgmlDType.Q4_0 -> loadQuantized(device.device, asTSlice(rawData, BlockQ40))
                    GgmlDType.Q4_1 -> loadQuantized(device.device, asTSlice(rawData, BlockQ41))
                    GgmlDType.Q5_0 -> loadQuantized(device.device, asTSlice(rawData, BlockQ50))
                    GgmlDType.Q5_1 -> loadQuantized(device.device, asTSlice(rawData, BlockQ51))
                    GgmlDType.Q8_0 -> loadQuantized(device.device, asTSlice(rawData, BlockQ80))
                    GgmlDType.Q8_1 -> loadQuantized(device.device, asTSlice(rawData, BlockQ81))
                    GgmlDType.Q2K -> loadQuantized(device.device, asTSlice(rawData, BlockQ2K))
                    GgmlDType.Q3K -> loadQuantized(device.device, asTSlice(rawData, BlockQ3K))
                    GgmlDType.Q4K -> loadQuantized(device.device, asTSlice(rawData, BlockQ4K))
                    GgmlDType.Q5K -> loadQuantized(device.device, asTSlice(rawData, BlockQ5K))
                    GgmlDType.Q6K -> loadQuantized(device.device, asTSlice(rawData, BlockQ6K))
                    GgmlDType.Q8K -> loadQuantized(device.device, asTSlice(rawData, BlockQ8K))
                    GgmlDType.BF16 -> loadQuantized(device.device, asTSlice(rawData, BF16))
                }
            }
        }
    }
}

private val DEQUANTIZE_ALL: ThreadLocal<Boolean> = ThreadLocal()
private val DEQUANTIZE_ALL_F16: ThreadLocal<Boolean> = ThreadLocal()

internal object QuantizedHelper {
    val sizeOf: (Any?) -> Int = { comp ->
        when (comp) {
            F32GgmlType, GgmlDType.F32, Float -> 4
            F16GgmlType, BF16GgmlType, GgmlDType.F16, GgmlDType.BF16, F16, BF16 -> 2
            BlockQ40 -> 18
            BlockQ41 -> 20
            BlockQ50 -> 22
            BlockQ51 -> 24
            BlockQ80 -> 34
            BlockQ81 -> 36
            BlockQ2K -> QK_K / 16 + QK_K / 4 + 2 * 2
            BlockQ3K -> QK_K / 8 + QK_K / 4 + 12 + 2
            BlockQ4K -> QK_K / 2 + K_SCALE_SIZE + 2 * 2
            BlockQ5K -> QK_K / 8 + QK_K / 2 + 2 * 2 + K_SCALE_SIZE
            BlockQ6K -> 3 * QK_K / 4 + QK_K / 16 + 2
            BlockQ8K -> 4 + QK_K + QK_K / 16 * 2
            else -> 1
        }
    }

    val alignOf: (Any?) -> Int = { _ -> 1 }

    val fromRawParts: (Int, Int, ByteArray, Any?) -> List<Any> = { _, count, data, comp ->
        val size = sizeOf(comp)
        val out = ArrayList<Any>(count)
        for (i in 0 until count) {
            val offset = i * size
            val block: Any = when (comp) {
                F32GgmlType, GgmlDType.F32, Float -> F32GgmlType.fromBytes(data, offset)
                F16GgmlType, GgmlDType.F16, F16 -> F16GgmlType.fromBytes(data, offset)
                BF16GgmlType, GgmlDType.BF16, BF16 -> BF16GgmlType.fromBytes(data, offset)
                BlockQ40 -> BlockQ40.fromBytes(data, offset)
                BlockQ41 -> BlockQ41.fromBytes(data, offset)
                BlockQ50 -> BlockQ50.fromBytes(data, offset)
                BlockQ51 -> BlockQ51.fromBytes(data, offset)
                BlockQ80 -> BlockQ80.fromBytes(data, offset)
                BlockQ81 -> BlockQ81.fromBytes(data, offset)
                BlockQ2K -> BlockQ2K.fromBytes(data, offset)
                BlockQ3K -> BlockQ3K.fromBytes(data, offset)
                BlockQ4K -> BlockQ4K.fromBytes(data, offset)
                BlockQ5K -> BlockQ5K.fromBytes(data, offset)
                BlockQ6K -> BlockQ6K.fromBytes(data, offset)
                BlockQ8K -> BlockQ8K.fromBytes(data, offset)
                else -> F32GgmlType.zeros()
            }
            out.add(block)
        }
        out
    }

    val write: (String, String) -> String = { _, s -> s }

    val asPtrBytes: (ByteArray, MutableList<out Any>, Any) -> ByteArray = { _, list, comp ->
        when (comp) {
            F32GgmlType, GgmlDType.F32, Float -> F32GgmlType.toBytes(list.map { it as F32Scalar })
            F16GgmlType, GgmlDType.F16, F16 -> F16GgmlType.toBytes(list.map { it as F16 })
            BF16GgmlType, GgmlDType.BF16, BF16 -> BF16GgmlType.toBytes(list.map { it as BF16 })
            BlockQ40 -> BlockQ40.toBytes(list.map { it as BlockQ40 })
            BlockQ41 -> BlockQ41.toBytes(list.map { it as BlockQ41 })
            BlockQ50 -> BlockQ50.toBytes(list.map { it as BlockQ50 })
            BlockQ51 -> BlockQ51.toBytes(list.map { it as BlockQ51 })
            BlockQ80 -> BlockQ80.toBytes(list.map { it as BlockQ80 })
            BlockQ81 -> BlockQ81.toBytes(list.map { it as BlockQ81 })
            BlockQ2K -> BlockQ2K.toBytes(list.map { it as BlockQ2K })
            BlockQ3K -> BlockQ3K.toBytes(list.map { it as BlockQ3K })
            BlockQ4K -> BlockQ4K.toBytes(list.map { it as BlockQ4K })
            BlockQ5K -> BlockQ5K.toBytes(list.map { it as BlockQ5K })
            BlockQ6K -> BlockQ6K.toBytes(list.map { it as BlockQ6K })
            BlockQ8K -> BlockQ8K.toBytes(list.map { it as BlockQ8K })
            else -> ByteArray(0)
        }
    }

    val cowBorrowed: (ByteArray) -> ByteArray = { b -> b }
    val cowFrom: (ByteArray) -> ByteArray = { b -> b }
    val cowFromParts: (ByteArray, Int) -> ByteArray = { ptr, len ->
        if (ptr.size == len) ptr else ptr.copyOf(len)
    }

    val devicePtrWithGuard: (QCudaStorage, Long) -> CandleResult<Pair<Long, Long>> = { storage, stream ->
        storage.devicePtr().map { ptr -> Pair(ptr, stream) }
    }

    val asSliceF32: (CpuStorage) -> CandleResult<FloatArray> = { s ->
        when (s) {
            is CpuStorage.F32 -> CandleResult.Ok(s.data)
            else -> CandleResult.Err(CandleError.Msg("expected f32 storage").bt())
        }
    }

    val asSliceF16: (CpuStorage) -> CandleResult<Array<F16>> = { s ->
        when (s) {
            is CpuStorage.F16 -> CandleResult.Ok(Array(s.data.size) { F16(s.data[it]) })
            else -> CandleResult.Err(CandleError.Msg("expected f16 storage").bt())
        }
    }

    val broadcastLeftT2: (Tensor, Int, Int) -> CandleResult<Tensor> = { w, b1, b2 ->
        try {
            CandleResult.Ok(w.broadcastLeft(Shape.fromDims(intArrayOf(b1, b2))).t())
        } catch (e: CandleException) {
            CandleResult.Err(e)
        }
    }

    val broadcastLeftT1: (Tensor, Int) -> CandleResult<Tensor> = { w, bsize ->
        try {
            CandleResult.Ok(w.broadcastLeft(Shape.fromDims(intArrayOf(bsize))).t())
        } catch (e: CandleException) {
            CandleResult.Err(e)
        }
    }

    val matmulViaF16: (Tensor, Tensor, DType) -> CandleResult<Tensor> = { xs, w, inDtype ->
        try {
            CandleResult.Ok(xs.toDtype(DType.F16).matmul(w).toDtype(inDtype))
        } catch (e: CandleException) {
            CandleResult.Err(e)
        }
    }

    val dequantizeAllFlag: () -> Boolean = {
        DEQUANTIZE_ALL.getOr {
            val s = getenv("CANDLE_DEQUANTIZE_ALL")
            s != null && s.isNotEmpty() && s != "0"
        }
    }

    val dequantizeAllF16Flag: () -> Boolean = {
        DEQUANTIZE_ALL_F16.getOr {
            val s = getenv("CANDLE_DEQUANTIZE_ALL_F16")
            s != null && s.isNotEmpty() && s != "0"
        }
    }

    val createQuantizedVec: (MutableList<out Any>, Any) -> QuantizedType = { list, comp ->
        when (comp) {
            F32GgmlType, GgmlDType.F32, Float -> QuantizedVec(list.map { it as F32Scalar }.toMutableList(), F32GgmlType)
            F16GgmlType, GgmlDType.F16, F16 -> QuantizedVec(list.map { it as F16 }.toMutableList(), F16GgmlType)
            BF16GgmlType, GgmlDType.BF16, BF16 -> QuantizedVec(list.map { it as BF16 }.toMutableList(), BF16GgmlType)
            BlockQ40 -> QuantizedVec(list.map { it as BlockQ40 }.toMutableList(), BlockQ40)
            BlockQ41 -> QuantizedVec(list.map { it as BlockQ41 }.toMutableList(), BlockQ41)
            BlockQ50 -> QuantizedVec(list.map { it as BlockQ50 }.toMutableList(), BlockQ50)
            BlockQ51 -> QuantizedVec(list.map { it as BlockQ51 }.toMutableList(), BlockQ51)
            BlockQ80 -> QuantizedVec(list.map { it as BlockQ80 }.toMutableList(), BlockQ80)
            BlockQ81 -> QuantizedVec(list.map { it as BlockQ81 }.toMutableList(), BlockQ81)
            BlockQ2K -> QuantizedVec(list.map { it as BlockQ2K }.toMutableList(), BlockQ2K)
            BlockQ3K -> QuantizedVec(list.map { it as BlockQ3K }.toMutableList(), BlockQ3K)
            BlockQ4K -> QuantizedVec(list.map { it as BlockQ4K }.toMutableList(), BlockQ4K)
            BlockQ5K -> QuantizedVec(list.map { it as BlockQ5K }.toMutableList(), BlockQ5K)
            BlockQ6K -> QuantizedVec(list.map { it as BlockQ6K }.toMutableList(), BlockQ6K)
            BlockQ8K -> QuantizedVec(list.map { it as BlockQ8K }.toMutableList(), BlockQ8K)
            else -> throw IllegalStateException("Unsupported quantized companion")
        }
    }
}
