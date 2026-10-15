package io.github.kotlinmania.candle

import io.github.kotlinmania.envie.getenv
import io.github.kotlinmania.threadlocal.ThreadLocal

/**
 * Quantized tensor storage, GGML data types, block quantization/dequantization,
 * and quantized matrix multiplication operations across CPU, CUDA, and Metal backends
 * in the Candle Kotlin Multiplatform machine learning framework.
 */
// port-lint: source quantized/mod.rs

private val ByteArray.len: () -> Int get() = { this.size }
private val ByteArray.asPtr: () -> Int get() = { this.size }
private val FloatArray.len: () -> Int get() = { this.size }
private val Int.isMultipleOf: (Int) -> Boolean get() = { d -> d > 0 && this % d == 0 }
private val <T> List<T>.toVec: () -> MutableList<T> get() = { this.toMutableList() }
private val <T> MutableList<T>.len: () -> Int get() = { this.size }
private val <T> MutableList<T>.asSlice: () -> MutableList<T> get() = { this }
private val <T> MutableList<T>.asPtr: () -> ByteArray get() = { ByteArray(0) }
private val <T> MutableList<T>.pop: () -> T? get() = { if (this.isEmpty()) null else this.removeAt(this.size - 1) }
private val <T> MutableList<T>.push: (T) -> Unit get() = { v -> this.add(v) }
private val QTensor.asRef: () -> QTensor get() = { this }

private fun <T : GgmlType> asTSlice(data: ByteArray, companion: GgmlTypeCompanion<T, *>? = null): List<GgmlType> {
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
        fun fromData(data: ByteArray, device: Device, dtype: GgmlDType): CandleResult<QStorage> {
            val rawData = data
            return when (device) {
                is Device.Cpu -> CandleResult.Ok(Cpu(dtype.fromData(QuantizedHelper.cowBorrowed(rawData))))
                is Device.Metal -> when (dtype) {
                    GgmlDType.F32 -> loadQuantized(device.device, asTSlice(rawData, F32Scalar))
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
                    GgmlDType.F32 -> loadQuantized(device.device, asTSlice(rawData, F32Scalar))
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

enum class GgmlDType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K;

    private val self: GgmlDType get() = this

    internal fun toU32(): UInt {
        return when (self) {
            F32 -> 0u
            F16 -> 1u
            Q4_0 -> 2u
            Q4_1 -> 3u
            Q5_0 -> 6u
            Q5_1 -> 7u
            Q8_0 -> 8u
            Q8_1 -> 9u
            Q2K -> 10u
            Q3K -> 11u
            Q4K -> 12u
            Q5K -> 13u
            Q6K -> 14u
            Q8K -> 15u
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            BF16 -> 30u
        }
    }

    /**
     * The block dtype
     */
    fun cpuZeros(elemCount: Int): QuantizedType {
        return when (self) {
            F32 -> QuantizedVec.new(MutableList(elemCount) { F32Scalar.zeros() }, F32Scalar)
            F16 -> QuantizedVec.new(MutableList(elemCount) { F16.zeros() }, F16)
            Q4_0 -> QuantizedVec.new(MutableList(elemCount / BlockQ40.BLCK_SIZE) { BlockQ40.zeros() }, BlockQ40)
            Q4_1 -> QuantizedVec.new(MutableList(elemCount / BlockQ41.BLCK_SIZE) { BlockQ41.zeros() }, BlockQ41)
            Q5_0 -> QuantizedVec.new(MutableList(elemCount / BlockQ50.BLCK_SIZE) { BlockQ50.zeros() }, BlockQ50)
            Q5_1 -> QuantizedVec.new(MutableList(elemCount / BlockQ51.BLCK_SIZE) { BlockQ51.zeros() }, BlockQ51)
            Q8_0 -> QuantizedVec.new(MutableList(elemCount / BlockQ80.BLCK_SIZE) { BlockQ80.zeros() }, BlockQ80)
            Q8_1 -> QuantizedVec.new(MutableList(elemCount / BlockQ81.BLCK_SIZE) { BlockQ81.zeros() }, BlockQ81)
            Q2K -> QuantizedVec.new(MutableList(elemCount / BlockQ2K.BLCK_SIZE) { BlockQ2K.zeros() }, BlockQ2K)
            Q3K -> QuantizedVec.new(MutableList(elemCount / BlockQ3K.BLCK_SIZE) { BlockQ3K.zeros() }, BlockQ3K)
            Q4K -> QuantizedVec.new(MutableList(elemCount / BlockQ4K.BLCK_SIZE) { BlockQ4K.zeros() }, BlockQ4K)
            Q5K -> QuantizedVec.new(MutableList(elemCount / BlockQ5K.BLCK_SIZE) { BlockQ5K.zeros() }, BlockQ5K)
            Q6K -> QuantizedVec.new(MutableList(elemCount / BlockQ6K.BLCK_SIZE) { BlockQ6K.zeros() }, BlockQ6K)
            Q8K -> QuantizedVec.new(MutableList(elemCount / BlockQ8K.BLCK_SIZE) { BlockQ8K.zeros() }, BlockQ8K)
            BF16 -> QuantizedVec.new(MutableList(elemCount) { BF16.zeros() }, BF16)
        }
    }

    fun fromData(data: ByteArray): QuantizedType {
        val rawData = data
        return when (self) {
            F32 -> QuantizedVec.new(asTSlice(rawData, F32Scalar).toVec(), F32Scalar)
            F16 -> QuantizedVec.new(asTSlice(rawData, F16).toVec(), F16)
            Q4_0 -> QuantizedVec.new(asTSlice(rawData, BlockQ40).toVec(), BlockQ40)
            Q4_1 -> QuantizedVec.new(asTSlice(rawData, BlockQ41).toVec(), BlockQ41)
            Q5_0 -> QuantizedVec.new(asTSlice(rawData, BlockQ50).toVec(), BlockQ50)
            Q5_1 -> QuantizedVec.new(asTSlice(rawData, BlockQ51).toVec(), BlockQ51)
            Q8_0 -> QuantizedVec.new(asTSlice(rawData, BlockQ80).toVec(), BlockQ80)
            Q8_1 -> QuantizedVec.new(asTSlice(rawData, BlockQ81).toVec(), BlockQ81)
            Q2K -> QuantizedVec.new(asTSlice(rawData, BlockQ2K).toVec(), BlockQ2K)
            Q3K -> QuantizedVec.new(asTSlice(rawData, BlockQ3K).toVec(), BlockQ3K)
            Q4K -> QuantizedVec.new(asTSlice(rawData, BlockQ4K).toVec(), BlockQ4K)
            Q5K -> QuantizedVec.new(asTSlice(rawData, BlockQ5K).toVec(), BlockQ5K)
            Q6K -> QuantizedVec.new(asTSlice(rawData, BlockQ6K).toVec(), BlockQ6K)
            Q8K -> QuantizedVec.new(asTSlice(rawData, BlockQ8K).toVec(), BlockQ8K)
            BF16 -> QuantizedVec.new(asTSlice(rawData, BF16).toVec(), BF16)
        }
    }

    /**
     * The type size for blocks in bytes.
     */
    fun typeSize(): Int {
        // Block sizes in bytes for each quantized GgmlDType variant.
        return when (self) {
            F32 -> 4
            F16, BF16 -> 2
            Q4_0 -> QuantizedHelper.sizeOf(BlockQ40)
            Q4_1 -> QuantizedHelper.sizeOf(BlockQ41)
            Q5_0 -> QuantizedHelper.sizeOf(BlockQ50)
            Q5_1 -> QuantizedHelper.sizeOf(BlockQ51)
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Q8_0 -> QuantizedHelper.sizeOf(BlockQ80)
            Q8_1 -> QuantizedHelper.sizeOf(BlockQ81)
            Q2K -> QuantizedHelper.sizeOf(BlockQ2K)
            Q3K -> QuantizedHelper.sizeOf(BlockQ3K)
            Q4K -> QuantizedHelper.sizeOf(BlockQ4K)
            Q5K -> QuantizedHelper.sizeOf(BlockQ5K)
            Q6K -> QuantizedHelper.sizeOf(BlockQ6K)
            Q8K -> QuantizedHelper.sizeOf(BlockQ8K)
        }
    }

    /**
     * The block size, i.e. the number of elements stored in each block.
     */
    fun blockSize(): Int {
        return when (self) {
            F32 -> 1
            F16, BF16 -> 1
            Q4_0 -> QK4_0
            Q4_1 -> QK4_1
            Q5_0 -> QK5_0
            Q5_1 -> QK5_1
            Q8_0 -> QK8_0
            Q8_1 -> QK8_1
            Q2K, Q3K, Q4K, Q5K, Q6K, Q8K -> QK_K
        }
    }

    companion object {
        val Q40: GgmlDType = Q4_0
        val Q41: GgmlDType = Q4_1
        val Q50: GgmlDType = Q5_0
        val Q51: GgmlDType = Q5_1
        val Q80: GgmlDType = Q8_0
        val Q81: GgmlDType = Q8_1

        internal fun fromU32(u: UInt): CandleResult<GgmlDType> {
            val dtype = when (u) {
                0u -> F32
                1u -> F16
                2u -> Q4_0
                3u -> Q4_1
                6u -> Q5_0
                7u -> Q5_1
                8u -> Q8_0
                9u -> Q8_1
                10u -> Q2K
                11u -> Q3K
                12u -> Q4K
                13u -> Q5K
                14u -> Q6K
                15u -> Q8K
                // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
                30u -> BF16
                else -> return CandleResult.Err(CandleError.Msg("unknown dtype for tensor $u").bt())
            }
            return CandleResult.Ok(dtype)
        }
    }
}

// A type-erased version of GgmlType for polymorphic quantized storage.
interface QuantizedType {
    fun dtype(): GgmlDType
    fun matmulT(mkn: Triple<Int, Int, Int>, lhs: FloatArray, dst: FloatArray): CandleResult<Unit>
    fun matmulTF16(mkn: Triple<Int, Int, Int>, lhs: Array<F16>, dst: Array<F16>): CandleResult<Unit>
    fun dequantize(elemCount: Int): CandleResult<CpuStorage>
    fun storageSizeInBytes(): Int
    fun asPtr(): ByteArray
    fun blockSize(): Int
    fun fromFloat(xs: FloatArray)
    fun fromFloatImatrix(xs: FloatArray, imatrixWeights: FloatArray, nPerRow: Int)
    fun size(): Int
}

class QuantizedVec<E : GgmlType>(
    val self: MutableList<E>,
    private val T: GgmlTypeCompanion<E, *>,
) : QuantizedType {
    val blocks: MutableList<E> get() = self

    override fun matmulT(mkn: Triple<Int, Int, Int>, lhs: FloatArray, dst: FloatArray): CandleResult<Unit> {
        return matmul(mkn, lhs, self.asSlice(), dst, T)
    }

    override fun matmulTF16(mkn: Triple<Int, Int, Int>, lhs: Array<F16>, dst: Array<F16>): CandleResult<Unit> {
        return matmulF16(mkn, lhs, self.asSlice(), dst, T)
    }

    override fun size(): Int {
        return self.len() * QuantizedHelper.sizeOf(T)
    }

    override fun fromFloat(xs: FloatArray) {
        T.fromFloat(xs, self)
    }

    override fun fromFloatImatrix(xs: FloatArray, imatrixWeights: FloatArray, nPerRow: Int) {
        T.fromFloatImatrix(xs, self, imatrixWeights, nPerRow)
    }

    override fun dtype(): GgmlDType {
        return T.DTYPE
    }

    override fun blockSize(): Int {
        return T.BLCK_SIZE
    }

    override fun dequantize(elemCount: Int): CandleResult<CpuStorage> {
        val ys = FloatArray(elemCount)
        T.toFloat(self.asSlice(), ys)
        return CandleResult.Ok(CpuStorage.F32(ys))
    }

    override fun storageSizeInBytes(): Int {
        return self.len() * QuantizedHelper.sizeOf(T)
    }

    override fun asPtr(): ByteArray {
        return QuantizedHelper.asPtrBytes(self.asPtr(), self, T)
    }

    companion object {
        val new: (MutableList<out GgmlType>, GgmlTypeCompanion<*, *>) -> QuantizedType = { list, comp ->
            QuantizedHelper.createQuantizedVec(list, comp)
        }
    }
}

private fun checkShape(shape: Shape, blockSize: Int): CandleResult<Unit> {
    val dims = shape.dims()
    if (dims.isEmpty()) {
        return CandleResult.Err(CandleError.Msg(
            "scalar tensor cannot be quantized $shape").bt())
    }
    if (!dims[dims.size - 1].isMultipleOf(blockSize)) {
        return CandleResult.Err(CandleError.Msg(
            "quantized tensor must have their last dim divisible by block size $shape $blockSize"
        ).bt())
    }
    return CandleResult.Ok(Unit)
}

class QTensor(
    val storage: QStorage,
    val shape: Shape,
) : CustomOp1 {
    private val self: QTensor get() = this

    fun fmt(f: String = ""): String {
        return QuantizedHelper.write(f, "QTensor[${self.shape}; ${self.dtype()}]")
    }

    fun dtype(): GgmlDType {
        return self.storage.dtype()
    }

    fun device(): Device {
        return self.storage.device()
    }

    fun rank(): Int {
        return self.shape.rank()
    }

    fun shape(): Shape {
        return self.shape
    }

    fun dequantize(device: Device): CandleResult<Tensor> {
        val storage = self.storage.dequantize(self.shape.elemCount()).unwrapOrElse { return CandleResult.Err(it) }
        val none = BackpropOp.none()
        return fromStorage(storage, self.shape.clone(), none, false).toDevice(device)
    }

    fun dequantizeF16(device: Device): CandleResult<Tensor> {
        // In the CUDA case, we have a specialized kernel as this can be useful for volta
        // architectures. https://github.com/huggingface/candle/issues/2136
        return when (val st = self.storage) {
            is QStorage.Cuda -> {
                val s = st.storage.dequantizeF16(self.shape.elemCount()).unwrapOrElse { return CandleResult.Err(it) }
                val none = BackpropOp.none()
                fromStorage(Storage.Cuda(s), self.shape.clone(), none, false)
                    .toDevice(device)
            }
            else -> {
                val s = self.dequantize(device).unwrapOrElse { return CandleResult.Err(it) }.toDtype(DType.F16).unwrapOrElse { return CandleResult.Err(it) }
                CandleResult.Ok(s)
            }
        }
    }

    fun storageSizeInBytes(): Int {
        return self.storage.sizeInBytes()
    }

    fun data(): CandleResult<ByteArray> {
        return self.storage.data()
    }

    fun indexedMoeForward(x: Tensor, ids: Tensor): CandleResult<Tensor> {
        return when (val st = self.storage) {
            is QStorage.Cuda -> {
                val xSt = x.storageAndLayout().first
                val idsSt = ids.storageAndLayout().first
                if (xSt is Storage.Cuda && idsSt is Storage.Cuda) {
                    val pair = st.storage.indexedMoeForward(
                        self.shape(),
                        xSt.storage,
                        x.layout(),
                        idsSt.storage,
                        ids.layout(),
                    ).unwrapOrElse { return CandleResult.Err(it) }
                    CandleResult.Ok(fromStorage(
                        Storage.Cuda(pair.first),
                        pair.second,
                        BackpropOp.none(),
                        false,
                    )
                    )
                } else {
                    throw IllegalStateException("Non-cuda indexedMoeForward is not implemented!")
                }
            }
            else -> throw IllegalStateException("indexedMoeForward is not implemented in this platform!")
        }
    }

    fun devicePtr(): CandleResult<Long> {
        return when (val st = self.storage) {
            is QStorage.Cuda -> st.storage.devicePtr()
            is QStorage.Metal, is QStorage.Cpu -> {
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
        return self.storage.devicePtrWithGuard(stream)
    }

    override fun name(): String {
        return "qmatmul"
    }

    override fun cpuFwd(
        storage: CpuStorage,
        layout: Layout,
    ): CandleResult<Pair<CpuStorage, Shape>> {
        if (!layout.isContiguous()) {
            return CandleResult.Err(CandleError.Msg("input tensor is not contiguous $layout").bt())
        }
        val srcShape = layout.shape()
        // self is transposed so n is first then k.
        val nk = self.shape.dims2().unwrapOrElse { return CandleResult.Err(it) }
        val n = nk.first
        val k = nk.second
        if (srcShape.rank() < 2) {
            return CandleResult.Err(CandleError.Msg("input tensor has only one dimension $layout").bt())
        }
        val dstShapeDims = srcShape.dims().toList().toVec()
        val lastK = dstShapeDims.pop()!!
        if (lastK != k) {
            return CandleResult.Err(CandleError.Msg("input tensor $layout incompatible with ${self.shape}").bt())
        }
        dstShapeDims.push(n)
        val dstShape = Shape.fromDims(dstShapeDims.toIntArray())
        val selfStorage = when (val st = self.storage) {
            is QStorage.Cpu -> st.storage
            is QStorage.Metal, is QStorage.Cuda -> return CandleResult.Err(CandleError.Msg("Invalid storage").bt())
        }
        return when (storage.dtype()) {
            DType.F32 -> {
                val slice = QuantizedHelper.asSliceF32(storage).unwrapOrElse { return CandleResult.Err(it) }
                val subSlice =
                    slice.copyOfRange(layout.startOffset(), layout.startOffset() + srcShape.elemCount())
                val dstStorage = FloatArray(dstShape.elemCount())
                selfStorage.matmulT(
                    Triple(dstShape.elemCount() / n, k, n),
                    subSlice,
                    dstStorage,
                ).unwrapOrElse { return CandleResult.Err(it) }
                CandleResult.Ok(Pair(CpuStorage.F32(dstStorage), dstShape))
            }
            DType.F16 -> {
                val slice = QuantizedHelper.asSliceF16(storage).unwrapOrElse { return CandleResult.Err(it) }
                val subSlice =
                    slice.copyOfRange(layout.startOffset(), layout.startOffset() + srcShape.elemCount())
                val dstStorage = Array(dstShape.elemCount()) { F16.ZERO }
                selfStorage.matmulTF16(
                    Triple(dstShape.elemCount() / n, k, n),
                    subSlice,
                    dstStorage,
                ).unwrapOrElse { return CandleResult.Err(it) }
                CandleResult.Ok(Pair(CpuStorage.F16(dstStorage), dstShape))
            }
            else -> CandleResult.Err(CandleError.Msg("Expected f32/f16").bt())
        }
    }

    override fun metalFwd(
        storage: MetalStorage,
        layout: Layout,
    ): CandleResult<Pair<MetalStorage, Shape>> {
        val selfStorage = when (val st = self.storage) {
            is QStorage.Metal -> st.storage
            else ->
                throw IllegalStateException("Cannot call metal matmul on non metal QTensor")
        }
        return selfStorage.fwd(self.shape, storage, layout)
    }

    override fun cudaFwd(
        storage: CudaStorage,
        layout: Layout,
    ): CandleResult<Pair<CudaStorage, Shape>> {
        val selfStorage = when (val st = self.storage) {
            is QStorage.Cuda -> st.storage
            else ->
                throw IllegalStateException("Cannot call cuda matmul on non cuda QTensor")
        }
        return selfStorage.fwd(self.shape, storage, layout)
    }

    companion object {
        fun new(storage: QStorage, shape: Shape): CandleResult<QTensor> {
            val s = shape.clone()
            checkShape(s, storage.blockSize()).unwrapOrElse { return CandleResult.Err(it) }
            return CandleResult.Ok(QTensor(storage, s))
        }

        fun quantize(src: Tensor, dtype: GgmlDType): CandleResult<QTensor> {
            val shape = src.shape()
            val blockSize = dtype.blockSize()
            checkShape(shape, blockSize).unwrapOrElse { return CandleResult.Err(it) }
            val flat = src.toDtype(DType.F32)
                .unwrapOrElse { return CandleResult.Err(it) }.flattenAll().unwrapOrElse { return CandleResult.Err(it) }
            val elemCount = shape.elemCount()
            if (!elemCount.isMultipleOf(blockSize)) {
                return CandleResult.Err(CandleError.Msg(
                    "tensor size ($shape) is not divisible by block size $blockSize"
                ).bt())
            }
            val storage = src.device().qzeros(elemCount, dtype).unwrapOrElse { return CandleResult.Err(it) }
            storage.quantize(flat.storageAndLayout().first).unwrapOrElse { return CandleResult.Err(it) }
            return CandleResult.Ok(QTensor(
                storage,
                shape.clone(),
            ))
        }

        fun quantizeImatrix(
            src: Tensor,
            imatrixWeights: FloatArray,
            dtype: GgmlDType,
        ): CandleResult<QTensor> {
            // (nPerRow/QK_K-1)*QK_K+(QK_K/32-1)*32+32=nPerRow
            // Size of imatrix == last dim of tensor
            val nPerRow = src.dim(D.Minus1).unwrapOrElse { return CandleResult.Err(it) }
            if (imatrixWeights.len() != nPerRow) {
                return CandleResult.Err(CandleError.Msg(
                    "imatrix weights must have the same length ${imatrixWeights.len()} as the last dim of src $nPerRow"
                ).bt())
            }

            val shape = src.shape()
            val blockSize = dtype.blockSize()
            checkShape(shape, blockSize).unwrapOrElse { return CandleResult.Err(it) }
            val flat = src.toDtype(DType.F32)
                .unwrapOrElse { return CandleResult.Err(it) }
                .flattenAll()
                .unwrapOrElse { return CandleResult.Err(it) }
            val elemCount = shape.elemCount()
            if (!elemCount.isMultipleOf(blockSize)) {
                return CandleResult.Err(CandleError.Msg(
                    "tensor size ($shape) is not divisible by block size $blockSize"
                ).bt())
            }
            val storage = src.device().qzeros(elemCount, dtype).unwrapOrElse { return CandleResult.Err(it) }
            storage.quantizeImatrix(flat.storageAndLayout().first, imatrixWeights, nPerRow).unwrapOrElse { return CandleResult.Err(it) }
            return CandleResult.Ok(QTensor(
                storage,
                shape.clone(),
            ))
        }

        /**
         * Quantize `src` (currently on the CPU) to a QTensor on `dev`
         */
        fun quantizeImatrixOnto(
            src: Tensor,
            imatrixWeights: FloatArray,
            dtype: GgmlDType,
            dev: Device,
        ): CandleResult<QTensor> {
            if (!src.device().isCpu()) {
                return CandleResult.Err(CandleError.Msg(
                    "`quantizeOnto` expects a `src` to be on the cpu, got ${src.device()}."
                ).bt())
            }
            // (nPerRow/QK_K-1)*QK_K+(QK_K/32-1)*32+32=nPerRow
            // Size of imatrix == last dim of tensor
            val nPerRow = src.dim(D.Minus1).unwrapOrElse { return CandleResult.Err(it) }
            if (imatrixWeights.len() != nPerRow) {
                return CandleResult.Err(CandleError.Msg(
                    "imatrix weights must have the same length ${imatrixWeights.len()} as the last dim of src $nPerRow"
                ).bt())
            }
            val shape = src.shape()
            val blockSize = dtype.blockSize()
            checkShape(shape, blockSize).unwrapOrElse { return CandleResult.Err(it) }
            val flat = src.toDtype(DType.F32)
                .unwrapOrElse { return CandleResult.Err(it) }
                .flattenAll()
                .unwrapOrElse { return CandleResult.Err(it) }
            val elemCount = shape.elemCount()
            if (!elemCount.isMultipleOf(blockSize)) {
                return CandleResult.Err(CandleError.Msg(
                    "tensor size ($shape) is not divisible by block size $blockSize"
                ).bt())
            }
            // storage is on the `dev`, src is on `cpu`
            val storage = dev.qzeros(elemCount, dtype).unwrapOrElse { return CandleResult.Err(it) }
            storage.quantizeImatrixOnto(flat.storageAndLayout().first, imatrixWeights, nPerRow)
                .unwrapOrElse { return CandleResult.Err(it) }
            return CandleResult.Ok(QTensor(
                storage,
                shape.clone(),
            ))
        }

        /**
         * Quantize `src` (currently on the CPU) to a QTensor on `dev`
         */
        fun quantizeOnto(src: Tensor, dtype: GgmlDType, dev: Device): CandleResult<QTensor> {
            if (!src.device().isCpu()) {
                return CandleResult.Err(CandleError.Msg(
                    "`quantizeOnto` expects a `src` to be on the cpu, got ${src.device()}."
                ).bt())
            }
            val shape = src.shape()
            val blockSize = dtype.blockSize()
            checkShape(shape, blockSize).unwrapOrElse { return CandleResult.Err(it) }
            val flat = src.toDtype(DType.F32)
                .unwrapOrElse { return CandleResult.Err(it) }
                .flattenAll().unwrapOrElse { return CandleResult.Err(it) }
            val elemCount = shape.elemCount()
            if (!elemCount.isMultipleOf(blockSize)) {
                return CandleResult.Err(CandleError.Msg(
                    "tensor size ($shape) is not divisible by block size $blockSize"
                ).bt())
            }
            // storage is on the `dev`, src is on `cpu`
            val storage = dev.qzeros(elemCount, dtype).unwrapOrElse { return CandleResult.Err(it) }
            storage.quantizeOnto(flat.storageAndLayout().first).unwrapOrElse { return CandleResult.Err(it) }
            return CandleResult.Ok(QTensor(
                storage,
                shape.clone(),
            ))
        }
    }
}

private val DEQUANTIZE_ALL: ThreadLocal<Boolean> = ThreadLocal()
private val DEQUANTIZE_ALL_F16: ThreadLocal<Boolean> = ThreadLocal()

sealed class QMatMul : Module {
    private val self: QMatMul get() = this

    data class QTensorVariant(val qtensor: QTensor) : QMatMul() {
        val t: QTensor get() = qtensor
    }
    data class TensorVariant(val tensor: Tensor) : QMatMul() {
        val t: Tensor get() = tensor
        val w: Tensor get() = tensor
    }
    data class TensorF16Variant(val tensor: Tensor) : QMatMul() {
        val t: Tensor get() = tensor
        val w: Tensor get() = tensor
    }

    fun dequantizeF16(): CandleResult<Tensor> {
        return when (self) {
            is QTensorVariant -> (this as QTensorVariant).t.dequantizeF16((this as QTensorVariant).t.device())
            is TensorVariant -> (this as TensorVariant).t.toDtype(DType.F16)
            is TensorF16Variant -> CandleResult.Ok((this as TensorF16Variant).t.clone())
        }
    }

    fun forwardViaF16(xs: Tensor): CandleResult<Tensor> {
        val w = self.dequantizeF16().unwrapOrElse { return CandleResult.Err(it) }
        val inDtype = xs.dtype()
        val wT = when (xs.dims().size) {
            4 -> QuantizedHelper.broadcastLeftT2(w, xs.dims()[0], xs.dims()[1]).unwrapOrElse { return CandleResult.Err(it) }
            3 -> QuantizedHelper.broadcastLeftT1(w, xs.dims()[0]).unwrapOrElse { return CandleResult.Err(it) }
            else -> w.t().unwrapOrElse { return CandleResult.Err(it) }
        }
        return QuantizedHelper.matmulViaF16(xs, wT, inDtype)
    }

    fun indexedMoeForward(x: Tensor, ids: Tensor): CandleResult<Tensor> {
        return when (self) {
            is QTensorVariant -> (this as QTensorVariant).t.indexedMoeForward(x, ids)
            else -> {
                throw IllegalStateException("Not implemented!")
            }
        }
    }

    override fun forward(xs: Tensor): CandleResult<Tensor> {
        return when (self) {
            is QTensorVariant -> xs.applyOp1NoBwd((this as QTensorVariant).t.asRef())
            is TensorVariant -> {
                val w = when (xs.dims().size) {
                    4 -> QuantizedHelper.broadcastLeftT2((this as TensorVariant).w, xs.dims()[0], xs.dims()[1]).unwrapOrElse { return CandleResult.Err(it) }
                    3 -> QuantizedHelper.broadcastLeftT1((this as TensorVariant).w, xs.dims()[0]).unwrapOrElse { return CandleResult.Err(it) }
                    else -> (this as TensorVariant).w.t().unwrapOrElse { return CandleResult.Err(it) }
                }
                xs.matmul(w)
            }
            is TensorF16Variant -> {
                val inDtype = xs.dtype()
                val w = when (xs.dims().size) {
                    4 -> QuantizedHelper.broadcastLeftT2((this as TensorF16Variant).w, xs.dims()[0], xs.dims()[1]).unwrapOrElse { return CandleResult.Err(it) }
                    3 -> QuantizedHelper.broadcastLeftT1((this as TensorF16Variant).w, xs.dims()[0]).unwrapOrElse { return CandleResult.Err(it) }
                    else -> (this as TensorF16Variant).w.t().unwrapOrElse { return CandleResult.Err(it) }
                }
                QuantizedHelper.matmulViaF16(xs, w, inDtype)
            }
        }
    }

    companion object {
        fun fromArc(qtensor: QTensor): CandleResult<QMatMul> {
            val dequantize = when (qtensor.dtype()) {
                GgmlDType.F32, GgmlDType.F16, GgmlDType.BF16 -> true
                else -> QuantizedHelper.dequantizeAllFlag()
            }
            val t = if (dequantize) {
                val tensor = qtensor.dequantize(qtensor.device()).unwrapOrElse { return CandleResult.Err(it) }
                TensorVariant(tensor)
            } else if (QuantizedHelper.dequantizeAllF16Flag()) {
                val tensor = qtensor.dequantizeF16(qtensor.device()).unwrapOrElse { return CandleResult.Err(it) }
                TensorF16Variant(tensor)
            } else {
                QTensorVariant(qtensor)
            }
            return CandleResult.Ok(t)
        }

        fun fromQtensor(qtensor: QTensor): CandleResult<QMatMul> {
            return fromArc(qtensor)
        }
    }
}

private object QuantizedHelper {
    val sizeOf: (GgmlTypeCompanion<*, *>?) -> Int = { comp ->
        when (comp) {
            F32Scalar -> 4
            F16, BF16 -> 2
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

    val alignOf: (GgmlTypeCompanion<*, *>?) -> Int = { _ -> 1 }

    val fromRawParts: (Int, Int, ByteArray, GgmlTypeCompanion<*, *>?) -> List<GgmlType> = { _, count, data, comp ->
        val size = sizeOf(comp)
        val out = ArrayList<GgmlType>(count)
        for (i in 0 until count) {
            val offset = i * size
            val block: GgmlType = when (comp) {
                F32Scalar -> F32Scalar.fromBytes(data, offset)
                F16 -> F16.fromBytes(data, offset)
                BF16 -> BF16.fromBytes(data, offset)
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
                else -> F32Scalar.zeros()
            }
            out.add(block)
        }
        out
    }

    val write: (String, String) -> String = { _, s -> s }

    val asPtrBytes: (ByteArray, MutableList<out GgmlType>, GgmlTypeCompanion<*, *>) -> ByteArray = { _, list, comp ->
        when (comp) {
            F32Scalar -> F32Scalar.toBytes(list.map { it as F32Scalar })
            F16 -> F16.toBytes(list.map { it as F16 })
            BF16 -> BF16.toBytes(list.map { it as BF16 })
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
            is CpuStorage.F16 -> CandleResult.Ok(s.data)
            else -> CandleResult.Err(CandleError.Msg("expected f16 storage").bt())
        }
    }

    val broadcastLeftT2: (Tensor, Int, Int) -> CandleResult<Tensor> = { w, b1, b2 ->
        w.broadcastLeft(Shape.fromDims(intArrayOf(b1, b2))).andThen { it.t() }
    }

    val broadcastLeftT1: (Tensor, Int) -> CandleResult<Tensor> = { w, bsize ->
        w.broadcastLeft(Shape.fromDims(intArrayOf(bsize))).andThen { it.t() }
    }

    val matmulViaF16: (Tensor, Tensor, DType) -> CandleResult<Tensor> = { xs, w, inDtype ->
        xs.toDtype(DType.F16).andThen { it.matmul(w) }.andThen { it.toDtype(inDtype) }
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

    val createQuantizedVec: (MutableList<out GgmlType>, GgmlTypeCompanion<*, *>) -> QuantizedType = { list, comp ->
        when (comp) {
            F32Scalar -> QuantizedVec(list.map { it as F32Scalar }.toMutableList(), F32Scalar)
            F16 -> QuantizedVec(list.map { it as F16 }.toMutableList(), F16)
            BF16 -> QuantizedVec(list.map { it as BF16 }.toMutableList(), BF16)
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
