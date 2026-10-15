// port-lint: source quantized/mod.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.envie.getenv
import io.github.kotlinmania.threadlocal.ThreadLocal

private fun <T> asTSlice(data: ByteArray, typeSize: Int, decodeBlock: (ByteArray, Int) -> T): List<T> {
    check(data.size % typeSize == 0) {
        "Data length must be a multiple of element size"
    }
    val count = data.size / typeSize
    val out = ArrayList<T>(count)
    for (i in 0 until count) {
        out.add(decodeBlock(data, i * typeSize))
    }
    return out
}

class QTensor(
    val storage: QStorage,
    private val tensorShape: Shape,
) : CustomOp1 {
    fun fmt(): String = "QTensor[${this.tensorShape}; ${this.dtype()}]"

    fun dtype(): GgmlDType = this.storage.dtype()

    fun device(): Device = this.storage.device()

    fun rank(): Int = this.tensorShape.rank()

    fun shape(): Shape = this.tensorShape

    fun dequantize(device: Device): CandleResult<Tensor> {
        val deqStorage = when (val res = this.storage.dequantize(this.tensorShape.elemCount())) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        val none = BackpropOp.none()
        return fromStorage(deqStorage, this.tensorShape.clone(), none, false).toDevice(device)
    }

    fun dequantizeF16(device: Device): CandleResult<Tensor> {
        // In the CUDA case, we have a specialized kernel as this can be useful for volta
        // architectures. https://github.com/huggingface/candle/issues/2136
        return when (val st = this.storage) {
            is QStorage.Cuda -> {
                val s = when (val res = st.storage.dequantizeF16(this.tensorShape.elemCount())) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val none = BackpropOp.none()
                fromStorage(Storage.Cuda(s), this.tensorShape.clone(), none, false).toDevice(device)
            }
            else -> {
                val deq = when (val res = this.dequantize(device)) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                deq.toDtype(DType.F16)
            }
        }
    }

    fun storageSizeInBytes(): Int = this.storage.sizeInBytes()

    fun data(): CandleResult<ByteArray> = this.storage.data()

    fun indexedMoeForward(x: Tensor, ids: Tensor): CandleResult<Tensor> {
        return when (val st = this.storage) {
            is QStorage.Cuda -> {
                val xSt = x.storageAndLayout().first
                val idsSt = ids.storageAndLayout().first
                if (xSt is Storage.Cuda && idsSt is Storage.Cuda) {
                    val (outStorage, outShape) = when (
                        val res = st.storage.indexedMoeForward(
                            this.shape(),
                            xSt.storage,
                            x.layout(),
                            idsSt.storage,
                            ids.layout(),
                        )
                    ) {
                        is CandleResult.Ok -> res.value
                        is CandleResult.Err -> return CandleResult.Err(res.error)
                    }
                    CandleResult.Ok(
                        fromStorage(
                            Storage.Cuda(outStorage),
                            outShape,
                            BackpropOp.none(),
                            false,
                        )
                    )
                } else {
                    throw IllegalStateException("Non-cuda indexedMoeForward is unsupported!")
                }
            }
            else -> throw IllegalStateException("indexedMoeForward is unsupported in this platform!")
        }
    }

    fun devicePtr(): CandleResult<Long> = when (val st = this.storage) {
        is QStorage.Cuda -> st.storage.devicePtr()
        is QStorage.Metal, is QStorage.Cpu -> CandleResult.Err(CandleError.Msg("unsupported").bt())
    }

    fun devicePtrWithGuard(stream: Long): CandleResult<Pair<Long, Long>> =
        this.storage.devicePtrWithGuard(stream)

    override fun name(): String = "qmatmul"

    override fun cpuFwd(
        storage: CpuStorage,
        layout: Layout,
    ): CandleResult<Pair<CpuStorage, Shape>> {
        if (!layout.isContiguous()) {
            return CandleResult.Err(CandleError.Msg("input tensor is not contiguous $layout").bt())
        }
        val srcShape = layout.shape()
        // self is transposed so n is first then k.
        val (n, k) = when (val res = this.tensorShape.dims2()) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        if (srcShape.rank() < 2) {
            return CandleResult.Err(CandleError.Msg("input tensor has only one dimension $layout").bt())
        }
        val dstDims = srcShape.dims().toMutableList()
        val lastK = dstDims.removeAt(dstDims.size - 1)
        if (lastK != k) {
            return CandleResult.Err(
                CandleError.Msg("input tensor $layout incompatible with ${this.tensorShape}").bt()
            )
        }
        dstDims.add(n)
        val dstShape = Shape.fromDims(dstDims.toIntArray())
        val selfStorage = when (val st = this.storage) {
            is QStorage.Cpu -> st.storage
            is QStorage.Metal, is QStorage.Cuda -> return CandleResult.Err(CandleError.Msg("Invalid storage").bt())
        }
        return when (storage) {
            is CpuStorage.F32 -> {
                val slice = storage.data.copyOfRange(
                    layout.startOffset(),
                    layout.startOffset() + srcShape.elemCount(),
                )
                val dstStorage = FloatArray(dstShape.elemCount())
                when (val res = selfStorage.matmulT(Triple(dstShape.elemCount() / n, k, n), slice, dstStorage)) {
                    is CandleResult.Ok -> {}
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                CandleResult.Ok(CpuStorage.F32(dstStorage) to dstShape)
            }
            is CpuStorage.F16 -> {
                val slice = storage.data.copyOfRange(
                    layout.startOffset(),
                    layout.startOffset() + srcShape.elemCount(),
                )
                val dstStorage = Array(dstShape.elemCount()) { F16.ZERO }
                when (val res = selfStorage.matmulTF16(Triple(dstShape.elemCount() / n, k, n), slice, dstStorage)) {
                    is CandleResult.Ok -> {}
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                CandleResult.Ok(CpuStorage.F16(dstStorage) to dstShape)
            }
            else -> CandleResult.Err(CandleError.Msg("Expected f32/f16").bt())
        }
    }

    override fun metalFwd(
        storage: MetalStorage,
        layout: Layout,
    ): CandleResult<Pair<MetalStorage, Shape>> {
        val selfStorage = when (val st = this.storage) {
            is QStorage.Metal -> st.storage
            else -> throw IllegalStateException("Cannot call metal matmul on non metal QTensor")
        }
        return selfStorage.fwd(this.tensorShape, storage, layout)
    }

    override fun cudaFwd(
        storage: CudaStorage,
        layout: Layout,
    ): CandleResult<Pair<CudaStorage, Shape>> {
        val selfStorage = when (val st = this.storage) {
            is QStorage.Cuda -> st.storage
            else -> throw IllegalStateException("Cannot call cuda matmul on non cuda QTensor")
        }
        return selfStorage.fwd(this.tensorShape, storage, layout)
    }

    companion object {
        fun new(storage: QStorage, shape: Shape): CandleResult<QTensor> {
            when (val res = checkShape(shape, storage.blockSize())) {
                is CandleResult.Ok -> {}
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            return CandleResult.Ok(QTensor(storage, shape))
        }

        fun quantize(src: Tensor, dtype: GgmlDType): CandleResult<QTensor> {
            val shape = src.shape()
            val blockSize = dtype.blockSize()
            when (val res = checkShape(shape, blockSize)) {
                is CandleResult.Ok -> {}
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val f32Tensor = when (val res = src.toDtype(DType.F32)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val flat = when (val res = f32Tensor.flattenAll()) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val elemCount = shape.elemCount()
            if (elemCount % blockSize != 0) {
                return CandleResult.Err(
                    CandleError.Msg("tensor size ($shape) is not divisible by block size $blockSize").bt()
                )
            }
            val storage = when (val res = src.device().qzeros(elemCount, dtype)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            when (val res = storage.quantize(flat.storageAndLayout().first)) {
                is CandleResult.Ok -> {}
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            return CandleResult.Ok(QTensor(storage, shape.clone()))
        }

        fun quantizeImatrix(
            src: Tensor,
            imatrixWeights: FloatArray,
            dtype: GgmlDType,
        ): CandleResult<QTensor> {
            // (nPerRow/QK_K-1)*QK_K+(QK_K/32-1)*32+32=nPerRow
            // Size of imatrix == last dim of tensor
            val nPerRow = when (val res = src.dim(D.Minus1)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            if (imatrixWeights.size != nPerRow) {
                return CandleResult.Err(
                    CandleError.Msg(
                        "imatrix weights must have the same length ${imatrixWeights.size} as the last dim of src $nPerRow"
                    ).bt()
                )
            }

            val shape = src.shape()
            val blockSize = dtype.blockSize()
            when (val res = checkShape(shape, blockSize)) {
                is CandleResult.Ok -> {}
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val f32Tensor = when (val res = src.toDtype(DType.F32)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val flat = when (val res = f32Tensor.flattenAll()) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val elemCount = shape.elemCount()
            if (elemCount % blockSize != 0) {
                return CandleResult.Err(
                    CandleError.Msg("tensor size ($shape) is not divisible by block size $blockSize").bt()
                )
            }
            val storage = when (val res = src.device().qzeros(elemCount, dtype)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            when (val res = storage.quantizeImatrix(flat.storageAndLayout().first, imatrixWeights, nPerRow)) {
                is CandleResult.Ok -> {}
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            return CandleResult.Ok(QTensor(storage, shape.clone()))
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
                return CandleResult.Err(
                    CandleError.Msg("`quantizeOnto` expects a `src` to be on the cpu, got ${src.device()}.").bt()
                )
            }
            // (nPerRow/QK_K-1)*QK_K+(QK_K/32-1)*32+32=nPerRow
            // Size of imatrix == last dim of tensor
            val nPerRow = when (val res = src.dim(D.Minus1)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            if (imatrixWeights.size != nPerRow) {
                return CandleResult.Err(
                    CandleError.Msg(
                        "imatrix weights must have the same length ${imatrixWeights.size} as the last dim of src $nPerRow"
                    ).bt()
                )
            }
            val shape = src.shape()
            val blockSize = dtype.blockSize()
            when (val res = checkShape(shape, blockSize)) {
                is CandleResult.Ok -> {}
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val f32Tensor = when (val res = src.toDtype(DType.F32)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val flat = when (val res = f32Tensor.flattenAll()) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val elemCount = shape.elemCount()
            if (elemCount % blockSize != 0) {
                return CandleResult.Err(
                    CandleError.Msg("tensor size ($shape) is not divisible by block size $blockSize").bt()
                )
            }
            // storage is on the `dev`, src is on `cpu`
            val storage = when (val res = dev.qzeros(elemCount, dtype)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            when (val res = storage.quantizeImatrixOnto(flat.storageAndLayout().first, imatrixWeights, nPerRow)) {
                is CandleResult.Ok -> {}
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            return CandleResult.Ok(QTensor(storage, shape.clone()))
        }

        /**
         * Quantize `src` (currently on the CPU) to a QTensor on `dev`
         */
        fun quantizeOnto(src: Tensor, dtype: GgmlDType, dev: Device): CandleResult<QTensor> {
            if (!src.device().isCpu()) {
                return CandleResult.Err(
                    CandleError.Msg("`quantizeOnto` expects a `src` to be on the cpu, got ${src.device()}.").bt()
                )
            }
            val shape = src.shape()
            val blockSize = dtype.blockSize()
            when (val res = checkShape(shape, blockSize)) {
                is CandleResult.Ok -> {}
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val f32Tensor = when (val res = src.toDtype(DType.F32)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val flat = when (val res = f32Tensor.flattenAll()) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val elemCount = shape.elemCount()
            if (elemCount % blockSize != 0) {
                return CandleResult.Err(
                    CandleError.Msg("tensor size ($shape) is not divisible by block size $blockSize").bt()
                )
            }
            // storage is on the `dev`, src is on `cpu`
            val storage = when (val res = dev.qzeros(elemCount, dtype)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            when (val res = storage.quantizeOnto(flat.storageAndLayout().first)) {
                is CandleResult.Ok -> {}
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            return CandleResult.Ok(QTensor(storage, shape.clone()))
        }
    }
}

internal fun Device.qzeros(elemCount: Int, dtype: GgmlDType): CandleResult<QStorage> = when (this) {
    is Device.Cpu -> {
        val storage = dtype.cpuZeros(elemCount)
        CandleResult.Ok(QStorage.Cpu(storage))
    }
    is Device.Metal -> {
        when (val res = QMetalStorage.zeros(this.device, elemCount, dtype)) {
            is CandleResult.Ok -> CandleResult.Ok(QStorage.Metal(res.value))
            is CandleResult.Err -> CandleResult.Err(res.error)
        }
    }
    is Device.Cuda -> {
        when (val res = QCudaStorage.zeros(this.device, elemCount, dtype)) {
            is CandleResult.Ok -> CandleResult.Ok(QStorage.Cuda(res.value))
            is CandleResult.Err -> CandleResult.Err(res.error)
        }
    }
}

sealed class QStorage {
    data class Cpu(val storage: QuantizedType) : QStorage()
    data class Metal(val storage: QMetalStorage) : QStorage()
    data class Cuda(val storage: QCudaStorage) : QStorage()

    fun blockSize(): Int = when (this) {
        is Cpu -> this.storage.blockSize()
        is Metal -> this.storage.dtype().blockSize()
        is Cuda -> this.storage.dtype().blockSize()
    }

    fun dtype(): GgmlDType = when (this) {
        is Cpu -> this.storage.dtype()
        is Metal -> this.storage.dtype()
        is Cuda -> this.storage.dtype()
    }

    fun device(): Device = when (this) {
        is Cpu -> Device.Cpu
        is Metal -> Device.Metal(this.storage.device())
        is Cuda -> Device.Cuda(this.storage.device())
    }

    fun sizeInBytes(): Int = when (this) {
        is Cpu -> this.storage.storageSizeInBytes()
        is Metal -> this.storage.storageSizeInBytes()
        is Cuda -> this.storage.storageSizeInBytes()
    }

    fun quantize(src: Storage): CandleResult<Unit> {
        when {
            this is Cpu && src is Storage.Cpu -> {
                val slice = when (val s = src.storage) {
                    is CpuStorage.F32 -> s.data
                    else -> return CandleResult.Err(CandleError.Msg("expected f32 storage").bt())
                }
                this.storage.fromFloat(slice)
            }
            this is Metal && src is Storage.Metal -> return this.storage.quantize(src.storage)
            this is Cuda && src is Storage.Cuda -> return this.storage.quantize(src.storage)
            else -> return CandleResult.Err(
                CandleError.Msg("Invalid quantize storage locations do not match").bt()
            )
        }
        return CandleResult.Ok(Unit)
    }

    fun quantizeImatrix(
        src: Storage,
        imatrixWeights: FloatArray,
        nPerRow: Int,
    ): CandleResult<Unit> {
        when {
            this is Cpu && src is Storage.Cpu -> {
                val slice = when (val s = src.storage) {
                    is CpuStorage.F32 -> s.data
                    else -> return CandleResult.Err(CandleError.Msg("expected f32 storage").bt())
                }
                this.storage.fromFloatImatrix(slice, imatrixWeights, nPerRow)
            }
            this is Metal && src is Storage.Metal ->
                return this.storage.quantizeImatrix(src.storage, imatrixWeights, nPerRow)
            this is Cuda && src is Storage.Cuda ->
                return this.storage.quantizeImatrix(src.storage, imatrixWeights, nPerRow)
            else -> return CandleResult.Err(
                CandleError.Msg("Invalid quantize storage locations do not match").bt()
            )
        }
        return CandleResult.Ok(Unit)
    }

    fun quantizeOnto(src: Storage): CandleResult<Unit> {
        when {
            this is Cpu && src is Storage.Cpu -> {
                val slice = when (val s = src.storage) {
                    is CpuStorage.F32 -> s.data
                    else -> return CandleResult.Err(CandleError.Msg("expected f32 storage").bt())
                }
                this.storage.fromFloat(slice)
            }
            this is Metal && src is Storage.Cpu -> return this.storage.quantizeOnto(src.storage)
            this is Cuda && src is Storage.Cpu -> return this.storage.quantizeOnto(src.storage)
            else -> return CandleResult.Err(
                CandleError.Msg("Invalid quantize source storage locations: not on cpu").bt()
            )
        }
        return CandleResult.Ok(Unit)
    }

    fun quantizeImatrixOnto(
        src: Storage,
        imatrixWeights: FloatArray,
        nPerRow: Int,
    ): CandleResult<Unit> {
        when {
            this is Cpu && src is Storage.Cpu -> {
                val slice = when (val s = src.storage) {
                    is CpuStorage.F32 -> s.data
                    else -> return CandleResult.Err(CandleError.Msg("expected f32 storage").bt())
                }
                this.storage.fromFloatImatrix(slice, imatrixWeights, nPerRow)
            }
            this is Metal && src is Storage.Cpu ->
                return this.storage.quantizeImatrixOnto(src.storage, imatrixWeights, nPerRow)
            this is Cuda && src is Storage.Cpu ->
                return this.storage.quantizeImatrixOnto(src.storage, imatrixWeights, nPerRow)
            else -> return CandleResult.Err(
                CandleError.Msg("Invalid quantize storage locations do not match").bt()
            )
        }
        return CandleResult.Ok(Unit)
    }

    fun dequantize(elemCount: Int): CandleResult<Storage> = when (this) {
        is Cpu -> when (val res = this.storage.dequantize(elemCount)) {
            is CandleResult.Ok -> CandleResult.Ok(Storage.Cpu(res.value))
            is CandleResult.Err -> CandleResult.Err(res.error)
        }
        is Metal -> when (val res = this.storage.dequantize(elemCount)) {
            is CandleResult.Ok -> CandleResult.Ok(Storage.Metal(res.value))
            is CandleResult.Err -> CandleResult.Err(res.error)
        }
        is Cuda -> when (val res = this.storage.dequantize(elemCount)) {
            is CandleResult.Ok -> CandleResult.Ok(Storage.Cuda(res.value))
            is CandleResult.Err -> CandleResult.Err(res.error)
        }
    }

    fun data(): CandleResult<ByteArray> = when (this) {
        is Cpu -> CandleResult.Ok(this.storage.asPtr())
        is Cuda -> this.storage.data()
        is Metal -> this.storage.data()
    }

    fun devicePtr(): CandleResult<Long> = when (this) {
        is Cuda -> this.storage.devicePtr()
        is Metal, is Cpu -> CandleResult.Err(CandleError.Msg("unsupported").bt())
    }

    fun devicePtrWithGuard(stream: Long): CandleResult<Pair<Long, Long>> = when (this) {
        is Cuda -> when (val res = this.storage.devicePtr()) {
            is CandleResult.Ok -> CandleResult.Ok(res.value to stream)
            is CandleResult.Err -> CandleResult.Err(res.error)
        }
        is Metal, is Cpu -> CandleResult.Err(CandleError.Msg("unsupported").bt())
    }

    companion object {
        fun fromData(data: ByteArray, device: Device, dtype: GgmlDType): CandleResult<QStorage> {
            return when (device) {
                is Device.Cpu -> CandleResult.Ok(Cpu(dtype.fromData(data)))
                is Device.Metal -> loadQuantized(device.device, asTSlice(data, dtype.typeSize()) { _, _ -> 0 })
                is Device.Cuda -> loadQuantized(device.device, asTSlice(data, dtype.typeSize()) { _, _ -> 0 })
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

    internal fun toU32(): UInt = when (this) {
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

    /**
     * The block dtype
     */
    fun cpuZeros(elemCount: Int): QuantizedType = when (this) {
        F32 -> QuantizedVec(MutableList(elemCount) { F32Scalar.zeros() }, F32Scalar)
        F16 -> QuantizedVec(MutableList(elemCount) { F16.zeros() }, F16)
        Q4_0 -> QuantizedVec(MutableList(elemCount / BlockQ40.BLCK_SIZE) { BlockQ40.zeros() }, BlockQ40)
        Q4_1 -> QuantizedVec(MutableList(elemCount / BlockQ41.BLCK_SIZE) { BlockQ41.zeros() }, BlockQ41)
        Q5_0 -> QuantizedVec(MutableList(elemCount / BlockQ50.BLCK_SIZE) { BlockQ50.zeros() }, BlockQ50)
        Q5_1 -> QuantizedVec(MutableList(elemCount / BlockQ51.BLCK_SIZE) { BlockQ51.zeros() }, BlockQ51)
        Q8_0 -> QuantizedVec(MutableList(elemCount / BlockQ80.BLCK_SIZE) { BlockQ80.zeros() }, BlockQ80)
        Q8_1 -> QuantizedVec(MutableList(elemCount / BlockQ81.BLCK_SIZE) { BlockQ81.zeros() }, BlockQ81)
        Q2K -> QuantizedVec(MutableList(elemCount / BlockQ2K.BLCK_SIZE) { BlockQ2K.zeros() }, BlockQ2K)
        Q3K -> QuantizedVec(MutableList(elemCount / BlockQ3K.BLCK_SIZE) { BlockQ3K.zeros() }, BlockQ3K)
        Q4K -> QuantizedVec(MutableList(elemCount / BlockQ4K.BLCK_SIZE) { BlockQ4K.zeros() }, BlockQ4K)
        Q5K -> QuantizedVec(MutableList(elemCount / BlockQ5K.BLCK_SIZE) { BlockQ5K.zeros() }, BlockQ5K)
        Q6K -> QuantizedVec(MutableList(elemCount / BlockQ6K.BLCK_SIZE) { BlockQ6K.zeros() }, BlockQ6K)
        Q8K -> QuantizedVec(MutableList(elemCount / BlockQ8K.BLCK_SIZE) { BlockQ8K.zeros() }, BlockQ8K)
        BF16 -> QuantizedVec(MutableList(elemCount) { BF16.zeros() }, BF16)
    }

    fun fromData(data: ByteArray): QuantizedType = when (this) {
        F32 -> QuantizedVec(asTSlice(data, 4) { b, o -> F32Scalar.fromBytes(b, o) }.toMutableList(), F32Scalar)
        F16 -> QuantizedVec(asTSlice(data, 2) { b, o -> F16.fromBytes(b, o) }.toMutableList(), F16)
        Q4_0 -> QuantizedVec(asTSlice(data, 18) { b, o -> BlockQ40.fromBytes(b, o) }.toMutableList(), BlockQ40)
        Q4_1 -> QuantizedVec(asTSlice(data, 20) { b, o -> BlockQ41.fromBytes(b, o) }.toMutableList(), BlockQ41)
        Q5_0 -> QuantizedVec(asTSlice(data, 22) { b, o -> BlockQ50.fromBytes(b, o) }.toMutableList(), BlockQ50)
        Q5_1 -> QuantizedVec(asTSlice(data, 24) { b, o -> BlockQ51.fromBytes(b, o) }.toMutableList(), BlockQ51)
        Q8_0 -> QuantizedVec(asTSlice(data, 34) { b, o -> BlockQ80.fromBytes(b, o) }.toMutableList(), BlockQ80)
        Q8_1 -> QuantizedVec(asTSlice(data, 36) { b, o -> BlockQ81.fromBytes(b, o) }.toMutableList(), BlockQ81)
        Q2K -> QuantizedVec(asTSlice(data, 84) { b, o -> BlockQ2K.fromBytes(b, o) }.toMutableList(), BlockQ2K)
        Q3K -> QuantizedVec(asTSlice(data, 110) { b, o -> BlockQ3K.fromBytes(b, o) }.toMutableList(), BlockQ3K)
        Q4K -> QuantizedVec(asTSlice(data, 144) { b, o -> BlockQ4K.fromBytes(b, o) }.toMutableList(), BlockQ4K)
        Q5K -> QuantizedVec(asTSlice(data, 176) { b, o -> BlockQ5K.fromBytes(b, o) }.toMutableList(), BlockQ5K)
        Q6K -> QuantizedVec(asTSlice(data, 210) { b, o -> BlockQ6K.fromBytes(b, o) }.toMutableList(), BlockQ6K)
        Q8K -> QuantizedVec(asTSlice(data, 292) { b, o -> BlockQ8K.fromBytes(b, o) }.toMutableList(), BlockQ8K)
        BF16 -> QuantizedVec(asTSlice(data, 2) { b, o -> BF16.fromBytes(b, o) }.toMutableList(), BF16)
    }

    /**
     * The type size for blocks in bytes.
     */
    fun typeSize(): Int = when (this) {
        F32 -> 4
        F16, BF16 -> 2
        Q4_0 -> 18
        Q4_1 -> 20
        Q5_0 -> 22
        Q5_1 -> 24
        // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
        Q8_0 -> 34
        Q8_1 -> 36
        Q2K -> QK_K / 16 + QK_K / 4 + 2 * 2
        Q3K -> QK_K / 8 + QK_K / 4 + 12 + 2
        Q4K -> QK_K / 2 + K_SCALE_SIZE + 2 * 2
        Q5K -> QK_K / 8 + QK_K / 2 + 2 * 2 + K_SCALE_SIZE
        Q6K -> 3 * QK_K / 4 + QK_K / 16 + 2
        Q8K -> 4 + QK_K + QK_K / 16 * 2
    }

    /**
     * The block size, i.e. the number of elements stored in each block.
     */
    fun blockSize(): Int = when (this) {
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

    companion object {
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

class QuantizedVec<T : GgmlType>(
    val blocks: MutableList<T>,
    private val companion: GgmlTypeCompanion<T, *>,
) : QuantizedType {
    override fun matmulT(mkn: Triple<Int, Int, Int>, lhs: FloatArray, dst: FloatArray): CandleResult<Unit> =
        matmul(mkn, lhs, this.blocks, dst, this.companion)

    override fun matmulTF16(mkn: Triple<Int, Int, Int>, lhs: Array<F16>, dst: Array<F16>): CandleResult<Unit> =
        matmulF16(mkn, lhs, this.blocks, dst, this.companion)

    override fun size(): Int = this.blocks.size * this.companion.DTYPE.typeSize()

    override fun fromFloat(xs: FloatArray) {
        this.companion.fromFloat(xs, this.blocks)
    }

    override fun fromFloatImatrix(xs: FloatArray, imatrixWeights: FloatArray, nPerRow: Int) {
        this.companion.fromFloatImatrix(xs, this.blocks, imatrixWeights, nPerRow)
    }

    override fun dtype(): GgmlDType = this.companion.DTYPE

    override fun blockSize(): Int = this.companion.BLCK_SIZE

    override fun dequantize(elemCount: Int): CandleResult<CpuStorage> {
        val ys = FloatArray(elemCount)
        this.companion.toFloat(this.blocks, ys)
        return CandleResult.Ok(CpuStorage.F32(ys))
    }

    override fun storageSizeInBytes(): Int = this.blocks.size * this.companion.DTYPE.typeSize()

    override fun asPtr(): ByteArray = this.companion.toBytes(this.blocks)
}

private fun checkShape(shape: Shape, blockSize: Int): CandleResult<Unit> {
    val dims = shape.dims()
    if (dims.isEmpty()) {
        return CandleResult.Err(CandleError.Msg("scalar tensor cannot be quantized $shape").bt())
    }
    if (dims[dims.size - 1] % blockSize != 0) {
        return CandleResult.Err(
            CandleError.Msg(
                "quantized tensor must have their last dim divisible by block size $shape $blockSize"
            ).bt()
        )
    }
    return CandleResult.Ok(Unit)
}

private val DEQUANTIZE_ALL: ThreadLocal<Boolean> = ThreadLocal()
private val DEQUANTIZE_ALL_F16: ThreadLocal<Boolean> = ThreadLocal()

sealed class QMatMul : Module {
    data class QTensorVariant(val qtensor: QTensor) : QMatMul()
    data class TensorVariant(val tensor: Tensor) : QMatMul()
    data class TensorF16Variant(val tensor: Tensor) : QMatMul()

    fun dequantizeF16(): CandleResult<Tensor> = when (this) {
        is QTensorVariant -> this.qtensor.dequantizeF16(this.qtensor.device())
        is TensorVariant -> this.tensor.toDtype(DType.F16)
        is TensorF16Variant -> CandleResult.Ok(this.tensor.clone())
    }

    fun forwardViaF16(xs: Tensor): CandleResult<Tensor> {
        val w0 = when (val res = this.dequantizeF16()) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        val inDtype = xs.dtype()
        val dims = xs.dims()
        val w = when (dims.size) {
            4 -> {
                val b = when (val r = w0.broadcastLeft(Shape.fromDims(intArrayOf(dims[0], dims[1])))) {
                    is CandleResult.Ok -> r.value
                    is CandleResult.Err -> return CandleResult.Err(r.error)
                }
                when (val r = b.t()) {
                    is CandleResult.Ok -> r.value
                    is CandleResult.Err -> return CandleResult.Err(r.error)
                }
            }
            3 -> {
                val b = when (val r = w0.broadcastLeft(Shape.fromDims(intArrayOf(dims[0])))) {
                    is CandleResult.Ok -> r.value
                    is CandleResult.Err -> return CandleResult.Err(r.error)
                }
                when (val r = b.t()) {
                    is CandleResult.Ok -> r.value
                    is CandleResult.Err -> return CandleResult.Err(r.error)
                }
            }
            else -> when (val r = w0.t()) {
                is CandleResult.Ok -> r.value
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }
        }
        val xsF16 = when (val r = xs.toDtype(DType.F16)) {
            is CandleResult.Ok -> r.value
            is CandleResult.Err -> return CandleResult.Err(r.error)
        }
        val mm = when (val r = xsF16.matmul(w)) {
            is CandleResult.Ok -> r.value
            is CandleResult.Err -> return CandleResult.Err(r.error)
        }
        return mm.toDtype(inDtype)
    }

    fun indexedMoeForward(x: Tensor, ids: Tensor): CandleResult<Tensor> = when (this) {
        is QTensorVariant -> this.qtensor.indexedMoeForward(x, ids)
        else -> throw IllegalStateException("Unsupported!")
    }

    override fun forward(xs: Tensor): CandleResult<Tensor> = when (this) {
        is QTensorVariant -> xs.applyOp1NoBwd(this.qtensor)
        is TensorVariant -> {
            val dims = xs.dims()
            val w = when (dims.size) {
                4 -> {
                    val b = when (val r = this.tensor.broadcastLeft(Shape.fromDims(intArrayOf(dims[0], dims[1])))) {
                        is CandleResult.Ok -> r.value
                        is CandleResult.Err -> return CandleResult.Err(r.error)
                    }
                    when (val r = b.t()) {
                        is CandleResult.Ok -> r.value
                        is CandleResult.Err -> return CandleResult.Err(r.error)
                    }
                }
                3 -> {
                    val b = when (val r = this.tensor.broadcastLeft(Shape.fromDims(intArrayOf(dims[0])))) {
                        is CandleResult.Ok -> r.value
                        is CandleResult.Err -> return CandleResult.Err(r.error)
                    }
                    when (val r = b.t()) {
                        is CandleResult.Ok -> r.value
                        is CandleResult.Err -> return CandleResult.Err(r.error)
                    }
                }
                else -> when (val r = this.tensor.t()) {
                    is CandleResult.Ok -> r.value
                    is CandleResult.Err -> return CandleResult.Err(r.error)
                }
            }
            xs.matmul(w)
        }
        is TensorF16Variant -> {
            val inDtype = xs.dtype()
            val dims = xs.dims()
            val w = when (dims.size) {
                4 -> {
                    val b = when (val r = this.tensor.broadcastLeft(Shape.fromDims(intArrayOf(dims[0], dims[1])))) {
                        is CandleResult.Ok -> r.value
                        is CandleResult.Err -> return CandleResult.Err(r.error)
                    }
                    when (val r = b.t()) {
                        is CandleResult.Ok -> r.value
                        is CandleResult.Err -> return CandleResult.Err(r.error)
                    }
                }
                3 -> {
                    val b = when (val r = this.tensor.broadcastLeft(Shape.fromDims(intArrayOf(dims[0])))) {
                        is CandleResult.Ok -> r.value
                        is CandleResult.Err -> return CandleResult.Err(r.error)
                    }
                    when (val r = b.t()) {
                        is CandleResult.Ok -> r.value
                        is CandleResult.Err -> return CandleResult.Err(r.error)
                    }
                }
                else -> when (val r = this.tensor.t()) {
                    is CandleResult.Ok -> r.value
                    is CandleResult.Err -> return CandleResult.Err(r.error)
                }
            }
            val xsF16 = when (val r = xs.toDtype(DType.F16)) {
                is CandleResult.Ok -> r.value
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }
            val mm = when (val r = xsF16.matmul(w)) {
                is CandleResult.Ok -> r.value
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }
            mm.toDtype(inDtype)
        }
    }

    companion object {
        fun fromArc(qtensor: QTensor): CandleResult<QMatMul> {
            val dequantize = when (qtensor.dtype()) {
                GgmlDType.F32, GgmlDType.F16, GgmlDType.BF16 -> true
                else -> DEQUANTIZE_ALL.getOr {
                    val s = getenv("CANDLE_DEQUANTIZE_ALL")
                    s != null && s.isNotEmpty() && s != "0"
                }
            }
            val t = if (dequantize) {
                val tensor = when (val res = qtensor.dequantize(qtensor.device())) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                TensorVariant(tensor)
            } else if (
                DEQUANTIZE_ALL_F16.getOr {
                    val s = getenv("CANDLE_DEQUANTIZE_ALL_F16")
                    s != null && s.isNotEmpty() && s != "0"
                }
            ) {
                val tensor = when (val res = qtensor.dequantizeF16(qtensor.device())) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                TensorF16Variant(tensor)
            } else {
                QTensorVariant(qtensor)
            }
            return CandleResult.Ok(t)
        }

        fun fromQtensor(qtensor: QTensor): CandleResult<QMatMul> =
            fromArc(qtensor)
    }
}
