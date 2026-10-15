package io.github.kotlinmania.candle

private val FloatArray.len: () -> Int get() = { this.size }
private val Int.isMultipleOf: (Int) -> Boolean get() = { d -> d > 0 && this % d == 0 }
private inline fun Tensor.unwrapOrElse(f: (CandleException) -> Tensor): Tensor = this
private inline fun Int.unwrapOrElse(f: (CandleException) -> Int): Int = this

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
        return CandleResult.Ok(fromStorage(storage, self.shape.clone(), none, false).toDevice(device))
    }

    fun dequantizeF16(device: Device): CandleResult<Tensor> {
        // In the CUDA case, we have a specialized kernel as this can be useful for volta
        // architectures. https://github.com/huggingface/candle/issues/2136
        return when (val st = self.storage) {
            is QStorage.Cuda -> {
                val s = st.storage.dequantizeF16(self.shape.elemCount()).unwrapOrElse { return CandleResult.Err(it) }
                val none = BackpropOp.none()
                CandleResult.Ok(fromStorage(Storage.Cuda(s), self.shape.clone(), none, false)
                    .toDevice(device))
            }
            else -> {
                val s = self.dequantize(device).unwrapOrElse { return CandleResult.Err(it) }.toDtype(DType.F16)
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
    ): Pair<CpuStorage, Shape> {
        if (!layout.isContiguous()) {
            throw CandleError.Msg("input tensor is not contiguous $layout").bt()
        }
        val srcShape = layout.shape()
        // self is transposed so n is first then k.
        val nk = self.shape.dims2().unwrapOrElse { throw it }
        val n = nk.first
        val k = nk.second
        if (srcShape.rank() < 2) {
            throw CandleError.Msg("input tensor has only one dimension $layout").bt()
        }
        val dstShapeDims = srcShape.dims().toList().toVec()
        val lastK = dstShapeDims.pop()!!
        if (lastK != k) {
            throw CandleError.Msg("input tensor $layout incompatible with ${self.shape}").bt()
        }
        dstShapeDims.push(n)
        val dstShape = Shape.fromDims(dstShapeDims.toIntArray())
        val selfStorage = when (val st = self.storage) {
            is QStorage.Cpu -> st.storage
            is QStorage.Metal, is QStorage.Cuda -> throw CandleError.Msg("Invalid storage").bt()
        }
        return when (storage.dtype()) {
            DType.F32 -> {
                val slice = QuantizedHelper.asSliceF32(storage).unwrapOrElse { throw it }
                val subSlice =
                    slice.copyOfRange(layout.startOffset(), layout.startOffset() + srcShape.elemCount())
                val dstStorage = FloatArray(dstShape.elemCount())
                selfStorage.matmulT(
                    Triple(dstShape.elemCount() / n, k, n),
                    subSlice,
                    dstStorage,
                ).unwrapOrElse { throw it }
                Pair(CpuStorage.F32(dstStorage), dstShape)
            }
            DType.F16 -> {
                val slice = QuantizedHelper.asSliceF16(storage).unwrapOrElse { throw it }
                val subSlice =
                    slice.copyOfRange(layout.startOffset(), layout.startOffset() + srcShape.elemCount())
                val dstStorage = Array(dstShape.elemCount()) { F16.ZERO }
                selfStorage.matmulTF16(
                    Triple(dstShape.elemCount() / n, k, n),
                    subSlice,
                    dstStorage,
                ).unwrapOrElse { throw it }
                Pair(CpuStorage.F16(ShortArray(dstStorage.size) { dstStorage[it].bits }), dstShape)
            }
            else -> throw CandleError.Msg("Expected f32/f16").bt()
        }
    }

    override fun metalFwd(
        storage: MetalStorage,
        layout: Layout,
    ): Pair<MetalStorage, Shape> {
        val selfStorage = when (val st = self.storage) {
            is QStorage.Metal -> st.storage
            else ->
                throw IllegalStateException("Cannot call metal matmul on non metal QTensor")
        }
        return selfStorage.fwd(self.shape, storage, layout).unwrapOrElse { throw it }
    }

    override fun cudaFwd(
        storage: CudaStorage,
        layout: Layout,
    ): Pair<CudaStorage, Shape> {
        val selfStorage = when (val st = self.storage) {
            is QStorage.Cuda -> st.storage
            else ->
                throw IllegalStateException("Cannot call cuda matmul on non cuda QTensor")
        }
        return selfStorage.fwd(self.shape, storage, layout).unwrapOrElse { throw it }
    }

    companion object {
        fun new(storage: QStorage, shape: Shape): CandleResult<QTensor> {
            val s = shape.clone()
            checkShape(s, storage.blockSize()).unwrapOrElse { return CandleResult.Err(it) }
            return CandleResult.Ok(QTensor(storage, s))
        }

        fun new(storage: QStorage, dims: List<Int>): CandleResult<QTensor> =
            new(storage, Shape.from(dims))

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
