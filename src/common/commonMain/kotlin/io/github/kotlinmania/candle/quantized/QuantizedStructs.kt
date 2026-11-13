package io.github.kotlinmania.candle.quantized

import io.github.kotlinmania.candle.cpu.Cpu
import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.cudabackend.CudaStorage
import io.github.kotlinmania.candle.customop.CustomOp1
import io.github.kotlinmania.candle.device.Device
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dummymetalbackend.MetalStorage
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.op.BackpropOp
import io.github.kotlinmania.candle.quantized.repack.PackedCache
import io.github.kotlinmania.candle.quantized.repack.tryGemvFused
import io.github.kotlinmania.candle.quantized.repack.tryIndexedGemv
import io.github.kotlinmania.candle.quantized.repack.tryMatmulF32
import io.github.kotlinmania.candle.quantized.repackx86.PackedX86
import io.github.kotlinmania.candle.quantized.repackx86.X86Level
import io.github.kotlinmania.candle.quantized.repackx86.gemvFused
import io.github.kotlinmania.candle.quantized.repackx86.level
import io.github.kotlinmania.candle.quantized.repackx86.pack
import io.github.kotlinmania.candle.quantized.repackx86.quantizeLhs
import io.github.kotlinmania.candle.quantized.repackx86.select
import io.github.kotlinmania.candle.shape.D
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.storage.Storage
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.candle.tensor.fromStorage
import io.github.kotlinmania.f16.ZERO
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr

public class QTensor(
    public var storage: QStorage = QStorage(),
    public var shape: Shape = Shape(),
    public var repackedQs: PackedCache = PackedCache(),
) : CustomOp1 {
    public fun fmt(f: StringBuilder) {
        f.append("QTensor[${this.shape}; ${this.dtype()}]")
    }

    public fun dtype(): GgmlDType = this.storage.dtype()

    public fun device(): Device = this.storage.device()

    public fun rank(): Int = this.shape.rank()

    public fun dequantize(device: Device): Result<Tensor> {
        val storage =
            when (val tryResult = this.storage.dequantize(this.shape.elemCount())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val none = BackpropOp.none()
        return fromStorage(storage, this.shape.clone(), none, false).toDevice(device)
    }

    public fun dequantizeF16(device: Device): Result<Tensor> {
        // In the CUDA case, we have a specialized kernel as this can be useful for volta
        // architectures. https://github.com/huggingface/candle/issues/2136
        return when (val matchVal = this.storage) {
            is QStorage.Cuda -> {
                val (s) = matchVal
                val s1 =
                    when (val tryResult = s.dequantizeF16(this.shape.elemCount())) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val none = BackpropOp.none()
                fromStorage(Storage.Cuda(s1), this.shape.clone(), none, false).toDevice(device)
            }
            else -> {
                val s2 =
                    when (
                        val tryResult =
                            when (val tryResult = this.dequantize(device)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.toDtype(DType.F16)
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(s2)
            }
        }
    }

    public fun embedding(ids: Tensor): Result<Tensor> {
        val (rows, hidden) =
            when (val tryResult = this.shape.dims2()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        if (!hidden.isMultipleOf(this.dtype().blockSize())) {
            bail("quantized embedding hidden size {hidden} is not divisible by block size {}", this.dtype().blockSize())
        }
        val outShape = ids.dims().toVec()
        outShape.push(hidden)
        val device = this.device()
        val ids1 =
            when (
                val tryResult =
                    when (
                        val tryResult =
                            when (
                                val tryResult =
                                    when (val tryResult = ids.toDevice(device)) {
                                        is Result.Ok<*> -> tryResult.value
                                        is Result.Err<*> -> return Result.Err(tryResult.error)
                                    }.toDtype(DType.U32)
                            ) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.flattenAll()
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.contiguous()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val storage3 =
            when (val matchVal = this.storage) {
                is QStorage.Cpu -> {
                    val (storage) = matchVal
                    val ids2 =
                        when (val tryResult = ids1.toVec1<UInt>()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Storage.Cpu(
                        when (val tryResult = storage.embedding(ids2, rows, hidden)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
                }
                is QStorage.Metal -> {
                    val (storage1) = matchVal
                    when (val matchVal1 = ids1.storage()) {
                        is Storage.Metal -> {
                            val (idsStorage) = matchVal1
                            Storage.Metal(
                                when (val tryResult = storage1.embedding(rows, hidden, idsStorage, ids1.layout())) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                },
                            )
                        }
                        else -> error("ids were moved to the QTensor device")
                    }
                }
                is QStorage.Cuda -> {
                    val (storage2) = matchVal
                    when (val matchVal2 = ids1.storage()) {
                        is Storage.Cuda -> {
                            val (idsStorage1) = matchVal2
                            Storage.Cuda(
                                when (val tryResult = storage2.embedding(rows, hidden, idsStorage1, ids1.layout())) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                },
                            )
                        }
                        else -> error("ids were moved to the QTensor device")
                    }
                }
            }
        val none = BackpropOp.none()
        return Result.Ok(fromStorage(storage3, outShape, none, false))
    }

    public fun storageSizeInBytes(): Int = this.storage.sizeInBytes()

    public fun data(): Result<List<UByte>> = this.storage.data()

    public fun indexedMoeForward(x: Tensor, ids: Tensor): Result<Tensor> {
        return when (val matchVal = this.storage) {
            is QStorage.Cuda -> {
                val (s) = matchVal
                run {
                    val matchVal1 = Pair(x.storage(), ids.storage())
                    when {
                        matchVal1.first is Storage.Cuda && matchVal1.second is Storage.Cuda -> {
                            val (xStorage) = matchVal1.first as Storage.Cuda
                            val (idsStorage) = matchVal1.second as Storage.Cuda
                            val (storage, outShape) =
                                when (val tryResult = s.indexedMoeForward(this.shape(), xStorage, x.layout(), idsStorage, ids.layout())) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }
                            Result.Ok(fromStorage(Storage.Cuda(storage), outShape, BackpropOp.none(), false))
                        }
                        else -> {
                            error("Non-cuda indexed_moe_forward is not implemented!")
                        }
                    }
                }
            }
            else -> {
                error("indexed_moe_forward is not implemented in this platform!")
            }
        }
    }

    public fun devicePtr(): Result<ConstPtr> =
        when (val matchVal = this.storage) {
            is QStorage.Cuda -> {
                val (storage) = matchVal
                storage.devicePtr()
            }
            is QStorage.Metal -> {
                bail("not implemented")
            }
            is QStorage.Cpu -> {
                bail("not implemented")
            }
        }

    public fun devicePtrWithGuard(stream: CudaStream): Result<Pair<ConstPtr, SyncOnDrop>> = this.storage.devicePtrWithGuard(stream)

    internal fun storageRef(): QuantizedType =
        when (val matchVal = this.storage) {
            is QStorage.Cpu -> {
                val (s) = matchVal
                s.asRef()
            }
            else -> error("cpu-only path")
        }

    /**
     * Indexed (MoE) matmul over stacked expert weights [nExperts, nOut, k]: each
     * (token, expert-id) pair gemvs against its expert's rows via the repacked cache.
     * Returns None when the layout, dtype, or device is unsupported.
     */
    public fun indexedGemv(x: Tensor, ids: Tensor): Result<Tensor?> {
        return run {
            if (!x.device().isCpu() ||
                !when (x.dtype()) {
                    DType.F32, DType.BF16 -> true
                    else -> false
                } || !x.isContiguous()
            ) {
                return Result.Ok(null)
            }
            val (nExperts, nOut, k) = this.shape.dims3() ?: return Result.Ok(null)
            val (batch, xT, xk) = x.dims3() ?: return Result.Ok(null)
            val (idsB, topk) = ids.dims2() ?: return Result.Ok(null)
            if (xk != k || idsB != batch || (xT != 1 && xT != topk)) {
                return Result.Ok(null)
            }
            val storage = this.storage ?: return Result.Ok(null)
            val idsV: List<UInt> =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = ids.toDtype(DType.U32)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.flattenAll()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.toVec1<UInt>()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            if (idsV.any({ e -> e.toInt() >= nExperts })) {
                bail("expert index out of range")
            }
            val guard = x.storage()
            val cpu = guard ?: return Result.Ok(null)
            val offset = x.layout().startOffset()
            val nRows = batch * xT
            val widened: Any?
            val lhs: List<Float> =
                if (x.dtype() == DType.BF16) {
                    val bslice =
                        when (val tryResult = cpu.asSlice<Bf16>()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    widened = widenBf16(((bslice).slice(offset until offset + nRows * k)))
                    widened
                } else {
                    val slice =
                        when (val tryResult = cpu.asSlice<Float>()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    ((slice).slice(offset until offset + nRows * k))
                }
            val dst = mutableListOf(0.0f, batch * topk * nOut)
            val ok =
                when (val tryResult = tryIndexedGemv(storage.asRef(), this.repackedQs, nExperts, nOut, k, lhs, nRows, idsV, topk, dst)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            drop(guard)
            if (!ok) {
                return Result.Ok(null)
            }
            val out =
                when (val tryResult = Tensor.fromVec(dst, Triple(batch, topk, nOut), Cpu)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            if (x.dtype() == DType.BF16) {
                Result.Ok(
                    when (val tryResult = out.toDtype(DType.BF16)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
            } else {
                Result.Ok(out)
            }
        }
    }

    public fun name(): String = "qmatmul"

    public fun cpuFwd(storage: CpuStorage, layout: Layout): Result<Pair<CpuStorage, Shape>> {
        if (!layout.isContiguous()) {
            bail("input tensor is not contiguous {layout:?}")
        }
        val srcShape = layout.shape()
        // self is transposed so n is first then k.
        val (n, k) =
            when (val tryResult = this.shape.dims2()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        if (srcShape.rank() < 2) {
            bail("input tensor has only one dimension {layout:?}")
        }
        val dstShape = srcShape.dims().toVec()
        val lastK = dstShape.pop()!!
        if (lastK != k) {
            bail("input tensor {layout:?} incompatible with {:?}", this.shape)
        }
        dstShape.push(n)
        val dstShape1 = Shape.from(dstShape)
        val selfStorage =
            when (val matchVal = this.storage) {
                is QStorage.Cpu -> {
                    val (storage1) = matchVal
                    storage1
                }
                is QStorage.Metal -> bail("Invalid storage")
                is QStorage.Cuda -> bail("Invalid storage")
            }
        return when (storage.dtype()) {
            DType.f32 -> {
                val slice =
                    when (val tryResult = storage.asSlice<Float>()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val slice1 = ((slice).slice(layout.startOffset() until layout.startOffset() + srcShape.elemCount()))
                val dstStorage = mutableListOf(0.0f, dstShape1.elemCount())
                val mkn = Triple(dstShape1.elemCount() / n, k, n)
                val usedPacked =
                    when (val tryResult = tryMatmulF32(selfStorage.asRef(), this.repackedQs, mkn, slice1, dstStorage)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                if (usedPacked) {
                    return Result.Ok(Pair(CpuStorage.F32(dstStorage), dstShape1))
                }
                when (val tryResult = selfStorage.matmulT(mkn, slice1, dstStorage)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
                Result.Ok(Pair(CpuStorage.F32(dstStorage), dstShape1))
            }
            DType.f16 -> {
                val slice2 =
                    when (val tryResult = storage.asSlice<F16>()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val slice3 = ((slice2).slice(layout.startOffset() until layout.startOffset() + srcShape.elemCount()))
                val dstStorage1 = mutableListOf(ZERO, dstShape1.elemCount())
                when (val tryResult = selfStorage.matmulTF16(Triple(dstShape1.elemCount() / n, k, n), slice3, dstStorage1)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
                Result.Ok(Pair(CpuStorage.F16(dstStorage1), dstShape1))
            }
            DType.bf16 -> {
                // widen to f32 once and take the repacked path; output stays bf16
                val slice4 =
                    when (val tryResult = storage.asSlice<Bf16>()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val slice5 = ((slice4).slice(layout.startOffset() until layout.startOffset() + srcShape.elemCount()))
                val lhs = widenBf16(slice5)
                val dstStorage2 = mutableListOf(0.0f, dstShape1.elemCount())
                val mkn1 = Triple(dstShape1.elemCount() / n, k, n)
                val usedPacked1 =
                    when (val tryResult = tryMatmulF32(selfStorage.asRef(), this.repackedQs, mkn1, lhs, dstStorage2)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                if (!usedPacked1) {
                    when (val tryResult = selfStorage.matmulT(mkn1, lhs, dstStorage2)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                val dst: List<Bf16> = narrowBf16(dstStorage2)
                Result.Ok(Pair(CpuStorage.BF16(dst), dstShape1))
            }
            else -> bail("Expected f32/f16/bf16")
        }
    }

    public fun metalFwd(storage: MetalStorage, layout: Layout): Result<Pair<MetalStorage, Shape>> {
        val selfStorage =
            when (val matchVal = this.storage) {
                is QStorage.Metal -> {
                    val (metal) = matchVal
                    metal
                }
                else -> error("Cannot call metal matmul on non metal QTensor")
            }
        return selfStorage.fwd(this.shape, storage, layout)
    }

    public fun cudaFwd(storage: CudaStorage, layout: Layout): Result<Pair<CudaStorage, Shape>> {
        val selfStorage =
            when (val matchVal = this.storage) {
                is QStorage.Cuda -> {
                    val (cuda) = matchVal
                    cuda
                }
                else -> error("Cannot call cuda matmul on non cuda QTensor")
            }
        return selfStorage.fwd(this.shape, storage, layout)
    }

    public companion object {
        public fun new(storage: QStorage, shape: Shape): Result<QTensor> {
            val shape1 = shape
            when (val tryResult = checkShape(shape1, storage.blockSize())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
            return Result.Ok(QTensor(storage = storage, shape = shape1, repackedQs = PackedCache.new()))
        }

        public fun quantize(src: Tensor, dtype: GgmlDType): Result<QTensor> {
            val shape = src.shape()
            val blockSize = dtype.blockSize()
            when (val tryResult = checkShape(shape, blockSize)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
            val src1 =
                when (
                    val tryResult =
                        when (val tryResult = src.toDtype(DType.F32)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.flattenAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val elemCount = shape.elemCount()
            if (!elemCount.isMultipleOf(blockSize)) {
                bail("tensor size ({shape:?}) is not divisible by block size {}", blockSize)
            }
            var storage =
                when (val tryResult = src1.device().qzeros(elemCount, dtype)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            when (val tryResult = storage.quantize(src1.storage())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
            return Result.Ok(QTensor(storage = storage, shape = shape.clone(), repackedQs = PackedCache.new()))
        }

        public fun quantizeImatrix(src: Tensor, imatrixWeights: List<Float>, dtype: GgmlDType): Result<QTensor> {
            // (nPerRow/QK_K-1)*QK_K+(QK_K/32-1)*32+32=nPerRow
            // Size of imatrix == last dim of tensor
            val nPerRow =
                when (val tryResult = src.dim(D.Minus1)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            if (imatrixWeights.size != nPerRow) {
                bail(
                    "imatrix weights must have the same length {} as the last dim of src {}",
                    imatrixWeights.size,
                    when (val tryResult = src.dim(D.Minus1)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
            }
            val shape = src.shape()
            val blockSize = dtype.blockSize()
            when (val tryResult = checkShape(shape, blockSize)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
            val src1 =
                when (
                    val tryResult =
                        when (val tryResult = src.toDtype(DType.F32)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.flattenAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val elemCount = shape.elemCount()
            if (!elemCount.isMultipleOf(blockSize)) {
                bail("tensor size ({shape:?}) is not divisible by block size {}", blockSize)
            }
            var storage =
                when (val tryResult = src1.device().qzeros(elemCount, dtype)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            when (val tryResult = storage.quantizeImatrix(src1.storage(), imatrixWeights, nPerRow)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
            return Result.Ok(QTensor(storage = storage, shape = shape.clone(), repackedQs = PackedCache.new()))
        }

        /**
         * Quantize `src` (currently on the CPU) to a QTensor on `dev`
         */
        public fun quantizeImatrixOnto(src: Tensor, imatrixWeights: List<Float>, dtype: GgmlDType, dev: Device): Result<QTensor> {
            if (!src.device().isCpu()) {
                bail("`quantize_onto` expects a `src` to be on the cpu, got {:?}.", src.device())
            }
            // (nPerRow/QK_K-1)*QK_K+(QK_K/32-1)*32+32=nPerRow
            // Size of imatrix == last dim of tensor
            val nPerRow =
                when (val tryResult = src.dim(D.Minus1)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            if (imatrixWeights.size != nPerRow) {
                bail(
                    "imatrix weights must have the same length {} as the last dim of src {}",
                    imatrixWeights.size,
                    when (val tryResult = src.dim(D.Minus1)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
            }
            val shape = src.shape()
            val blockSize = dtype.blockSize()
            when (val tryResult = checkShape(shape, blockSize)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
            val src1 =
                when (
                    val tryResult =
                        when (val tryResult = src.toDtype(DType.F32)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.flattenAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val elemCount = shape.elemCount()
            if (!elemCount.isMultipleOf(blockSize)) {
                bail("tensor size ({shape:?}) is not divisible by block size {}", blockSize)
            }
            // storage is on the `dev`, src is on `cpu`
            var storage =
                when (val tryResult = dev.qzeros(elemCount, dtype)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            when (val tryResult = storage.quantizeImatrixOnto(src1.storage(), imatrixWeights, nPerRow)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
            return Result.Ok(QTensor(storage = storage, shape = shape.clone(), repackedQs = PackedCache.new()))
        }

        /**
         * Quantize `src` (currently on the CPU) to a QTensor on `dev`
         */
        public fun quantizeOnto(src: Tensor, dtype: GgmlDType, dev: Device): Result<QTensor> {
            if (!src.device().isCpu()) {
                bail("`quantize_onto` expects a `src` to be on the cpu, got {:?}.", src.device())
            }
            val shape = src.shape()
            val blockSize = dtype.blockSize()
            when (val tryResult = checkShape(shape, blockSize)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
            val src1 =
                when (
                    val tryResult =
                        when (val tryResult = src.toDtype(DType.F32)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.flattenAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val elemCount = shape.elemCount()
            if (!elemCount.isMultipleOf(blockSize)) {
                bail("tensor size ({shape:?}) is not divisible by block size {}", blockSize)
            }
            // storage is on the `dev`, src is on `cpu`
            var storage =
                when (val tryResult = dev.qzeros(elemCount, dtype)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            when (val tryResult = storage.quantizeOnto(src1.storage())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
            return Result.Ok(QTensor(storage = storage, shape = shape.clone(), repackedQs = PackedCache.new()))
        }

        /**
         * Fused m==1 matmul over same-dtype tensors sharing one lhs (e.g. qkv or gate+up in
         * decode): one lhs quantization and one parallel region. None when unsupported.
         */
        public fun gemvFusedSharedLhs(ts: List<QTensor>, lhs: Tensor): Result<List<Tensor>?> {
            run {
                if (ts.isEmpty() || !lhs.device().isCpu() || lhs.dtype() != DType.F32) {
                    return Result.Ok(null)
                }
                if (!lhs.isContiguous()) {
                    return Result.Ok(null)
                }
                val dims = lhs.dims()
                val (k, batch) = dims.splitLast() ?: return Result.Ok(null)
                if (batch.iter().product<Int>() != 1) {
                    return Result.Ok(null)
                }
                val ns = ArrayList(ts.size)
                val parts: List<Pair<QuantizedType, PackedCache>> = ArrayList(ts.size)
                for (t in ts) {
                    val (n, tk) =
                        when (val tryResult = t.shape.dims2()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    if (tk != k) {
                        return Result.Ok(null)
                    }
                    val s = t.storage ?: return Result.Ok(null)
                    ns.add(n)
                    parts.add(Pair(s.asRef(), t.repackedQs))
                }
                val storage = lhs.storage()
                val cpu = storage ?: return Result.Ok(null)
                val slice =
                    when (val tryResult = cpu.asSlice<Float>()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val offset = lhs.layout().startOffset()
                val slice1 = ((slice).slice(offset until offset + k))
                val dsts: List<List<Float>> = ns.map({ n1 -> mutableListOf(0.0f, n1) }).toMutableList()
                if (!when (val tryResult = tryGemvFused(k, slice1, parts, dsts)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                ) {
                    return Result.Ok(null)
                }
                drop(storage)
                val outs =
                    when (
                        val tryResult =
                            (dsts.zip(ns))
                                .map({ (d, n1) ->
                                    var shape = dims.toVec()
                                    shape.lastMut()!! = n1
                                    Tensor.fromVec(d, shape, Cpu)
                                })
                                .collect<Result<Vec>>()
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(outs)
            }
            run {
                // the fused kernel is 512-bit only; downlevel tiers fall through to
                // per-matmul calls which dispatch to the 256-bit kernels
                if (level() != X86Level.Avx512Vnni) {
                    return Result.Ok(null)
                }
                if (ts.isEmpty() || !lhs.device().isCpu() || lhs.dtype() != DType.F32) {
                    return Result.Ok(null)
                }
                if (!lhs.isContiguous()) {
                    return Result.Ok(null)
                }
                val dims1 = lhs.dims()
                val (k1, batch1) = dims1.splitLast() ?: return Result.Ok(null)
                val m: Int = batch1.iter().product()
                if (m == 0) {
                    return Result.Ok(null)
                }
                val dtype = ts[0].dtype()
                val shapes = ArrayList(ts.size)
                for (t1 in ts) {
                    val (n1, tk1) = t1.shape.dims2() ?: return Result.Ok(null)
                    if (tk1 != k1 || t1.dtype() != dtype ||
                        !when (val matchVal = t1.storage) {
                            is QStorage.Cpu -> true
                            else -> false
                        } || !select(dtype, m, n1, k1)
                    ) {
                        return Result.Ok(null)
                    }
                    shapes.add(n1)
                }
                val guard = lhs.storage()
                val cpu1 = guard ?: return Result.Ok(null)
                val slice2 =
                    when (val tryResult = cpu1.asSlice<Float>()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val offset1 = lhs.layout().startOffset()
                val lhsData = ((slice2).slice(offset1 until offset1 + m * k1))
                val lhsQ = quantizeLhs(lhsData, m, k1)
                val packs: List<PackedX86> =
                    (
                        ts.mapIndexed { index, item -> Pair(index, item) }.map({ (i, t2) ->
                            val n2 = shapes[i]
                            t2.repackedQs.x86GetOrInit({ pack(dtype, t2.storageRef(), n2, k1) })
                        })
                    ).toMutableList()
                val parts1: List<Pair<PackedX86, Int>> = (packs.zip(shapes)).map({ (p, n2) -> Pair(p, n2) }).toMutableList()
                val dsts1: List<List<Float>> = shapes.map({ n2 -> mutableListOf(0.0f, m * n2) }).toMutableList()
                gemvFused(parts1, lhsQ, m, k1, dsts1)
                drop(guard)
                val out = ArrayList(ts.size)
                val outDims = dims1.toVec()
                for ((dst, n2) in (dsts1.zip(shapes))) {
                    outDims.lastMut()!! = n2
                    out.add(
                        when (val tryResult = Tensor.fromVec(dst, outDims.clone(), Cpu)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
                }
                Result.Ok(out)
            }
            return run {
                Result.Ok(null)
            }
        }
    }
}
