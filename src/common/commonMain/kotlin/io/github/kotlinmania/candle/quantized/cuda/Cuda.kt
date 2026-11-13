// port-lint: source quantized/cuda.rs
package io.github.kotlinmania.candle.quantized.cuda

import io.github.kotlinmania.candle.cpu.Cpu
import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.cudabackend.CudaStorage
import io.github.kotlinmania.candle.cudabackend.CudaStorageSlice
import io.github.kotlinmania.candle.dummycudabackend.CudaDevice
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.quantized.GgmlDType
import io.github.kotlinmania.candle.quantized.QStorage
import io.github.kotlinmania.candle.quantized.fastmmvq.tryFwd
import io.github.kotlinmania.candle.quantized.kquants.BlockQ2K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ3K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ40
import io.github.kotlinmania.candle.quantized.kquants.BlockQ41
import io.github.kotlinmania.candle.quantized.kquants.BlockQ4K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ50
import io.github.kotlinmania.candle.quantized.kquants.BlockQ51
import io.github.kotlinmania.candle.quantized.kquants.BlockQ5K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ6K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ80
import io.github.kotlinmania.candle.quantized.kquants.BlockQ81
import io.github.kotlinmania.candle.quantized.kquants.BlockQ8K
import io.github.kotlinmania.candle.quantized.kquants.GgmlType
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candlekernels.QUANTIZED
import io.github.kotlinmania.cudarc.driver.CudaSlice
import io.github.kotlinmania.cudarc.driver.CudaStream
import io.github.kotlinmania.cudarc.driver.CudaView
import io.github.kotlinmania.cudarc.driver.LaunchConfig
import io.github.kotlinmania.cudarc.driver.SyncOnDrop
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.rustlang.core.mem.constPtrFromRustCast
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.candle.cudabackend.device.builderArg as barg

public class PaddedCudaSlice internal constructor(
    innerCtorArg: CudaSlice<UByte>,
    public var len: Int,
) {
    public var inner: CudaSlice<UByte> = innerCtorArg
        internal set

    public constructor() : this(
        innerCtorArg = CudaSlice<UByte>(),
        len = 0,
    )

    public operator fun component1(): CudaSlice<UByte> = this.inner

    public operator fun component2(): Int = this.len

    public fun copy(inner: CudaSlice<UByte> = this.inner, len: Int = this.len): PaddedCudaSlice = PaddedCudaSlice(innerCtorArg = inner, len = len)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is PaddedCudaSlice) return false
        if (this.inner != other.inner) return false
        if (this.len != other.len) return false
        return true
    }

    override fun hashCode(): Int {
        var result = this.inner.hashCode()
        result = 31 * result + this.len.hashCode()
        return result
    }

    public companion object {
        public operator fun invoke(
            inner: CudaSlice<UByte> = CudaSlice<UByte>(),
            len: Int = 0,
        ): PaddedCudaSlice =
            PaddedCudaSlice(
                innerCtorArg = inner,
                len = len,
            )
    }
}

public data class QCudaStorage(
    public var data: PaddedCudaSlice = PaddedCudaSlice(),
    public var dtype: GgmlDType = GgmlDType.entries.first(),
    public var device: CudaDevice = CudaDevice(),
) {
    public fun indexedMoeForward(selfShape: Shape, input: CudaStorage, inputL: Layout, ids: CudaStorage, idsL: Layout): Result<Pair<CudaStorage, Shape>> {
        return if (when (this.dtype()) {
                GgmlDType.q80, GgmlDType.q2k, GgmlDType.q3k, GgmlDType.q4k, GgmlDType.q5k, GgmlDType.q6k -> true
                else -> false
            }
        ) {
            val inputStorage =
                when (val tryResult = input.asCudaSlice<Float>()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val idsStorage =
                when (val tryResult = ids.asCudaSlice<UInt>()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            indexedMoeForwardFusedQ81Input(this.data.inner.slice(0 until Int.MAX_VALUE), selfShape, this.dtype(), inputStorage, inputL.shape(), idsStorage.slice(0 until Int.MAX_VALUE), idsL.shape(), this.device)
        } else {
            bail("The given quantized dtype {:?} is not supported for indexed_moe_forward!", this.dtype())
        }
    }

    public fun dequantize(elemCount: Int): Result<CudaStorage> {
        fun <VecDotType> deq(buffer: List<UByte>, n: Int, dst: MutableList<Float>, toFloatTFn: (Any?) -> Any?) {
            val slice = fromRawParts(constPtrFromRustCast(buffer.asPtr()), n)
            val vec = slice.toVec()
            toFloatTFn(vec, dst)
        }
        val fastKernel =
            when (this.dtype) {
                GgmlDType.q40, GgmlDType.q41, GgmlDType.q50, GgmlDType.q51, GgmlDType.q80, GgmlDType.q2k, GgmlDType.q3k, GgmlDType.q4k, GgmlDType.q5k, GgmlDType.q6k, GgmlDType.q8k -> true
                else -> false
            }
        if (fastKernel) {
            return dequantizeF32(this.data, this.dtype, elemCount, this.device())
        }
        // Run the dequantization on cpu.
        val buffer =
            when (val tryResult = this.device.cloneDtoh(this.data.inner.slice(0 until this.data.len))) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val out = mutableListOf(0.0, elemCount)
        val blockLen = elemCount / this.dtype.blockSize()
        when (this.dtype) {
            GgmlDType.f32 -> deq<Float>(buffer, blockLen, out)
            GgmlDType.f16 -> deq<F16>(buffer, blockLen, out)
            GgmlDType.bf16 -> deq<Bf16>(buffer, blockLen, out)
            GgmlDType.q40 -> deq<BlockQ40>(buffer, blockLen, out)
            GgmlDType.q41 -> deq<BlockQ41>(buffer, blockLen, out)
            GgmlDType.q50 -> deq<BlockQ50>(buffer, blockLen, out)
            GgmlDType.q51 -> deq<BlockQ51>(buffer, blockLen, out)
            GgmlDType.q80 -> deq<BlockQ80>(buffer, blockLen, out)
            GgmlDType.q81 -> deq<BlockQ81>(buffer, blockLen, out)
            GgmlDType.q2k -> deq<BlockQ2K>(buffer, blockLen, out)
            GgmlDType.q3k -> deq<BlockQ3K>(buffer, blockLen, out)
            GgmlDType.q4k -> deq<BlockQ4K>(buffer, blockLen, out)
            GgmlDType.q5k -> deq<BlockQ5K>(buffer, blockLen, out)
            GgmlDType.q6k -> deq<BlockQ6K>(buffer, blockLen, out)
            GgmlDType.q8k -> deq<BlockQ8K>(buffer, blockLen, out)
        }
        return this.device.storageFromCpuStorage(CpuStorage.F32(out))
    }

    public fun dequantizeF16(elemCount: Int): Result<CudaStorage> = dequantizeF16(this.data, this.dtype, elemCount, this.device())

    public fun quantize(src: CudaStorage): Result<Unit> {
        // Run the quantization on cpu.
        val src1 =
            when (val matchVal = src.slice) {
                is CudaStorageSlice.F32 -> {
                    val (data) = matchVal
                    when (val tryResult = this.device.cloneDtoh(data)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                else -> bail("only f32 can be quantized")
            }
        val srcLen = src1.len()
        val src2 = Cpu(CpuStorage.F32(src1))
        val qcpuStorage =
            when (val tryResult = Cpu.qzeros(srcLen, this.dtype)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        when (val tryResult = qcpuStorage.quantize(src2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        val data1 =
            when (val tryResult = qcpuStorage.data()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val paddedLen = data1.len() + MATRIX_ROW_PADDING * this.dtype.typeSize() / this.dtype.blockSize()
        var inner =
            when (val tryResult = this.device.alloc<UByte>(paddedLen)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        when (val tryResult = this.device.memcpyHtod(data1, inner.sliceMut(0 until data1.len()))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        this.data = PaddedCudaSlice(inner = inner, len = data1.len())
        return Result.Ok(Unit)
    }

    public fun quantizeImatrix(src: CudaStorage, imatrixWeights: List<Float>, nPerRow: Int): Result<Unit> {
        // Run the quantization on cpu.
        val src1 =
            when (val matchVal = src.slice) {
                is CudaStorageSlice.F32 -> {
                    val (data) = matchVal
                    when (val tryResult = this.device.cloneDtoh(data)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                else -> bail("only f32 can be quantized")
            }
        val srcLen = src1.len()
        val src2 = Cpu(CpuStorage.F32(src1))
        val qcpuStorage =
            when (val tryResult = Cpu.qzeros(srcLen, this.dtype)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        when (val tryResult = qcpuStorage.quantizeImatrix(src2, imatrixWeights, nPerRow)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        val data1 =
            when (val tryResult = qcpuStorage.data()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val paddedLen = data1.len() + MATRIX_ROW_PADDING * this.dtype.typeSize() / this.dtype.blockSize()
        var inner =
            when (val tryResult = this.device.alloc<UByte>(paddedLen)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        when (val tryResult = this.device.memcpyHtod(data1, inner.sliceMut(0 until data1.len()))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        this.data = PaddedCudaSlice(inner = inner, len = data1.len())
        return Result.Ok(Unit)
    }

    public fun quantizeImatrixOnto(src: CpuStorage, imatrixWeights: List<Float>, nPerRow: Int): Result<Unit> {
        // Run the quantization on cpu.
        val srcLen =
            when (val tryResult = src.asSlice<Float>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }.len()
        val qcpuStorage =
            when (val tryResult = Cpu.qzeros(srcLen, this.dtype)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        if (qcpuStorage is QStorage.Cpu) {
            val (storage) = qcpuStorage
            storage.fromFloatImatrix(
                when (val tryResult = src.asSlice<Float>()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                },
                imatrixWeights,
                nPerRow,
            )
        } else {
            error("unreachable")
        }
        val data =
            when (val tryResult = qcpuStorage.data()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val paddedLen = data.len() + MATRIX_ROW_PADDING * this.dtype.typeSize() / this.dtype.blockSize()
        var inner =
            when (val tryResult = this.device.alloc<UByte>(paddedLen)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        when (val tryResult = this.device.memcpyHtod(data, inner.sliceMut(0 until data.len()))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        this.data = PaddedCudaSlice(inner = inner, len = data.len())
        return Result.Ok(Unit)
    }

    public fun quantizeOnto(src: CpuStorage): Result<Unit> {
        // Run the quantization on cpu.
        val srcLen =
            when (val tryResult = src.asSlice<Float>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }.len()
        val qcpuStorage =
            when (val tryResult = Cpu.qzeros(srcLen, this.dtype)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        if (qcpuStorage is QStorage.Cpu) {
            val (storage) = qcpuStorage
            storage.fromFloat(
                when (val tryResult = src.asSlice<Float>()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                },
            )
        } else {
            error("unreachable")
        }
        val data =
            when (val tryResult = qcpuStorage.data()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val paddedLen = data.len() + MATRIX_ROW_PADDING * this.dtype.typeSize() / this.dtype.blockSize()
        var inner =
            when (val tryResult = this.device.alloc<UByte>(paddedLen)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        when (val tryResult = this.device.memcpyHtod(data, inner.sliceMut(0 until data.len()))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        this.data = PaddedCudaSlice(inner = inner, len = data.len())
        return Result.Ok(Unit)
    }

    public fun storageSizeInBytes(): Int = this.data.len

    public fun embedding(rows: Int, hidden: Int, ids: CudaStorage, idsL: Layout): Result<CudaStorage> {
        if (!idsL.isContiguous()) {
            bail("quantized embedding requires contiguous ids")
        }
        if (!hidden.isMultipleOf(this.dtype.blockSize())) {
            bail("quantized embedding hidden size {hidden} is not divisible by block size {}", this.dtype.blockSize())
        }
        val expectedSize = rows * hidden * this.dtype.typeSize() / this.dtype.blockSize()
        if (this.storageSizeInBytes() != expectedSize) {
            bail("quantized tensor has {} bytes, expected {expected_size}", this.storageSizeInBytes())
        }
        val ids1 =
            when (val tryResult = ids.asCudaSlice<UInt>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val ids2 =
            when (val matchVal = idsL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Result.Err(Error.RequiresContiguous(op = "quantized-embedding").bt())) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                else -> {
                    val o1 = matchVal.first
                    val o2 = matchVal.second
                    ids1.slice(o1 until o2)
                }
            }
        return getRows(this.data, this.dtype, hidden, ids2, this.device())
    }

    public fun fwd(selfShape: Shape, storage: CudaStorage, layout: Layout): Result<Pair<CudaStorage, Shape>> {
        // Optimized MMVQ and MMQ paths (support most paths: BF16/F16/F32, batch 1-8, all quant types, reuses per-device workspace).
        if (!FORCE_DMMV.load(Ordering.Relaxed)) {
            when (
                val matchVal =
                    when (val tryResult = tryFwd(this, selfShape, storage, layout)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
            ) {
                null -> {}
                else -> {
                    val result = matchVal
                    return Result.Ok(result)
                }
            }
            when (
                val matchVal1 =
                    when (
                        val tryResult =
                            io.github.kotlinmania.candle.quantized.fastmmq
                                .tryFwd(this, selfShape, storage, layout)
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
            ) {
                null -> {}
                else -> {
                    val result1 = matchVal1
                    return Result.Ok(result1)
                }
            }
        }
        // Fallback
        val maxBm =
            if (FORCE_DMMV.load(Ordering.Relaxed)) {
                1
            } else {
                8
            }
        val useVecKernel =
            run {
                val matchVal2 = layout.shape().dims()
                when {
                    matchVal2.size == 3 -> {
                        val b = matchVal2[0]
                        val m = matchVal2[1]
                        b * m <= maxBm
                    }
                    matchVal2.size == 2 -> {
                        val b1 = matchVal2[0]
                        b1 <= maxBm
                    }
                    else -> false
                }
            }
        return if (useVecKernel) {
            this.dequantizeMatmulVec(selfShape, storage, layout)
        } else {
            this.dequantizeMatmul(selfShape, storage, layout)
        }
    }

    public fun devicePtr(): Result<ConstPtr> =
        Result.Ok(
            constPtrFromRustCast(
                this.data.inner
                    .devicePtr(this.data.inner.stream())
                    .first,
            ),
        )

    public fun devicePtrWithGuard(stream: CudaStream): Result<Pair<ConstPtr, SyncOnDrop>> {
        val (ptr, guard) = this.data.inner.devicePtr(stream)
        return Result.Ok(Pair(constPtrFromRustCast(ptr), guard))
    }

    public fun dequantizeMatmulVec(selfShape: Shape, rhs: CudaStorage, rhsL: Layout): Result<Pair<CudaStorage, Shape>> {
        val (nrows, ncols) =
            when (val tryResult = selfShape.dims2()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val rhs1 =
            when (val tryResult = rhs.asCudaSlice<Float>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val rhs2 =
            when (val matchVal = rhsL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Result.Err(Error.RequiresContiguous(op = "dmmv").bt())) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                else -> {
                    val o1 = matchVal.first
                    val o2 = matchVal.second
                    rhs1.slice(o1 until o2)
                }
            }
        val (bSize, k2) =
            run {
                val matchVal1 = rhsL.shape().dims()
                when {
                    matchVal1.size == 3 -> {
                        val b = matchVal1[0]
                        val m = matchVal1[1]
                        val k = matchVal1[2]
                        Pair(b * m, k)
                    }
                    matchVal1.size == 2 -> {
                        val b1 = matchVal1[0]
                        val k1 = matchVal1[1]
                        Pair(b1, k1)
                    }
                    else -> bail("unexpected rhs shape in dmmv {:?}", rhsL.shape())
                }
            }
        if (ncols != k2) {
            bail("mismatch on matmul dim {self_shape:?} {:?}", rhsL.shape())
        }
        val out =
            if (FORCE_DMMV.load(Ordering.Relaxed)) {
                when (val tryResult = dequantizeMulMatVec(this.data, rhs2, this.dtype, ncols, nrows, this.device())) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            } else {
                when (val tryResult = mulMatVecViaQ81(this.data, rhs2, this.dtype, ncols, nrows, bSize, this.device())) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
        val outShape = rhsL.shape().dims().toVec()
        outShape.pop()
        outShape.push(nrows)
        return Result.Ok(Pair(out, outShape.into()))
    }

    public fun dequantizeMatmul(selfShape: Shape, storage: CudaStorage, layout: Layout): Result<Pair<CudaStorage, Shape>> {
        val (n, k) =
            when (val tryResult = selfShape.dims2()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val (b1, m2, k22) =
            run {
                val matchVal = layout.shape().dims()
                when {
                    matchVal.size == 3 -> {
                        val b = matchVal[0]
                        val m = matchVal[1]
                        val k2 = matchVal[2]
                        Triple(b, m, k2)
                    }
                    matchVal.size == 2 -> {
                        val m1 = matchVal[0]
                        val k21 = matchVal[1]
                        Triple(1, m1, k21)
                    }
                    else -> {
                        val s = matchVal
                        bail("unexpected shape for input {s:?}")
                    }
                }
            }
        if (k22 != k) {
            bail("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }
        val out =
            if (FORCE_DMMV.load(Ordering.Relaxed)) {
                val dataF32 =
                    when (val tryResult = this.dequantize(n * k)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val rhsL =
                    when (val tryResult = Layout.new(Pair(k, n).into(), mutableListOf(1, k), 0).broadcastAs(Triple(b1, k, n))) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                when (val tryResult = storage.matmul(dataF32, listOf(b1, m2, n, k), layout, rhsL)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            } else {
                val storage1 =
                    when (val tryResult = storage.asCudaSlice<Float>()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val storage2 =
                    when (val matchVal1 = layout.contiguousOffsets()) {
                        null ->
                            when (val tryResult = Result.Err(Error.RequiresContiguous(op = "quantized-matmul").bt())) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }
                        else -> {
                            val o1 = matchVal1.first
                            val o2 = matchVal1.second
                            storage1.slice(o1 until o2)
                        }
                    }
                when (val tryResult = mulMatViaQ81(this.data, storage2, this.dtype, n, k, k, b1 * m2, this.device())) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
        val outShape = layout.shape().dims().toVec()
        outShape.pop()
        outShape.push(n)
        return Result.Ok(Pair(out, outShape.into()))
    }

    public companion object {
        public fun zeros(device: CudaDevice, elCount: Int, dtype: GgmlDType): Result<QCudaStorage> {
            val sizeInBytes = ceilDiv(elCount, dtype.blockSize()) * dtype.typeSize()
            val paddedSizeInBytes = ceilDiv(elCount + MATRIX_ROW_PADDING, dtype.blockSize()) * dtype.typeSize()
            val inner =
                when (val tryResult = device.allocZeros<UByte>(paddedSizeInBytes)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            return Result.Ok(QCudaStorage(data = PaddedCudaSlice(inner = inner, len = sizeInBytes), device = device.clone(), dtype = dtype))
        }
    }
}

internal var FORCE_DMMV: AtomicBool = AtomicBool.new(false)

public fun setForceDmmv(f: Boolean) {
    FORCE_DMMV.store(f, Ordering.Relaxed)
}

public val WARP_SIZE: Int = 32

public val MMQ_X_Q4_0_AMPERE: Int = 4

public val MMQ_Y_Q4_0_AMPERE: Int = 32

public val NWARPS_Q4_0_AMPERE: Int = 4

public val GGML_CUDA_MMV_X: Int = 32

public val GGML_CUDA_MMV_Y: Int = 1

public val CUDA_QUANTIZE_BLOCK_SIZE: Int = 256

public val CUDA_DEQUANTIZE_BLOCK_SIZE: Int = 256

public val CUDA_GET_ROWS_BLOCK_SIZE: Int = 256

public val MATRIX_ROW_PADDING: Int = 512

public fun ceilDiv(p: Int, q: Int): Int = p.divCeil(q)

public fun pad(p: Int, q: Int): Int = ceilDiv(p, q) * q

public fun quantizeQ81(src: CudaView<Float>, dst: CudaSlice<UByte>, k: Int, ky: Int, dev: CudaDevice): Result<Unit> {
    val CHUNK_SIZE: Int = 65535
    val kxPadded = pad(k, MATRIX_ROW_PADDING)
    val numBlocks = ceilDiv(kxPadded, CUDA_QUANTIZE_BLOCK_SIZE)
    val totalRows = ky
    // Get Q8_1 metadata.
    val q81BlockSize = GgmlDType.Q8_1.blockSize()
    val q81TypeSize = GgmlDType.Q8_1.typeSize()
    // Calculate the size of the output buffer in bytes.
    val numBlocksPerRow = kxPadded / q81BlockSize
    val dstRowSizeBytes = numBlocksPerRow * q81TypeSize
    // gridDim.y limit
    val func =
        when (val tryResult = dev.getOrLoadFunc("quantize_q8_1", QUANTIZED)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    var rowsProcessed = 0
    while (rowsProcessed < totalRows) {
        // --- calculate the number of rows for this chunk ---
        val remainingRows = totalRows - rowsProcessed
        // This is our gridDim.y, now <= 65535
        val rowsInChunk = minOf(CHUNK_SIZE, remainingRows)
        // --- slice the source (f32) tensor by elements ---
        val srcStartElem = rowsProcessed * k
        val srcNumElems = rowsInChunk * k
        val srcChunk = src.slice(srcStartElem until (srcStartElem + srcNumElems))
        // --- slice the destination (u8) tensor by bytes ---
        val dstStartByte = rowsProcessed * dstRowSizeBytes
        val dstNumBytes = rowsInChunk * dstRowSizeBytes
        val dstChunk = dst.slice(dstStartByte until (dstStartByte + dstNumBytes))
        val cfg = LaunchConfig(gridDim = Triple(numBlocks.toUInt(), rowsInChunk.toUInt(), 1), blockDim = Triple(CUDA_QUANTIZE_BLOCK_SIZE.toUInt(), 1, 1), sharedMemBytes = 0)
        val builder = func.builder()
        builder.arg(srcChunk)
        builder.arg(dstChunk)
        barg(builder, k, kxPadded)
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        rowsProcessed += rowsInChunk
    }
    return Result.Ok(Unit)
}

public fun dequantizeF32(data: PaddedCudaSlice, dtype: GgmlDType, elemCount: Int, dev: CudaDevice): Result<CudaStorage> {
    val nb = elemCount.divCeil(256)
    val (kernelName, isK, blockDim, numBlocks) =
        when (dtype) {
            GgmlDType.q40 -> listOf("dequantize_block_q4_0_f32", false, 32, nb)
            GgmlDType.q41 -> listOf("dequantize_block_q4_1_f32", false, 32, nb)
            GgmlDType.q50 -> listOf("dequantize_block_q5_0_f32", false, CUDA_DEQUANTIZE_BLOCK_SIZE, ceilDiv(elemCount, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE))
            GgmlDType.q51 -> listOf("dequantize_block_q5_1_f32", false, CUDA_DEQUANTIZE_BLOCK_SIZE, ceilDiv(elemCount, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE))
            GgmlDType.q80 -> listOf("dequantize_block_q8_0_f32", false, 32, nb)
            GgmlDType.q2k -> listOf("dequantize_block_q2_K_f32", true, 64, nb)
            GgmlDType.q3k -> listOf("dequantize_block_q3_K_f32", true, 64, nb)
            GgmlDType.q4k -> listOf("dequantize_block_q4_K_f32", true, 32, nb)
            GgmlDType.q5k -> listOf("dequantize_block_q5_K_f32", true, 64, nb)
            GgmlDType.q6k -> listOf("dequantize_block_q6_K_f32", true, 64, nb)
            GgmlDType.q8k -> listOf("dequantize_block_q8_K_f32", true, 32, nb)
            else -> bail("unsupported dtype for dequantize {dtype:?}")
        }
    val func =
        when (val tryResult = dev.getOrLoadFunc(kernelName, QUANTIZED)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst =
        when (val tryResult = dev.alloc<Float>(elemCount)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    val cfg = LaunchConfig(gridDim = Triple(numBlocks.toUInt(), 1, 1), blockDim = Triple(blockDim.toUInt(), 1, 1), sharedMemBytes = 0)
    if (isK) {
        val builder = func.builder()
        builder.arg(data.inner)
        builder.arg(dst)
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    } else {
        val nb32 =
            when (dtype) {
                GgmlDType.q50, GgmlDType.q51 -> elemCount
                else -> elemCount / 32
            }
        val builder1 = func.builder()
        builder1.arg(data.inner)
        builder1.arg(dst)
        barg(builder1, nb32.toInt())
        when (val tryResult = builder1.launch(cfg).w()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    }
    return Result.Ok(CudaStorage.wrapCudaSlice(dst, dev.clone()))
}

public fun dequantizeF16(data: PaddedCudaSlice, dtype: GgmlDType, elemCount: Int, dev: CudaDevice): Result<CudaStorage> {
    val nb = elemCount.divCeil(256)
    val (kernelName, isK, blockDim, numBlocks) =
        when (dtype) {
            GgmlDType.q40 -> listOf("dequantize_block_q4_0_f16", false, 32, nb)
            GgmlDType.q41 -> listOf("dequantize_block_q4_1_f16", false, 32, nb)
            GgmlDType.q50 -> listOf("dequantize_block_q5_0_f16", false, CUDA_DEQUANTIZE_BLOCK_SIZE, ceilDiv(elemCount, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE))
            GgmlDType.q51 -> listOf("dequantize_block_q5_1_f16", false, CUDA_DEQUANTIZE_BLOCK_SIZE, ceilDiv(elemCount, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE))
            GgmlDType.q80 -> listOf("dequantize_block_q8_0_f16", false, 32, nb)
            GgmlDType.q2k -> listOf("dequantize_block_q2_K_f16", true, 64, nb)
            GgmlDType.q3k -> listOf("dequantize_block_q3_K_f16", true, 64, nb)
            GgmlDType.q4k -> listOf("dequantize_block_q4_K_f16", true, 32, nb)
            GgmlDType.q5k -> listOf("dequantize_block_q5_K_f16", true, 64, nb)
            GgmlDType.q6k -> listOf("dequantize_block_q6_K_f16", true, 64, nb)
            GgmlDType.q8k -> listOf("dequantize_block_q8_K_f16", true, 32, nb)
            else -> bail("unsupported dtype for dequantize {dtype:?}")
        }
    val func =
        when (val tryResult = dev.getOrLoadFunc(kernelName, QUANTIZED)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst =
        when (val tryResult = dev.alloc<F16>(elemCount)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    val cfg = LaunchConfig(gridDim = Triple(numBlocks.toUInt(), 1, 1), blockDim = Triple(blockDim.toUInt(), 1, 1), sharedMemBytes = 0)
    if (isK) {
        val builder = func.builder()
        builder.arg(data.inner)
        builder.arg(dst)
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    } else {
        val nb32 =
            when (dtype) {
                GgmlDType.q50, GgmlDType.q51 -> elemCount
                else -> elemCount / 32
            }
        val builder1 = func.builder()
        builder1.arg(data.inner)
        builder1.arg(dst)
        barg(builder1, nb32.toInt())
        when (val tryResult = builder1.launch(cfg).w()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    }
    return Result.Ok(CudaStorage.wrapCudaSlice(dst, dev.clone()))
}

public fun getRows(data: PaddedCudaSlice, dtype: GgmlDType, hidden: Int, ids: CudaView<UInt>, dev: CudaDevice): Result<CudaStorage> {
    val (kernelName, blockDim, blockNumY, canStrideY) =
        when (dtype) {
            GgmlDType.f32 -> listOf("get_rows_f32", CUDA_GET_ROWS_BLOCK_SIZE, ceilDiv(hidden, CUDA_GET_ROWS_BLOCK_SIZE), true)
            GgmlDType.f16 -> listOf("get_rows_f16", CUDA_GET_ROWS_BLOCK_SIZE, ceilDiv(hidden, CUDA_GET_ROWS_BLOCK_SIZE), true)
            GgmlDType.bf16 -> listOf("get_rows_bf16", CUDA_GET_ROWS_BLOCK_SIZE, ceilDiv(hidden, CUDA_GET_ROWS_BLOCK_SIZE), true)
            GgmlDType.q40 -> listOf("get_rows_q4_0", CUDA_GET_ROWS_BLOCK_SIZE, ceilDiv(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE), true)
            GgmlDType.q41 -> listOf("get_rows_q4_1", CUDA_GET_ROWS_BLOCK_SIZE, ceilDiv(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE), true)
            GgmlDType.q50 -> listOf("get_rows_q5_0", CUDA_GET_ROWS_BLOCK_SIZE, ceilDiv(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE), true)
            GgmlDType.q51 -> listOf("get_rows_q5_1", CUDA_GET_ROWS_BLOCK_SIZE, ceilDiv(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE), true)
            GgmlDType.q80 -> listOf("get_rows_q8_0", CUDA_GET_ROWS_BLOCK_SIZE, ceilDiv(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE), true)
            GgmlDType.q2k -> listOf("get_rows_q2_K", 64, hidden / dtype.blockSize(), false)
            GgmlDType.q3k -> listOf("get_rows_q3_K", 64, hidden / dtype.blockSize(), false)
            GgmlDType.q4k -> listOf("get_rows_q4_K", 32, hidden / dtype.blockSize(), false)
            GgmlDType.q5k -> listOf("get_rows_q5_K", 64, hidden / dtype.blockSize(), false)
            GgmlDType.q6k -> listOf("get_rows_q6_K", 64, hidden / dtype.blockSize(), false)
            else -> bail("unsupported dtype for CUDA quantized embedding {dtype:?}")
        }
    val func =
        when (val tryResult = dev.getOrLoadFunc(kernelName, QUANTIZED)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val idsLen = ids.len()
    val dst =
        when (val tryResult = dev.alloc<Float>(idsLen * hidden)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    if (idsLen == 0) {
        return Result.Ok(CudaStorage.wrapCudaSlice(dst, dev.clone()))
    }
    if (!canStrideY && blockNumY > UShort.MAX_VALUE.toInt()) {
        bail("quantized embedding hidden size {hidden} exceeds CUDA grid y limit")
    }
    val gridY =
        if (canStrideY) {
            blockNumY.min(UShort.MAX_VALUE.toInt())
        } else {
            blockNumY
        }
    val cfg = LaunchConfig(gridDim = Triple(idsLen.toUInt(), gridY.toUInt(), 1), blockDim = Triple(blockDim.toUInt(), 1, 1), sharedMemBytes = 0)
    val rowStride = hidden * dtype.typeSize() / dtype.blockSize()
    val builder = func.builder()
    builder.arg(data.inner)
    builder.arg(ids)
    builder.arg(dst)
    barg(builder, hidden.toLong(), rowStride)
    when (val tryResult = builder.launch(cfg).w()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(CudaStorage.wrapCudaSlice(dst, dev.clone()))
}

public fun dequantizeMulMatVec(data: PaddedCudaSlice, y: CudaView<Float>, dtype: GgmlDType, ncols: Int, nrows: Int, dev: CudaDevice): Result<CudaStorage> {
    val dataElems = data.len / dtype.typeSize() * dtype.blockSize()
    if (dataElems < ncols * nrows) {
        bail("unexpected data size {}, ncols {ncols} {nrows}", dataElems)
    }
    if (y.len() != ncols) {
        bail("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    val kernelName =
        when (dtype) {
            GgmlDType.q40 -> "dequantize_mul_mat_vec_q4_0_cuda"
            GgmlDType.q41 -> "dequantize_mul_mat_vec_q4_1_cuda"
            GgmlDType.q50 -> "dequantize_mul_mat_vec_q5_0_cuda"
            GgmlDType.q51 -> "dequantize_mul_mat_vec_q5_1_cuda"
            GgmlDType.q80 -> "dequantize_mul_mat_vec_q8_0_cuda"
            GgmlDType.q2k -> "dequantize_mul_mat_vec_q2_k"
            GgmlDType.q3k -> "dequantize_mul_mat_vec_q3_k"
            GgmlDType.q4k -> "dequantize_mul_mat_vec_q4_k"
            GgmlDType.q5k -> "dequantize_mul_mat_vec_q5_k"
            GgmlDType.q6k -> "dequantize_mul_mat_vec_q6_k"
            else -> bail("unsupported dtype for quantized matmul {dtype:?}")
        }
    val func =
        when (val tryResult = dev.getOrLoadFunc(kernelName, QUANTIZED)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst =
        when (val tryResult = dev.alloc<Float>(nrows)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val blockNumY = ceilDiv(nrows, GGML_CUDA_MMV_Y)
    val cfg = LaunchConfig(gridDim = Triple(blockNumY.toUInt(), 1, 1), blockDim = Triple(WARP_SIZE.toUInt(), GGML_CUDA_MMV_Y.toUInt(), 1), sharedMemBytes = 0)
    val builder = func.builder()
    builder.arg(data.inner)
    builder.arg(y)
    builder.arg(dst)
    barg(builder, ncols, nrows)
    when (val tryResult = builder.launch(cfg).w()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(CudaStorage.wrapCudaSlice(dst, dev.clone()))
}

public fun mulMatVecViaQ81(data: PaddedCudaSlice, y: CudaView<Float>, dtype: GgmlDType, ncols: Int, nrows: Int, bSize: Int, dev: CudaDevice): Result<CudaStorage> {
    val dataElems = data.len / dtype.typeSize() * dtype.blockSize()
    if (dataElems < ncols * nrows) {
        bail("unexpected data size {}, ncols {ncols} {nrows}", dataElems)
    }
    if (y.len() != ncols * bSize) {
        bail("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    if (bSize == 0 || bSize > 8) {
        bail("only bsize between 1 and 8 are supported, got {b_size}")
    }
    // Start by quantizing y
    val ncolsPadded = pad(ncols, MATRIX_ROW_PADDING)
    val ySizeInBytes = bSize * ncolsPadded * GgmlDType.Q8_1.typeSize() / GgmlDType.Q8_1.blockSize()
    val yQ81 =
        when (val tryResult = dev.allocZeros<UByte>(ySizeInBytes)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val tryResult = quantizeQ81(y, yQ81, ncols, bSize, dev)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    val kernelName =
        when (dtype) {
            GgmlDType.q40 -> "mul_mat_vec_q4_0_q8_1_cuda"
            GgmlDType.q41 -> "mul_mat_vec_q4_1_q8_1_cuda"
            GgmlDType.q50 -> "mul_mat_vec_q5_0_q8_1_cuda"
            GgmlDType.q51 -> "mul_mat_vec_q5_1_q8_1_cuda"
            GgmlDType.q80 -> "mul_mat_vec_q8_0_q8_1_cuda"
            GgmlDType.q2k -> "mul_mat_vec_q2_K_q8_1_cuda"
            GgmlDType.q3k -> "mul_mat_vec_q3_K_q8_1_cuda"
            GgmlDType.q4k -> "mul_mat_vec_q4_K_q8_1_cuda"
            GgmlDType.q5k -> "mul_mat_vec_q5_K_q8_1_cuda"
            GgmlDType.q6k -> "mul_mat_vec_q6_K_q8_1_cuda"
            else -> bail("unsupported dtype for quantized matmul {dtype:?}")
        }
    val kernelName1 = "$kernelName$bSize"
    val func =
        when (val tryResult = dev.getOrLoadFunc(kernelName1, QUANTIZED)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst =
        when (val tryResult = dev.allocZeros<Float>(nrows * bSize)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // https://github.com/ggerganov/llama.cpp/blob/facb8b56f8fd3bb10a693bf0943ae9d69d0828ef/ggml-cuda/mmvq.cu#L98
    val (nblocks, nwarps) =
        when (bSize) {
            1 -> {
                Pair(nrows.toUInt(), 4)
            }
            in 2..4 -> {
                Pair((nrows.toUInt()).divCeil(2), 4)
            }
            in 5..8 -> Pair((nrows.toUInt()).divCeil(2), 2)
            else -> bail("unexpected bsize {b_size}")
        }
    val cfg = LaunchConfig(gridDim = Triple(nblocks, 1, 1), blockDim = Triple(WARP_SIZE.toUInt(), nwarps, 1), sharedMemBytes = 0)
    val builder = func.builder()
    builder.arg(data.inner)
    builder.arg(yQ81)
    builder.arg(dst)
    barg(builder, ncols, nrows, ncolsPadded, nrows)
    when (val tryResult = builder.launch(cfg).w()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(CudaStorage.wrapCudaSlice(dst, dev.clone()))
}

public fun mulMatViaQ81(data: PaddedCudaSlice, y: CudaView<Float>, dtype: GgmlDType, xRows: Int, xCols: Int, yRows: Int, yCols: Int, dev: CudaDevice): Result<CudaStorage> {
    val dataElems = data.len / dtype.typeSize() * dtype.blockSize()
    if (dataElems < xRows * xCols) {
        bail("unexpected lhs size {}, {x_rows} {x_cols}", dataElems)
    }
    if (y.len() != yRows * yCols) {
        bail("unexpected y size {}, {y_rows} {y_cols}", y.len())
    }
    if (xCols != yRows) {
        bail("unexpected x/y size {x_rows} {x_cols} {y_rows} {y_cols}")
    }
    val k = xCols
    // Start by quantizing y
    val kPadded = pad(k, MATRIX_ROW_PADDING)
    val ySizeInBytes = kPadded * yCols * GgmlDType.Q8_1.typeSize() / GgmlDType.Q8_1.blockSize()
    val yQ81 =
        when (val tryResult = dev.allocZeros<UByte>(ySizeInBytes)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val tryResult = quantizeQ81(y, yQ81, k, yCols, dev)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    val (kernelName, mmqX, mmqY) =
        when (dtype) {
            GgmlDType.q40 -> Triple("mul_mat_q4_0", 64, 128)
            GgmlDType.q41 -> Triple("mul_mat_q4_1", 64, 128)
            GgmlDType.q50 -> Triple("mul_mat_q5_0", 128, 64)
            GgmlDType.q51 -> Triple("mul_mat_q5_1", 128, 64)
            GgmlDType.q80 -> Triple("mul_mat_q8_0", 128, 64)
            GgmlDType.q2k -> Triple("mul_mat_q2_K", 64, 128)
            GgmlDType.q3k -> Triple("mul_mat_q3_K", 128, 128)
            GgmlDType.q4k -> Triple("mul_mat_q4_K", 64, 128)
            GgmlDType.q5k -> Triple("mul_mat_q5_K", 64, 128)
            GgmlDType.q6k -> Triple("mul_mat_q6_K", 64, 64)
            else -> bail("unsupported dtype for quantized matmul {dtype:?}")
        }
    val func =
        when (val tryResult = dev.getOrLoadFunc(kernelName, QUANTIZED)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst =
        when (val tryResult = dev.allocZeros<Float>(xRows * yCols)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val cfg = LaunchConfig(gridDim = Triple(ceilDiv(xRows, mmqY).toUInt(), ceilDiv(yCols, mmqX).toUInt(), 1), blockDim = Triple(WARP_SIZE.toUInt(), 4, 1), sharedMemBytes = 0)
    val builder = func.builder()
    builder.arg(data.inner)
    builder.arg(yQ81)
    builder.arg(dst)
    barg(builder, xCols, xRows, yCols, kPadded, xRows)
    when (val tryResult = builder.launch(cfg).w()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(CudaStorage.wrapCudaSlice(dst, dev.clone()))
}

public fun indexedMoeForwardFusedQ81Input(weight: CudaView<UByte>, wShape: Shape, wDtype: GgmlDType, input: CudaSlice<Float>, inShape: Shape, ids: CudaView<UInt>, idxShape: Shape, dev: CudaDevice): Result<Pair<CudaStorage, Shape>> {
    val (_, n, k) =
        when (val tryResult = wShape.dims3()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val batch = inShape.dims()[0]
    val inputDim1 = inShape.dims()[1]
    val topk = idxShape.dims()[1]
    check(batch == idxShape.dims()[0]) { "batch dim not match!" }
    // Quantize input into q81.
    val totalRows = batch * inputDim1
    val kPadded = pad(k, MATRIX_ROW_PADDING)
    // Get Q8_1 metadata.
    val q81BlockSize = GgmlDType.Q8_1.blockSize()
    val q81TypeSize = GgmlDType.Q8_1.typeSize()
    // Calculate the size of the output buffer in bytes.
    val numBlocksPerRow = kPadded / q81BlockSize
    val dstRowSizeBytes = numBlocksPerRow * q81TypeSize
    val ySizeInBytes = totalRows * dstRowSizeBytes
    val inputQuant =
        when (val tryResult = dev.allocZeros<UByte>(ySizeInBytes)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val inputView = input.slice(0 until Int.MAX_VALUE)
    when (val tryResult = quantizeQ81(inputView, inputQuant, k, totalRows, dev)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    // output buffer
    val outsize = batch * topk * n
    val out =
        when (val tryResult = dev.allocZeros<Float>(outsize)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val kernelName =
        when (wDtype) {
            GgmlDType.q2k -> "indexed_moe_forward_q2k_q8_1"
            GgmlDType.q3k -> "indexed_moe_forward_q3k_q8_1"
            GgmlDType.q4k -> "indexed_moe_forward_q4k_q8_1"
            GgmlDType.q5k -> "indexed_moe_forward_q5k_q8_1"
            GgmlDType.q6k -> "indexed_moe_forward_q6k_q8_1"
            GgmlDType.q80 -> "indexed_moe_forward_q8_0_q8_1"
            else -> bail("unsupported dtype for indexed_moe_forward {w_dtype:?}")
        }
    val func =
        when (val tryResult = dev.getOrLoadFunc(kernelName, QUANTIZED)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val (nblocks, nwarps) = Pair(n.toUInt(), 4)
    val cfg = LaunchConfig(gridDim = Triple(nblocks, batch.toUInt(), topk.toUInt()), blockDim = Triple(WARP_SIZE.toUInt(), nwarps, 1), sharedMemBytes = 0)
    val builder = func.builder()
    builder.arg(weight)
    builder.arg(inputQuant)
    builder.arg(ids)
    builder.arg(out)
    barg(builder, n.toInt(), k.toInt(), batch.toInt(), topk.toInt(), kPadded, inputDim1.toInt())
    when (val tryResult = builder.launch(cfg).w()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    val outShape = inShape.dims().toVec()
    outShape.pop()
    outShape.push(n)
    outShape[1] = topk
    return Result.Ok(Pair(CudaStorage.wrapCudaSlice(out, dev.clone()), outShape.into()))
}

public fun <VecDotType, T : GgmlType> loadQuantized(device: CudaDevice, data: List<T>): Result<QStorage> {
    val data1 = fromRawParts(constPtrFromRustCast(data.asPtr()), sizeOfVal(data))
    val dtype = T.DTYPE
    val paddedLen = data1.len() + MATRIX_ROW_PADDING * dtype.typeSize() / dtype.blockSize()
    var inner =
        when (val tryResult = device.allocZeros<UByte>(paddedLen)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val tryResult = device.memcpyHtod(data1, inner.sliceMut(0 until data1.len()))) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(QStorage.Cuda(QCudaStorage(data = PaddedCudaSlice(inner = inner, len = data1.len()), device = device.clone(), dtype = dtype)))
}
