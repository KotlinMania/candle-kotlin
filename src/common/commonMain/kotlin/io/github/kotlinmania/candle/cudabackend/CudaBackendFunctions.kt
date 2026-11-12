package io.github.kotlinmania.candle.cudabackend

import io.github.kotlinmania.candle.cudabackend.device.CudaDevice
import io.github.kotlinmania.candle.cudabackend.error.CudaError
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.op.BinaryOpT
import io.github.kotlinmania.candle.op.UnaryOpT
import io.github.kotlinmania.candle.scalar.Scalar
import io.github.kotlinmania.cudarc.cublas.CudaBlas
import io.github.kotlinmania.cudarc.cublas.GemmConfig
import io.github.kotlinmania.cudarc.cublas.StridedBatchedConfig
import io.github.kotlinmania.cudarc.cublas.result.CublasError
import io.github.kotlinmania.cudarc.cublas.result.gemmStridedBatchedEx
import io.github.kotlinmania.cudarc.cublas.sys
import io.github.kotlinmania.cudarc.driver.CudaSlice
import io.github.kotlinmania.cudarc.driver.CudaView
import io.github.kotlinmania.cudarc.driver.CudaViewMut
import io.github.kotlinmania.cudarc.driver.DeviceRepr
import io.github.kotlinmania.cudarc.driver.LaunchArgs
import io.github.kotlinmania.cudarc.driver.LaunchConfig
import io.github.kotlinmania.cudarc.driver.SyncOnDrop
import io.github.kotlinmania.cudarc.driver.ValidAsZeroBits
import io.github.kotlinmania.float8.F8e4m3
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.rustlang.core.mem.AddressableLocal
import io.github.kotlinmania.rustlang.core.mem.ObjectSlotLayout
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.constPtrFromRustCast
import io.github.kotlinmania.rustlang.core.mem.mutPtrFromRustCast
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.result.fromResidual
import io.github.kotlinmania.candle.cudabackend.device.builderArg as barg

public fun <T : WithDType> kernelName(root: String): String {
    val dtype = T.DTYPE.asStr()
    return "${root}_$dtype"
}

public fun <T : DeviceRepr> slicePtr(v: CudaSlice<T>, lo: Int): Pair<ULong, SyncOnDrop> {
    val (_, guard) = v.devicePtr(v.stream())
    val (ptr, _) = v.slice(lo until Int.MAX_VALUE).devicePtr(v.stream())
    return Pair(ptr, guard)
}

public fun <T> sliceSrcAndDst(src: CudaSlice<T>, srcL: Layout, dst: CudaSlice<T>, dstOffset: Int): Pair<CudaView<T>, CudaViewMut<T>> {
    val srcOffset = srcL.startOffset()
    // Copy exactly the number of elements the layout describes. Deriving the
    // count from the storage lengths alone over-copies whenever `src` is a view
    // whose storage continues past it -- anything produced by `narrow` -- and
    // that overruns into the elements sitting after `dstOffset` in `dst`. The
    // two storage bounds are kept as a defensive clamp.
    val toCopy =
        srcL
            .shape()
            .elemCount()
            .min(dst.len().saturatingSub(dstOffset))
            .min(src.len().saturatingSub(srcOffset))
    val src1 = src.slice(srcOffset until srcOffset + toCopy)
    val dst1 = dst.sliceMut(dstOffset until dstOffset + toCopy)
    return Pair(src1, dst1)
}

public fun <T> gemmConfig(alpha: T, beta: T, tupleArg2: List<Any?>, lhsL: Layout, rhsL: Layout): Result<StridedBatchedConfig<T>> {
    val lhsStride = lhsL.stride()
    val rhsStride = rhsL.stride()
    val rhsM1 = rhsStride[rhsStride.len() - 1]
    val rhsM2 = rhsStride[rhsStride.len() - 2]
    val lhsM1 = lhsStride[lhsStride.len() - 1]
    val lhsM2 = lhsStride[lhsStride.len() - 2]
    // The a tensor has dims batching, k, n (rhs)
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    val (lda, transa) =
        if ((rhsM1 == 1 || n == 1) && (rhsM2 == n || k == 1)) {
            Pair(n.toInt(), cublasoperationt.CUBLAS_OP_N)
        } else if ((rhsM1 == k || n == 1) && (rhsM2 == 1 || k == 1)) {
            Pair(k.toInt(), cublasoperationt.CUBLAS_OP_T)
        } else {
            when (val tryResult = Err(CudaError.MatMulNonContiguous(lhsStride = lhsL.clone(), rhsStride = rhsL.clone(), mnk = Triple(m, n, k)))) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
    // The b tensor has dims batching, m, k (lhs)
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    val (ldb, transb) =
        if ((lhsM1 == 1 || k == 1) && (lhsM2 == k || m == 1)) {
            Pair(k.toInt(), cublasoperationt.CUBLAS_OP_N)
        } else if ((lhsM1 == m || k == 1) && (lhsM2 == 1 || m == 1)) {
            Pair(m.toInt(), cublasoperationt.CUBLAS_OP_T)
        } else {
            when (val tryResult = Err(CudaError.MatMulNonContiguous(lhsStride = lhsL.clone(), rhsStride = rhsL.clone(), mnk = Triple(m, n, k)))) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
    // The setup below was copied from:
    // https://github.com/lebedov/scikit-cuda/blob/7e7300474286019c917a6c8a4bca59405c64fbce/tests/testCublas.py#L531
    val gemm = GemmConfig(alpha = alpha, beta = beta, m = n.toInt(), n = m.toInt(), k = k.toInt(), lda = lda, ldb = ldb, ldc = n.toInt(), transa = transa, transb = transb)
    val strideB: Int =
        run {
            val matchVal = ((lhsStride).slice(0 until lhsStride.len() - 2))
            when {
                matchVal.size == 2 && matchVal[0] == matchVal[1] * lhsL.dims()[1] -> {
                    val s1 = matchVal[0]
                    val stride = matchVal[1]
                    stride
                }
                matchVal.size == 2 && lhsL.dims()[0] == 1 -> {
                    val stride1 = matchVal[1]
                    stride1
                }
                matchVal.size == 2 && lhsL.dims()[1] == 1 -> {
                    val stride2 = matchVal[0]
                    stride2
                }
                matchVal.size == 1 -> {
                    val stride3 = matchVal[0]
                    stride3
                }
                matchVal.isEmpty() -> m * k
                else ->
                    when (val tryResult = Err(CudaError.MatMulNonContiguous(lhsStride = lhsL.clone(), rhsStride = rhsL.clone(), mnk = Triple(m, n, k)))) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
            }
        }
    val strideA: Int =
        run {
            val matchVal1 = ((rhsStride).slice(0 until rhsStride.len() - 2))
            when {
                matchVal1.size == 2 && matchVal1[0] == matchVal1[1] * rhsL.dims()[1] -> {
                    val s11 = matchVal1[0]
                    val stride4 = matchVal1[1]
                    stride4
                }
                matchVal1.size == 2 && rhsL.dims()[0] == 1 -> {
                    val stride5 = matchVal1[1]
                    stride5
                }
                matchVal1.size == 2 && rhsL.dims()[1] == 1 -> {
                    val stride6 = matchVal1[0]
                    stride6
                }
                matchVal1.size == 1 -> {
                    val stride7 = matchVal1[0]
                    stride7
                }
                matchVal1.isEmpty() -> n * k
                else ->
                    when (val tryResult = Err(CudaError.MatMulNonContiguous(lhsStride = lhsL.clone(), rhsStride = rhsL.clone(), mnk = Triple(m, n, k)))) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
            }
        }
    return Ok(StridedBatchedConfig(batchSize = b.toInt(), gemm = gemm, strideA = strideA.toLong(), strideB = strideB.toLong(), strideC = (m * n).toLong()))
}

/**
 * This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
 * allowed with f32 GEMMs.
 */
public fun gemmReducedPrecisionF32(): Boolean = MM_F32_REDUCED_PRECISION.load(Ordering.Relaxed)

/**
 * This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
 * allowed with f32 GEMMs.
 */
public fun setGemmReducedPrecisionF32(b: Boolean) {
    MM_F32_REDUCED_PRECISION.store(b, Ordering.Relaxed)
}

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with f16 GEMMs.
 */
public fun gemmReducedPrecisionF16(): Boolean = MM_F16_REDUCED_PRECISION.load(Ordering.Relaxed)

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with f16 GEMMs.
 */
public fun setGemmReducedPrecisionF16(b: Boolean) {
    MM_F16_REDUCED_PRECISION.store(b, Ordering.Relaxed)
}

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with bf16 GEMMs.
 */
public fun gemmReducedPrecisionBf16(): Boolean = MM_BF16_REDUCED_PRECISION.load(Ordering.Relaxed)

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with bf16 GEMMs.
 */
public fun setGemmReducedPrecisionBf16(b: Boolean) {
    MM_BF16_REDUCED_PRECISION.store(b, Ordering.Relaxed)
}

public fun gemmStridedBatchedF32(cublas: CudaBlas, cfg: StridedBatchedConfig<Float>, a: CudaView<Float>, b: CudaView<Float>, c: CudaSlice<Float>): Result<Unit, CublasError> {
    val computeType =
        if (gemmReducedPrecisionF32()) {
            sys.CUBLAS_COMPUTE_32F_FAST_TF32
        } else {
            sys.CUBLAS_COMPUTE_32F
        }
    val alpha = constPtrFromRustCast(cfg.gemm.alpha)
    val beta = constPtrFromRustCast(cfg.gemm.beta)
    val stream = c.stream().clone()
    val (a1, guardA) = a.devicePtr(stream)
    val (b1, guardB) = b.devicePtr(stream)
    val (c1, guardC) = c.devicePtrMut(stream)
    return gemmStridedBatchedEx(cublas.handle(), cfg.gemm.transa, cfg.gemm.transb, cfg.gemm.m, cfg.gemm.n, cfg.gemm.k, alpha, constPtrFromRustCast(a1), sys.CUDA_R_32F, cfg.gemm.lda, cfg.strideA, constPtrFromRustCast(b1), sys.CUDA_R_32F, cfg.gemm.ldb, cfg.strideB, beta, mutPtrFromRustCast(c1), sys.CUDA_R_32F, cfg.gemm.ldc, cfg.strideC, cfg.batchSize, computeType, sys.CUBLAS_GEMM_DEFAULT_TENSOR_OP)
}

public fun gemmStridedBatchedF16(cublas: CudaBlas, cfg: StridedBatchedConfig<F16>, a: CudaView<F16>, b: CudaView<F16>, c: CudaSlice<F16>): Result<Unit, CublasError> {
    val localStorageL2646C4 = AddressableLocal(cfg.gemm.alpha, ObjectSlotLayout<Any?>())
    var alpha by localStorageL2646C4
    val localStorageL2647C4 = AddressableLocal(cfg.gemm.beta, ObjectSlotLayout<Any?>())
    var beta by localStorageL2647C4
    val localStorageL2648C4 = AddressableLocal(cfg.gemm.alpha.toF32(), PrimitiveLayouts.F32)
    var alphaF32 by localStorageL2648C4
    val localStorageL2649C4 = AddressableLocal(cfg.gemm.beta.toF32(), PrimitiveLayouts.F32)
    var betaF32 by localStorageL2649C4
    val (computeType, alpha1, beta1) =
        if (gemmReducedPrecisionF16()) {
            Triple(sys.CUBLAS_COMPUTE_16F, ConstPtr(localStorageL2646C4.asMutPtr().rawAddress), ConstPtr(localStorageL2647C4.asMutPtr().rawAddress))
        } else {
            Triple(sys.CUBLAS_COMPUTE_32F, ConstPtr(localStorageL2648C4.asMutPtr().rawAddress), ConstPtr(localStorageL2649C4.asMutPtr().rawAddress))
        }
    val stream = c.stream().clone()
    val (a1, guardA) = a.devicePtr(stream)
    val (b1, guardB) = b.devicePtr(stream)
    val (c1, guardC) = c.devicePtrMut(stream)
    return gemmStridedBatchedEx(cublas.handle(), cfg.gemm.transa, cfg.gemm.transb, cfg.gemm.m, cfg.gemm.n, cfg.gemm.k, alpha1, constPtrFromRustCast(a1), sys.CUDA_R_16F, cfg.gemm.lda, cfg.strideA, constPtrFromRustCast(b1), sys.CUDA_R_16F, cfg.gemm.ldb, cfg.strideB, beta1, mutPtrFromRustCast(c1), sys.CUDA_R_16F, cfg.gemm.ldc, cfg.strideC, cfg.batchSize, computeType, sys.CUBLAS_GEMM_DEFAULT_TENSOR_OP)
}

public fun gemmStridedBatchedBf16(cublas: CudaBlas, cfg: StridedBatchedConfig<Bf16>, a: CudaView<Bf16>, b: CudaView<Bf16>, c: CudaSlice<Bf16>): Result<Unit, CublasError> {
    val localStorageL2705C4 = AddressableLocal(cfg.gemm.alpha.toF32(), PrimitiveLayouts.F32)
    var alphaF32 by localStorageL2705C4
    val localStorageL2706C4 = AddressableLocal(cfg.gemm.beta.toF32(), PrimitiveLayouts.F32)
    var betaF32 by localStorageL2706C4
    // The type for alpha and beta depends on the computeType.
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex
    val (computeType, alpha, beta) =
        if (gemmReducedPrecisionBf16()) {
            Triple(sys.CUBLAS_COMPUTE_32F_FAST_16BF, ConstPtr(localStorageL2705C4.asMutPtr().rawAddress), ConstPtr(localStorageL2706C4.asMutPtr().rawAddress))
        } else {
            Triple(sys.CUBLAS_COMPUTE_32F, ConstPtr(localStorageL2705C4.asMutPtr().rawAddress), ConstPtr(localStorageL2706C4.asMutPtr().rawAddress))
        }
    val stream = c.stream().clone()
    val (a1, guardA) = a.devicePtr(stream)
    val (b1, guardB) = b.devicePtr(stream)
    val (c1, guardC) = c.devicePtrMut(stream)
    return gemmStridedBatchedEx(cublas.handle(), cfg.gemm.transa, cfg.gemm.transb, cfg.gemm.m, cfg.gemm.n, cfg.gemm.k, alpha, constPtrFromRustCast(a1), sys.CUDA_R_16BF, cfg.gemm.lda, cfg.strideA, constPtrFromRustCast(b1), sys.CUDA_R_16BF, cfg.gemm.ldb, cfg.strideB, beta, mutPtrFromRustCast(c1), sys.CUDA_R_16BF, cfg.gemm.ldc, cfg.strideC, cfg.batchSize, computeType, sys.CUBLAS_GEMM_DEFAULT_TENSOR_OP)
}

public fun Scalar.builderArg(builder: LaunchArgs) {
    when (this) {
        is Scalar.U8 -> {
            val (v) = this
            builder.arg(v)
        }
        is Scalar.U32 -> {
            val (v1) = this
            builder.arg(v1)
        }
        is Scalar.I16 -> {
            val (v2) = this
            builder.arg(v2)
        }
        is Scalar.I32 -> {
            val (v3) = this
            builder.arg(v3)
        }
        is Scalar.I64 -> {
            val (v4) = this
            builder.arg(v4)
        }
        is Scalar.F32 -> {
            val (v5) = this
            builder.arg(v5)
        }
        is Scalar.F64 -> {
            val (v6) = this
            builder.arg(v6)
        }
        is Scalar.F16 -> {
            val (v7) = this
            builder.arg(v7)
        }
        is Scalar.Bf16 -> {
            val (v8) = this
            builder.arg(v8)
        }
        is Scalar.F8e4m3 -> {
            val (v9) = this
            builder.arg(v9)
        }
    }
}

public fun <U : UnaryOpT, T> U.f(src: CudaSlice<T>, dev: CudaDevice, layout: Layout): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
    val shape = layout.shape()
    val dims = shape.dims()
    val elCount = shape.elemCount()
    val cfg = LaunchConfig.forNumElems(elCount.toUInt())
    val ds =
        when (val tryResult = SlicePtrOrNull.paramsFromLayout(dev, layout)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val src1 = src.slice(layout.startOffset() until Int.MAX_VALUE)
    val func =
        when (val tryResult = dev.getOrLoadFunc(kernelName<T>(U.KERNEL), UNARY)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    // SAFETY: Set later by running the kernel.
    val out =
        when (val tryResult = dev.alloc<T>(elCount)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val builder = func.builder()
    barg(builder, elCount)
    barg(builder, dims.len())
    ds.builderArg(builder)
    builder.arg(src1)
    builder.arg(out)
    // SAFETY: ffi.
    when (val tryResult = builder.launch(cfg).w()) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    return Ok(out)
}

public fun <U : BinaryOpT, T> U.f(lhs: CudaSlice<T>, lhsL: Layout, rhs: CudaSlice<T>, rhsL: Layout, dev: CudaDevice): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
    val shape = lhsL.shape()
    val dims = shape.dims()
    val elemCount = shape.elemCount()
    val cfg = LaunchConfig.forNumElems(elemCount.toUInt())
    val dimsAndStrides =
        if (lhsL.isContiguous() && rhsL.isContiguous()) {
            SlicePtrOrNull.Null
        } else {
            when (val tryResult = SlicePtrOrNull.paramsFromVec(dev, listOf(dims, lhsL.stride(), rhsL.stride()).concat())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
    val lhs1 = lhs.slice(lhsL.startOffset() until Int.MAX_VALUE)
    val rhs1 = rhs.slice(rhsL.startOffset() until Int.MAX_VALUE)
    val func =
        when (val tryResult = dev.getOrLoadFunc(kernelName<T>(U.KERNEL), BINARY)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    // SAFETY: Set later by running the kernel.
    val out =
        when (val tryResult = dev.alloc<T>(elemCount)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val builder = func.builder()
    barg(builder, elemCount)
    barg(builder, dims.len())
    dimsAndStrides.builderArg(builder)
    builder.arg(lhs1)
    builder.arg(rhs1)
    builder.arg(out)
    // SAFETY: ffi
    when (val tryResult = builder.launch(cfg).w()) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    return Ok(out)
}

public fun cudaDTypeUByte(s: CudaStorage): Result<CudaSlice<UByte>> =
    when (val matchVal = s.slice) {
        is CudaStorageSlice.U8 -> {
            val (data) = matchVal
            Ok(data)
        }
        else -> Err(Error.UnexpectedDType(expected = DType.U8, got = s.dtype(), msg = "unexpected dtype").bt())
    }

public fun cudaDTypeUByte(slice: CudaSlice<UByte>, device: CudaDevice): CudaStorage {
    val slice1 = CudaStorageSlice.u8(slice)
    return CudaStorage(slice = slice1, device = device)
}

public fun cudaDTypeUInt(s: CudaStorage): Result<CudaSlice<UInt>> =
    when (val matchVal = s.slice) {
        is CudaStorageSlice.U32 -> {
            val (data) = matchVal
            Ok(data)
        }
        else -> Err(Error.UnexpectedDType(expected = DType.U32, got = s.dtype(), msg = "unexpected dtype").bt())
    }

public fun cudaDTypeUInt(slice: CudaSlice<UInt>, device: CudaDevice): CudaStorage {
    val slice1 = CudaStorageSlice.u32(slice)
    return CudaStorage(slice = slice1, device = device)
}

public fun cudaDTypeShort(s: CudaStorage): Result<CudaSlice<Short>> =
    when (val matchVal = s.slice) {
        is CudaStorageSlice.I16 -> {
            val (data) = matchVal
            Ok(data)
        }
        else -> Err(Error.UnexpectedDType(expected = DType.I16, got = s.dtype(), msg = "unexpected dtype").bt())
    }

public fun cudaDTypeShort(slice: CudaSlice<Short>, device: CudaDevice): CudaStorage {
    val slice1 = CudaStorageSlice.i16(slice)
    return CudaStorage(slice = slice1, device = device)
}

public fun cudaDTypeInt(s: CudaStorage): Result<CudaSlice<Int>> =
    when (val matchVal = s.slice) {
        is CudaStorageSlice.I32 -> {
            val (data) = matchVal
            Ok(data)
        }
        else -> Err(Error.UnexpectedDType(expected = DType.I32, got = s.dtype(), msg = "unexpected dtype").bt())
    }

public fun cudaDTypeInt(slice: CudaSlice<Int>, device: CudaDevice): CudaStorage {
    val slice1 = CudaStorageSlice.i32(slice)
    return CudaStorage(slice = slice1, device = device)
}

public fun cudaDTypeLong(s: CudaStorage): Result<CudaSlice<Long>> =
    when (val matchVal = s.slice) {
        is CudaStorageSlice.I64 -> {
            val (data) = matchVal
            Ok(data)
        }
        else -> Err(Error.UnexpectedDType(expected = DType.I64, got = s.dtype(), msg = "unexpected dtype").bt())
    }

public fun cudaDTypeLong(slice: CudaSlice<Long>, device: CudaDevice): CudaStorage {
    val slice1 = CudaStorageSlice.i64(slice)
    return CudaStorage(slice = slice1, device = device)
}

public fun cudaDTypeF16(s: CudaStorage): Result<CudaSlice<F16>> =
    when (val matchVal = s.slice) {
        is CudaStorageSlice.F16 -> {
            val (data) = matchVal
            Ok(data)
        }
        else -> Err(Error.UnexpectedDType(expected = DType.F16, got = s.dtype(), msg = "unexpected dtype").bt())
    }

public fun cudaDTypeF16(slice: CudaSlice<F16>, device: CudaDevice): CudaStorage {
    val slice1 = CudaStorageSlice.f16(slice)
    return CudaStorage(slice = slice1, device = device)
}

public fun cudaDTypeBf16(s: CudaStorage): Result<CudaSlice<Bf16>> =
    when (val matchVal = s.slice) {
        is CudaStorageSlice.Bf16 -> {
            val (data) = matchVal
            Ok(data)
        }
        else -> Err(Error.UnexpectedDType(expected = DType.BF16, got = s.dtype(), msg = "unexpected dtype").bt())
    }

public fun cudaDTypeBf16(slice: CudaSlice<Bf16>, device: CudaDevice): CudaStorage {
    val slice1 = CudaStorageSlice.bf16(slice)
    return CudaStorage(slice = slice1, device = device)
}

public fun cudaDTypeFloat(s: CudaStorage): Result<CudaSlice<Float>> =
    when (val matchVal = s.slice) {
        is CudaStorageSlice.F32 -> {
            val (data) = matchVal
            Ok(data)
        }
        else -> Err(Error.UnexpectedDType(expected = DType.F32, got = s.dtype(), msg = "unexpected dtype").bt())
    }

public fun cudaDTypeFloat(slice: CudaSlice<Float>, device: CudaDevice): CudaStorage {
    val slice1 = CudaStorageSlice.f32(slice)
    return CudaStorage(slice = slice1, device = device)
}

public fun cudaDTypeDouble(s: CudaStorage): Result<CudaSlice<Double>> =
    when (val matchVal = s.slice) {
        is CudaStorageSlice.F64 -> {
            val (data) = matchVal
            Ok(data)
        }
        else -> Err(Error.UnexpectedDType(expected = DType.F64, got = s.dtype(), msg = "unexpected dtype").bt())
    }

public fun cudaDTypeDouble(slice: CudaSlice<Double>, device: CudaDevice): CudaStorage {
    val slice1 = CudaStorageSlice.f64(slice)
    return CudaStorage(slice = slice1, device = device)
}

public fun cudaDTypeF8e4m3(s: CudaStorage): Result<CudaSlice<F8e4m3>> =
    when (val matchVal = s.slice) {
        is CudaStorageSlice.F8e4m3 -> {
            val (data) = matchVal
            Ok(data)
        }
        else -> Err(Error.UnexpectedDType(expected = DType.F8E4M3, got = s.dtype(), msg = "unexpected dtype").bt())
    }

public fun cudaDTypeF8e4m3(slice: CudaSlice<F8e4m3>, device: CudaDevice): CudaStorage {
    val slice1 = CudaStorageSlice.f8e4m3(slice)
    return CudaStorage(slice = slice1, device = device)
}
