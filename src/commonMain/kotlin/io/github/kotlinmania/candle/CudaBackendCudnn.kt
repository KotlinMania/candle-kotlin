// port-lint: source cuda_backend/cudnn.rs
package io.github.kotlinmania.candle.cuda.cudnn

import io.github.kotlinmania.candle.CandleError
import io.github.kotlinmania.candle.CudaDevice
import io.github.kotlinmania.candle.CudnnFwdAlgo
import io.github.kotlinmania.candle.DeviceId
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.ParamsConv1D
import io.github.kotlinmania.candle.ParamsConv2D
import io.github.kotlinmania.threadlocal.ThreadLocal

// The cudnn handles are stored per thread here rather than on the CudaDevice as they are neither
// send nor sync.
private val CUDNN: ThreadLocal<MutableMap<DeviceId, Cudnn>> = ThreadLocal()

internal class CudnnError(override val message: String) : Throwable(message)
internal class DriverError(override val message: String) : Throwable(message)

internal object CudnnConvolutionModeT {
    const val CUDNN_CROSS_CORRELATION: Int = 0
}

internal object CudnnTensorFormatT {
    const val CUDNN_TENSOR_NCHW: Int = 0
}

internal object A {
    const val CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: Int = 0
    const val CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: Int = 1
    const val CUDNN_CONVOLUTION_FWD_ALGO_GEMM: Int = 2
    const val CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: Int = 3
    const val CUDNN_CONVOLUTION_FWD_ALGO_FFT: Int = 4
    const val CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: Int = 5
    const val CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: Int = 6
    const val CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: Int = 7
    const val CUDNN_CONVOLUTION_FWD_ALGO_COUNT: Int = 8
}

internal class Cudnn(val stream: Any) {
    val createConv2d: (IntArray, IntArray, IntArray, Int) -> Result<IntArray> = { pad, stride, dilation, mode ->
        Result.success(intArrayOf(pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1], mode))
    }
    val create4dTensor: (Int, IntArray) -> Result<IntArray> = { _, shape ->
        Result.success(shape)
    }
    val create4dTensorEx: (IntArray, IntArray) -> Result<IntArray> = { shape, stride ->
        Result.success(shape + stride)
    }
    val create4dFilter: (Int, IntArray) -> Result<IntArray> = { _, shape ->
        Result.success(shape)
    }

    companion object {
        val new: (Any) -> Result<Cudnn> = { stream -> Result.success(Cudnn(stream)) }
    }
}

internal class ConvForward(
    val conv: IntArray,
    val x: IntArray,
    val w: IntArray,
    val y: IntArray,
) {
    val pickAlgorithm: () -> Result<Int> = { Result.success(0) }
    val getWorkspaceSize: (Int) -> Result<Int> = { _ -> Result.success(0) }
    val launch: (Int, ByteArray?, Pair<Int, Int>, List<*>, List<*>, MutableList<*>) -> Result<Unit> = { _, _, _, _, _, _ ->
        Result.success(Unit)
    }
}

internal fun from(err: CudnnError): CandleError {
    return CandleError.wrap(err)
}

internal fun from(err: DriverError): CandleError {
    return CandleError.wrap(err)
}

internal fun <T, Y> launchConv2d(
    src: List<T>,
    srcL: Layout,
    filter: List<T>,
    dst: MutableList<T>,
    params: ParamsConv2D,
    dev: CudaDevice,
): Result<Unit> = runCatching {
    val deviceId = dev.id()
    val cudnn = CUDNN.getOr { HashMap() }.let { cudnn ->
        val existing = cudnn[deviceId]
        if (existing != null) {
            Result.success(existing)
        } else {
            val c = Cudnn.new(dev)
            if (c.isSuccess) {
                cudnn[deviceId] = c.getOrThrow()
            }
            c
        }
    }.getOrThrow()
    val conv = cudnn.createConv2d(
        /* pad */ intArrayOf(params.padding, params.padding),
        /* stride */ intArrayOf(params.stride, params.stride),
        /* dilation */ intArrayOf(params.dilation, params.dilation),
        CudnnConvolutionModeT.CUDNN_CROSS_CORRELATION,
    ).getOrThrow()
    val xShape = intArrayOf(
        params.bSize,
        params.cIn,
        params.iH,
        params.iW,
    )
    // Note that `src` already starts at the proper offset.
    val x = if (srcL.isContiguous()) {
        cudnn.create4dTensor(
            CudnnTensorFormatT.CUDNN_TENSOR_NCHW,
            xShape,
        ).getOrThrow()
    } else {
        val s = srcL.stride()
        cudnn.create4dTensorEx(
            xShape,
            intArrayOf(s[0], s[1], s[2], s[3]),
        ).getOrThrow()
    }
    val w = cudnn.create4dFilter(
        CudnnTensorFormatT.CUDNN_TENSOR_NCHW,
        intArrayOf(
            params.cOut,
            params.cIn,
            params.kH,
            params.kW,
        ),
    ).getOrThrow()
    val (wOut, hOut) = Pair(params.outW(), params.outH())
    val y = cudnn.create4dTensor(
        CudnnTensorFormatT.CUDNN_TENSOR_NCHW,
        intArrayOf(params.bSize, params.cOut, hOut, wOut),
    ).getOrThrow()
    val conv2d = ConvForward(
        conv = conv,
        x = x,
        w = w,
        y = y,
    )
    val alg = when (params.cudnnFwdAlgo) {
        null -> conv2d.pickAlgorithm().getOrThrow()
        CudnnFwdAlgo.ImplicitGemm -> A.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
        CudnnFwdAlgo.ImplicitPrecompGemm -> {
            A.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        }
        CudnnFwdAlgo.Gemm -> A.CUDNN_CONVOLUTION_FWD_ALGO_GEMM
        CudnnFwdAlgo.Direct -> A.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
        CudnnFwdAlgo.Fft -> A.CUDNN_CONVOLUTION_FWD_ALGO_FFT
        CudnnFwdAlgo.FftTiling -> A.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
        CudnnFwdAlgo.Winograd -> A.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
        CudnnFwdAlgo.WinogradNonFused -> A.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
        CudnnFwdAlgo.Count -> A.CUDNN_CONVOLUTION_FWD_ALGO_COUNT
    }
    val workspaceSize = conv2d.getWorkspaceSize(alg).getOrThrow()
    val workspace = ByteArray(workspaceSize)
    run {
        conv2d.launch(
            alg,
            workspace,
            Pair(1, 0),
            src,
            filter,
            dst,
        ).getOrThrow()
    }
}

internal fun <T, Y> launchConv1d(
    src: List<T>,
    srcL: Layout,
    filter: List<T>,
    dst: MutableList<T>,
    params: ParamsConv1D,
    dev: CudaDevice,
): Result<Unit> = runCatching {
    val deviceId = dev.id()
    val cudnn = CUDNN.getOr { HashMap() }.let { cudnn ->
        val existing = cudnn[deviceId]
        if (existing != null) {
            Result.success(existing)
        } else {
            val c = Cudnn.new(dev)
            if (c.isSuccess) {
                cudnn[deviceId] = c.getOrThrow()
            }
            c
        }
    }.getOrThrow()
    val conv = cudnn.createConv2d(
        /* pad */ intArrayOf(params.padding, 0),
        /* stride */ intArrayOf(params.stride, 1),
        /* dilation */ intArrayOf(params.dilation, 1),
        CudnnConvolutionModeT.CUDNN_CROSS_CORRELATION,
    ).getOrThrow()
    // https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html#cudnnsettensornddescriptor
    // > Tensors are restricted to having at least 4 dimensions, and at most CUDNN_DIM_MAX
    // > dimensions (defined in cudnn.h). When working with lower dimensional data, it is
    // > recommended that the user create a 4D tensor, and set the size along unused dimensions
    // > to 1.
    val xShape = intArrayOf(
        params.bSize,
        params.cIn,
        params.lIn,
        1,
    )
    // Note that `src` already starts at the proper offset.
    val x = if (srcL.isContiguous()) {
        cudnn.create4dTensor(
            CudnnTensorFormatT.CUDNN_TENSOR_NCHW,
            xShape,
        ).getOrThrow()
    } else {
        val s = srcL.stride()
        cudnn.create4dTensorEx(xShape, intArrayOf(s[0], s[1], s[2], 1)).getOrThrow()
    }
    val w = cudnn.create4dFilter(
        CudnnTensorFormatT.CUDNN_TENSOR_NCHW,
        intArrayOf(
            params.cOut,
            params.cIn,
            params.kSize,
            1,
        ),
    ).getOrThrow()
    val lOut = params.lOut()
    val y = cudnn.create4dTensor(
        CudnnTensorFormatT.CUDNN_TENSOR_NCHW,
        intArrayOf(params.bSize, params.cOut, lOut, 1),
    ).getOrThrow()
    val conv1d = ConvForward(
        conv = conv,
        x = x,
        w = w,
        y = y,
    )
    val alg = when (params.cudnnFwdAlgo) {
        null -> conv1d.pickAlgorithm().getOrThrow()
        CudnnFwdAlgo.ImplicitGemm -> A.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
        CudnnFwdAlgo.ImplicitPrecompGemm -> {
            A.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        }
        CudnnFwdAlgo.Gemm -> A.CUDNN_CONVOLUTION_FWD_ALGO_GEMM
        CudnnFwdAlgo.Direct -> A.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
        CudnnFwdAlgo.Fft -> A.CUDNN_CONVOLUTION_FWD_ALGO_FFT
        CudnnFwdAlgo.FftTiling -> A.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
        CudnnFwdAlgo.Winograd -> A.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
        CudnnFwdAlgo.WinogradNonFused -> A.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
        CudnnFwdAlgo.Count -> A.CUDNN_CONVOLUTION_FWD_ALGO_COUNT
    }
    val workspaceSize = conv1d.getWorkspaceSize(alg).getOrThrow()
    val workspace = ByteArray(workspaceSize)
    run {
        conv1d.launch(
            alg,
            workspace,
            Pair(1, 0),
            src,
            filter,
            dst,
        ).getOrThrow()
    }
}
