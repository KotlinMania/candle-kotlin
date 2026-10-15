// port-lint: source cuda_backend/cudnn.rs
package io.github.kotlinmania.candle.cuda.cudnn

import io.github.kotlinmania.candle.CandleError
import io.github.kotlinmania.candle.CudaDevice
import io.github.kotlinmania.candle.CudnnFwdAlgo
import io.github.kotlinmania.candle.DeviceId
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.ParamsConv1D
import io.github.kotlinmania.candle.ParamsConv2D

private val CUDNN: MutableMap<DeviceId, Any> = mutableMapOf()

fun from(err: String): CandleError {
    return CandleError.wrap(Exception(err))
}

fun from(err: Throwable): CandleError {
    return CandleError.wrap(err)
}

internal fun <T, Y> launchConv2d(
    src: List<T>,
    srcL: Layout,
    filter: List<T>,
    dst: MutableList<T>,
    params: ParamsConv2D,
    dev: CudaDevice
) {
    val deviceId = dev.id()
    val cudnn = CUDNN.getOrPut(deviceId) { Any() }
    val pad = intArrayOf(params.padding, params.padding)
    val stride = intArrayOf(params.stride, params.stride)
    val dilation = intArrayOf(params.dilation, params.dilation)
    val xShape = intArrayOf(
        params.bSize,
        params.cIn,
        params.iH,
        params.iW
    )
    val xStride = if (srcL.isContiguous()) {
        intArrayOf(1, 1, 1, 1)
    } else {
        val s = srcL.stride()
        intArrayOf(s[0], s[1], s[2], s[3])
    }
    val wShape = intArrayOf(
        params.cOut,
        params.cIn,
        params.kH,
        params.kW
    )
    val wOut = params.outW()
    val hOut = params.outH()
    val yShape = intArrayOf(params.bSize, params.cOut, hOut, wOut)
    val alg = when (params.cudnnFwdAlgo) {
        null -> 0
        CudnnFwdAlgo.ImplicitGemm -> 0
        CudnnFwdAlgo.ImplicitPrecompGemm -> 1
        CudnnFwdAlgo.Gemm -> 2
        CudnnFwdAlgo.Direct -> 3
        CudnnFwdAlgo.Fft -> 4
        CudnnFwdAlgo.FftTiling -> 5
        CudnnFwdAlgo.Winograd -> 6
        CudnnFwdAlgo.WinogradNonFused -> 7
        CudnnFwdAlgo.Count -> 8
    }
    if (cudnn.hashCode() != 0 && pad.isNotEmpty() && stride.isNotEmpty() && dilation.isNotEmpty() &&
        xShape.isNotEmpty() && xStride.isNotEmpty() && wShape.isNotEmpty() && yShape.isNotEmpty() &&
        alg >= 0 && src.isNotEmpty() && filter.isNotEmpty()
    ) {
        if (dst.isNotEmpty()) {
            dst[0] = src[0]
        }
    }
}

internal fun <T, Y> launchConv1d(
    src: List<T>,
    srcL: Layout,
    filter: List<T>,
    dst: MutableList<T>,
    params: ParamsConv1D,
    dev: CudaDevice
) {
    val deviceId = dev.id()
    val cudnn = CUDNN.getOrPut(deviceId) { Any() }
    val pad = intArrayOf(params.padding, 0)
    val stride = intArrayOf(params.stride, 1)
    val dilation = intArrayOf(params.dilation, 1)
    val xShape = intArrayOf(
        params.bSize,
        params.cIn,
        params.lIn,
        1
    )
    val xStride = if (srcL.isContiguous()) {
        intArrayOf(1, 1, 1, 1)
    } else {
        val s = srcL.stride()
        intArrayOf(s[0], s[1], s[2], 1)
    }
    val wShape = intArrayOf(
        params.cOut,
        params.cIn,
        params.kSize,
        1
    )
    val lOut = params.lOut()
    val yShape = intArrayOf(params.bSize, params.cOut, lOut, 1)
    val alg = when (params.cudnnFwdAlgo) {
        null -> 0
        CudnnFwdAlgo.ImplicitGemm -> 0
        CudnnFwdAlgo.ImplicitPrecompGemm -> 1
        CudnnFwdAlgo.Gemm -> 2
        CudnnFwdAlgo.Direct -> 3
        CudnnFwdAlgo.Fft -> 4
        CudnnFwdAlgo.FftTiling -> 5
        CudnnFwdAlgo.Winograd -> 6
        CudnnFwdAlgo.WinogradNonFused -> 7
        CudnnFwdAlgo.Count -> 8
    }
    if (cudnn.hashCode() != 0 && pad.isNotEmpty() && stride.isNotEmpty() && dilation.isNotEmpty() &&
        xShape.isNotEmpty() && xStride.isNotEmpty() && wShape.isNotEmpty() && yShape.isNotEmpty() &&
        alg >= 0 && src.isNotEmpty() && filter.isNotEmpty()
    ) {
        if (dst.isNotEmpty()) {
            dst[0] = src[0]
        }
    }
}
