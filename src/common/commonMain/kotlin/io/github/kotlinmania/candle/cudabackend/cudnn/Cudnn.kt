// port-lint: source cuda_backend/cudnn.rs
package io.github.kotlinmania.candle.cudabackend.cudnn

import io.github.kotlinmania.candle.ParamsConv1D
import io.github.kotlinmania.candle.ParamsConv2D
import io.github.kotlinmania.candle.cudabackend.CudaDevice
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.cudarc.cudnn.safe.ConvForward
import io.github.kotlinmania.cudarc.cudnn.safe.Cudnn
import io.github.kotlinmania.cudarc.driver.CudaSlice
import io.github.kotlinmania.cudarc.driver.CudaView
import io.github.kotlinmania.cudarc.driver.DeviceRepr
import io.github.kotlinmania.cudarc.driver.ValidAsZeroBits
import io.github.kotlinmania.candle.CudnnFwdAlgo as CandleAlgo
import io.github.kotlinmania.cudarc.cudnn.sys.cudnnConvolutionFwdAlgoT as A

public fun fromError(err: cudarc.CudnnError): Error = Error.wrap(err)

public fun fromError(err: cudarc.DriverError): Error = Error.wrap(err)

internal fun <T, Y : cudarc.CudnnDataType> launchConv2d(src: CudaView<T>, srcL: Layout, filter: CudaView<T>, dst: CudaSlice<T>, params: ParamsConv2D, dev: CudaDevice, oneTFn: (Any?) -> Any?, zeroTFn: (Any?) -> Any?): Result<Unit> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits, T : cudarc.CudnnDataType {
    val deviceId = dev.id()
    val cudnn =
        CUDNN.with(closure@{ cudnn ->
            when (val matchVal = cudnn.borrow().get(deviceId)) {
                null -> {}
                else -> {
                    val cudnn1 = matchVal
                    return@closure Result.success(cudnn1.clone())
                }
            }
            val c = Cudnn.new(dev.cudaStream())
            if (c is Ok) {
                val (c1) = c
                cudnn.borrowMut().insert(deviceId, c1.clone())
            }
            c
        }).getOrElse { return Result.failure(it) }
    val conv = cudnn.createConv2d<Y>(listOf(params.padding, params.padding), listOf(params.stride, params.stride), listOf(params.dilation, params.dilation), cudarc.CUDNN_CROSS_CORRELATION).getOrElse { return Result.failure(it) }
    val xShape = listOf(params.bSize, params.cIn, params.iH, params.iW)
    // Note that `src` already starts at the proper offset.
    val x =
        if (srcL.isContiguous()) {
            cudnn.create4dTensor<T>(cudarc.CUDNN_TENSOR_NCHW, xShape).getOrElse { return Result.failure(it) }
        } else {
            val s = srcL.stride()
            cudnn.create4dTensorEx<T>(xShape, listOf(s[0].toInt(), s[1].toInt(), s[2].toInt(), s[3].toInt())).getOrElse { return Result.failure(it) }
        }
    val w = cudnn.create4dFilter<T>(cudarc.CUDNN_TENSOR_NCHW, listOf(params.cOut, params.cIn, params.kH, params.kW)).getOrElse { return Result.failure(it) }
    val (wOut, hOut) = Pair(params.outW(), params.outH())
    val y = cudnn.create4dTensor<T>(cudarc.CUDNN_TENSOR_NCHW, listOf(params.bSize, params.cOut, hOut, wOut)).getOrElse { return Result.failure(it) }
    val conv2d = ConvForward(conv = conv, x = x, w = w, y = y)
    val alg =
        when (val matchVal = params.cudnnFwdAlgo) {
            null -> conv2d.pickAlgorithm().getOrElse { return Result.failure(it) }
            CandleAlgo.ImplicitGemm -> A.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
            CandleAlgo.ImplicitPrecompGemm -> {
                A.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
            }
            CandleAlgo.Gemm -> A.CUDNN_CONVOLUTION_FWD_ALGO_GEMM
            CandleAlgo.Direct -> A.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
            CandleAlgo.Fft -> A.CUDNN_CONVOLUTION_FWD_ALGO_FFT
            CandleAlgo.FftTiling -> A.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
            CandleAlgo.Winograd -> A.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
            CandleAlgo.WinogradNonFused -> A.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
            CandleAlgo.Count -> A.CUDNN_CONVOLUTION_FWD_ALGO_COUNT
        }
    val workspaceSize = conv2d.getWorkspaceSize(alg).getOrElse { return Result.failure(it) }
    val workspace = dev.cudaStream().allocZeros<UByte>(workspaceSize).getOrElse { return Result.failure(it) }
    run {
        conv2d.launch<CudaSlice<UByte>>(alg, workspace, Pair(oneTFn(), zeroTFn()), src, filter, dst).getOrElse { return Result.failure(it) }
    }
    return Result.success(Unit)
}

internal fun <T, Y : cudarc.CudnnDataType> launchConv1d(src: CudaView<T>, srcL: Layout, filter: CudaView<T>, dst: CudaSlice<T>, params: ParamsConv1D, dev: CudaDevice, oneTFn: (Any?) -> Any?, zeroTFn: (Any?) -> Any?): Result<Unit> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits, T : cudarc.CudnnDataType {
    val deviceId = dev.id()
    val cudnn =
        CUDNN.with(closure1@{ cudnn ->
            when (val matchVal = cudnn.borrow().get(deviceId)) {
                null -> {}
                else -> {
                    val cudnn1 = matchVal
                    return@closure1 Result.success(cudnn1.clone())
                }
            }
            val c = Cudnn.new(dev.cudaStream())
            if (c is Ok) {
                val (c1) = c
                cudnn.borrowMut().insert(deviceId, c1.clone())
            }
            c
        }).getOrElse { return Result.failure(it) }
    val conv = cudnn.createConv2d<Y>(listOf(params.padding, 0), listOf(params.stride, 1), listOf(params.dilation, 1), cudarc.CUDNN_CROSS_CORRELATION).getOrElse { return Result.failure(it) }
    // https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html#cudnnsettensornddescriptor
    // > Tensors are restricted to having at least 4 dimensions, and at most CUDNN_DIM_MAX
    // > dimensions (defined in cudnn.h). When working with lower dimensional data, it is
    // > recommended that the user create a 4D tensor, and set the size along unused dimensions
    // > to 1.
    val xShape = listOf(params.bSize, params.cIn, params.lIn, 1)
    // Note that `src` already starts at the proper offset.
    val x =
        if (srcL.isContiguous()) {
            cudnn.create4dTensor<T>(cudarc.CUDNN_TENSOR_NCHW, xShape).getOrElse { return Result.failure(it) }
        } else {
            val s = srcL.stride()
            cudnn.create4dTensorEx<T>(xShape, listOf(s[0].toInt(), s[1].toInt(), s[2].toInt(), 1)).getOrElse { return Result.failure(it) }
        }
    val w = cudnn.create4dFilter<T>(cudarc.CUDNN_TENSOR_NCHW, listOf(params.cOut, params.cIn, params.kSize, 1)).getOrElse { return Result.failure(it) }
    val lOut = params.lOut()
    val y = cudnn.create4dTensor<T>(cudarc.CUDNN_TENSOR_NCHW, listOf(params.bSize, params.cOut, lOut, 1)).getOrElse { return Result.failure(it) }
    val conv1d = ConvForward(conv = conv, x = x, w = w, y = y)
    val alg =
        when (val matchVal = params.cudnnFwdAlgo) {
            null -> conv1d.pickAlgorithm().getOrElse { return Result.failure(it) }
            CandleAlgo.ImplicitGemm -> A.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
            CandleAlgo.ImplicitPrecompGemm -> {
                A.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
            }
            CandleAlgo.Gemm -> A.CUDNN_CONVOLUTION_FWD_ALGO_GEMM
            CandleAlgo.Direct -> A.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
            CandleAlgo.Fft -> A.CUDNN_CONVOLUTION_FWD_ALGO_FFT
            CandleAlgo.FftTiling -> A.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
            CandleAlgo.Winograd -> A.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
            CandleAlgo.WinogradNonFused -> A.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
            CandleAlgo.Count -> A.CUDNN_CONVOLUTION_FWD_ALGO_COUNT
        }
    val workspaceSize = conv1d.getWorkspaceSize(alg).getOrElse { return Result.failure(it) }
    val workspace = dev.cudaStream().allocZeros<UByte>(workspaceSize).getOrElse { return Result.failure(it) }
    run {
        conv1d.launch<CudaSlice<UByte>>(alg, workspace, Pair(oneTFn(), zeroTFn()), src, filter, dst).getOrElse { return Result.failure(it) }
    }
    return Result.success(Unit)
}
