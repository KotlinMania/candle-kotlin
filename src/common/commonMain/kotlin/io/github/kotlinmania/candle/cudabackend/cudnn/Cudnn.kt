// port-lint: source cuda_backend/cudnn.rs
package io.github.kotlinmania.candle.cudabackend.cudnn

import io.github.kotlinmania.candle.conv.ParamsConv1D
import io.github.kotlinmania.candle.conv.ParamsConv2D
import io.github.kotlinmania.candle.cudabackend.device.CudaDevice
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.cudarc.cudnn.safe.ConvForward
import io.github.kotlinmania.cudarc.cudnn.safe.Cudnn
import io.github.kotlinmania.cudarc.driver.CudaSlice
import io.github.kotlinmania.cudarc.driver.CudaView
import io.github.kotlinmania.cudarc.driver.DeviceRepr
import io.github.kotlinmania.cudarc.driver.ValidAsZeroBits
import io.github.kotlinmania.rustlang.core.result.Result
import io.github.kotlinmania.rustlang.core.result.fromResidual
import io.github.kotlinmania.candle.conv.CudnnFwdAlgo as CandleAlgo
import io.github.kotlinmania.cudarc.cudnn.sys.cudnnConvolutionFwdAlgoT as A

public fun fromError(err: cudarc.CudnnError): Error = Error.wrap(err)

public fun fromError(err: cudarc.DriverError): Error = Error.wrap(err)

internal fun <T, Y : cudarc.CudnnDataType> launchConv2d(src: CudaView<T>, srcL: Layout, filter: CudaView<T>, dst: CudaSlice<T>, params: ParamsConv2D, dev: CudaDevice, oneTFn: (Any?) -> Any?, zeroTFn: (Any?) -> Any?): Result<Unit> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits, T : cudarc.CudnnDataType {
    val deviceId = dev.id()
    val cudnn =
        when (
            val tryResult =
                CUDNN.with(
                    closure@{ cudnn ->
                        when (val matchVal = cudnn.borrow().get(deviceId)) {
                            null -> {}
                            else -> {
                                val cudnn1 = matchVal
                                return@closure Ok(cudnn1.clone())
                            }
                        }
                        val c = Cudnn.new(dev.cudaStream())
                        if (c is Ok) {
                            val (c1) = c
                            cudnn.borrowMut().insert(deviceId, c1.clone())
                        }
                        c
                    },
                )
        ) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val conv =
        when (val tryResult = cudnn.createConv2d<Y>(listOf(params.padding, params.padding), listOf(params.stride, params.stride), listOf(params.dilation, params.dilation), cudarc.CUDNN_CROSS_CORRELATION)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val xShape = listOf(params.bSize, params.cIn, params.iH, params.iW)
    // Note that `src` already starts at the proper offset.
    val x =
        if (srcL.isContiguous()) {
            when (val tryResult = cudnn.create4dTensor<T>(cudarc.CUDNN_TENSOR_NCHW, xShape)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        } else {
            val s = srcL.stride()
            when (val tryResult = cudnn.create4dTensorEx<T>(xShape, listOf(s[0].toInt(), s[1].toInt(), s[2].toInt(), s[3].toInt()))) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
    val w =
        when (val tryResult = cudnn.create4dFilter<T>(cudarc.CUDNN_TENSOR_NCHW, listOf(params.cOut, params.cIn, params.kH, params.kW))) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val (wOut, hOut) = Pair(params.outW(), params.outH())
    val y =
        when (val tryResult = cudnn.create4dTensor<T>(cudarc.CUDNN_TENSOR_NCHW, listOf(params.bSize, params.cOut, hOut, wOut))) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val conv2d = ConvForward(conv = conv, x = x, w = w, y = y)
    val alg =
        when (val matchVal = params.cudnnFwdAlgo) {
            null ->
                when (val tryResult = conv2d.pickAlgorithm()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
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
    val workspaceSize =
        when (val tryResult = conv2d.getWorkspaceSize(alg)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val workspace =
        when (val tryResult = dev.cudaStream().allocZeros<UByte>(workspaceSize)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    run {
        when (val tryResult = conv2d.launch<CudaSlice<UByte>>(alg, workspace, Pair(oneTFn(), zeroTFn()), src, filter, dst)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    }
    return Ok(Unit)
}

internal fun <T, Y : cudarc.CudnnDataType> launchConv1d(src: CudaView<T>, srcL: Layout, filter: CudaView<T>, dst: CudaSlice<T>, params: ParamsConv1D, dev: CudaDevice, oneTFn: (Any?) -> Any?, zeroTFn: (Any?) -> Any?): Result<Unit> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits, T : cudarc.CudnnDataType {
    val deviceId = dev.id()
    val cudnn =
        when (
            val tryResult =
                CUDNN.with(
                    closure1@{ cudnn ->
                        when (val matchVal = cudnn.borrow().get(deviceId)) {
                            null -> {}
                            else -> {
                                val cudnn1 = matchVal
                                return@closure1 Ok(cudnn1.clone())
                            }
                        }
                        val c = Cudnn.new(dev.cudaStream())
                        if (c is Ok) {
                            val (c1) = c
                            cudnn.borrowMut().insert(deviceId, c1.clone())
                        }
                        c
                    },
                )
        ) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val conv =
        when (val tryResult = cudnn.createConv2d<Y>(listOf(params.padding, 0), listOf(params.stride, 1), listOf(params.dilation, 1), cudarc.CUDNN_CROSS_CORRELATION)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    // https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html#cudnnsettensornddescriptor
    // > Tensors are restricted to having at least 4 dimensions, and at most CUDNN_DIM_MAX
    // > dimensions (defined in cudnn.h). When working with lower dimensional data, it is
    // > recommended that the user create a 4D tensor, and set the size along unused dimensions
    // > to 1.
    val xShape = listOf(params.bSize, params.cIn, params.lIn, 1)
    // Note that `src` already starts at the proper offset.
    val x =
        if (srcL.isContiguous()) {
            when (val tryResult = cudnn.create4dTensor<T>(cudarc.CUDNN_TENSOR_NCHW, xShape)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        } else {
            val s = srcL.stride()
            when (val tryResult = cudnn.create4dTensorEx<T>(xShape, listOf(s[0].toInt(), s[1].toInt(), s[2].toInt(), 1))) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
    val w =
        when (val tryResult = cudnn.create4dFilter<T>(cudarc.CUDNN_TENSOR_NCHW, listOf(params.cOut, params.cIn, params.kSize, 1))) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val lOut = params.lOut()
    val y =
        when (val tryResult = cudnn.create4dTensor<T>(cudarc.CUDNN_TENSOR_NCHW, listOf(params.bSize, params.cOut, lOut, 1))) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val conv1d = ConvForward(conv = conv, x = x, w = w, y = y)
    val alg =
        when (val matchVal = params.cudnnFwdAlgo) {
            null ->
                when (val tryResult = conv1d.pickAlgorithm()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
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
    val workspaceSize =
        when (val tryResult = conv1d.getWorkspaceSize(alg)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val workspace =
        when (val tryResult = dev.cudaStream().allocZeros<UByte>(workspaceSize)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    run {
        when (val tryResult = conv1d.launch<CudaSlice<UByte>>(alg, workspace, Pair(oneTFn(), zeroTFn()), src, filter, dst)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    }
    return Ok(Unit)
}
