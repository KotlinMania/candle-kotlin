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
private val CUDNN: ThreadLocal<MutableMap<DeviceId, Any>> = ThreadLocal()

private val cudnnNew: (Any) -> Result<Any> = { stream -> Result.success(stream) }
private val createConv2d: (Any, IntArray, IntArray, IntArray, Int) -> Result<IntArray> = { _, pad, stride, dilation, mode ->
    Result.success(intArrayOf(pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1], mode))
}
private val create4dTensor: (Any, Int, IntArray) -> Result<IntArray> = { _, _, shape ->
    Result.success(shape)
}
private val create4dTensorEx: (Any, IntArray, IntArray) -> Result<IntArray> = { _, shape, stride ->
    Result.success(shape + stride)
}
private val create4dFilter: (Any, Int, IntArray) -> Result<IntArray> = { _, _, shape ->
    Result.success(shape)
}
private val pickAlgorithm: (Array<IntArray>) -> Result<Int> = { _ -> Result.success(0) }
private val getWorkspaceSize: (Array<IntArray>, Int) -> Result<Int> = { _, _ -> Result.success(0) }
private val allocZerosU8: (Any, Int) -> Result<ByteArray> = { _, size -> Result.success(ByteArray(size)) }
private val launchForward: (Array<IntArray>, Int, ByteArray, List<*>, List<*>, MutableList<*>) -> Result<Unit> = { _, _, _, _, _, _ ->
    Result.success(Unit)
}

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
    dev: CudaDevice,
): Result<Unit> {
    val deviceId = dev.id()
    val map = CUDNN.getOr { HashMap() }
    val cudnn = map[deviceId]?.let { Result.success(it) } ?: run {
        val c = cudnnNew(dev)
        c.getOrNull()?.let { map[deviceId] = it }
        c
    }
    val handle = cudnn.getOrThrow()
    val conv = createConv2d(
        handle,
        intArrayOf(params.padding, params.padding),
        intArrayOf(params.stride, params.stride),
        intArrayOf(params.dilation, params.dilation),
        0,
    ).getOrThrow()
    val xShape = intArrayOf(
        params.bSize,
        params.cIn,
        params.iH,
        params.iW,
    )
    // Note that `src` already starts at the proper offset.
    val x = if (srcL.isContiguous()) {
        create4dTensor(
            handle,
            0,
            xShape,
        ).getOrThrow()
    } else {
        val s = srcL.stride()
        create4dTensorEx(
            handle,
            xShape,
            intArrayOf(s[0], s[1], s[2], s[3]),
        ).getOrThrow()
    }
    val w = create4dFilter(
        handle,
        0,
        intArrayOf(
            params.cOut,
            params.cIn,
            params.kH,
            params.kW,
        ),
    ).getOrThrow()
    val wOut = params.outW()
    val hOut = params.outH()
    val y = create4dTensor(
        handle,
        0,
        intArrayOf(params.bSize, params.cOut, hOut, wOut),
    ).getOrThrow()
    val conv2d = arrayOf(conv, x, w, y)
    val alg = when (params.cudnnFwdAlgo) {
        null -> pickAlgorithm(conv2d).getOrThrow()
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
    val workspaceSize = getWorkspaceSize(conv2d, alg).getOrThrow()
    val workspace = allocZerosU8(dev, workspaceSize).getOrThrow()
    launchForward(
        conv2d,
        alg,
        workspace,
        src,
        filter,
        dst,
    ).getOrThrow()
    return Result.success(Unit)
}

internal fun <T, Y> launchConv1d(
    src: List<T>,
    srcL: Layout,
    filter: List<T>,
    dst: MutableList<T>,
    params: ParamsConv1D,
    dev: CudaDevice,
): Result<Unit> {
    val deviceId = dev.id()
    val map = CUDNN.getOr { HashMap() }
    val cudnn = map[deviceId]?.let { Result.success(it) } ?: run {
        val c = cudnnNew(dev)
        c.getOrNull()?.let { map[deviceId] = it }
        c
    }
    val handle = cudnn.getOrThrow()
    val conv = createConv2d(
        handle,
        intArrayOf(params.padding, 0),
        intArrayOf(params.stride, 1),
        intArrayOf(params.dilation, 1),
        0,
    ).getOrThrow()
    val xShape = intArrayOf(
        params.bSize,
        params.cIn,
        params.lIn,
        1,
    )
    // Note that `src` already starts at the proper offset.
    val x = if (srcL.isContiguous()) {
        create4dTensor(
            handle,
            0,
            xShape,
        ).getOrThrow()
    } else {
        val s = srcL.stride()
        create4dTensorEx(
            handle,
            xShape,
            intArrayOf(s[0], s[1], s[2], 1),
        ).getOrThrow()
    }
    val w = create4dFilter(
        handle,
        0,
        intArrayOf(
            params.cOut,
            params.cIn,
            params.kSize,
            1,
        ),
    ).getOrThrow()
    val lOut = params.lOut()
    val y = create4dTensor(
        handle,
        0,
        intArrayOf(params.bSize, params.cOut, lOut, 1),
    ).getOrThrow()
    val conv1d = arrayOf(conv, x, w, y)
    val alg = when (params.cudnnFwdAlgo) {
        null -> pickAlgorithm(conv1d).getOrThrow()
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
    val workspaceSize = getWorkspaceSize(conv1d, alg).getOrThrow()
    val workspace = allocZerosU8(dev, workspaceSize).getOrThrow()
    launchForward(
        conv1d,
        alg,
        workspace,
        src,
        filter,
        dst,
    ).getOrThrow()
    return Result.success(Unit)
}
