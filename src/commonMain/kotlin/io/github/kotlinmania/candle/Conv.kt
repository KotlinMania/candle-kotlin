// port-lint: source conv.rs
package io.github.kotlinmania.candle

/**
 * 1D and 2D Convolutions
 */

public enum class CudnnFwdAlgo {
    ImplicitGemm,
    ImplicitPrecompGemm,
    Gemm,
    Direct,
    Fft,
    FftTiling,
    Winograd,
    WinogradNonFused,
    Count,
}

public data class ParamsConv1D(
    val bSize: Int,
    // Maybe we should have a version without lIn as this bit depends on the input and not only on
    // the weights.
    val lIn: Int,
    val cOut: Int,
    val cIn: Int,
    val kSize: Int,
    val padding: Int,
    val stride: Int,
    val dilation: Int,
    val cudnnFwdAlgo: CudnnFwdAlgo? = null,
) {
    public fun lOut(): Int =
        (lIn + 2 * padding - dilation * (kSize - 1) - 1) / stride + 1

    public fun outDims(): List<Int> {
        val lOut = lOut()
        return listOf(bSize, cOut, lOut)
    }
}

public data class ParamsConvTranspose1D(
    val bSize: Int,
    val lIn: Int,
    val cOut: Int,
    val cIn: Int,
    val kSize: Int,
    val padding: Int,
    val outputPadding: Int,
    val stride: Int,
    val dilation: Int,
) {
    public fun lOut(): Int =
        (lIn - 1) * stride - 2 * padding + dilation * (kSize - 1) + outputPadding + 1

    public fun outDims(): List<Int> {
        val lOut = lOut()
        return listOf(bSize, cOut, lOut)
    }
}

public data class ParamsConv2D(
    val bSize: Int,
    val iH: Int,
    val iW: Int,
    val kH: Int,
    val kW: Int,
    val cOut: Int,
    val cIn: Int,
    val padding: Int,
    val stride: Int,
    val dilation: Int,
    val cudnnFwdAlgo: CudnnFwdAlgo? = null,
) {
    public fun outH(): Int =
        (iH + 2 * padding - dilation * (kH - 1) - 1) / stride + 1

    public fun outW(): Int =
        (iW + 2 * padding - dilation * (kW - 1) - 1) / stride + 1

    public fun outDims(): List<Int> = listOf(bSize, cOut, outH(), outW())
}

public data class ParamsConvTranspose2D(
    val bSize: Int,
    val iH: Int,
    val iW: Int,
    val kH: Int,
    val kW: Int,
    val cOut: Int,
    val cIn: Int,
    val padding: Int,
    val outputPadding: Int,
    val stride: Int,
    val dilation: Int,
) {
    public fun outH(): Int =
        (iH - 1) * stride + dilation * (kH - 1) + outputPadding + 1 - 2 * padding

    public fun outW(): Int =
        (iW - 1) * stride + dilation * (kW - 1) + outputPadding + 1 - 2 * padding

    public fun outDims(): List<Int> = listOf(bSize, cOut, outH(), outW())
}

public fun Tensor.conv1dSingleGroup(kernel: Tensor, params: ParamsConv1D): Tensor {
    val storage = storage().conv1d(layout(), kernel.storage(), kernel.layout(), params)
    val op = BackpropOp.new2(this, kernel) { arg, k ->
        Op.Conv1D(
            arg = arg,
            kernel = k,
            padding = params.padding,
            stride = params.stride,
            dilation = params.dilation,
        )
    }
    val outDims = params.outDims()
    return fromStorage(storage, Shape(outDims), op, false)
}

/**
 * Applies a 1D convolution over the input tensor.
 */
public fun Tensor.conv1d(
    kernel: Tensor,
    padding: Int,
    stride: Int,
    dilation: Int,
    groups: Int,
): Tensor = conv1dWithAlgo(kernel, padding, stride, dilation, groups, null)

/**
 * Applies a 1D convolution over the input tensor.
 */
public fun Tensor.conv1dWithAlgo(
    kernel: Tensor,
    padding: Int,
    stride: Int,
    dilation: Int,
    groups: Int,
    cudnnFwdAlgo: CudnnFwdAlgo?,
): Tensor {
    val (cOut, cInK, kSize) = kernel.dims3()
    val (bSize, cIn, lIn) = dims3()
    if (cIn != cInK * groups) {
        throw CandleException.Conv1dInvalidArgs(
            inpShape = shape(),
            kShape = kernel.shape(),
            padding = padding,
            stride = stride,
            msg = "the number of in-channels on the input doesn't match the kernel size",
        )
    }
    val params = ParamsConv1D(
        bSize = bSize,
        lIn = lIn,
        cOut = cOut / groups,
        cIn = cIn / groups,
        kSize = kSize,
        padding = padding,
        stride = stride,
        dilation = dilation,
        cudnnFwdAlgo = cudnnFwdAlgo,
    )
    return if (groups == 1) {
        conv1dSingleGroup(kernel, params)
    } else {
        val blocks = chunk(groups, 1)
        val kernels = kernel.chunk(groups, 0)
        val outBlocks = blocks.zip(kernels).map { (block, k) ->
            block.conv1dSingleGroup(k, params)
        }
        Tensor.cat(outBlocks, 1)
    }
}

public fun Tensor.convTranspose1dSingleGroup(
    kernel: Tensor,
    params: ParamsConvTranspose1D,
): Tensor {
    val storage = storage().convTranspose1d(layout(), kernel.storage(), kernel.layout(), params)
    val op = BackpropOp.new2(this, kernel) { arg, k ->
        Op.ConvTranspose1D(
            arg = arg,
            kernel = k,
            padding = params.padding,
            outputPadding = params.outputPadding,
            stride = params.stride,
            dilation = params.dilation,
        )
    }
    val outDims = params.outDims()
    return fromStorage(storage, Shape(outDims), op, false)
}

/**
 * Applies a 1D transposed convolution over the input tensor.
 */
public fun Tensor.convTranspose1d(
    kernel: Tensor,
    padding: Int,
    outputPadding: Int,
    stride: Int,
    dilation: Int,
    groups: Int,
): Tensor {
    val (cInK, cOut, kSize) = kernel.dims3()
    val (bSize, cIn, lIn) = dims3()
    if (cIn != cInK) {
        throw CandleException.Msg("inChannel mismatch between input ($cIn) and kernel ($cInK)")
    }
    if (cIn % groups != 0) {
        throw CandleException.Msg("inChannel $cIn is not divisible by the number of groups")
    }
    val params = ParamsConvTranspose1D(
        bSize = bSize,
        lIn = lIn,
        kSize = kSize,
        cOut = cOut,
        cIn = cIn / groups,
        padding = padding,
        outputPadding = outputPadding,
        stride = stride,
        dilation = dilation,
    )
    return if (groups == 1) {
        convTranspose1dSingleGroup(kernel, params)
    } else {
        val blocks = chunk(groups, 1)
        val kernels = kernel.chunk(groups, 0)
        val outBlocks = blocks.zip(kernels).map { (block, k) ->
            block.convTranspose1dSingleGroup(k, params)
        }
        Tensor.cat(outBlocks, 1)
    }
}

public fun Tensor.conv2dSingleGroup(kernel: Tensor, params: ParamsConv2D): Tensor {
    val storage = storage().conv2d(layout(), kernel.storage(), kernel.layout(), params)
    val op = BackpropOp.new2(this, kernel) { arg, k ->
        Op.Conv2D(
            arg = arg,
            kernel = k,
            padding = params.padding,
            stride = params.stride,
            dilation = params.dilation,
        )
    }
    val outDims = params.outDims()
    return fromStorage(storage, Shape(outDims), op, false)
}

/**
 * Applies a 2D convolution over the input tensor.
 */
public fun Tensor.conv2d(
    kernel: Tensor,
    padding: Int,
    stride: Int,
    dilation: Int,
    groups: Int,
): Tensor = conv2dWithAlgo(kernel, padding, stride, dilation, groups, null)

public fun Tensor.conv2dWithAlgo(
    kernel: Tensor,
    padding: Int,
    stride: Int,
    dilation: Int,
    groups: Int,
    cudnnFwdAlgo: CudnnFwdAlgo?,
): Tensor {
    val (bSize, cIn, iH, iW) = dims4()
    val (cOut, cInK, kH, kW) = kernel.dims4()
    if (cIn != cInK * groups) {
        throw CandleException.Msg(
            "inChannel mismatch between input ($cIn, groups $groups) and kernel ($cInK)",
        )
    }
    val params = ParamsConv2D(
        bSize = bSize,
        iH = iH,
        iW = iW,
        kH = kH,
        kW = kW,
        cOut = cOut / groups,
        cIn = cIn / groups,
        padding = padding,
        stride = stride,
        dilation = dilation,
        cudnnFwdAlgo = cudnnFwdAlgo,
    )
    return if (groups == 1) {
        conv2dSingleGroup(kernel, params)
    } else {
        val blocks = chunk(groups, 1)
        val kernels = kernel.chunk(groups, 0)
        val outBlocks = blocks.zip(kernels).map { (block, k) ->
            block.conv2dSingleGroup(k, params)
        }
        Tensor.cat(outBlocks, 1)
    }
}

/**
 * Applies a 2D transposed convolution over the input tensor.
 */
public fun Tensor.convTranspose2d(
    kernel: Tensor,
    padding: Int,
    outputPadding: Int,
    stride: Int,
    dilation: Int,
): Tensor {
    val (bSize, cIn, iH, iW) = dims4()
    val (cInK, cOut, kH, kW) = kernel.dims4()
    if (cIn != cInK) {
        throw CandleException.Msg("inChannel mismatch between input ($cIn) and kernel ($cInK)")
    }
    val params = ParamsConvTranspose2D(
        bSize = bSize,
        iH = iH,
        iW = iW,
        kH = kH,
        kW = kW,
        cOut = cOut,
        cIn = cIn,
        padding = padding,
        outputPadding = outputPadding,
        stride = stride,
        dilation = dilation,
    )
    val storage = storage().convTranspose2d(layout(), kernel.storage(), kernel.layout(), params)
    val op = BackpropOp.new2(this, kernel) { arg, k ->
        Op.ConvTranspose2D(
            arg = arg,
            kernel = k,
            padding = params.padding,
            outputPadding = params.outputPadding,
            stride = params.stride,
            dilation = params.dilation,
        )
    }
    val outDims = params.outDims()
    return fromStorage(storage, Shape(outDims), op, false)
}
