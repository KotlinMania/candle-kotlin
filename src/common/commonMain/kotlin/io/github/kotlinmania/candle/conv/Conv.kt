// port-lint: source conv.rs
package io.github.kotlinmania.candle.conv

import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.op.BackpropOp
import io.github.kotlinmania.candle.op.Op
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.candle.tensor.fromStorage

// 1D and 2D Convolutions

public open class ParamsConv1D(
    internal var bSize: Int,
    // Maybe we should have a version without lIn as this bit depends on the input and not only on
    // the weights.
    internal var lIn: Int,
    internal var cOut: Int,
    internal var cIn: Int,
    internal var kSize: Int,
    internal var padding: Int,
    internal var stride: Int,
    internal var dilation: Int,
    internal var cudnnFwdAlgo: CudnnFwdAlgo?,
) {
    internal open fun lOut(): Int = (this.lIn + 2 * this.padding - this.dilation * (this.kSize - 1) - 1) / this.stride + 1

    internal open fun outDims(): List<Int> {
        val lOut = this.lOut()
        return mutableListOf(this.bSize, this.cOut, lOut)
    }
}

public open class ParamsConvTranspose1D(
    internal var bSize: Int,
    internal var lIn: Int,
    internal var cOut: Int,
    internal var cIn: Int,
    internal var kSize: Int,
    internal var padding: Int,
    internal var outputPadding: Int,
    internal var stride: Int,
    internal var dilation: Int,
) {
    internal open fun lOut(): Int = (this.lIn - 1) * this.stride - 2 * this.padding + this.dilation * (this.kSize - 1) + this.outputPadding + 1

    internal open fun outDims(): List<Int> {
        val lOut = this.lOut()
        return mutableListOf(this.bSize, this.cOut, lOut)
    }
}

public sealed class CudnnFwdAlgo {
    public data object ImplicitGemm : CudnnFwdAlgo()
    public data object ImplicitPrecompGemm : CudnnFwdAlgo()
    public data object Gemm : CudnnFwdAlgo()
    public data object Direct : CudnnFwdAlgo()
    public data object Fft : CudnnFwdAlgo()
    public data object FftTiling : CudnnFwdAlgo()
    public data object Winograd : CudnnFwdAlgo()
    public data object WinogradNonFused : CudnnFwdAlgo()
    public data object Count : CudnnFwdAlgo()

    public fun __rustEq(other: CudnnFwdAlgo): Boolean = this == other
}

public open class ParamsConv2D(
    internal var bSize: Int,
    internal var iH: Int,
    internal var iW: Int,
    internal var kH: Int,
    internal var kW: Int,
    internal var cOut: Int,
    internal var cIn: Int,
    internal var padding: Int,
    internal var stride: Int,
    internal var dilation: Int,
    public var cudnnFwdAlgo: CudnnFwdAlgo?,
) {
    internal open fun outH(): Int = (this.iH + 2 * this.padding - this.dilation * (this.kH - 1) - 1) / this.stride + 1

    internal open fun outW(): Int = (this.iW + 2 * this.padding - this.dilation * (this.kW - 1) - 1) / this.stride + 1

    internal open fun outDims(): List<Int> = mutableListOf(this.bSize, this.cOut, this.outH(), this.outW())
}

public open class ParamsConvTranspose2D(
    internal var bSize: Int,
    internal var iH: Int,
    internal var iW: Int,
    internal var kH: Int,
    internal var kW: Int,
    internal var cOut: Int,
    internal var cIn: Int,
    internal var padding: Int,
    internal var outputPadding: Int,
    internal var stride: Int,
    internal var dilation: Int,
) {
    internal open fun outH(): Int = (this.iH - 1) * this.stride + this.dilation * (this.kH - 1) + this.outputPadding + 1 - 2 * this.padding

    internal open fun outW(): Int = (this.iW - 1) * this.stride + this.dilation * (this.kW - 1) + this.outputPadding + 1 - 2 * this.padding

    internal open fun outDims(): List<Int> = mutableListOf(this.bSize, this.cOut, this.outH(), this.outW())
}

public fun Tensor.conv1dSingleGroup(kernel: Tensor, params: ParamsConv1D): Result<Tensor> {
    val storage = this.storage().conv1d(this.layout(), kernel.storage(), kernel.layout(), params).getOrElse { return Result.failure(it) }
    val op = BackpropOp.new2(this, kernel, { arg, kernel1 -> Op.Conv1D(arg = arg, kernel = kernel1, padding = params.padding, stride = params.stride, dilation = params.dilation) })
    val outDims = params.outDims()
    return Result.success(fromStorage(storage, outDims, op, false))
}

/**
 * Applies a 1D convolution over the input tensor.
 */
public fun Tensor.conv1d(kernel: Tensor, padding: Int, stride: Int, dilation: Int, groups: Int): Result<Tensor> = this.conv1dWithAlgo(kernel, padding, stride, dilation, groups, null)

/**
 * Applies a 1D convolution over the input tensor.
 */
public fun Tensor.conv1dWithAlgo(kernel: Tensor, padding: Int, stride: Int, dilation: Int, groups: Int, cudnnFwdAlgo: CudnnFwdAlgo?): Result<Tensor> {
    val (cOut, cInK, kSize) = kernel.dims3().getOrElse { return Result.failure(it) }
    val (bSize, cIn, lIn) = this.dims3().getOrElse { return Result.failure(it) }
    if (cIn != cInK * groups) {
        Result.failure(Error.Conv1dInvalidArgs(inpShape = this.shape().clone(), kShape = kernel.shape().clone(), padding = padding, stride = stride, msg = "the number of in-channels on the input doesn't match the kernel size").bt()).getOrElse { return Result.failure(it) }
    }
    val params = ParamsConv1D(bSize = bSize, lIn = lIn, cOut = cOut / groups, cIn = cIn / groups, kSize = kSize, padding = padding, stride = stride, dilation = dilation, cudnnFwdAlgo = cudnnFwdAlgo)
    return if (groups == 1) {
        this.conv1dSingleGroup(kernel, params)
    } else {
        val blocks = this.chunk(groups, 1).getOrElse { return Result.failure(it) }
        val kernel1 = kernel.chunk(groups, 0).getOrElse { return Result.failure(it) }
        val blocks1 = (blocks.iter().zip(kernel1)).map({ (block, kernel2) -> block.conv1dSingleGroup(kernel2, params) }).collect<Result<Vec>>().getOrElse { return Result.failure(it) }
        Tensor.cat(blocks1, 1)
    }
}

public fun Tensor.convTranspose1dSingleGroup(kernel: Tensor, params: ParamsConvTranspose1D): Result<Tensor> {
    val storage = this.storage().convTranspose1d(this.layout(), kernel.storage(), kernel.layout(), params).getOrElse { return Result.failure(it) }
    val op = BackpropOp.new2(this, kernel, { arg, kernel1 -> Op.ConvTranspose1D(arg = arg, kernel = kernel1, padding = params.padding, outputPadding = params.outputPadding, stride = params.stride, dilation = params.dilation) })
    val outDims = params.outDims()
    return Result.success(fromStorage(storage, outDims, op, false))
}

/**
 * Applies a 1D transposed convolution over the input tensor.
 */
public fun Tensor.convTranspose1d(kernel: Tensor, padding: Int, outputPadding: Int, stride: Int, dilation: Int, groups: Int): Result<Tensor> {
    val (cInK, cOut, kSize) = kernel.dims3().getOrElse { return Result.failure(it) }
    val (bSize, cIn, lIn) = this.dims3().getOrElse { return Result.failure(it) }
    if (cIn != cInK) {
        bail("in_channel mismatch between input ({c_in}) and kernel ({c_in_k})")
    }
    if (cIn % groups != 0) {
        bail("in_channel {c_in} is not divisible by the number of groups")
    }
    val params = ParamsConvTranspose1D(bSize = bSize, lIn = lIn, kSize = kSize, cOut = cOut, cIn = cIn / groups, padding = padding, outputPadding = outputPadding, stride = stride, dilation = dilation)
    return if (groups == 1) {
        this.convTranspose1dSingleGroup(kernel, params)
    } else {
        val blocks = this.chunk(groups, 1).getOrElse { return Result.failure(it) }
        val kernel1 = kernel.chunk(groups, 0).getOrElse { return Result.failure(it) }
        val blocks1 = (blocks.iter().zip(kernel1)).map({ (block, kernel2) -> block.convTranspose1dSingleGroup(kernel2, params) }).collect<Result<Vec>>().getOrElse { return Result.failure(it) }
        Tensor.cat(blocks1, 1)
    }
}

public fun Tensor.conv2dSingleGroup(kernel: Tensor, params: ParamsConv2D): Result<Tensor> {
    val storage = this.storage().conv2d(this.layout(), kernel.storage(), kernel.layout(), params).getOrElse { return Result.failure(it) }
    val op = BackpropOp.new2(this, kernel, { arg, kernel1 -> Op.Conv2D(arg = arg, kernel = kernel1, padding = params.padding, stride = params.stride, dilation = params.dilation) })
    val outDims = params.outDims()
    return Result.success(fromStorage(storage, outDims, op, false))
}

/**
 * Applies a 2D convolution over the input tensor.
 */
public fun Tensor.conv2d(kernel: Tensor, padding: Int, stride: Int, dilation: Int, groups: Int): Result<Tensor> = this.conv2dWithAlgo(kernel, padding, stride, dilation, groups, null)

public fun Tensor.conv2dWithAlgo(kernel: Tensor, padding: Int, stride: Int, dilation: Int, groups: Int, cudnnFwdAlgo: CudnnFwdAlgo?): Result<Tensor> {
    val (bSize, cIn, iH, iW) = this.dims4().getOrElse { return Result.failure(it) }
    val (cOut, cInK, kH, kW) = kernel.dims4().getOrElse { return Result.failure(it) }
    if (cIn != cInK * groups) {
        bail("in_channel mismatch between input ({c_in}, groups {groups}) and kernel ({c_in_k})")
    }
    val params = ParamsConv2D(bSize = bSize, iH = iH, iW = iW, kH = kH, kW = kW, cOut = cOut / groups, cIn = cIn / groups, padding = padding, stride = stride, dilation = dilation, cudnnFwdAlgo = cudnnFwdAlgo)
    return if (groups == 1) {
        this.conv2dSingleGroup(kernel, params)
    } else {
        val blocks = this.chunk(groups, 1).getOrElse { return Result.failure(it) }
        val kernel1 = kernel.chunk(groups, 0).getOrElse { return Result.failure(it) }
        val blocks1 = (blocks.iter().zip(kernel1)).map({ (block, kernel2) -> block.conv2dSingleGroup(kernel2, params) }).collect<Result<Vec>>().getOrElse { return Result.failure(it) }
        Tensor.cat(blocks1, 1)
    }
}

/**
 * Applies a 2D transposed convolution over the input tensor.
 */
public fun Tensor.convTranspose2d(kernel: Tensor, padding: Int, outputPadding: Int, stride: Int, dilation: Int): Result<Tensor> {
    val (bSize, cIn, iH, iW) = this.dims4().getOrElse { return Result.failure(it) }
    val (cInK, cOut, kH, kW) = kernel.dims4().getOrElse { return Result.failure(it) }
    if (cIn != cInK) {
        bail("in_channel mismatch between input ({c_in}) and kernel ({c_in_k})")
    }
    val params = ParamsConvTranspose2D(bSize = bSize, iH = iH, iW = iW, kH = kH, kW = kW, cOut = cOut, cIn = cIn, padding = padding, outputPadding = outputPadding, stride = stride, dilation = dilation)
    val storage = this.storage().convTranspose2d(this.layout(), kernel.storage(), kernel.layout(), params).getOrElse { return Result.failure(it) }
    val op = BackpropOp.new2(this, kernel, { arg, kernel1 -> Op.ConvTranspose2D(arg = arg, kernel = kernel1, padding = params.padding, outputPadding = params.outputPadding, stride = params.stride, dilation = params.dilation) })
    val outDims = params.outDims()
    return Result.success(fromStorage(storage, outDims, op, false))
}
