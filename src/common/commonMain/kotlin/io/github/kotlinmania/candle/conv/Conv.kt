// port-lint: source conv.rs
package io.github.kotlinmania.candle.conv

import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.op.BackpropOp
import io.github.kotlinmania.candle.op.Op
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.candle.tensor.fromStorage
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout

// 1D and 2D Convolutions

public data class ParamsConv1D(
    internal var bSize: Int = 0,
    // Maybe we should have a version without lIn as this bit depends on the input and not only on
    // the weights.
    internal var lIn: Int = 0,
    internal var cOut: Int = 0,
    internal var cIn: Int = 0,
    internal var kSize: Int = 0,
    internal var padding: Int = 0,
    internal var stride: Int = 0,
    internal var dilation: Int = 0,
    internal var cudnnFwdAlgo: CudnnFwdAlgo? = null,
) {
    internal fun lOut(): Int = (this.lIn + 2 * this.padding - this.dilation * (this.kSize - 1) - 1) / this.stride + 1

    internal fun outDims(): List<Int> {
        val lOut = this.lOut()
        return mutableListOf(this.bSize, this.cOut, lOut)
    }
}

public data class ParamsConvTranspose1D(
    internal var bSize: Int = 0,
    internal var lIn: Int = 0,
    internal var cOut: Int = 0,
    internal var cIn: Int = 0,
    internal var kSize: Int = 0,
    internal var padding: Int = 0,
    internal var outputPadding: Int = 0,
    internal var stride: Int = 0,
    internal var dilation: Int = 0,
) {
    internal fun lOut(): Int = (this.lIn - 1) * this.stride - 2 * this.padding + this.dilation * (this.kSize - 1) + this.outputPadding + 1

    internal fun outDims(): List<Int> {
        val lOut = this.lOut()
        return mutableListOf(this.bSize, this.cOut, lOut)
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<ParamsConvTranspose1D> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<ParamsConvTranspose1D> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 72L
                override fun readNative(addr: Long): ParamsConvTranspose1D =
                    ParamsConvTranspose1D(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 8L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 16L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 24L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 32L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 40L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 48L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 56L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 64L),
                    )
                override fun writeNative(addr: Long, value: ParamsConvTranspose1D) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr, value.bSize)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 8L, value.lIn)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 16L, value.cOut)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 24L, value.cIn)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 32L, value.kSize)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 40L, value.padding)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 48L, value.outputPadding)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 56L, value.stride)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 64L, value.dilation)
                }
            }
    }
}

public enum class CudnnFwdAlgo(
    public val value: UInt,
) {
    ImplicitGemm(0u),
    ImplicitPrecompGemm(1u),
    Gemm(2u),
    Direct(3u),
    Fft(4u),
    FftTiling(5u),
    Winograd(6u),
    WinogradNonFused(7u),
    Count(8u),
}

public data class ParamsConv2D(
    internal var bSize: Int = 0,
    internal var iH: Int = 0,
    internal var iW: Int = 0,
    internal var kH: Int = 0,
    internal var kW: Int = 0,
    internal var cOut: Int = 0,
    internal var cIn: Int = 0,
    internal var padding: Int = 0,
    internal var stride: Int = 0,
    internal var dilation: Int = 0,
    public var cudnnFwdAlgo: CudnnFwdAlgo? = null,
) {
    internal fun outH(): Int = (this.iH + 2 * this.padding - this.dilation * (this.kH - 1) - 1) / this.stride + 1

    internal fun outW(): Int = (this.iW + 2 * this.padding - this.dilation * (this.kW - 1) - 1) / this.stride + 1

    internal fun outDims(): List<Int> = mutableListOf(this.bSize, this.cOut, this.outH(), this.outW())
}

public data class ParamsConvTranspose2D(
    internal var bSize: Int = 0,
    internal var iH: Int = 0,
    internal var iW: Int = 0,
    internal var kH: Int = 0,
    internal var kW: Int = 0,
    internal var cOut: Int = 0,
    internal var cIn: Int = 0,
    internal var padding: Int = 0,
    internal var outputPadding: Int = 0,
    internal var stride: Int = 0,
    internal var dilation: Int = 0,
) {
    internal fun outH(): Int = (this.iH - 1) * this.stride + this.dilation * (this.kH - 1) + this.outputPadding + 1 - 2 * this.padding

    internal fun outW(): Int = (this.iW - 1) * this.stride + this.dilation * (this.kW - 1) + this.outputPadding + 1 - 2 * this.padding

    internal fun outDims(): List<Int> = mutableListOf(this.bSize, this.cOut, this.outH(), this.outW())

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<ParamsConvTranspose2D> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<ParamsConvTranspose2D> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 88L
                override fun readNative(addr: Long): ParamsConvTranspose2D =
                    ParamsConvTranspose2D(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 8L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 16L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 24L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 32L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 40L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 48L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 56L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 64L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 72L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 80L),
                    )
                override fun writeNative(addr: Long, value: ParamsConvTranspose2D) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr, value.bSize)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 8L, value.iH)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 16L, value.iW)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 24L, value.kH)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 32L, value.kW)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 40L, value.cOut)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 48L, value.cIn)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 56L, value.padding)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 64L, value.outputPadding)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 72L, value.stride)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 80L, value.dilation)
                }
            }
    }
}

public fun Tensor.conv1dSingleGroup(kernel: Tensor, params: ParamsConv1D): Result<Tensor> {
    val storage =
        when (val tryResult = this.storage().conv1d(this.layout(), kernel.storage(), kernel.layout(), params)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val op = BackpropOp.new2(this, kernel, { arg, kernel1 -> Op.Conv1D(arg = arg, kernel = kernel1, padding = params.padding, stride = params.stride, dilation = params.dilation) })
    val outDims = params.outDims()
    return Result.Ok(fromStorage(storage, outDims, op, false))
}

/**
 * Applies a 1D convolution over the input tensor.
 */
public fun Tensor.conv1d(kernel: Tensor, padding: Int, stride: Int, dilation: Int, groups: Int): Result<Tensor> = this.conv1dWithAlgo(kernel, padding, stride, dilation, groups, null)

/**
 * Applies a 1D convolution over the input tensor.
 */
public fun Tensor.conv1dWithAlgo(kernel: Tensor, padding: Int, stride: Int, dilation: Int, groups: Int, cudnnFwdAlgo: CudnnFwdAlgo?): Result<Tensor> {
    val (cOut, cInK, kSize) =
        when (val tryResult = kernel.dims3()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val (bSize, cIn, lIn) =
        when (val tryResult = this.dims3()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    if (cIn != cInK * groups) {
        when (val tryResult = Result.Err(Error.Conv1dInvalidArgs(inpShape = this.shape().clone(), kShape = kernel.shape().clone(), padding = padding, stride = stride, msg = "the number of in-channels on the input doesn't match the kernel size").bt())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    }
    val params = ParamsConv1D(bSize = bSize, lIn = lIn, cOut = cOut / groups, cIn = cIn / groups, kSize = kSize, padding = padding, stride = stride, dilation = dilation, cudnnFwdAlgo = cudnnFwdAlgo)
    return if (groups == 1) {
        this.conv1dSingleGroup(kernel, params)
    } else {
        val blocks =
            when (val tryResult = this.chunk(groups, 1)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val kernel1 =
            when (val tryResult = kernel.chunk(groups, 0)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val blocks1 =
            when (val tryResult = (blocks.iter().zip(kernel1)).map({ (block, kernel2) -> block.conv1dSingleGroup(kernel2, params) }).collect<Result<Vec>>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        Tensor.cat(blocks1, 1)
    }
}

public fun Tensor.convTranspose1dSingleGroup(kernel: Tensor, params: ParamsConvTranspose1D): Result<Tensor> {
    val storage =
        when (val tryResult = this.storage().convTranspose1d(this.layout(), kernel.storage(), kernel.layout(), params)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val op = BackpropOp.new2(this, kernel, { arg, kernel1 -> Op.ConvTranspose1D(arg = arg, kernel = kernel1, padding = params.padding, outputPadding = params.outputPadding, stride = params.stride, dilation = params.dilation) })
    val outDims = params.outDims()
    return Result.Ok(fromStorage(storage, outDims, op, false))
}

/**
 * Applies a 1D transposed convolution over the input tensor.
 */
public fun Tensor.convTranspose1d(kernel: Tensor, padding: Int, outputPadding: Int, stride: Int, dilation: Int, groups: Int): Result<Tensor> {
    val (cInK, cOut, kSize) =
        when (val tryResult = kernel.dims3()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val (bSize, cIn, lIn) =
        when (val tryResult = this.dims3()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
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
        val blocks =
            when (val tryResult = this.chunk(groups, 1)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val kernel1 =
            when (val tryResult = kernel.chunk(groups, 0)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val blocks1 =
            when (val tryResult = (blocks.iter().zip(kernel1)).map({ (block, kernel2) -> block.convTranspose1dSingleGroup(kernel2, params) }).collect<Result<Vec>>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        Tensor.cat(blocks1, 1)
    }
}

public fun Tensor.conv2dSingleGroup(kernel: Tensor, params: ParamsConv2D): Result<Tensor> {
    val storage =
        when (val tryResult = this.storage().conv2d(this.layout(), kernel.storage(), kernel.layout(), params)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val op = BackpropOp.new2(this, kernel, { arg, kernel1 -> Op.Conv2D(arg = arg, kernel = kernel1, padding = params.padding, stride = params.stride, dilation = params.dilation) })
    val outDims = params.outDims()
    return Result.Ok(fromStorage(storage, outDims, op, false))
}

/**
 * Applies a 2D convolution over the input tensor.
 */
public fun Tensor.conv2d(kernel: Tensor, padding: Int, stride: Int, dilation: Int, groups: Int): Result<Tensor> = this.conv2dWithAlgo(kernel, padding, stride, dilation, groups, null)

public fun Tensor.conv2dWithAlgo(kernel: Tensor, padding: Int, stride: Int, dilation: Int, groups: Int, cudnnFwdAlgo: CudnnFwdAlgo?): Result<Tensor> {
    val (bSize, cIn, iH, iW) =
        when (val tryResult = this.dims4()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val (cOut, cInK, kH, kW) =
        when (val tryResult = kernel.dims4()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    if (cIn != cInK * groups) {
        bail("in_channel mismatch between input ({c_in}, groups {groups}) and kernel ({c_in_k})")
    }
    val params = ParamsConv2D(bSize = bSize, iH = iH, iW = iW, kH = kH, kW = kW, cOut = cOut / groups, cIn = cIn / groups, padding = padding, stride = stride, dilation = dilation, cudnnFwdAlgo = cudnnFwdAlgo)
    return if (groups == 1) {
        this.conv2dSingleGroup(kernel, params)
    } else {
        val blocks =
            when (val tryResult = this.chunk(groups, 1)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val kernel1 =
            when (val tryResult = kernel.chunk(groups, 0)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val blocks1 =
            when (val tryResult = (blocks.iter().zip(kernel1)).map({ (block, kernel2) -> block.conv2dSingleGroup(kernel2, params) }).collect<Result<Vec>>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        Tensor.cat(blocks1, 1)
    }
}

/**
 * Applies a 2D transposed convolution over the input tensor.
 */
public fun Tensor.convTranspose2d(kernel: Tensor, padding: Int, outputPadding: Int, stride: Int, dilation: Int): Result<Tensor> {
    val (bSize, cIn, iH, iW) =
        when (val tryResult = this.dims4()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val (cInK, cOut, kH, kW) =
        when (val tryResult = kernel.dims4()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    if (cIn != cInK) {
        bail("in_channel mismatch between input ({c_in}) and kernel ({c_in_k})")
    }
    val params = ParamsConvTranspose2D(bSize = bSize, iH = iH, iW = iW, kH = kH, kW = kW, cOut = cOut, cIn = cIn, padding = padding, outputPadding = outputPadding, stride = stride, dilation = dilation)
    val storage =
        when (val tryResult = this.storage().convTranspose2d(this.layout(), kernel.storage(), kernel.layout(), params)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val op = BackpropOp.new2(this, kernel, { arg, kernel1 -> Op.ConvTranspose2D(arg = arg, kernel = kernel1, padding = params.padding, outputPadding = params.outputPadding, stride = params.stride, dilation = params.dilation) })
    val outDims = params.outDims()
    return Result.Ok(fromStorage(storage, outDims, op, false))
}
