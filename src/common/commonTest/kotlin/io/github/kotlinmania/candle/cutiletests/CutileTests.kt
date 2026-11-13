package io.github.kotlinmania.candle.cutiletests

import io.github.kotlinmania.candlecore.CpuStorage
import io.github.kotlinmania.candlecore.CudaStorage
import io.github.kotlinmania.candlecore.CustomOp2
import io.github.kotlinmania.candlecore.DType
import io.github.kotlinmania.candlecore.Device
import io.github.kotlinmania.candlecore.Layout
import io.github.kotlinmania.candlecore.Result
import io.github.kotlinmania.candlecore.Shape
import io.github.kotlinmania.candlecore.Tensor
import io.github.kotlinmania.candlecore.bail
import io.github.kotlinmania.candlecore.cutile
import io.github.kotlinmania.candlecore.utils.cudaIsAvailable
import io.github.kotlinmania.cutile.core.*
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

public val BLOCK_SIZE: Int = 32

public val OUTPUT_SEED: Float = -7.0f

public val WARMUP_SIZE: Int = BLOCK_SIZE * 2

public var GPU_TEST_LOCK: Mutex<Unit> = Mutex.new()

public fun tensor(ptr: MutPtr, len: Int): Tensor<Float> {
    val shape: Shape = Shape(dims = listOf(len))
    val strides: Array = Array(dims = listOf(1))
    val pointer: PointerTile<MutPtr> = pointerToTile(ptr)
    return makeTensorView(pointer, shape, strides, newTokenUnordered())
}

public fun add(out: MutPtr, lhs: MutPtr, rhs: MutPtr, len: Int) {
    val out1 = tensor(out, len)
    val lhs1 = tensor(lhs, len)
    val rhs1 = tensor(rhs, len)
    val block = constShape(32)
    val blockId = getTileBlockId().first
    val value = lhs1.partition(block).load(listOf(blockId)) + rhs1.partition(block).load(listOf(blockId))
    out1.partitionMut(block).store(value, listOf(blockId))
}

public enum class Mode(
    public val value: UInt,
) {
    Compile(0u),
    Launch(1u),
}

public class Add(
    public var value: Mode = Mode(),
) : CustomOp2 {
    public var field0: Mode
        get() = value
        set(v) {
            value = v
        }

    public fun name(): String = "cutile-test-add"

    public fun cpuFwd(lhs: CpuStorage, lhsLayout: Layout, rhs: CpuStorage, rhsLayout: Layout): Result<Pair<CpuStorage, Shape>> = bail("cutile-test-add requires CUDA")

    public fun cudaFwd(lhs: CudaStorage, lhsLayout: Layout, rhs: CudaStorage, rhsLayout: Layout): Result<Pair<CudaStorage, Shape>> {
        if (!lhsLayout.isContiguous() || !rhsLayout.isContiguous()) {
            bail("cutile-test-add requires contiguous tensors")
        }
        if (lhsLayout.dims() != rhsLayout.dims()) {
            bail("cutile-test-add requires equal shapes")
        }
        val len = lhsLayout.shape().elemCount()
        if (!len.isMultipleOf(BLOCK_SIZE)) {
            bail("cutile-test-add requires a multiple of {BLOCK_SIZE} elements")
        }
        val lenI32 =
            when (val tryResult = Int.tryFrom(len)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val device = lhs.device
        val context =
            when (val tryResult = cutile.CutileContext.new(device)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val lhs1 =
            when (val tryResult = context.readStorage<Float>(lhs, lhsLayout)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val rhs1 =
            when (val tryResult = context.readStorage<Float>(rhs, rhsLayout)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val seed = mutableListOf(OUTPUT_SEED, len)
        val output =
            when (val tryResult = device.cloneHtod(seed)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val out =
            when (val tryResult = context.write(output, 0)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val launcher = add(out.devicePointer(), lhs1.devicePointer(), rhs1.devicePointer(), lenI32).grid(Triple((len / BLOCK_SIZE).toUInt(), 1, 1))
        when (this.value) {
            Mode.Compile -> {
                when (val tryResult = cutile.kernel("test add warmup", { launcher.compileOn(context.stream()) })) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
            Mode.Launch -> {
                when (val tryResult = cutile.kernel("test add launch", { launcher.asyncOn(context.stream()) })) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
        }
        drop(Triple(out, lhs1, rhs1))
        return Result.Ok(Pair(CudaStorage.wrapCudaSlice(output, device.clone()), lhsLayout.shape().clone()))
    }
}

public fun cudaDevice(): Result<Device?> {
    return if (cudaIsAvailable()) {
        Result.Ok(
            when (val tryResult = Device.newCuda(0)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
        )
    } else {
        Result.Ok(null)
    }
}

@kotlin.test.Test
public fun compileOnlyWarmupDoesNotLaunch(): Result<Unit> {
    val device =
        when (val tryResult = cudaDevice()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ?: return Result.Ok(Unit)
    val lhs =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, (WARMUP_SIZE + 1).toFloat(), device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.narrow(0, 1, WARMUP_SIZE)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs =
        when (
            val tryResult =
                when (val tryResult = Tensor.ones(WARMUP_SIZE + 1, DType.F32, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.narrow(0, 1, WARMUP_SIZE)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val compileCount = cutile.jitCompileCount()
    val output =
        when (val tryResult = lhs.applyOp2NoBwd(rhs, Add(Mode.Compile))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = output.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == mutableListOf(OUTPUT_SEED, WARMUP_SIZE),
    )
    val warmedCount = cutile.jitCompileCount()
    check(warmedCount == compileCount + 1)
    val output1 =
        when (val tryResult = lhs.applyOp2NoBwd(rhs, Add(Mode.Launch))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(cutile.jitCompileCount() == warmedCount)
    val expected = (0 until WARMUP_SIZE).map({ index -> index.toFloat() + 2.0f }).collect<Vec>()
    check(
        when (val tryResult = output1.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == expected,
    )
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun customOpPreservesCandleStreamOrdering(): Result<Unit> {
    val device =
        when (val tryResult = cudaDevice()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ?: return Result.Ok(Unit)
    val input =
        when (val tryResult = Tensor.arange(0.0f, BLOCK_SIZE.toFloat(), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val lhs =
        when (val tryResult = (input + 1.0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs =
        when (val tryResult = (input * 2.0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val output =
        when (val tryResult = lhs.applyOp2NoBwd(rhs, Add(Mode.Launch))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    drop(Triple(input, lhs, rhs))
    val output1 =
        when (val tryResult = (output + 3.0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val expected = (0 until BLOCK_SIZE).map({ index -> 3.0f * index.toFloat() + 4.0f }).collect<Vec>()
    check(
        when (val tryResult = output1.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == expected,
    )
    return Result.Ok(Unit)
}
