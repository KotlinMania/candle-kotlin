package io.github.kotlinmania.candle.customoptests

import io.github.kotlinmania.candlecore.CpuStorage
import io.github.kotlinmania.candlecore.CustomOp1
import io.github.kotlinmania.candlecore.DType
import io.github.kotlinmania.candlecore.Device
import io.github.kotlinmania.candlecore.InplaceOp1
import io.github.kotlinmania.candlecore.InplaceOp2
import io.github.kotlinmania.candlecore.InplaceOp3
import io.github.kotlinmania.candlecore.Layout
import io.github.kotlinmania.candlecore.Result
import io.github.kotlinmania.candlecore.Shape
import io.github.kotlinmania.candlecore.Tensor
import io.github.kotlinmania.candlecore.Var
import io.github.kotlinmania.candlecore.bail
import io.github.kotlinmania.candlecore.mapDtype
import io.github.kotlinmania.candlecore.testutils.toVec1Round
import io.github.kotlinmania.rustlang.core.F16
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.result.unwrapOr

public fun <T : io.github.kotlinmania.numtraits.Float> fwd(v: T, alpha: Double, fromTFn: (Any?) -> T, nanTFn: (Any?) -> Any?, oneTFn: (Any?) -> Any?): T =
    if (v.isSignPositive()) {
        v
    } else {
        val alpha1 = fromTFn(alpha).unwrapOr(nanTFn())
        (v.exp() - oneTFn()) * alpha1
    }

public class Elu(
    public var alpha: Double = 0.0,
) : CustomOp1,
    InplaceOp1 {
    public fun name(): String = "elu"

    public fun cpuFwd(s: CpuStorage, l: Layout): Result<Pair<CpuStorage, Shape>> {
        val storage = mapDtype("elu", s, { s1 -> cpubackend.unaryMap(s1, l, { v -> fwd(v, this.alpha) }) }, listOf(F8E4M3, BF16, F16, F32, F64))
        return Result.Ok(Pair(storage, l.shape().clone()))
    }

    public fun nameInplaceOp1(): String = "elu"

    public fun cpuFwdInplaceOp1(s: CpuStorage, l: Layout): Result<Unit> {
        val alpha = this.alpha
        when (s) {
            is CpuStorage.F8e4m3 -> {
                val (s1) = s
                s1.iterMut().forEach({ v -> v = fwd(v, alpha) })
            }
            is CpuStorage.Bf16 -> {
                val (s2) = s
                s2.iterMut().forEach({ v -> v = fwd(v, alpha) })
            }
            is CpuStorage.F16 -> {
                val (s3) = s
                s3.iterMut().forEach({ v -> v = fwd(v, alpha) })
            }
            is CpuStorage.F32 -> {
                val (s4) = s
                s4.iterMut().forEach({ v -> v = fwd(v, alpha) })
            }
            is CpuStorage.F64 -> {
                val (s5) = s
                s5.iterMut().forEach({ v -> v = fwd(v, alpha) })
            }
            else -> bail("unsupported dtype for inplace elu")
        }
        return Result.Ok(Unit)
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Elu> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Elu> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 8L
                override fun readNative(addr: Long): Elu =
                    Elu(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F64).readNative(addr),
                    )
                override fun writeNative(addr: Long, value: Elu) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F64).writeNative(addr, value.alpha)
                }
            }
    }
}

@kotlin.test.Test
public fun customOp1NoBackward(): Result<Unit> {
    val cpu = Device.Cpu
    val t =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0u, 12u, cpu)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toDtype(DType.F32)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t1 =
        when (val tryResult = (t - 5.0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val eluT =
        when (val tryResult = t1.applyOp1NoBwd(Elu(alpha = 1.0))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = toVec1Round(eluT, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(-0.9933, -0.9817, -0.9502, -0.8647, -0.6321, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    )
    return Result.Ok(Unit)
}

// Define a similar struct as Elu but with backward support.
public fun <T : io.github.kotlinmania.numtraits.Float> bwd(v: T, alpha: Double, fromTFn: (Any?) -> T, oneTFn: (Any?) -> Any?, nanTFn: (Any?) -> Any?): T =
    if (v.isSignPositive()) {
        oneTFn()
    } else {
        val alpha1 = fromTFn(alpha).unwrapOr(nanTFn())
        v.exp() * alpha1
    }

public class EluBackward(
    public var alpha: Double = 0.0,
) : CustomOp1 {
    public fun name(): String = "elu-bwd"

    public fun cpuFwd(s: CpuStorage, l: Layout): Result<Pair<CpuStorage, Shape>> {
        val storage = mapDtype("elu-bwd", s, { s1 -> cpubackend.unaryMap(s1, l, { v -> bwd(v, this.alpha) }) }, listOf(F8E4M3, BF16, F16, F32, F64))
        return Result.Ok(Pair(storage, l.shape().clone()))
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<EluBackward> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<EluBackward> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 8L
                override fun readNative(addr: Long): EluBackward =
                    EluBackward(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F64).readNative(addr),
                    )
                override fun writeNative(addr: Long, value: EluBackward) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F64).writeNative(addr, value.alpha)
                }
            }
    }
}

public class EluWithBackward(
    public var value: Elu = Elu(),
) : CustomOp1 {
    public var field0: Elu
        get() = value
        set(v) {
            value = v
        }

    public fun name(): String = "elu"

    public fun cpuFwd(s: CpuStorage, l: Layout): Result<Pair<CpuStorage, Shape>> = this.value.cpuFwd(s, l)

    public fun bwd(arg: Tensor, res: Tensor, gradRes: Tensor): Result<Tensor?> {
        val alpha = this.value.alpha
        val bwd =
            when (val tryResult = arg.applyOp1(EluBackward(alpha = alpha))) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        return Result.Ok(
            when (val tryResult = gradRes.mul(bwd)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
        )
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<EluWithBackward> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<EluWithBackward> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 8L
                override fun readNative(addr: Long): EluWithBackward =
                    EluWithBackward(
                        (Elu.typeLayout()).readNative(addr),
                    )
                override fun writeNative(addr: Long, value: EluWithBackward) {
                    (Elu.typeLayout()).writeNative(addr, value.value)
                }
            }

        public fun new(alpha: Double): EluWithBackward = EluWithBackward(Elu(alpha = alpha))
    }
}

@kotlin.test.Test
public fun customOp1WithBackward(): Result<Unit> {
    val cpu = Device.Cpu
    val t =
        when (val tryResult = Var.new(listOf(-2.0f, 0.0f, 2.0f), cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val eluT =
        when (val tryResult = t.applyOp1(EluWithBackward.new(2.0))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = toVec1Round(eluT, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(-1.7293, 0.0, 2.0),
    )
    val grads =
        when (val tryResult = eluT.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX = grads.get(t)!!
    check(
        when (val tryResult = toVec1Round(gradX, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(0.2707, 1.0, 1.0),
    )
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun inplaceOp1(): Result<Unit> {
    val cpu = Device.Cpu
    val t =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0u, 12u, cpu)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toDtype(DType.F32)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t1 =
        when (val tryResult = (t - 5.0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val tryResult = t1.inplaceOp1(Elu(alpha = 1.0))) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    check(
        when (val tryResult = toVec1Round(t1, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(-0.9933, -0.9817, -0.9502, -0.8647, -0.6321, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    )
    return Result.Ok(Unit)
}

public class InplaceAdd :
    InplaceOp2,
    InplaceOp3 {
    public fun name(): String = "inplace-add"

    public fun cpuFwd(s1: CpuStorage, l1: Layout, s2: CpuStorage, l2: Layout): Result<Unit> {
        val elems = l1.shape().elemCount()
        val start1 = l1.startOffset()
        val start2 = l2.startOffset()
        val (d1, d2) = Pair(s1, s2) ?: bail("unsupported dtype for inplace-add")
        val rhs: List<Float> = ((d2).slice(start2 until start2 + elems)).toVec()
        for ((v, r) in (((d1).slice(start1 until start1 + elems)).zip(rhs))) {
            v += r
        }
        return Result.Ok(Unit)
    }

    public fun nameInplaceOp3(): String = "inplace-add3"

    public fun cpuFwd(arg0: CpuStorage, arg1: Layout, arg2: CpuStorage, arg3: Layout, arg4: CpuStorage, arg5: Layout): Result<Unit> = Result.Ok(Unit)

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<InplaceAdd> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<InplaceAdd> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): InplaceAdd =
                    InplaceAdd()
                override fun writeNative(addr: Long, value: InplaceAdd) {
                }
            }
    }
}

@kotlin.test.Test
public fun inplaceOpSharedStorageErrorsInsteadOfDeadlocking(): Result<Unit> {
    val cpu = Device.Cpu
    val t =
        when (val tryResult = Tensor.fromVec(mutableListOf(1.0f, 2.0, 3.0, 4.0), 4, cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst =
        when (val tryResult = t.narrow(0, 0, 2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val src =
        when (val tryResult = t.narrow(0, 2, 2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val res = dst.inplaceOp2(src, InplaceAdd())
    check(res.isErr()) { "shared-storage inplace_op2 must error" }
    val res1 = dst.inplaceOp3(src, src, InplaceAdd())
    res1.expectErr("shared-storage inplace_op3 must error")
    // Distinct storages keep working.
    val a =
        when (val tryResult = Tensor.fromVec(mutableListOf(1.0f, 2.0), 2, cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val b =
        when (val tryResult = Tensor.fromVec(mutableListOf(10.0f, 20.0), 2, cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val tryResult = a.inplaceOp2(b, InplaceAdd())) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    check(
        when (val tryResult = a.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(11.0, 22.0),
    )
    return Result.Ok(Unit)
}
