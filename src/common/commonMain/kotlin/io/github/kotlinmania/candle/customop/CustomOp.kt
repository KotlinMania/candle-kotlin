// port-lint: source custom_op.rs
package io.github.kotlinmania.candle.customop

import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.cudabackend.CudaStorage
import io.github.kotlinmania.candle.cudabackend.utils.S
import io.github.kotlinmania.candle.dummymetalbackend.MetalStorage
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.layout.LayoutRelation
import io.github.kotlinmania.candle.op.BackpropOp
import io.github.kotlinmania.candle.op.Op
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.storage.Storage
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.candle.tensor.fromStorage
import io.github.kotlinmania.rustlang.core.result.fromResidual

/**
 * Unary ops that can be defined in user-land.
 */
public interface CustomOp1 {
    public fun name(): String

    /**
     * The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cpuFwd(storage: CpuStorage, layout: Layout): Result<Pair<CpuStorage, Shape>>

    /**
     * The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cudaFwd(storage: CudaStorage, layout: Layout): Result<Pair<CudaStorage, Shape>> = Err(Error.Cuda("no cuda implementation for ${this.name()}".into()))

    /**
     * The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun metalFwd(storage: MetalStorage, layout: Layout): Result<Pair<MetalStorage, Shape>> = Err(Error.Metal("no metal implementation for ${this.name()}".into()))

    /**
     * This function takes as argument the argument `arg` used in the forward pass, the result
     * produced by the forward operation `res` and the gradient of the result `gradRes`.
     * The function should return the gradient of the argument.
     */
    public fun bwd(arg: Tensor, res: Tensor, gradRes: Tensor): Result<Tensor?> = Err(Error.BackwardNotSupported(op = this.name()))
}

public interface CustomOp2 {
    public fun name(): String

    /**
     * The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cpuFwd(s1: CpuStorage, l1: Layout, s2: CpuStorage, l2: Layout): Result<Pair<CpuStorage, Shape>>

    /**
     * The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cudaFwd(arg0: CudaStorage, arg1: Layout, arg2: CudaStorage, arg3: Layout): Result<Pair<CudaStorage, Shape>> = Err(Error.Cuda("no cuda implementation for ${this.name()}".into()))

    /**
     * The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun metalFwd(arg0: MetalStorage, arg1: Layout, arg2: MetalStorage, arg3: Layout): Result<Pair<MetalStorage, Shape>> = Err(Error.Metal("no metal implementation for ${this.name()}".into()))

    public fun bwd(arg1: Tensor, arg2: Tensor, res: Tensor, gradRes: Tensor): Result<Pair<Tensor?, Tensor?>> = Err(Error.BackwardNotSupported(op = this.name()))
}

public interface CustomOp3 {
    public fun name(): String

    /**
     * The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cpuFwd(s1: CpuStorage, l1: Layout, s2: CpuStorage, l2: Layout, s3: CpuStorage, l3: Layout): Result<Pair<CpuStorage, Shape>>

    /**
     * The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cudaFwd(arg0: CudaStorage, arg1: Layout, arg2: CudaStorage, arg3: Layout, arg4: CudaStorage, arg5: Layout): Result<Pair<CudaStorage, Shape>> = Err(Error.Cuda("no cuda implementation for ${this.name()}".into()))

    /**
     * The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun metalFwd(arg0: MetalStorage, arg1: Layout, arg2: MetalStorage, arg3: Layout, arg4: MetalStorage, arg5: Layout): Result<Pair<MetalStorage, Shape>> = Err(Error.Metal("no metal implementation for ${this.name()}".into()))

    public fun bwd(arg1: Tensor, arg2: Tensor, arg3: Tensor, res: Tensor, gradRes: Tensor): Result<Triple<Tensor?, Tensor?, Tensor?>> = Err(Error.BackwardNotSupported(op = this.name()))
}

/**
 * Applies a unary custom op without backward support
 */
public fun <C : CustomOp1> Tensor.applyOp1NoBwd(c: C): Result<Tensor> {
    val (storage, shape) =
        when (val tryResult = this.storage().applyOp1(this.layout(), c)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    return Ok(fromStorage(storage, shape, BackpropOp.none(), false))
}

/**
 * Applies a binary custom op without backward support
 */
public fun <C : CustomOp2> Tensor.applyOp2NoBwd(rhs: Tensor, c: C): Result<Tensor> {
    val (storage, shape) =
        when (val tryResult = this.storage().applyOp2(this.layout(), rhs.storage(), rhs.layout(), c)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    return Ok(fromStorage(storage, shape, BackpropOp.none(), false))
}

/**
 * Applies a ternary custom op without backward support
 */
public fun <C : CustomOp3> Tensor.applyOp3NoBwd(t2: Tensor, t3: Tensor, c: C): Result<Tensor> {
    val (storage, shape) =
        when (val tryResult = this.storage().applyOp3(this.layout(), t2.storage(), t2.layout(), t3.storage(), t3.layout(), c)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    return Ok(fromStorage(storage, shape, BackpropOp.none(), false))
}

/**
 * Applies a unary custom op.
 */
public fun Tensor.applyOp1Arc(c: CustomOp1): Result<Tensor> {
    val (storage, shape) =
        when (val tryResult = this.storage().applyOp1(this.layout(), c.asRef().asRef())) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val op = BackpropOp.new1(this, { s -> Op.CustomOp1(s, c.clone()) })
    return Ok(fromStorage(storage, shape, op, false))
}

public fun <C : CustomOp1> Tensor.applyOp1(c: C): Result<Tensor> = this.applyOp1Arc(c)

/**
 * Applies a binary custom op.
 */
public fun Tensor.applyOp2Arc(rhs: Tensor, c: CustomOp2): Result<Tensor> {
    val (storage, shape) =
        when (val tryResult = this.storage().applyOp2(this.layout(), rhs.storage(), rhs.layout(), c.asRef().asRef())) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val op = BackpropOp.new2(this, rhs, { t1, t2 -> Op.CustomOp2(t1, t2, c.clone()) })
    return Ok(fromStorage(storage, shape, op, false))
}

public fun <C : CustomOp2> Tensor.applyOp2(r: Tensor, c: C): Result<Tensor> = this.applyOp2Arc(r, c)

/**
 * Applies a ternary custom op.
 */
public fun Tensor.applyOp3Arc(t2: Tensor, t3: Tensor, c: CustomOp3): Result<Tensor> {
    val (storage, shape) =
        when (val tryResult = this.storage().applyOp3(this.layout(), t2.storage(), t2.layout(), t3.storage(), t3.layout(), c.asRef().asRef())) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val op =
        BackpropOp.new3(this, t2, t3, { t1, t21, t31 ->
            Op.CustomOp3(t1, t21, t31, c.clone())
        })
    return Ok(fromStorage(storage, shape, op, false))
}

public fun <C : CustomOp3> Tensor.applyOp3(t2: Tensor, t3: Tensor, c: C): Result<Tensor> = this.applyOp3Arc(t2, t3, c)

// In place ops.

/**
 * Unary ops that can be defined in user-land.
 * These ops work in place and as such back-prop is unsupported.
 */
public interface InplaceOp1 {
    public fun name(): String

    /**
     * The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cpuFwd(storage: CpuStorage, layout: Layout): Result<Unit>

    /**
     * The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cudaFwd(storage: CudaStorage, layout: Layout): Result<Unit> = Err(Error.Cuda("no cuda implementation for ${this.name()}".into()))

    /**
     * The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun metalFwd(storage: MetalStorage, layout: Layout): Result<Unit> = Err(Error.Metal("no metal implementation for ${this.name()}".into()))
}

/**
 * In-place ops that can be defined in user-land.
 * These ops work in-place and as such back-propagation is unsupported.
 */
public interface InplaceOpN {
    public fun name(): String

    /**
     * Defines the source access pattern of this in-place op.
     * Defaults to `None`, which rejects all in-place source aliasing.
     */
    public fun srcAccessPattern(): AccessPattern? = null

    public fun cpuFwd(dst: CpuStorage, dstL: Layout, srcs: List<Pair<CpuStorage, Layout>>): Result<Unit> = bail("no cpu implementation for {}", this.name())

    public fun cpuFwdAliased(dst: CpuStorage, dstL: Layout, srcs: List<Pair<Src<CpuStorage>, Layout>>): Result<Unit> = bail("no aliased cpu implementation for {}", this.name())

    public fun cudaFwd(dst: CudaStorage, dstL: Layout, srcs: List<Pair<CudaStorage, Layout>>): Result<Unit> = bail("no cuda implementation for {}", this.name())

    public fun cudaFwdAliased(dst: CudaStorage, dstL: Layout, srcs: List<Pair<Src<CudaStorage>, Layout>>): Result<Unit> = bail("no aliased cpu implementation for {}", this.name())

    public fun metalFwd(dst: MetalStorage, dstL: Layout, srcs: List<Pair<MetalStorage, Layout>>): Result<Unit> = bail("no metal implementation for {}", this.name())

    public fun metalFwdAliased(dst: MetalStorage, dstL: Layout, srcs: List<Pair<Src<MetalStorage>, Layout>>): Result<Unit> = bail("no aliased metal implementation for {}", this.name())
}

public sealed class Src<S> {
    public data class Distinct<S>(
        public var value: S,
    ) : Src<S>()
    public data class Aliased<S>(
        public var value: LayoutRelation,
    ) : Src<S>()

    public fun clone(): Src<S> = this
}

// If all sources are distinct we return the entire array of tensors without the `Src` wrapper.
internal fun <B> allDistinct(srcs: List<Pair<Src<B>, Layout>>, n: Int): List<Pair<B, Layout>>? =
    srcs
        .all({ (s, _) ->
            when (s) {
                is Src.Distinct -> true
                else -> false
            }
        })
        .then({
            fromFn({ i ->
                run {
                    val matchVal = srcs[i]
                    when {
                        matchVal.first is Src.Distinct -> {
                            val (s) = matchVal.first as Src.Distinct<*>
                            val l = matchVal.second
                            Pair(s, l)
                        }
                        else -> error("checked immediately above")
                    }
                }
            })
        })

/**
 * Indicates which indices a kernel reads, relative to the destination indices it writes.
 *
 * When used with [LayoutRelation] we can describe safe access patterns.
 * For example `Elementwise` is safe to use with both `LayoutRelation.Identical` and `LayoutRelation.Disjoint`,
 * while `Arbitrary` is only guaranteed to be safe with `LayoutRelation.Disjoint`.
 */
public enum class AccessPattern(
    public val value: UInt,
) {
    /**
     * Reads only source index `i` when writing destination index `i`.
     */
    Elementwise(0u),

    /**
     * Reads arbitrary source indices.
     */
    Arbitrary(1u),
    ;

    public fun supports(rel: LayoutRelation): Boolean =
        run {
            val matchVal = Pair(this, rel)
            when {
                matchVal.first == AccessPattern.Elementwise && (matchVal.second == LayoutRelation.Identical || matchVal.second == LayoutRelation.Disjoint) -> true
                matchVal.first == AccessPattern.Arbitrary && matchVal.second == LayoutRelation.Disjoint -> true
                else -> false
            }
        }
}

public fun <C : InplaceOp1> C.name(): String = this.name()

public fun <C : InplaceOp1> C.cpuFwd(dst: CpuStorage, dl: Layout, arg2: List<Pair<CpuStorage, Layout>>): Result<Unit> = this.cpuFwd(dst, dl)

public fun <C : InplaceOp1> C.cpuFwdAliased(dst: CpuStorage, dl: Layout, arg2: List<Pair<Src<CpuStorage>, Layout>>): Result<Unit> = this.cpuFwd(dst, dl)

public fun <C : InplaceOp1> C.cudaFwd(dst: CudaStorage, dl: Layout, arg2: List<Pair<CudaStorage, Layout>>): Result<Unit> = this.cudaFwd(dst, dl)

public fun <C : InplaceOp1> C.cudaFwdAliased(dst: CudaStorage, dl: Layout, arg2: List<Pair<Src<CudaStorage>, Layout>>): Result<Unit> = this.cudaFwd(dst, dl)

public fun <C : InplaceOp1> C.metalFwd(dst: MetalStorage, dl: Layout, arg2: List<Pair<MetalStorage, Layout>>): Result<Unit> = this.metalFwd(dst, dl)

public fun <C : InplaceOp1> C.metalFwdAliased(dst: MetalStorage, dl: Layout, arg2: List<Pair<Src<MetalStorage>, Layout>>): Result<Unit> = this.metalFwd(dst, dl)

public interface InplaceOp2 {
    public fun name(): String

    /**
     * The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cpuFwd(s1: CpuStorage, l1: Layout, s2: CpuStorage, l2: Layout): Result<Unit>

    /**
     * The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cudaFwd(s1: CudaStorage, l1: Layout, s2: CudaStorage, l2: Layout): Result<Unit> = Err(Error.Cuda("no cuda implementation for ${this.name()}".into()))

    /**
     * The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun metalFwd(s1: MetalStorage, l1: Layout, s2: MetalStorage, l2: Layout): Result<Unit> = Err(Error.Metal("no metal implementation for ${this.name()}".into()))
}

public interface InplaceOp3 {
    public fun name(): String

    /**
     * The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cpuFwd(s1: CpuStorage, l1: Layout, s2: CpuStorage, l2: Layout, s3: CpuStorage, l3: Layout): Result<Unit>

    /**
     * The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cudaFwd(arg0: CudaStorage, arg1: Layout, arg2: CudaStorage, arg3: Layout, arg4: CudaStorage, arg5: Layout): Result<Unit> = Err(Error.Cuda("no cuda implementation for ${this.name()}".into()))

    /**
     * The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun metalFwd(arg0: MetalStorage, arg1: Layout, arg2: MetalStorage, arg3: Layout, arg4: MetalStorage, arg5: Layout): Result<Unit> = Err(Error.Metal("no metal implementation for ${this.name()}".into()))
}

/**
 * Applies a custom op in-place for the `self` tensor.
 *
 * Tensors sharing underlying storage with `self` are classified and passed as [Src.Aliased].
 * Separate tensors are locked and passed as [Src.Distinct].
 */
public fun <C : InplaceOpN<N>> Tensor.inplaceOp(srcs: List<Tensor>, c: C, n: Int): Result<Unit> {
    val name = c.name()
    // Ensure writes cannot collide with themselves
    if (this.layout().hasInternalOverlap()) {
        bail("{name}: dst has repeated elements (zero-stride). Can not write in-place")
    }
    // Classify srcs wrt dst
    val access = c.srcAccessPattern()
    val rels: List<LayoutRelation> = List((N).toInt()) { null }
    for (i in 0 until N) {
        if (!this.sameStorage(srcs[i])) {
            continue
        }
        val rel = Layout.relation(this.layout(), srcs[i].layout())
        when {
            access != null && access.supports(rel) -> {
                val a = access
                rels[i] = rel
            }
            access != null -> {
                val a1 = access
                bail("src {i} shares storage with dst ({rel:?}), which is not supported for the access pattern of `{name}` ({a:?}).")
            }
            else -> bail("src {i} shares storage with dst, and `{name}` does not support aliased operands.")
        }
    }
    // Acquire locks in order sorted by `Tensor.storageKey`.
    // Avoids deadlock from cycle of waiting locks.
    val dstKey = this.storageKey()
    val order: IntArray = fromFn({ i1 -> i1 })
    order.sortUnstableByKey({ i1 -> srcs[i1].storageKey() })
    val guards: List<Option> = fromFn({ null })
    var dst: Option? = null
    for (i1 in order) {
        if (rels[i1].isSome()) {
            continue
            // Aliased. Read through `dst`
        }
        val key = srcs[i1].storageKey()
        if (key > dstKey && dst == null) {
            dst = this.storageMut()
        }
        // Two sources sharing the same allocation, but distinct from dst.
        // `readRecursive` allows for shared read access
        guards[i1] = srcs[i1].storage()
    }
    // If `dst` is not yet set we acquire it from `self` now.
    var dst1 =
        when (dst) {
            null -> this.storageMut()
            else -> {
                val g = dst
                g
            }
        }
    val operands: List<Pair<Src<Storage>, Layout>> =
        fromFn({ i2 ->
            val s =
                run {
                    val matchVal = Pair(guards[i2], rels[i2])
                    when {
                        matchVal.first != null && matchVal.second == null -> {
                            val g1 = matchVal.first
                            Src.Distinct(g1)
                        }
                        matchVal.first == null && matchVal.second != null -> {
                            val rel1 = matchVal.second
                            Src.Aliased(rel1)
                        }
                        else -> error("Source is either distinct or aliased. Other match patterns should be impossible")
                    }
                }
            Pair(s, srcs[i2].layout())
        })
    return dst1.inplaceOp(this.layout(), operands, c)
}

/**
 * Applies a unary custom op in place.
 */
public fun <C : InplaceOp1> Tensor.inplaceOp1(c: C): Result<Unit> = this.inplaceOp(listOf(), c)

/**
 * Applies a binary custom op in place (for the first tensor).
 */
public fun <C : InplaceOpN> Tensor.inplaceOp2(rhs: Tensor, c: C): Result<Unit> = this.inplaceOp(listOf(rhs), c)

/**
 * Applies a ternary custom op in place (for the first tensor).
 */
public fun <C : InplaceOpN> Tensor.inplaceOp3(t2: Tensor, t3: Tensor, c: C): Result<Unit> = this.inplaceOp(listOf(t2, t3), c)
