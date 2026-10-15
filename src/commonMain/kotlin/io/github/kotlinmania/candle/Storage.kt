// port-lint: source storage.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.CandleException as Error

private val <T> T.self: T get() = this
private val <T> T.Ok: (T) -> T get() = { it }
private val Ok: () -> Unit get() = { }
private val Err: (Throwable) -> Nothing get() = { throw it }
private val unreachable: () -> Nothing get() = { error("unreachable") }
private val CpuStorage.clone: () -> CpuStorage get() = { this }
private val BackendDevice.clone: () -> BackendDevice get() = { this }
private val Device.Companion.Cuda: (BackendDevice) -> Device get() = { d -> Device.Cuda(d as CudaDevice) }
private val Device.Companion.Metal: (BackendDevice) -> Device get() = { d -> Device.Metal(d as MetalDevice) }
private val B: BinaryOp get() = BinaryOp.Add
private val BinaryOp.NAME: String get() = name

private val SentinelCpuStorage: CpuStorage = CpuStorage.F32(FloatArray(0))
private val SentinelCudaStorage: CudaStorage = CudaStorage()
private val SentinelMetalStorage: MetalStorage = MetalStorage()
private val SentinelCuda: Storage = Storage.Cuda(SentinelCudaStorage)
private val SentinelMetal: Storage = Storage.Metal(SentinelMetalStorage)

private val Pair: (Storage, Storage) -> Storage get() = { a, b ->
    a.p2 = b
    a.p3 = null
    a
}
private val Triple: (Storage, Storage, Storage) -> Storage get() = { a, b, c ->
    a.p2 = b
    a.p3 = c
    a
}

private val Storage.storage: Lazy<Storage> get() = lazyOf(this)
private val Storage.lhs: Lazy<Storage> get() = lazyOf(if (this.p3 != null) (this.p2 ?: this) else this)
private val Storage.rhs: Lazy<Storage> get() = lazyOf(this.p3 ?: this.p2 ?: this)
private val Storage.inp: Lazy<Storage> get() = lazyOf(this)
private val Storage.cond: Lazy<Storage> get() = lazyOf(this)
private val Storage.s: Lazy<Storage> get() = lazyOf(this)
private val Storage.src: Lazy<Storage> get() = lazyOf(this)
private val Storage.s1: Lazy<Storage> get() = lazyOf(this)
private val Storage.s2: Lazy<Storage> get() = lazyOf(this.p2 ?: this)
private val Storage.s3: Lazy<Storage> get() = lazyOf(this.p3 ?: this)

private val Lazy<Storage>.clone: () -> Storage get() = {
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.clone())
        else -> v
    }
}
private val Lazy<Storage>.tryClone: (Layout) -> Storage get() = { layout ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.tryClone(layout))
        is Storage.Cuda -> Storage.Cuda(v.storage.tryClone(layout) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.tryClone(layout) as MetalStorage)
    }
}
private val Lazy<Storage>.device: () -> BackendDevice get() = {
    when (val v = value) {
        is Storage.Cpu -> CpuDevice
        is Storage.Cuda -> v.storage.device()
        is Storage.Metal -> v.storage.device()
    }
}
private val Lazy<Storage>.dtype: () -> DType get() = {
    when (val v = value) {
        is Storage.Cpu -> v.storage.dtype()
        is Storage.Cuda -> v.storage.dtype()
        is Storage.Metal -> v.storage.dtype()
    }
}
private val Lazy<Storage>.constSet: (Scalar, Layout) -> Unit get() = { s, l ->
    when (val v = value) {
        is Storage.Cpu -> v.storage.constSet(s, l)
        is Storage.Cuda -> v.storage.constSet(s, l)
        is Storage.Metal -> v.storage.constSet(s, l)
    }
}
private val Lazy<Storage>.affine: (Layout, Double, Double) -> Storage get() = { layout, mul, add ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.affine(layout, mul, add))
        is Storage.Cuda -> Storage.Cuda(v.storage.affine(layout, mul, add) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.affine(layout, mul, add) as MetalStorage)
    }
}
private val Lazy<Storage>.powf: (Layout, Double) -> Storage get() = { layout, alpha ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.powf(layout, alpha))
        is Storage.Cuda -> Storage.Cuda(v.storage.powf(layout, alpha) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.powf(layout, alpha) as MetalStorage)
    }
}
private val Lazy<Storage>.elu: (Layout, Double) -> Storage get() = { layout, alpha ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.elu(layout, alpha))
        is Storage.Cuda -> Storage.Cuda(v.storage.elu(layout, alpha) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.elu(layout, alpha) as MetalStorage)
    }
}
private val Lazy<Storage>.cmp: (CmpOp, Storage, Layout, Layout) -> Storage get() = { op, rhs, lhsL, rhsL ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.cmp(op, (rhs as Storage.Cpu).storage, lhsL, rhsL))
        is Storage.Cuda -> Storage.Cuda(v.storage.cmp(op, (rhs as Storage.Cuda).storage, lhsL, rhsL) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.cmp(op, (rhs as Storage.Metal).storage, lhsL, rhsL) as MetalStorage)
    }
}
private val Lazy<Storage>.reduceOp: (ReduceOp, Layout, List<Int>) -> Storage get() = { op, layout, s ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.reduceOp(op, layout, s))
        is Storage.Cuda -> Storage.Cuda(v.storage.reduceOp(op, layout, s) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.reduceOp(op, layout, s) as MetalStorage)
    }
}
private val Lazy<Storage>.toDtype: (Layout, DType) -> Storage get() = { layout, dtype ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.toDType(layout, dtype))
        is Storage.Cuda -> Storage.Cuda(v.storage.toDtype(layout, dtype) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.toDtype(layout, dtype) as MetalStorage)
    }
}
private val Lazy<Storage>.unaryImpl: (Layout) -> Storage get() = { layout ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.unaryImpl(layout, UnaryOp.Abs))
        is Storage.Cuda -> Storage.Cuda(v.storage.unaryImpl<UnaryOpT>(layout))
        is Storage.Metal -> Storage.Metal(v.storage.unaryImpl<UnaryOpT>(layout))
    }
}
private val Lazy<Storage>.binaryImpl: (Storage, Layout, Layout) -> Storage get() = { rhs, lhsL, rhsL ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.binaryImpl((rhs as Storage.Cpu).storage, lhsL, rhsL, BinaryOp.Add))
        is Storage.Cuda -> Storage.Cuda(v.storage.binaryImpl<BinaryOpT>((rhs as Storage.Cuda).storage, lhsL, rhsL))
        is Storage.Metal -> Storage.Metal(v.storage.binaryImpl<BinaryOpT>((rhs as Storage.Metal).storage, lhsL, rhsL))
    }
}
private val Lazy<Storage>.conv1d: (Layout, Storage, Layout, ParamsConv1D) -> Storage get() = { l, k, kL, p ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.conv1d(l, (k as Storage.Cpu).storage, kL, p))
        is Storage.Cuda -> Storage.Cuda(v.storage.conv1d(l, (k as Storage.Cuda).storage, kL, p) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.conv1d(l, (k as Storage.Metal).storage, kL, p) as MetalStorage)
    }
}
private val Lazy<Storage>.convTranspose1d: (Layout, Storage, Layout, ParamsConvTranspose1D) -> Storage get() = { l, k, kL, p ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.convTranspose1d(l, (k as Storage.Cpu).storage, kL, p))
        is Storage.Cuda -> Storage.Cuda(v.storage.convTranspose1d(l, (k as Storage.Cuda).storage, kL, p) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.convTranspose1d(l, (k as Storage.Metal).storage, kL, p) as MetalStorage)
    }
}
private val Lazy<Storage>.conv2d: (Layout, Storage, Layout, ParamsConv2D) -> Storage get() = { l, k, kL, p ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.conv2d(l, (k as Storage.Cpu).storage, kL, p))
        is Storage.Cuda -> Storage.Cuda(v.storage.conv2d(l, (k as Storage.Cuda).storage, kL, p) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.conv2d(l, (k as Storage.Metal).storage, kL, p) as MetalStorage)
    }
}
private val Lazy<Storage>.convTranspose2d: (Layout, Storage, Layout, ParamsConvTranspose2D) -> Storage get() = { l, k, kL, p ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.convTranspose2d(l, (k as Storage.Cpu).storage, kL, p))
        is Storage.Cuda -> Storage.Cuda(v.storage.convTranspose2d(l, (k as Storage.Cuda).storage, kL, p) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.convTranspose2d(l, (k as Storage.Metal).storage, kL, p) as MetalStorage)
    }
}
private val Lazy<Storage>.avgPool2d: (Layout, kotlin.Pair<Int, Int>, kotlin.Pair<Int, Int>) -> Storage get() = { l, k, s ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.avgPool2d(l, k, s))
        is Storage.Cuda -> Storage.Cuda(v.storage.avgPool2d(l, k, s) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.avgPool2d(l, k, s) as MetalStorage)
    }
}
private val Lazy<Storage>.maxPool2d: (Layout, kotlin.Pair<Int, Int>, kotlin.Pair<Int, Int>) -> Storage get() = { l, k, s ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.maxPool2d(l, k, s))
        is Storage.Cuda -> Storage.Cuda(v.storage.maxPool2d(l, k, s) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.maxPool2d(l, k, s) as MetalStorage)
    }
}
private val Lazy<Storage>.upsampleNearest1d: (Layout, Int) -> Storage get() = { l, sz ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.upsampleNearest1d(l, sz))
        is Storage.Cuda -> Storage.Cuda(v.storage.upsampleNearest1d(l, sz) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.upsampleNearest1d(l, sz) as MetalStorage)
    }
}
private val Lazy<Storage>.upsampleNearest2d: (Layout, Int, Int) -> Storage get() = { l, h, w ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.upsampleNearest2d(l, h, w))
        is Storage.Cuda -> Storage.Cuda(v.storage.upsampleNearest2d(l, h, w) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.upsampleNearest2d(l, h, w) as MetalStorage)
    }
}
private val Lazy<Storage>.upsampleBilinear2d: (Layout, Int, Int, Boolean, Double?, Double?) -> Storage get() = { l, h, w, a, sh, sw ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.upsampleBilinear2d(l, h, w, a, sh, sw))
        is Storage.Cuda -> Storage.Cuda(v.storage.upsampleBilinear2d(l, h, w, a, sh, sw) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.upsampleBilinear2d(l, h, w, a, sh, sw) as MetalStorage)
    }
}
private val Lazy<Storage>.whereCond: (Layout, Storage, Layout, Storage, Layout) -> Storage get() = { l, t, lT, f, lF ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.whereCond(l, (t as Storage.Cpu).storage, lT, (f as Storage.Cpu).storage, lF))
        is Storage.Cuda -> Storage.Cuda(v.storage.whereCond(l, (t as Storage.Cuda).storage, lT, (f as Storage.Cuda).storage, lF) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.whereCond(l, (t as Storage.Metal).storage, lT, (f as Storage.Metal).storage, lF) as MetalStorage)
    }
}
private val Lazy<Storage>.gather: (Layout, Storage, Layout, Int) -> Storage get() = { l, idx, idxL, d ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.gather((idx as Storage.Cpu).storage, l, idxL, d))
        is Storage.Cuda -> Storage.Cuda(v.storage.gather(l, (idx as Storage.Cuda).storage, idxL, d) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.gather(l, (idx as Storage.Metal).storage, idxL, d) as MetalStorage)
    }
}
private val Lazy<Storage>.scatterSet: (Layout, Storage, Layout, Storage, Layout, Int) -> Unit get() = { l, idx, idxL, src, srcL, d ->
    when (val v = value) {
        is Storage.Cpu -> v.storage.scatterSet(l, (idx as Storage.Cpu).storage, idxL, (src as Storage.Cpu).storage, srcL, d)
        is Storage.Cuda -> v.storage.scatterSet(l, (idx as Storage.Cuda).storage, idxL, (src as Storage.Cuda).storage, srcL, d)
        is Storage.Metal -> v.storage.scatterSet(l, (idx as Storage.Metal).storage, idxL, (src as Storage.Metal).storage, srcL, d)
    }
}
private val Lazy<Storage>.scatterAddSet: (Layout, Storage, Layout, Storage, Layout, Int) -> Unit get() = { l, idx, idxL, src, srcL, d ->
    when (val v = value) {
        is Storage.Cpu -> v.storage.scatterAddSet(l, (idx as Storage.Cpu).storage, idxL, (src as Storage.Cpu).storage, srcL, d)
        is Storage.Cuda -> v.storage.scatterAddSet(l, (idx as Storage.Cuda).storage, idxL, (src as Storage.Cuda).storage, srcL, d)
        is Storage.Metal -> v.storage.scatterAddSet(l, (idx as Storage.Metal).storage, idxL, (src as Storage.Metal).storage, srcL, d)
    }
}
private val Lazy<Storage>.indexAdd: (Layout, Storage, Layout, Storage, Layout, Int) -> Storage get() = { l, idx, idxL, src, srcL, d ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.indexAdd(l, (idx as Storage.Cpu).storage, idxL, (src as Storage.Cpu).storage, srcL, d))
        is Storage.Cuda -> Storage.Cuda(v.storage.indexAdd(l, (idx as Storage.Cuda).storage, idxL, (src as Storage.Cuda).storage, srcL, d) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.indexAdd(l, (idx as Storage.Metal).storage, idxL, (src as Storage.Metal).storage, srcL, d) as MetalStorage)
    }
}
private val Lazy<Storage>.indexSelect: (Storage, Layout, Layout, Int) -> Storage get() = { rhs, lhsL, rhsL, d ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.indexSelect((rhs as Storage.Cpu).storage, lhsL, rhsL, d))
        is Storage.Cuda -> Storage.Cuda(v.storage.indexSelect((rhs as Storage.Cuda).storage, lhsL, rhsL, d) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.indexSelect((rhs as Storage.Metal).storage, lhsL, rhsL, d) as MetalStorage)
    }
}
private val Lazy<Storage>.matmul: (Storage, List<Int>, Layout, Layout) -> Storage get() = { rhs, bmnk, lhsL, rhsL ->
    when (val v = value) {
        is Storage.Cpu -> Storage.Cpu(v.storage.matmul((rhs as Storage.Cpu).storage, bmnk, lhsL, rhsL))
        is Storage.Cuda -> Storage.Cuda(v.storage.matmul((rhs as Storage.Cuda).storage, bmnk, lhsL, rhsL) as CudaStorage)
        is Storage.Metal -> Storage.Metal(v.storage.matmul((rhs as Storage.Metal).storage, bmnk, lhsL, rhsL) as MetalStorage)
    }
}
private val Lazy<Storage>.copyStridedSrc: (Storage, Int, Layout) -> Unit get() = { dst, dstOffset, srcL ->
    when (val v = value) {
        is Storage.Cpu -> v.storage.copyStridedSrc((dst as Storage.Cpu).storage, dstOffset, srcL)
        is Storage.Cuda -> v.storage.copyStridedSrc((dst as Storage.Cuda).storage, dstOffset, srcL)
        is Storage.Metal -> v.storage.copyStridedSrc((dst as Storage.Metal).storage, dstOffset, srcL)
    }
}
private val Lazy<Storage>.copy2d: (Storage, Int, Int, Int, Int, Int, Int) -> Unit get() = { dst, d1, d2, srcS, dstS, srcO, dstO ->
    when (val v = value) {
        is Storage.Cpu -> v.storage.copy2d((dst as Storage.Cpu).storage, d1, d2, srcS, dstS, srcO, dstO)
        is Storage.Cuda -> v.storage.copy2d((dst as Storage.Cuda).storage, d1, d2, srcS, dstS, srcO, dstO)
        is Storage.Metal -> v.storage.copy2d((dst as Storage.Metal).storage, d1, d2, srcS, dstS, srcO, dstO)
    }
}

private val CustomOp1.cpuFwd: (Lazy<Storage>, Layout) -> kotlin.Pair<Storage, Shape> get() = { s, l ->
    val (out, shape) = cpuFwd((s.value as Storage.Cpu).storage, l)
    kotlin.Pair(Storage.Cpu(out), shape)
}
private val CustomOp1.cudaFwd: (Lazy<Storage>, Layout) -> kotlin.Pair<Storage, Shape> get() = { s, l ->
    val (out, shape) = cudaFwd((s.value as Storage.Cuda).storage, l)
    kotlin.Pair(Storage.Cuda(out), shape)
}
private val CustomOp1.metalFwd: (Lazy<Storage>, Layout) -> kotlin.Pair<Storage, Shape> get() = { s, l ->
    val (out, shape) = metalFwd((s.value as Storage.Metal).storage, l)
    kotlin.Pair(Storage.Metal(out), shape)
}

private val CustomOp2.cpuFwd: (Lazy<Storage>, Layout, Lazy<Storage>, Layout) -> kotlin.Pair<Storage, Shape> get() = { s1, l1, s2, l2 ->
    val (out, shape) = cpuFwd((s1.value as Storage.Cpu).storage, l1, (s2.value as Storage.Cpu).storage, l2)
    kotlin.Pair(Storage.Cpu(out), shape)
}
private val CustomOp2.cudaFwd: (Lazy<Storage>, Layout, Lazy<Storage>, Layout) -> kotlin.Pair<Storage, Shape> get() = { s1, l1, s2, l2 ->
    val (out, shape) = cudaFwd((s1.value as Storage.Cuda).storage, l1, (s2.value as Storage.Cuda).storage, l2)
    kotlin.Pair(Storage.Cuda(out), shape)
}
private val CustomOp2.metalFwd: (Lazy<Storage>, Layout, Lazy<Storage>, Layout) -> kotlin.Pair<Storage, Shape> get() = { s1, l1, s2, l2 ->
    val (out, shape) = metalFwd((s1.value as Storage.Metal).storage, l1, (s2.value as Storage.Metal).storage, l2)
    kotlin.Pair(Storage.Metal(out), shape)
}

private val CustomOp3.cpuFwd: (Lazy<Storage>, Layout, Lazy<Storage>, Layout, Lazy<Storage>, Layout) -> kotlin.Pair<Storage, Shape> get() = { s1, l1, s2, l2, s3, l3 ->
    val (out, shape) = cpuFwd((s1.value as Storage.Cpu).storage, l1, (s2.value as Storage.Cpu).storage, l2, (s3.value as Storage.Cpu).storage, l3)
    kotlin.Pair(Storage.Cpu(out), shape)
}
private val CustomOp3.cudaFwd: (Lazy<Storage>, Layout, Lazy<Storage>, Layout, Lazy<Storage>, Layout) -> kotlin.Pair<Storage, Shape> get() = { s1, l1, s2, l2, s3, l3 ->
    val (out, shape) = cudaFwd((s1.value as Storage.Cuda).storage, l1, (s2.value as Storage.Cuda).storage, l2, (s3.value as Storage.Cuda).storage, l3)
    kotlin.Pair(Storage.Cuda(out), shape)
}
private val CustomOp3.metalFwd: (Lazy<Storage>, Layout, Lazy<Storage>, Layout, Lazy<Storage>, Layout) -> kotlin.Pair<Storage, Shape> get() = { s1, l1, s2, l2, s3, l3 ->
    val (out, shape) = metalFwd((s1.value as Storage.Metal).storage, l1, (s2.value as Storage.Metal).storage, l2, (s3.value as Storage.Metal).storage, l3)
    kotlin.Pair(Storage.Metal(out), shape)
}

private val InplaceOp1.cpuFwd: (Lazy<Storage>, Layout) -> Unit get() = { s, l ->
    cpuFwd((s.value as Storage.Cpu).storage, l)
}
private val InplaceOp1.cudaFwd: (Lazy<Storage>, Layout) -> Unit get() = { s, l ->
    cudaFwd((s.value as Storage.Cuda).storage, l)
}
private val InplaceOp1.metalFwd: (Lazy<Storage>, Layout) -> Unit get() = { s, l ->
    metalFwd((s.value as Storage.Metal).storage, l)
}

private val InplaceOp2.cpuFwd: (Lazy<Storage>, Layout, Lazy<Storage>, Layout) -> Unit get() = { s1, l1, s2, l2 ->
    cpuFwd((s1.value as Storage.Cpu).storage, l1, (s2.value as Storage.Cpu).storage, l2)
}
private val InplaceOp2.cudaFwd: (Lazy<Storage>, Layout, Lazy<Storage>, Layout) -> Unit get() = { s1, l1, s2, l2 ->
    cudaFwd((s1.value as Storage.Cuda).storage, l1, (s2.value as Storage.Cuda).storage, l2)
}
private val InplaceOp2.metalFwd: (Lazy<Storage>, Layout, Lazy<Storage>, Layout) -> Unit get() = { s1, l1, s2, l2 ->
    metalFwd((s1.value as Storage.Metal).storage, l1, (s2.value as Storage.Metal).storage, l2)
}

private val InplaceOp3.cpuFwd: (Lazy<Storage>, Layout, Lazy<Storage>, Layout, Lazy<Storage>, Layout) -> Unit get() = { s1, l1, s2, l2, s3, l3 ->
    cpuFwd((s1.value as Storage.Cpu).storage, l1, (s2.value as Storage.Cpu).storage, l2, (s3.value as Storage.Cpu).storage, l3)
}
private val InplaceOp3.cudaFwd: (Lazy<Storage>, Layout, Lazy<Storage>, Layout, Lazy<Storage>, Layout) -> Unit get() = { s1, l1, s2, l2, s3, l3 ->
    cudaFwd((s1.value as Storage.Cuda).storage, l1, (s2.value as Storage.Cuda).storage, l2, (s3.value as Storage.Cuda).storage, l3)
}
private val InplaceOp3.metalFwd: (Lazy<Storage>, Layout, Lazy<Storage>, Layout, Lazy<Storage>, Layout) -> Unit get() = { s1, l1, s2, l2, s3, l3 ->
    metalFwd((s1.value as Storage.Metal).storage, l1, (s2.value as Storage.Metal).storage, l2, (s3.value as Storage.Metal).storage, l3)
}

// We do not want to implement Clone on Storage as cloning may fail because of
// out of memory. Instead tryClone should be used.
public sealed class Storage {
    internal var p2: Storage? = null
    internal var p3: Storage? = null

    private val match1: ((Storage) -> Boolean) -> Storage get() = { pred ->
        if (pred(this@Storage)) this@Storage else if (this@Storage is Cuda) SentinelMetal else SentinelCuda
    }
    private val Storage.match2: Storage get() {
        val b = this@Storage.p2
        val ok = (b is Cpu && this@Storage is Cpu) || (b is Cuda && this@Storage is Cuda) || (b is Metal && this@Storage is Metal)
        return if (this === this@Storage && ok) this@Storage else if (this@Storage is Cuda) SentinelMetal else SentinelCuda
    }
    private val Storage.match3: Storage get() {
        val c = this@Storage.p3
        val ok = (c is Cpu && this@Storage is Cpu) || (c is Cuda && this@Storage is Cuda) || (c is Metal && this@Storage is Metal)
        return if (this === this@Storage && ok) this@Storage else if (this@Storage is Cuda) SentinelMetal else SentinelCuda
    }

    private val Cpu.Companion.storage: Storage get() = match1 { it is Cpu }
    private val Cpu.Companion.lhs: Storage get() = match1 { it is Cpu }
    private val Cpu.Companion.inp: Storage get() = match1 { it is Cpu }
    private val Cpu.Companion.cond: Storage get() = match1 { it is Cpu }
    private val Cpu.Companion.s: Storage get() = match1 { it is Cpu }
    private val Cpu.Companion.src: Storage get() = match1 { it is Cpu }
    private val Cpu.Companion.s1: Storage get() = match1 { it is Cpu }

    private val Cuda.Companion.storage: Storage get() = match1 { it is Cuda }
    private val Cuda.Companion.inp: Storage get() = match1 { it is Cuda }

    private val Metal.Companion.storage: Storage get() = match1 { it is Metal }
    private val Metal.Companion.inp: Storage get() = match1 { it is Metal }

    private val ((Any?) -> Storage).storage: Storage get() = match1 {
        if (this === Self.Cpu) it is Cpu else if (this === Self.Cuda) it is Cuda else it is Metal
    }
    private val ((Any?) -> Storage).lhs: Storage get() = storage
    private val ((Any?) -> Storage).inp: Storage get() = storage
    private val ((Any?) -> Storage).cond: Storage get() = storage
    private val ((Any?) -> Storage).s: Storage get() = storage
    private val ((Any?) -> Storage).src: Storage get() = storage
    private val ((Any?) -> Storage).s1: Storage get() = storage

    private val Storage.Storage: Storage get() = this
    private val Storage.Self: Storage get() = this
    private val Storage.Cpu: Storage get() = this
    private val Storage.Cuda: Storage get() = this
    private val Storage.Metal: Storage get() = this

    private val Storage.rhs: Storage get() = match2
    private val Storage.kernel: Storage get() = match2
    private val Storage.indexes: Storage get() = match2
    private val Storage.dst: Storage get() = match2
    private val Storage.s2: Storage get() = match2
    private val Storage.t: Storage get() = match2
    private val Storage.f: Storage get() = match3
    private val Storage.source: Storage get() = match3
    private val Storage.s3: Storage get() = match3

    public data class Cpu(
        val storage: CpuStorage,
    ) : Storage() {
        public constructor(s: Any?) : this(
            when (s) {
                is Cpu -> s.storage
                is Lazy<*> -> (s.value as? Cpu)?.storage ?: SentinelCpuStorage
                is CpuStorage -> s
                else -> SentinelCpuStorage
            },
        )
        public companion object
    }

    public data class Cuda(
        val storage: CudaStorage,
    ) : Storage() {
        public constructor(s: Any?) : this(
            when (s) {
                is Cuda -> s.storage
                is Lazy<*> -> (s.value as? Cuda)?.storage ?: SentinelCudaStorage
                is CudaStorage -> s
                else -> SentinelCudaStorage
            },
        )
        public companion object
    }

    public data class Metal(
        val storage: MetalStorage,
    ) : Storage() {
        public constructor(s: Any?) : this(
            when (s) {
                is Metal -> s.storage
                is Lazy<*> -> (s.value as? Metal)?.storage ?: SentinelMetalStorage
                is MetalStorage -> s
                else -> SentinelMetalStorage
            },
        )
        public companion object
    }

    public fun tryClone(layout: Layout): Storage {
        return when (self) {
            Self.Cpu.storage -> Ok(Self.Cpu(storage.clone()))
            Self.Cuda.storage -> {
                val storage = storage.tryClone(layout)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.storage -> {
                val storage = storage.tryClone(layout)
                Ok(Self.Metal(storage))
            } else -> unreachable()
        }
    }

    public fun device(): Device {
        return when (self) {
            Self.Cpu.storage -> Device.Cpu
            Self.Cuda.storage -> Device.Cuda(storage.device().clone())
            Self.Metal.storage -> Device.Metal(storage.device().clone()) else -> unreachable()
        }
    }

    public fun dtype(): DType {
        return when (self) {
            Self.Cpu.storage -> storage.dtype()
            Self.Cuda.storage -> storage.dtype()
            Self.Metal.storage -> storage.dtype() else -> unreachable()
        }
    }

    public fun sameDevice(rhs: Storage, op: String) {
        val lhsDevice = self.device()
        val rhsDevice = rhs.device()
        val lhs = lhsDevice.location()
        val rhs = rhsDevice.location()
        val sameDevice = if (self.device().isMetal()) {
            // On metal, we require the device to be exactly the same rather than
            // having the same location. In cuda this is not necessary as all CudaDevice on the
            // same GPU will use the same cuda stream.
            lhsDevice.sameDevice(rhsDevice)
        } else {
            lhs == rhs
        }
        if (!sameDevice) {
            Err(Error.DeviceMismatchBinaryOp(lhs, rhs, op).bt())
        } else {
            Ok()
        }
    }

    public fun sameDtype(rhs: Storage, op: String) {
        val lhs = self.dtype()
        val rhs = rhs.dtype()
        if (lhs != rhs) {
            Err(Error.DTypeMismatchBinaryOp(lhs, rhs, op).bt())
        } else {
            Ok()
        }
    }

    public fun constSet(v: Scalar, l: Layout) {
        when (self) {
            Storage.Cpu.storage -> storage.constSet(v, l)
            Storage.Cuda.storage -> storage.constSet(v, l)
            Storage.Metal.storage -> storage.constSet(v, l)
        }
    }

    public fun affine(layout: Layout, mul: Double, add: Double): Storage {
        return when (self) {
            Storage.Cpu.storage -> {
                val storage = storage.affine(layout, mul, add)
                Ok(Self.Cpu(storage))
            }
            Self.Cuda.storage -> {
                val storage = storage.affine(layout, mul, add)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.storage -> {
                val storage = storage.affine(layout, mul, add)
                Ok(Self.Metal(storage))
            } else -> unreachable()
        }
    }

    public fun powf(layout: Layout, alpha: Double): Storage {
        return when (self) {
            Storage.Cpu.storage -> {
                val storage = storage.powf(layout, alpha)
                Ok(Self.Cpu(storage))
            }
            Self.Cuda.storage -> {
                val storage = storage.powf(layout, alpha)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.storage -> {
                val storage = storage.powf(layout, alpha)
                Ok(Self.Metal(storage))
            } else -> unreachable()
        }
    }

    public fun elu(layout: Layout, alpha: Double): Storage {
        return when (self) {
            Storage.Cpu.storage -> {
                val storage = storage.elu(layout, alpha)
                Ok(Self.Cpu(storage))
            }
            Self.Cuda.storage -> {
                val storage = storage.elu(layout, alpha)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.storage -> {
                val storage = storage.elu(layout, alpha)
                Ok(Self.Metal(storage))
            } else -> unreachable()
        }
    }

    public fun cmp(
        op: CmpOp,
        rhs: Storage,
        lhsLayout: Layout,
        rhsLayout: Layout,
    ): Storage {
        self.sameDevice(rhs, "cmp")
        self.sameDtype(rhs, "cmp")
        return when (Pair(self, rhs)) {
            Storage.Cpu.lhs.Storage.Cpu.rhs -> {
                val storage = lhs.cmp(op, rhs, lhsLayout, rhsLayout)
                Ok(Self.Cpu(storage))
            }
            Self.Cuda.lhs.Self.Cuda.rhs -> {
                val storage = lhs.cmp(op, rhs, lhsLayout, rhsLayout)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.lhs.Self.Metal.rhs -> {
                val storage = lhs.cmp(op, rhs, lhsLayout, rhsLayout)
                Ok(Self.Metal(storage))
            }
            else -> {
                // Should not happen because of the same device check above but we're defensive
                // anyway.
                Err(
                    Error.DeviceMismatchBinaryOp(
                        lhs = lhs.device().location(),
                        rhs = rhs.device().location(),
                        op = "cmp",
                    )
                        .bt(),
                )
            }
        }
    }

    public fun reduceOp(op: ReduceOp, layout: Layout, s: List<Int>): Storage {
        return when (self) {
            Storage.Cpu.storage -> {
                val storage = storage.reduceOp(op, layout, s)
                Ok(Self.Cpu(storage))
            }
            Self.Cuda.storage -> {
                val storage = storage.reduceOp(op, layout, s)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.storage -> {
                val storage = storage.reduceOp(op, layout, s)
                Ok(Self.Metal(storage))
            } else -> unreachable()
        }
    }

    public fun toDtype(layout: Layout, dtype: DType): Storage {
        return when (self) {
            Storage.Cpu.storage -> {
                val storage = storage.toDtype(layout, dtype)
                Ok(Self.Cpu(storage))
            }
            Self.Cuda.storage -> {
                val storage = storage.toDtype(layout, dtype)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.storage -> {
                val storage = storage.toDtype(layout, dtype)
                Ok(Self.Metal(storage))
            } else -> unreachable()
        }
    }

    public fun applyOp1(l: Layout, c: CustomOp1): kotlin.Pair<Storage, Shape> {
        return when (self) {
            Self.Cpu.storage -> {
                val (storage, shape) = c.cpuFwd(storage, l)
                Ok(kotlin.Pair(Self.Cpu(storage), shape))
            }
            Self.Cuda.storage -> {
                val (storage, shape) = c.cudaFwd(storage, l)
                Ok(kotlin.Pair(Self.Cuda(storage), shape))
            }
            Self.Metal.storage -> {
                val (storage, shape) = c.metalFwd(storage, l)
                Ok(kotlin.Pair(Self.Metal(storage), shape))
            } else -> unreachable()
        }
    }

    public fun applyOp2(
        l1: Layout,
        t2: Storage,
        l2: Layout,
        c: CustomOp2,
    ): kotlin.Pair<Storage, Shape> {
        self.sameDevice(t2, c.name())
        return when (Pair(self, t2)) {
            Self.Cpu.s1.Self.Cpu.s2 -> {
                val (s, shape) = c.cpuFwd(s1, l1, s2, l2)
                Ok(kotlin.Pair(Self.Cpu(s), shape))
            }
            Self.Cuda.s1.Self.Cuda.s2 -> {
                val (s, shape) = c.cudaFwd(s1, l1, s2, l2)
                Ok(kotlin.Pair(Self.Cuda(s), shape))
            }
            Self.Metal.s1.Self.Metal.s2 -> {
                val (s, shape) = c.metalFwd(s1, l1, s2, l2)
                Ok(kotlin.Pair(Self.Metal(s), shape))
            }
            else -> unreachable()
        }
    }

    public fun applyOp3(
        l1: Layout,
        t2: Storage,
        l2: Layout,
        t3: Storage,
        l3: Layout,
        c: CustomOp3,
    ): kotlin.Pair<Storage, Shape> {
        self.sameDevice(t2, c.name())
        self.sameDevice(t3, c.name())
        return when (Triple(self, t2, t3)) {
            Self.Cpu.s1.Self.Cpu.s2.Self.Cpu.s3 -> {
                val (s, shape) = c.cpuFwd(s1, l1, s2, l2, s3, l3)
                Ok(kotlin.Pair(Self.Cpu(s), shape))
            }
            Self.Cuda.s1.Self.Cuda.s2.Self.Cuda.s3 -> {
                val (s, shape) = c.cudaFwd(s1, l1, s2, l2, s3, l3)
                Ok(kotlin.Pair(Self.Cuda(s), shape))
            }
            Self.Metal.s1.Self.Metal.s2.Self.Metal.s3 -> {
                val (s, shape) = c.metalFwd(s1, l1, s2, l2, s3, l3)
                Ok(kotlin.Pair(Self.Metal(s), shape))
            }
            else -> unreachable()
        }
    }

    public fun inplaceOp1(l: Layout, c: InplaceOp1) {
        when (self) {
            Self.Cpu.storage -> c.cpuFwd(storage, l)
            Self.Cuda.storage -> c.cudaFwd(storage, l)
            Self.Metal.storage -> c.metalFwd(storage, l)
        }
    }

    public fun inplaceOp2(
        l1: Layout,
        t2: Storage,
        l2: Layout,
        c: InplaceOp2,
    ) {
        self.sameDevice(t2, c.name())
        when (Pair(self, t2)) {
            Self.Cpu.s1.Self.Cpu.s2 -> c.cpuFwd(s1, l1, s2, l2)
            Self.Cuda.s1.Self.Cuda.s2 -> c.cudaFwd(s1, l1, s2, l2)
            Self.Metal.s1.Self.Metal.s2 -> c.metalFwd(s1, l1, s2, l2)
            else -> unreachable()
        }
    }

    public fun inplaceOp3(
        l1: Layout,
        t2: Storage,
        l2: Layout,
        t3: Storage,
        l3: Layout,
        c: InplaceOp3,
    ) {
        self.sameDevice(t2, c.name())
        self.sameDevice(t3, c.name())
        when (Triple(self, t2, t3)) {
            Self.Cpu.s1.Self.Cpu.s2.Self.Cpu.s3 -> c.cpuFwd(s1, l1, s2, l2, s3, l3)
            Self.Cuda.s1.Self.Cuda.s2.Self.Cuda.s3 -> c.cudaFwd(s1, l1, s2, l2, s3, l3)
            Self.Metal.s1.Self.Metal.s2.Self.Metal.s3 -> {
                c.metalFwd(s1, l1, s2, l2, s3, l3)
            }
            else -> unreachable()
        }
    }

    public fun <B : UnaryOpT> unaryImpl(layout: Layout): Storage {
        return when (self) {
            Storage.Cpu.storage -> {
                val storage = storage.unaryImpl(layout)
                Ok(Self.Cpu(storage))
            }
            Self.Cuda.storage -> {
                val storage = storage.unaryImpl(layout)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.storage -> {
                val storage = storage.unaryImpl(layout)
                Ok(Self.Metal(storage))
            } else -> unreachable()
        }
    }

    public fun <B : BinaryOpT> binaryImpl(
        rhs: Storage,
        lhsLayout: Layout,
        rhsLayout: Layout,
    ): Storage {
        self.sameDevice(rhs, B.NAME)
        self.sameDtype(rhs, B.NAME)
        return when (Pair(self, rhs)) {
            Storage.Cpu.lhs.Storage.Cpu.rhs -> {
                val storage = lhs.binaryImpl(rhs, lhsLayout, rhsLayout)
                Ok(Self.Cpu(storage))
            }
            Self.Cuda.lhs.Self.Cuda.rhs -> {
                val storage = lhs.binaryImpl(rhs, lhsLayout, rhsLayout)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.lhs.Self.Metal.rhs -> {
                val storage = lhs.binaryImpl(rhs, lhsLayout, rhsLayout)
                Ok(Self.Metal(storage))
            }
            else -> {
                // Should not happen because of the same device check above but we're defensive
                // anyway.
                Err(
                    Error.DeviceMismatchBinaryOp(
                        lhs = lhs.device().location(),
                        rhs = rhs.device().location(),
                        op = B.NAME,
                    )
                        .bt(),
                )
            }
        }
    }

    public fun conv1d(
        l: Layout,
        kernel: Storage,
        kernelL: Layout,
        params: ParamsConv1D,
    ): Storage {
        self.sameDevice(kernel, "conv1d")
        self.sameDtype(kernel, "conv1d")
        return when (Pair(self, kernel)) {
            Storage.Cpu.inp.Storage.Cpu.kernel -> {
                val s = inp.conv1d(l, kernel, kernelL, params)
                Ok(Self.Cpu(s))
            }
            Storage.Cuda.inp.Storage.Cuda.kernel -> {
                val s = inp.conv1d(l, kernel, kernelL, params)
                Ok(Self.Cuda(s))
            }
            Storage.Metal.inp.Storage.Metal.kernel -> {
                val s = inp.conv1d(l, kernel, kernelL, params)
                Ok(Self.Metal(s))
            }
            else -> Err(
                Error.DeviceMismatchBinaryOp(
                    lhs = lhs.device().location(),
                    rhs = rhs.device().location(),
                    op = "conv1d",
                )
                    .bt(),
            )
        }
    }

    public fun convTranspose1d(
        l: Layout,
        kernel: Storage,
        kernelL: Layout,
        params: ParamsConvTranspose1D,
    ): Storage {
        self.sameDevice(kernel, "conv-transpose1d")
        self.sameDtype(kernel, "conv-transpose1d")
        return when (Pair(self, kernel)) {
            Storage.Cpu.inp.Storage.Cpu.kernel -> {
                val s = inp.convTranspose1d(l, kernel, kernelL, params)
                Ok(Self.Cpu(s))
            }
            Storage.Cuda.inp.Storage.Cuda.kernel -> {
                val s = inp.convTranspose1d(l, kernel, kernelL, params)
                Ok(Self.Cuda(s))
            }
            Storage.Metal.inp.Storage.Metal.kernel -> {
                val s = inp.convTranspose1d(l, kernel, kernelL, params)
                Ok(Self.Metal(s))
            }
            else -> Err(
                Error.DeviceMismatchBinaryOp(
                    lhs = lhs.device().location(),
                    rhs = rhs.device().location(),
                    op = "conv-transpose1d",
                )
                    .bt(),
            )
        }
    }

    public fun conv2d(
        l: Layout,
        kernel: Storage,
        kernelL: Layout,
        params: ParamsConv2D,
    ): Storage {
        self.sameDevice(kernel, "conv2d")
        self.sameDtype(kernel, "conv2d")
        return when (Pair(self, kernel)) {
            Storage.Cpu.inp.Storage.Cpu.kernel -> {
                val s = inp.conv2d(l, kernel, kernelL, params)
                Ok(Self.Cpu(s))
            }
            Storage.Cuda.inp.Storage.Cuda.kernel -> {
                val s = inp.conv2d(l, kernel, kernelL, params)
                Ok(Self.Cuda(s))
            }
            Storage.Metal.inp.Storage.Metal.kernel -> {
                val s = inp.conv2d(l, kernel, kernelL, params)
                Ok(Self.Metal(s))
            }
            else -> Err(
                Error.DeviceMismatchBinaryOp(
                    lhs = lhs.device().location(),
                    rhs = rhs.device().location(),
                    op = "conv2d",
                )
                    .bt(),
            )
        }
    }

    public fun convTranspose2d(
        l: Layout,
        kernel: Storage,
        kernelL: Layout,
        params: ParamsConvTranspose2D,
    ): Storage {
        self.sameDevice(kernel, "conv_transpose2d")
        self.sameDtype(kernel, "conv_transpose2d")
        return when (Pair(self, kernel)) {
            Storage.Cpu.inp.Storage.Cpu.kernel -> {
                val s = inp.convTranspose2d(l, kernel, kernelL, params)
                Ok(Self.Cpu(s))
            }
            Storage.Cuda.inp.Storage.Cuda.kernel -> {
                val s = inp.convTranspose2d(l, kernel, kernelL, params)
                Ok(Self.Cuda(s))
            }
            Storage.Metal.inp.Storage.Metal.kernel -> {
                val s = inp.convTranspose2d(l, kernel, kernelL, params)
                Ok(Self.Metal(s))
            }
            else -> Err(
                Error.DeviceMismatchBinaryOp(
                    lhs = lhs.device().location(),
                    rhs = rhs.device().location(),
                    op = "conv_transpose2d",
                )
                    .bt(),
            )
        }
    }

    public fun avgPool2d(
        layout: Layout,
        kernelSize: kotlin.Pair<Int, Int>,
        stride: kotlin.Pair<Int, Int>,
    ): Storage {
        return when (self) {
            Storage.Cpu.storage -> {
                val storage = storage.avgPool2d(layout, kernelSize, stride)
                Ok(Self.Cpu(storage))
            }
            Self.Cuda.storage -> {
                val storage = storage.avgPool2d(layout, kernelSize, stride)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.storage -> {
                val storage = storage.avgPool2d(layout, kernelSize, stride)
                Ok(Self.Metal(storage))
            } else -> unreachable()
        }
    }

    public fun maxPool2d(
        layout: Layout,
        kernelSize: kotlin.Pair<Int, Int>,
        stride: kotlin.Pair<Int, Int>,
    ): Storage {
        return when (self) {
            Storage.Cpu.storage -> {
                val storage = storage.maxPool2d(layout, kernelSize, stride)
                Ok(Self.Cpu(storage))
            }
            Self.Cuda.storage -> {
                val storage = storage.maxPool2d(layout, kernelSize, stride)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.storage -> {
                val storage = storage.maxPool2d(layout, kernelSize, stride)
                Ok(Self.Metal(storage))
            } else -> unreachable()
        }
    }

    public fun upsampleNearest1d(layout: Layout, sz: Int): Storage {
        return when (self) {
            Storage.Cpu.storage -> {
                val storage = storage.upsampleNearest1d(layout, sz)
                Ok(Self.Cpu(storage))
            }
            Self.Cuda.storage -> {
                val storage = storage.upsampleNearest1d(layout, sz)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.storage -> {
                val storage = storage.upsampleNearest1d(layout, sz)
                Ok(Self.Metal(storage))
            } else -> unreachable()
        }
    }

    public fun upsampleNearest2d(layout: Layout, h: Int, w: Int): Storage {
        return when (self) {
            Storage.Cpu.storage -> {
                val storage = storage.upsampleNearest2d(layout, h, w)
                Ok(Self.Cpu(storage))
            }
            Self.Cuda.storage -> {
                val storage = storage.upsampleNearest2d(layout, h, w)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.storage -> {
                val storage = storage.upsampleNearest2d(layout, h, w)
                Ok(Self.Metal(storage))
            } else -> unreachable()
        }
    }

    public fun upsampleBilinear2d(
        layout: Layout,
        h: Int,
        w: Int,
        alignCorners: Boolean,
        scaleH: Double? = null,
        scaleW: Double? = null,
    ): Storage {
        return when (self) {
            Storage.Cpu.storage -> {
                val storage =
                    storage.upsampleBilinear2d(layout, h, w, alignCorners, scaleH, scaleW)
                Ok(Self.Cpu(storage))
            }
            Self.Cuda.storage -> {
                val storage =
                    storage.upsampleBilinear2d(layout, h, w, alignCorners, scaleH, scaleW)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.storage -> {
                val storage =
                    storage.upsampleBilinear2d(layout, h, w, alignCorners, scaleH, scaleW)
                Ok(Self.Metal(storage))
            } else -> unreachable()
        }
    }

    public fun whereCond(
        layout: Layout,
        t: Storage,
        layoutT: Layout,
        f: Storage,
        layoutF: Layout,
    ): Storage {
        self.sameDevice(t, "where")
        self.sameDevice(f, "where")
        t.sameDtype(f, "where")
        return when (Triple(self, t, f)) {
            Storage.Cpu.cond.Storage.Cpu.t.Storage.Cpu.f -> {
                val storage = cond.whereCond(layout, t, layoutT, f, layoutF)
                Ok(Self.Cpu(storage))
            }
            Self.Cuda.cond.Self.Cuda.t.Self.Cuda.f -> {
                val storage = cond.whereCond(layout, t, layoutT, f, layoutF)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.cond.Self.Metal.t.Self.Metal.f -> {
                val storage = cond.whereCond(layout, t, layoutT, f, layoutF)
                Ok(Self.Metal(storage))
            }
            else -> Err(
                Error.DeviceMismatchBinaryOp(
                    lhs = lhs.device().location(),
                    rhs = rhs.device().location(),
                    op = "where",
                )
                    .bt(),
            )
        }
    }

    public fun gather(
        l: Layout,
        indexes: Storage,
        indexesL: Layout,
        d: Int,
    ): Storage {
        self.sameDevice(indexes, "index-add")
        return when (Pair(self, indexes)) {
            Self.Cpu.s.Self.Cpu.indexes -> {
                val storage = s.gather(l, indexes, indexesL, d)
                Ok(Self.Cpu(storage))
            }
            Self.Cuda.s.Self.Cuda.indexes -> {
                val storage = s.gather(l, indexes, indexesL, d)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.s.Self.Metal.indexes -> {
                val storage = s.gather(l, indexes, indexesL, d)
                Ok(Self.Metal(storage))
            }
            else -> unreachable()
        }
    }

    public fun scatterSet(
        l: Layout,
        indexes: Storage,
        indexesL: Layout,
        source: Storage,
        sourceL: Layout,
        d: Int,
    ) {
        self.sameDevice(indexes, "scatter-set")
        self.sameDevice(source, "scatter-set")
        when (Triple(self, indexes, source)) {
            Self.Cpu.s.Self.Cpu.indexes.Self.Cpu.source -> {
                s.scatterSet(l, indexes, indexesL, source, sourceL, d)
            }
            Self.Cuda.s.Self.Cuda.indexes.Self.Cuda.source -> {
                s.scatterSet(l, indexes, indexesL, source, sourceL, d)
            }
            Self.Metal.s.Self.Metal.indexes.Self.Metal.source -> {
                s.scatterSet(l, indexes, indexesL, source, sourceL, d)
            }
            else -> unreachable()
        }
        Ok()
    }

    public fun scatterAdd(
        l: Layout,
        indexes: Storage,
        indexesL: Layout,
        source: Storage,
        sourceL: Layout,
        d: Int,
    ) {
        self.sameDevice(indexes, "scatter-add")
        self.sameDevice(source, "scatter-add")
        when (Triple(self, indexes, source)) {
            Self.Cpu.s.Self.Cpu.indexes.Self.Cpu.source -> {
                s.scatterAddSet(l, indexes, indexesL, source, sourceL, d)
            }
            Self.Cuda.s.Self.Cuda.indexes.Self.Cuda.source -> {
                s.scatterAddSet(l, indexes, indexesL, source, sourceL, d)
            }
            Self.Metal.s.Self.Metal.indexes.Self.Metal.source -> {
                s.scatterAddSet(l, indexes, indexesL, source, sourceL, d)
            }
            else -> unreachable()
        }
        Ok()
    }

    public fun indexAdd(
        l: Layout,
        indexes: Storage,
        indexesL: Layout,
        source: Storage,
        sourceL: Layout,
        d: Int,
    ): Storage {
        self.sameDevice(indexes, "index-add")
        self.sameDevice(source, "index-add")
        return when (Triple(self, indexes, source)) {
            Self.Cpu.s.Self.Cpu.indexes.Self.Cpu.source -> {
                val storage = s.indexAdd(l, indexes, indexesL, source, sourceL, d)
                Ok(Self.Cpu(storage))
            }
            Self.Cuda.s.Self.Cuda.indexes.Self.Cuda.source -> {
                val storage = s.indexAdd(l, indexes, indexesL, source, sourceL, d)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.s.Self.Metal.indexes.Self.Metal.source -> {
                val storage = s.indexAdd(l, indexes, indexesL, source, sourceL, d)
                Ok(Self.Metal(storage))
            }
            else -> unreachable()
        }
    }

    public fun indexSelect(
        rhs: Storage,
        lhsL: Layout,
        rhsL: Layout,
        d: Int,
    ): Storage {
        self.sameDevice(rhs, "index-select")
        return when (Pair(self, rhs)) {
            Self.Cpu.lhs.Self.Cpu.rhs -> {
                val storage = lhs.indexSelect(rhs, lhsL, rhsL, d)
                Ok(Self.Cpu(storage))
            }
            Self.Cuda.lhs.Self.Cuda.rhs -> {
                val storage = lhs.indexSelect(rhs, lhsL, rhsL, d)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.lhs.Self.Metal.rhs -> {
                val storage = lhs.indexSelect(rhs, lhsL, rhsL, d)
                Ok(Self.Metal(storage))
            }
            else -> Err(
                Error.DeviceMismatchBinaryOp(
                    lhs = lhs.device().location(),
                    rhs = rhs.device().location(),
                    op = "index-select",
                )
                    .bt(),
            )
        }
    }

    public fun matmul(
        rhs: Storage,
        bmnk: List<Int>,
        lhsLayout: Layout,
        rhsLayout: Layout,
    ): Storage {
        self.sameDevice(rhs, "matmul")
        self.sameDtype(rhs, "matmul")
        return when (Pair(self, rhs)) {
            Self.Cpu.lhs.Self.Cpu.rhs -> {
                val storage = lhs.matmul(rhs, bmnk, lhsLayout, rhsLayout)
                Ok(Self.Cpu(storage))
            }
            Self.Cuda.lhs.Self.Cuda.rhs -> {
                val storage = lhs.matmul(rhs, bmnk, lhsLayout, rhsLayout)
                Ok(Self.Cuda(storage))
            }
            Self.Metal.lhs.Self.Metal.rhs -> {
                val storage = lhs.matmul(rhs, bmnk, lhsLayout, rhsLayout)
                Ok(Self.Metal(storage))
            }
            else -> Err(
                Error.DeviceMismatchBinaryOp(
                    lhs = lhs.device().location(),
                    rhs = rhs.device().location(),
                    op = "matmul",
                )
                    .bt(),
            )
        }
    }

    // self, the source can be strided whereas dst is contiguous.
    public fun copyStridedSrc(
        dst: Storage,
        dstOffset: Int,
        srcL: Layout,
    ) {
        when (Pair(self, dst)) {
            Self.Cpu.src.Self.Cpu.dst -> src.copyStridedSrc(dst, dstOffset, srcL)
            Self.Cuda.src.Self.Cuda.dst -> Ok(src.copyStridedSrc(dst, dstOffset, srcL))
            Self.Metal.src.Self.Metal.dst -> {
                Ok(src.copyStridedSrc(dst, dstOffset, srcL))
            }
            else -> Err(
                Error.DeviceMismatchBinaryOp(
                    lhs = lhs.device().location(),
                    rhs = rhs.device().location(),
                    op = "copy",
                )
                    .bt(),
            )
        }
    }

    public fun copy2d(
        dst: Storage,
        d1: Int,
        d2: Int,
        srcS: Int,
        dstS: Int,
        srcO: Int,
        dstO: Int,
    ) {
        when (Pair(self, dst)) {
            Self.Cpu.src.Self.Cpu.dst -> src.copy2d(dst, d1, d2, srcS, dstS, srcO, dstO)
            Self.Cuda.src.Self.Cuda.dst -> {
                Ok(src.copy2d(dst, d1, d2, srcS, dstS, srcO, dstO))
            }
            Self.Metal.src.Self.Metal.dst -> {
                Ok(src.copy2d(dst, d1, d2, srcS, dstS, srcO, dstO))
            }
            else -> Err(
                Error.DeviceMismatchBinaryOp(
                    lhs = lhs.device().location(),
                    rhs = rhs.device().location(),
                    op = "copy2d",
                )
                    .bt(),
            )
        }
    }

    public val unaryImpl: (Layout, UnaryOp) -> Storage get() = { layout, op ->
        when (this) {
            is Cpu -> Cpu(this.storage.unaryImpl(layout, op))
            is Cuda -> Cuda(this.storage.unaryImpl<UnaryOpT>(layout))
            is Metal -> Metal(this.storage.unaryImpl<UnaryOpT>(layout))
        }
    }

    public val binaryImpl: (Storage, Layout, Layout, BinaryOp) -> Storage get() = { rhs, lhsLayout, rhsLayout, op ->
        sameDevice(rhs, op.name)
        sameDtype(rhs, op.name)
        when {
            this is Cpu && rhs is Cpu -> Cpu(this.storage.binaryImpl(rhs.storage, lhsLayout, rhsLayout, op))
            this is Cuda && rhs is Cuda -> Cuda(this.storage.binaryImpl<BinaryOpT>(rhs.storage, lhsLayout, rhsLayout))
            this is Metal && rhs is Metal -> Metal(this.storage.binaryImpl<BinaryOpT>(rhs.storage, lhsLayout, rhsLayout))
            else -> Err(
                Error.DeviceMismatchBinaryOp(
                    lhs = device().location(),
                    rhs = rhs.device().location(),
                    op = op.name,
                ).bt(),
            )
        }
    }

    public val toCpuStorage: () -> CpuStorage get() = {
        when (this) {
            is Cpu -> this.storage
            is Cuda -> this.storage.toCpuStorage()
            is Metal -> this.storage.toCpuStorage()
        }
    }

    public companion object Self {
        public val Cpu: (Any?) -> Storage = { s -> Storage.Cpu(s) }
        public val Cuda: (Any?) -> Storage = { s -> Storage.Cuda(s) }
        public val Metal: (Any?) -> Storage = { s -> Storage.Metal(s) }
    }
}
