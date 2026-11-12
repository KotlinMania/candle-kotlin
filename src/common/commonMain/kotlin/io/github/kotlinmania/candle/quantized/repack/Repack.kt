// port-lint: source quantized/repack.rs
@file:OptIn(ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.quantized.repack

import io.github.kotlinmania.candle.cpu.features.get
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.quantized.GgmlDType
import io.github.kotlinmania.candle.quantized.QuantizedType
import io.github.kotlinmania.candle.quantized.kquants.BlockQ40
import io.github.kotlinmania.candle.quantized.kquants.BlockQ4K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ5K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ6K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ80
import io.github.kotlinmania.candle.quantized.kquants.BlockQ8K
import io.github.kotlinmania.candle.quantized.kquants.QK4_0
import io.github.kotlinmania.candle.quantized.kquants.QK8_0
import io.github.kotlinmania.candle.quantized.kquants.QK_K
import io.github.kotlinmania.candle.quantized.kquants.matmulQ4kX8
import io.github.kotlinmania.candle.quantized.neon.matmulQ40X4
import io.github.kotlinmania.candle.quantized.neon.matmulQ40X4I8mm
import io.github.kotlinmania.candle.quantized.neon.matmulQ4kX8I8mm
import io.github.kotlinmania.candle.quantized.neon.matmulQ5kX8
import io.github.kotlinmania.candle.quantized.neon.matmulQ6kX8
import io.github.kotlinmania.candle.quantized.neon.matmulQ80X4
import io.github.kotlinmania.candle.quantized.neon.matmulQ80X4I8mm
import io.github.kotlinmania.candle.quantized.neon.vecDot4Q80x4Q80
import io.github.kotlinmania.candle.quantized.neon.vecDot8Q4kQ8k
import io.github.kotlinmania.candle.quantized.neon.vecDot8Q6kx8Q8k
import io.github.kotlinmania.candle.quantized.repackx86.PackedX86
import io.github.kotlinmania.candle.quantized.repackx86.TileQ4KAmx
import io.github.kotlinmania.candle.quantized.repackx86.amxAvailable
import io.github.kotlinmania.candle.quantized.repackx86.matmulAmxQ4k
import io.github.kotlinmania.candle.quantized.repackx86.matmulTiles
import io.github.kotlinmania.candle.quantized.repackx86.pack
import io.github.kotlinmania.candle.quantized.repackx86.packQ4kAmx
import io.github.kotlinmania.candle.quantized.repackx86.quantizeLhs
import io.github.kotlinmania.candle.quantized.repackx86.select
import io.github.kotlinmania.candle.utils.barrierPool
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.rustlang.core.mem.ObjectSlotLayout
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.mem.constPtrFromRustCast
import io.github.kotlinmania.rustlang.core.mem.mutPtrFromRustCast
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr
import io.github.kotlinmania.rustlang.core.result.fromResidual
import io.github.kotlinmania.rustlang.std.sync.OnceLock

public enum class PackedKind(
    public val value: UInt,
) {
    Q40x4(0u),
    Q4Kx8(1u),
    Q5Kx8(2u),
    Q6Kx8(3u),
    Q80x4(4u),
    ;

    public fun pack(storage: QuantizedType, n: Int): PackedStorage {
        val interleave =
            if (get().i8mm) {
                8
            } else {
                4
            }
        return when (this) {
            PackedKind.Q40x4 -> {
                val totalBlocks = storage.storageSizeInBytes() / sizeOf<BlockQ40>(typeLayoutT = ObjectSlotLayout<BlockQ40>())
                val blocks = fromRawParts(constPtrFromRustCast(storage.asPtr()), totalBlocks)
                PackedStorage.Q40x4(packToQ40x4(blocks, n, interleave))
            }
            PackedKind.Q4Kx8 -> {
                val totalBlocks1 = storage.storageSizeInBytes() / sizeOf<BlockQ4K>(typeLayoutT = ObjectSlotLayout<BlockQ4K>())
                val blocks1 = fromRawParts(constPtrFromRustCast(storage.asPtr()), totalBlocks1)
                PackedStorage.Q4Kx8(packToQ4kx8(blocks1, n))
            }
            PackedKind.Q5Kx8 -> {
                val totalBlocks2 = storage.storageSizeInBytes() / sizeOf<BlockQ5K>(typeLayoutT = ObjectSlotLayout<BlockQ5K>())
                val blocks2 = fromRawParts(constPtrFromRustCast(storage.asPtr()), totalBlocks2)
                PackedStorage.Q5Kx8(packToQ5kx8(blocks2, n, interleave))
            }
            PackedKind.Q6Kx8 -> {
                val totalBlocks3 = storage.storageSizeInBytes() / sizeOf<BlockQ6K>(typeLayoutT = ObjectSlotLayout<BlockQ6K>())
                val blocks3 = fromRawParts(constPtrFromRustCast(storage.asPtr()), totalBlocks3)
                PackedStorage.Q6Kx8(packToQ6kx8(blocks3, n, interleave))
            }
            PackedKind.Q80x4 -> {
                val totalBlocks4 = storage.storageSizeInBytes() / sizeOf<BlockQ80>(typeLayoutT = ObjectSlotLayout<BlockQ80>())
                val blocks4 = fromRawParts(constPtrFromRustCast(storage.asPtr()), totalBlocks4)
                PackedStorage.Q80x4(packToQ80x4(blocks4, n, interleave))
            }
        }
    }

    // matmul against one expert's row block inside stacked packed storage; subslices
    // per arm so nothing is copied
    public fun matmulExpertRange(mkn: Triple<Int, Int, Int>, lhs: List<Float>, packed: PackedStorage, expert: Int, dst: List<Float>): Result<Unit> {
        val (m, k, nOut) = mkn
        val features = get()
        return run {
            val matchVal = Pair(this, packed)
            when {
                matchVal.first == PackedKind.Q4Kx8 && matchVal.second is PackedStorage.Q4Kx8 -> {
                    val (v) = matchVal.second as PackedStorage.Q4Kx8
                    val p =
                        run {
                            val kb = k / QK_K
                            val groups = nOut / 8
                            ((v).slice(expert * groups * kb until (expert + 1) * groups * kb))
                        }
                    if (features.i8mm && mkn.first >= 4 && mkn.first.isMultipleOf(4)) {
                        matmulQ4kX8I8mm(mkn, lhs, p, dst)
                    } else {
                        matmulQ4kX8(mkn, lhs, p, dst)
                    }
                }
                matchVal.first == PackedKind.Q6Kx8 && matchVal.second is PackedStorage.Q6Kx8 -> {
                    val (v1) = matchVal.second as PackedStorage.Q6Kx8
                    matmulQ6kX8(
                        mkn,
                        lhs,
                        run {
                            val kb1 = k / QK_K
                            val groups1 = nOut / 8
                            ((v1).slice(expert * groups1 * kb1 until (expert + 1) * groups1 * kb1))
                        },
                        dst,
                    )
                }
                matchVal.first == PackedKind.Q80x4 && matchVal.second is PackedStorage.Q80x4 -> {
                    val (v2) = matchVal.second as PackedStorage.Q80x4
                    val p1 =
                        run {
                            val kb2 = k / QK8_0
                            val groups2 = nOut / 4
                            ((v2).slice(expert * groups2 * kb2 until (expert + 1) * groups2 * kb2))
                        }
                    if (features.i8mm && mkn.first >= 4 && mkn.first.isMultipleOf(4)) {
                        matmulQ80X4I8mm(mkn, lhs, p1, dst)
                    } else {
                        matmulQ80X4(mkn, lhs, p1, dst)
                    }
                }
                else -> bail("unsupported expert-range kind")
            }
        }
    }

    public fun matmul(mkn: Triple<Int, Int, Int>, lhs: List<Float>, packed: PackedStorage, dst: List<Float>): Result<Unit> {
        val features = get()
        return run {
            val matchVal = Pair(this, packed)
            when {
                matchVal.first == PackedKind.Q40x4 && matchVal.second is PackedStorage.Q40x4 -> {
                    val (packed1) = matchVal.second as PackedStorage.Q40x4
                    if (features.i8mm && mkn.first >= 4) {
                        matmulQ40X4I8mm(mkn, lhs, packed1, dst)
                    } else {
                        matmulQ40X4(mkn, lhs, packed1, dst)
                    }
                }
                matchVal.first == PackedKind.Q4Kx8 && matchVal.second is PackedStorage.Q4Kx8 -> {
                    val (packed2) = matchVal.second as PackedStorage.Q4Kx8
                    if (features.i8mm && mkn.first >= 4) {
                        matmulQ4kX8I8mm(mkn, lhs, packed2, dst)
                    } else {
                        matmulQ4kX8(mkn, lhs, packed2, dst)
                    }
                }
                matchVal.first == PackedKind.Q5Kx8 && matchVal.second is PackedStorage.Q5Kx8 -> {
                    val (packed3) = matchVal.second as PackedStorage.Q5Kx8
                    matmulQ5kX8(mkn, lhs, packed3, dst)
                }
                matchVal.first == PackedKind.Q6Kx8 && matchVal.second is PackedStorage.Q6Kx8 -> {
                    val (packed4) = matchVal.second as PackedStorage.Q6Kx8
                    matmulQ6kX8(mkn, lhs, packed4, dst)
                }
                matchVal.first == PackedKind.Q80x4 && matchVal.second is PackedStorage.Q80x4 -> {
                    val (packed5) = matchVal.second as PackedStorage.Q80x4
                    if (features.i8mm && mkn.first >= 4) {
                        matmulQ80X4I8mm(mkn, lhs, packed5, dst)
                    } else {
                        matmulQ80X4(mkn, lhs, packed5, dst)
                    }
                }
                else -> bail("packed storage kind mismatch")
            }
        }
    }

    public companion object {
        public fun select(dtype: GgmlDType, tupleArg1: Triple<Int, Int, Int>): PackedKind? {
            val features = get()
            val hasDotprodGemv = features.dotprod && m == 1
            val hasTiledMatmul = (features.dotprod || features.i8mm) && m >= 4 && m.isMultipleOf(4)
            return when {
                dtype == GgmlDType.Q4_0 && (n.isMultipleOf(4) && (hasDotprodGemv || hasTiledMatmul)) -> {
                    PackedKind.Q40x4
                }
                dtype == GgmlDType.Q4K && (n.isMultipleOf(8) && (hasDotprodGemv || hasTiledMatmul)) -> {
                    PackedKind.Q4Kx8
                }
                dtype == GgmlDType.Q5K && (n.isMultipleOf(8) && (hasDotprodGemv || hasTiledMatmul)) -> {
                    PackedKind.Q5Kx8
                }
                dtype == GgmlDType.Q6K && (n.isMultipleOf(8) && (hasDotprodGemv || hasTiledMatmul)) -> {
                    PackedKind.Q6Kx8
                }
                dtype == GgmlDType.Q8_0 && (n.isMultipleOf(4) && (hasDotprodGemv || hasTiledMatmul)) -> {
                    PackedKind.Q80x4
                }
                else -> null
            }
        }
    }
}

internal sealed class PackedStorage {
    public data class Q40x4(
        public var value: List<BlockQ40x4>,
    ) : PackedStorage()
    public data class Q4Kx8(
        public var value: List<BlockQ4Kx8>,
    ) : PackedStorage()
    public data class Q5Kx8(
        public var value: List<BlockQ5Kx8>,
    ) : PackedStorage()
    public data class Q6Kx8(
        public var value: List<BlockQ6Kx8>,
    ) : PackedStorage()
    public data class Q80x4(
        public var value: List<BlockQ80x4>,
    ) : PackedStorage()
}

internal data class PackedCache(
    public var x86: OnceLock<PackedX86> = OnceLock<PackedX86>(),
    public var x86Amx: OnceLock<List<TileQ4KAmx>> = OnceLock<List<TileQ4KAmx>>(),
    public var q40x4: OnceLock<PackedStorage> = OnceLock<PackedStorage>(),
    public var q4kx8: OnceLock<PackedStorage> = OnceLock<PackedStorage>(),
    public var q5kx8: OnceLock<PackedStorage> = OnceLock<PackedStorage>(),
    public var q6kx8: OnceLock<PackedStorage> = OnceLock<PackedStorage>(),
    public var q80x4: OnceLock<PackedStorage> = OnceLock<PackedStorage>(),
) {
    internal fun x86GetOrInit(init: () -> PackedX86): PackedX86 = this.x86.getOrInit(init)

    public fun getOrInit(kind: PackedKind, init: () -> PackedStorage): PackedStorage =
        when (kind) {
            PackedKind.Q40x4 -> this.q40x4.getOrInit(init)
            PackedKind.Q4Kx8 -> this.q4kx8.getOrInit(init)
            PackedKind.Q5Kx8 -> this.q5kx8.getOrInit(init)
            PackedKind.Q6Kx8 -> this.q6kx8.getOrInit(init)
            PackedKind.Q80x4 -> this.q80x4.getOrInit(init)
        }

    public companion object {
        internal fun new(): PackedCache = PackedCache(x86 = OnceLock.new(), x86Amx = OnceLock.new(), q40x4 = OnceLock.new(), q4kx8 = OnceLock.new(), q5kx8 = OnceLock.new(), q6kx8 = OnceLock.new(), q80x4 = OnceLock.new())
    }
}

internal data class BlockQ80x4(
    internal var d: List<F16> = List(4) { F16() },
    internal var qs: ByteArray = ByteArray(QK8_0 * 4),
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is BlockQ80x4) return false
        if (this.d != other.d) return false
        if (!this.qs.contentEquals(other.qs)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = this.d.hashCode()
        result = 31 * result + this.qs.contentHashCode()
        return result
    }
}

internal data class BlockQ40x4(
    internal var d: List<F16> = List(4) { F16() },
    internal var qs: ByteArray = ByteArray(QK4_0 * 2),
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is BlockQ40x4) return false
        if (this.d != other.d) return false
        if (!this.qs.contentEquals(other.qs)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = this.d.hashCode()
        result = 31 * result + this.qs.contentHashCode()
        return result
    }
}

internal data class BlockQ4Kx8(
    internal var d: List<F16> = List(8) { F16() },
    internal var dmin: List<F16> = List(8) { F16() },
    internal var scales: UByteArray = UByteArray(96),
    internal var qs: UByteArray = UByteArray(1024),
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is BlockQ4Kx8) return false
        if (this.d != other.d) return false
        if (this.dmin != other.dmin) return false
        if (!this.scales.contentEquals(other.scales)) return false
        if (!this.qs.contentEquals(other.qs)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = this.d.hashCode()
        result = 31 * result + this.dmin.hashCode()
        result = 31 * result + this.scales.contentHashCode()
        result = 31 * result + this.qs.contentHashCode()
        return result
    }
}

internal data class BlockQ5Kx8(
    internal var d: List<F16> = List(8) { F16() },
    internal var dmin: List<F16> = List(8) { F16() },
    internal var scales: UByteArray = UByteArray(96),
    internal var qh: UByteArray = UByteArray(QK_K),
    internal var qs: UByteArray = UByteArray(QK_K / 2 * 8),
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is BlockQ5Kx8) return false
        if (this.d != other.d) return false
        if (this.dmin != other.dmin) return false
        if (!this.scales.contentEquals(other.scales)) return false
        if (!this.qh.contentEquals(other.qh)) return false
        if (!this.qs.contentEquals(other.qs)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = this.d.hashCode()
        result = 31 * result + this.dmin.hashCode()
        result = 31 * result + this.scales.contentHashCode()
        result = 31 * result + this.qh.contentHashCode()
        result = 31 * result + this.qs.contentHashCode()
        return result
    }
}

internal data class BlockQ6Kx8(
    internal var d: List<F16> = List(8) { F16() },
    internal var scales: ByteArray = ByteArray(QK_K / 16 * 8),
    internal var ql: UByteArray = UByteArray(QK_K / 2 * 8),
    internal var qh: UByteArray = UByteArray(QK_K / 4 * 8),
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is BlockQ6Kx8) return false
        if (this.d != other.d) return false
        if (!this.scales.contentEquals(other.scales)) return false
        if (!this.ql.contentEquals(other.ql)) return false
        if (!this.qh.contentEquals(other.qh)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = this.d.hashCode()
        result = 31 * result + this.scales.contentHashCode()
        result = 31 * result + this.ql.contentHashCode()
        result = 31 * result + this.qh.contentHashCode()
        return result
    }
}

internal data class BlockQ8Kx4(
    internal var d: FloatArray = FloatArray(4),
    internal var qs: ByteArray = ByteArray(QK_K * 4),
    internal var bsums: ShortArray = ShortArray(QK_K / 4),
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is BlockQ8Kx4) return false
        if (!this.d.contentEquals(other.d)) return false
        if (!this.qs.contentEquals(other.qs)) return false
        if (!this.bsums.contentEquals(other.bsums)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = this.d.contentHashCode()
        result = 31 * result + this.qs.contentHashCode()
        result = 31 * result + this.bsums.contentHashCode()
        return result
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<BlockQ8Kx4> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<BlockQ8Kx4> {
                override val alignBytes: Long = 4L
                private val off2: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes((16L + ((QK_K * 4).toLong() * (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I8).strideBytes)), 2L)
                override val sizeBytes: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes((off2 + ((QK_K / 4).toLong() * (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I16).strideBytes)), 4L)
                override fun readNative(addr: Long): BlockQ8Kx4 =
                    BlockQ8Kx4(
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32, (4).toInt())
                        ).readNative(addr).toFloatArray(),
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I8, (QK_K * 4).toInt())
                        ).readNative(addr + 16L).toByteArray(),
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I16, (QK_K / 4).toInt())
                        ).readNative(addr + off2).toShortArray(),
                    )
                override fun writeNative(addr: Long, value: BlockQ8Kx4) {
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32, (4).toInt())
                    ).writeNative(addr, value.d.toList())
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I8, (QK_K * 4).toInt())
                    ).writeNative(addr + 16L, value.qs.toList())
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I16, (QK_K / 4).toInt())
                    ).writeNative(addr + off2, value.bsums.toList())
                }
            }
    }
}

internal fun tryMatmulF32(storage: QuantizedType, packed: PackedCache, mkn: Triple<Int, Int, Int>, lhs: List<Float>, dst: List<Float>): Result<Boolean> {
    run {
        val (m, k, n) = mkn
        val dtype = storage.dtype()
        when (val matchVal = PackedKind.select(dtype, mkn)) {
            null -> {}
            else -> {
                val kind = matchVal
                val packed1 = packed.getOrInit(kind, { kind.pack(storage, n) })
                when (val tryResult = kind.matmul(mkn, lhs, packed1, dst)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
                return Ok(true)
            }
        }
        // tiled kernels need m % 4 == 0: run the aligned rows tiled and the tail rows as
        // gemvs on the same packing instead of dropping the whole matmul to the unpacked path
        val kind1 = PackedKind.select(dtype, Triple(1, k, n)) ?: return Ok(false)
        val mTiled = m and 3.inv()
        if (mTiled > 0 && PackedKind.select(dtype, Triple(mTiled, k, n)) != kind1) {
            return Ok(false)
        }
        val packed2 = packed.getOrInit(kind1, { kind1.pack(storage, n) })
        val (lhsTiled, lhsTail) = lhs.splitAt(mTiled * k)
        val (dstTiled, dstTail) = dst.splitAtMut(mTiled * n)
        if (mTiled > 0) {
            when (val tryResult = kind1.matmul(Triple(mTiled, k, n), lhsTiled, packed2, dstTiled)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        for ((lhsRow, dstRow) in (lhsTail.chunksExact(k).zip(dstTail.chunksExactMut(n)))) {
            when (val tryResult = kind1.matmul(Triple(1, k, n), lhsRow, packed2, dstRow)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        Ok(true)
    }
    run {
        val (m1, k1, n1) = mkn
        if (!select(storage.dtype(), m1, n1, k1)) {
            return Ok(false)
        }
        val dtype1 = storage.dtype()
        val lhsQ = quantizeLhs(lhs, m1, k1)
        if (m1 >= 32 && dtype1 == GgmlDType.Q4K && amxAvailable()) {
            val tiles = packed.x86Amx.getOrInit({ packQ4kAmx(storage, n1, k1) })
            matmulAmxQ4k(tiles, lhsQ, m1, k1, n1, dst)
            return Ok(true)
        }
        val tiles1 = packed.x86.getOrInit({ pack(dtype1, storage, n1, k1) })
        matmulTiles(tiles1, lhsQ, m1, k1, n1, dst)
        Ok(true)
    }
    return run {
        Ok(false)
    }
}

public fun packToQ4kx8(blocks: List<BlockQ4K>, n: Int): List<BlockQ4Kx8> {
    check(n.isMultipleOf(8))
    check(blocks.size % n == 0)
    val kBlocks = blocks.size / n
    val nGroups = n / 8
    val packed = ArrayList<BlockQ4Kx8>(nGroups * kBlocks)
    for (g in 0 until nGroups) {
        for (b in 0 until kBlocks) {
            val src: List<BlockQ4K> = fromFn({ i -> blocks[(g * 8 + i) * kBlocks + b] })
            val p = BlockQ4Kx8(d = List(8) { ZERO }, dmin = List(8) { ZERO }, scales = UByteArray(96) { 0u.toUByte() }, qs = UByteArray(1024) { 0u.toUByte() })
            for ((i, s) in src.enumerate()) {
                p.d[i] = s.d
                p.dmin[i] = s.dmin
            }
            for (i1 in 0 until 128) {
                val srcId = i1 % 8
                val srcOffset = (i1 / 8) * 8
                val dstOffset = i1 * 8
                ((p.qs).slice(dstOffset until dstOffset + 8)).copyFromSlice(((src[srcId].qs).slice(srcOffset until srcOffset + 8)))
            }
            packKScaleMinX8(src.map({ s1 -> s1.scales }), p.scales)
            packed.add(p)
        }
    }
    return packed
}

public fun packToQ5kx8(blocks: List<BlockQ5K>, n: Int, blockLen: Int): List<BlockQ5Kx8> {
    check(n.isMultipleOf(8))
    check((QK_K / 2).isMultipleOf(blockLen))
    check((QK_K / 8).isMultipleOf(blockLen))
    check(blocks.size % n == 0)
    val kBlocks = blocks.size / n
    val nGroups = n / 8
    val packed = ArrayList<BlockQ5Kx8>(nGroups * kBlocks)
    for (g in 0 until nGroups) {
        for (b in 0 until kBlocks) {
            val src: List<BlockQ5K> = fromFn({ i -> blocks[(g * 8 + i) * kBlocks + b] })
            val p = BlockQ5Kx8(d = List(8) { ZERO }, dmin = List(8) { ZERO }, scales = UByteArray(96) { 0u.toUByte() }, qh = UByteArray(QK_K) { 0u.toUByte() }, qs = UByteArray(QK_K / 2 * 8) { 0u.toUByte() })
            for ((i, s) in src.enumerate()) {
                p.d[i] = s.d
                p.dmin[i] = s.dmin
            }
            for (i1 in 0 until (QK_K / 2 * 8) / blockLen) {
                val srcId = i1 % 8
                val srcOffset = (i1 / 8) * blockLen
                val dstOffset = i1 * blockLen
                ((p.qs).slice(dstOffset until dstOffset + blockLen)).copyFromSlice(((src[srcId].qs).slice(srcOffset until srcOffset + blockLen)))
            }
            for (i2 in 0 until (QK_K / 8 * 8) / blockLen) {
                val srcId1 = i2 % 8
                val srcOffset1 = (i2 / 8) * blockLen
                val dstOffset1 = i2 * blockLen
                ((p.qh).slice(dstOffset1 until dstOffset1 + blockLen)).copyFromSlice(((src[srcId1].qh).slice(srcOffset1 until srcOffset1 + blockLen)))
            }
            packKScaleMinX8(src.map({ s1 -> s1.scales }), p.scales)
            packed.add(p)
        }
    }
    return packed
}

public fun packToQ6kx8(blocks: List<BlockQ6K>, n: Int, blockLen: Int): List<BlockQ6Kx8> {
    check(n.isMultipleOf(8))
    check((QK_K / 2).isMultipleOf(blockLen))
    check((QK_K / 4).isMultipleOf(blockLen))
    check(blocks.size % n == 0)
    val kBlocks = blocks.size / n
    val nGroups = n / 8
    val packed = ArrayList<BlockQ6Kx8>(nGroups * kBlocks)
    for (g in 0 until nGroups) {
        for (b in 0 until kBlocks) {
            val src: List<BlockQ6K> = fromFn({ i -> blocks[(g * 8 + i) * kBlocks + b] })
            val p = BlockQ6Kx8(d = List(8) { ZERO }, scales = ByteArray(QK_K / 16 * 8) { 0.toByte() }, ql = UByteArray(QK_K / 2 * 8) { 0u.toUByte() }, qh = UByteArray(QK_K / 4 * 8) { 0u.toUByte() })
            for ((i, s) in src.enumerate()) {
                p.d[i] = s.d
            }
            for (i1 in 0 until (QK_K / 2 * 8) / blockLen) {
                val srcId = i1 % 8
                val srcOffset = (i1 / 8) * blockLen
                val dstOffset = i1 * blockLen
                ((p.ql).slice(dstOffset until dstOffset + blockLen)).copyFromSlice(((src[srcId].ql).slice(srcOffset until srcOffset + blockLen)))
            }
            for (i2 in 0 until (QK_K / 4 * 8) / blockLen) {
                val srcId1 = i2 % 8
                val srcOffset1 = (i2 / 8) * blockLen
                val dstOffset1 = i2 * blockLen
                ((p.qh).slice(dstOffset1 until dstOffset1 + blockLen)).copyFromSlice(((src[srcId1].qh).slice(srcOffset1 until srcOffset1 + blockLen)))
            }
            for ((col, s1) in src.enumerate()) {
                for (j in 0 until QK_K / 16) {
                    p.scales[j * 8 + col] = s1.scales[j]
                }
            }
            packed.add(p)
        }
    }
    return packed
}

public fun packKScaleMinX8(src: List<UByteArray>, dst: UByteArray) {
    for (i in 0 until 4) {
        val s = List(8) { 0u.toUByte() }
        val m = List(8) { 0u.toUByte() }
        for (j in 0 until 8) {
            s[j] = src[j][i] and 63u.toUByte()
            m[j] = src[j][i + 4] and 63u.toUByte()
        }
        val b12 = i * 12
        dst[b12] = ((s[0] and 63u) + (((s[4] and 48u).toUInt() shl 2).toUByte())).toUByte()
        dst[b12 + 1] = ((s[1] and 63u) + (((s[5] and 48u).toUInt() shl 2).toUByte())).toUByte()
        dst[b12 + 2] = ((s[2] and 63u) + (((s[6] and 48u).toUInt() shl 2).toUByte())).toUByte()
        dst[b12 + 3] = ((s[3] and 63u) + (((s[7] and 48u).toUInt() shl 2).toUByte())).toUByte()
        dst[b12 + 4] = ((m[0] and 63u) + (((m[4] and 48u).toUInt() shl 2).toUByte())).toUByte()
        dst[b12 + 5] = ((m[1] and 63u) + (((m[5] and 48u).toUInt() shl 2).toUByte())).toUByte()
        dst[b12 + 6] = ((m[2] and 63u) + (((m[6] and 48u).toUInt() shl 2).toUByte())).toUByte()
        dst[b12 + 7] = ((m[3] and 63u) + (((m[7] and 48u).toUInt() shl 2).toUByte())).toUByte()
        dst[b12 + 8] = ((s[4] and 15u) + (((m[4] and 15u).toUInt() shl 4).toUByte())).toUByte()
        dst[b12 + 9] = ((s[5] and 15u) + (((m[5] and 15u).toUInt() shl 4).toUByte())).toUByte()
        dst[b12 + 10] = ((s[6] and 15u) + (((m[6] and 15u).toUInt() shl 4).toUByte())).toUByte()
        dst[b12 + 11] = ((s[7] and 15u) + (((m[7] and 15u).toUInt() shl 4).toUByte())).toUByte()
    }
    for (i1 in 0 until 4) {
        val s1 = List(8) { 0u.toUByte() }
        val m1 = List(8) { 0u.toUByte() }
        for (j1 in 0 until 8) {
            s1[j1] = (((src[j1][i1] and 192u.toUByte()).toUInt() shr 2).toUByte()) or (src[j1][i1 + 8] and 15u.toUByte())
            m1[j1] = (((src[j1][i1 + 4] and 192u.toUByte()).toUInt() shr 2).toUByte()) or (((src[j1][i1 + 8] and 240u.toUByte()).toUInt() shr 4).toUByte())
        }
        val b121 = i1 * 12 + 48
        dst[b121] = ((s1[0] and 63u) + (((s1[4] and 48u).toUInt() shl 2).toUByte())).toUByte()
        dst[b121 + 1] = ((s1[1] and 63u) + (((s1[5] and 48u).toUInt() shl 2).toUByte())).toUByte()
        dst[b121 + 2] = ((s1[2] and 63u) + (((s1[6] and 48u).toUInt() shl 2).toUByte())).toUByte()
        dst[b121 + 3] = ((s1[3] and 63u) + (((s1[7] and 48u).toUInt() shl 2).toUByte())).toUByte()
        dst[b121 + 4] = ((m1[0] and 63u) + (((m1[4] and 48u).toUInt() shl 2).toUByte())).toUByte()
        dst[b121 + 5] = ((m1[1] and 63u) + (((m1[5] and 48u).toUInt() shl 2).toUByte())).toUByte()
        dst[b121 + 6] = ((m1[2] and 63u) + (((m1[6] and 48u).toUInt() shl 2).toUByte())).toUByte()
        dst[b121 + 7] = ((m1[3] and 63u) + (((m1[7] and 48u).toUInt() shl 2).toUByte())).toUByte()
        dst[b121 + 8] = ((s1[4] and 15u) + (((m1[4] and 15u).toUInt() shl 4).toUByte())).toUByte()
        dst[b121 + 9] = ((s1[5] and 15u) + (((m1[5] and 15u).toUInt() shl 4).toUByte())).toUByte()
        dst[b121 + 10] = ((s1[6] and 15u) + (((m1[6] and 15u).toUInt() shl 4).toUByte())).toUByte()
        dst[b121 + 11] = ((s1[7] and 15u) + (((m1[7] and 15u).toUInt() shl 4).toUByte())).toUByte()
    }
}

public fun packToQ40x4(blocks: List<BlockQ40>, n: Int, blockLen: Int): List<BlockQ40x4> {
    check(n.isMultipleOf(4))
    check((QK4_0 / 2).isMultipleOf(blockLen))
    check(blocks.size % n == 0)
    val kBlocks = blocks.size / n
    val nGroups = n / 4
    val packed = ArrayList<BlockQ40x4>(nGroups * kBlocks)
    for (g in 0 until nGroups) {
        for (b in 0 until kBlocks) {
            val src: List<BlockQ40> = fromFn({ i -> blocks[(g * 4 + i) * kBlocks + b] })
            val p = BlockQ40x4(d = List(4) { ZERO }, qs = ByteArray(QK4_0 * 2) { 0.toByte() })
            for ((i, s) in src.enumerate()) {
                p.d[i] = s.d
            }
            for (i1 in 0 until (QK4_0 / 2) / blockLen) {
                val offset = i1 * blockLen
                val dstOffset = i1 * blockLen * 4
                for ((j, s1) in src.enumerate()) {
                    val dstOffset1 = dstOffset + j * blockLen
                    for (k in 0 until blockLen) {
                        p.qs[dstOffset1 + k] = (s1.qs[offset + k] xor 0x88).toByte()
                    }
                }
            }
            packed.add(p)
        }
    }
    return packed
}

public fun packToQ80x4(blocks: List<BlockQ80>, n: Int, blockLen: Int): List<BlockQ80x4> {
    check(n.isMultipleOf(4))
    check(QK8_0.isMultipleOf(blockLen))
    check(blocks.size % n == 0)
    val kBlocks = blocks.size / n
    val nGroups = n / 4
    val packed = ArrayList<BlockQ80x4>(nGroups * kBlocks)
    for (g in 0 until nGroups) {
        for (b in 0 until kBlocks) {
            val src: List<BlockQ80> = fromFn({ i -> blocks[(g * 4 + i) * kBlocks + b] })
            val p = BlockQ80x4(d = List(4) { ZERO }, qs = ByteArray(QK8_0 * 4) { 0.toByte() })
            for ((i, s) in src.enumerate()) {
                p.d[i] = s.d
            }
            for (i1 in 0 until QK8_0 / blockLen) {
                val offset = i1 * blockLen
                val dstOffset = i1 * blockLen * 4
                for ((j, s1) in src.enumerate()) {
                    val dstOffset1 = dstOffset + j * blockLen
                    ((p.qs).slice(dstOffset1 until dstOffset1 + blockLen)).copyFromSlice(((s1.qs).slice(offset until offset + blockLen)))
                }
            }
            packed.add(p)
        }
    }
    return packed
}

public fun gemvFusedQ4k(kind: PackedKind, k: Int, lhs: List<Float>, parts: List<Pair<QuantizedType, PackedCache>>, dsts: List<List<Float>>) {
    val kInBlocks = k / QK_K
    threadLocal(FUSED_LHS_SCRATCH, std, cell, RefCell, Vec, std, cell, RefCell, new, Vec, new)
    FUSED_LHS_SCRATCH.with(
        closure@{ cell ->
            var scratch = cell.borrowMut()
            val elemSize = sizeOf<BlockQ8K>(typeLayoutT = ObjectSlotLayout<BlockQ8K>())
            val requiredLen = (kInBlocks * elemSize).divCeil(8)
            if (scratch.len() < requiredLen) {
                scratch.resize(requiredLen, 0)
            }
            val lhsB: List<BlockQ8K> = fromRawPartsMut(mutPtrFromRustCast(scratch.asMutPtr()), kInBlocks)
            BlockQ8K.fromFloat(lhs, lhsB)
            var metas: List<List<Any?>> = ArrayList(parts.size)
            var total = 0
            for ((i, storage, cache) in parts.enumerate()) {
                val n = dsts[i].size
                val ps = cache.getOrInit(kind, { kind.pack(storage, n) })
                val v = ps ?: return@closure
                metas.add(listOf(v.asPtr().toInt(), n / 8, dsts[i].asMutPtr().toInt(), total))
                total += n / 8
            }
            val blkBytes = sizeOf<BlockQ4Kx8>(typeLayoutT = ObjectSlotLayout<BlockQ4Kx8>())
            val lhsPtr = lhsB.asPtr().toInt()
            val pool = barrierPool()
            pool.executeStatic(total, { range ->
                val lhsRow: List<BlockQ8K> = fromRawParts(ConstPtr(lhsPtr.toLong()), kInBlocks)
                for (gg in range) {
                    val (pptr, ng, dptr, off) = (metas.indexOf({ (_, ng, _, off) -> gg < off + ng }).takeIf { it >= 0 })!!
                    val g = gg - off
                    val xs = fromRawParts(ConstPtr((pptr + g * kInBlocks * blkBytes).toLong()), kInBlocks)
                    val r = vecDot8Q4kQ8k(k, xs, lhsRow)
                    run {
                        copyNonoverlapping(r.asPtr(), MutPtr(((mutPtrFromRustCast(dptr))).rawAddress + ((g * 8).toLong() * (PrimitiveLayouts.F32).strideBytes)), 8)
                    }
                }
            })
        },
    )
}

public fun gemvFusedQ6k(kind: PackedKind, k: Int, lhs: List<Float>, parts: List<Pair<QuantizedType, PackedCache>>, dsts: List<List<Float>>) {
    val kInBlocks = k / QK_K
    threadLocal(FUSED_LHS_SCRATCH, std, cell, RefCell, Vec, std, cell, RefCell, new, Vec, new)
    FUSED_LHS_SCRATCH.with(
        closure1@{ cell ->
            var scratch = cell.borrowMut()
            val elemSize = sizeOf<BlockQ8K>(typeLayoutT = ObjectSlotLayout<BlockQ8K>())
            val requiredLen = (kInBlocks * elemSize).divCeil(8)
            if (scratch.len() < requiredLen) {
                scratch.resize(requiredLen, 0)
            }
            val lhsB: List<BlockQ8K> = fromRawPartsMut(mutPtrFromRustCast(scratch.asMutPtr()), kInBlocks)
            BlockQ8K.fromFloat(lhs, lhsB)
            var metas: List<List<Any?>> = ArrayList(parts.size)
            var total = 0
            for ((i, storage, cache) in parts.enumerate()) {
                val n = dsts[i].size
                val ps = cache.getOrInit(kind, { kind.pack(storage, n) })
                val v = ps ?: return@closure1
                metas.add(listOf(v.asPtr().toInt(), n / 8, dsts[i].asMutPtr().toInt(), total))
                total += n / 8
            }
            val blkBytes = sizeOf<BlockQ6Kx8>(typeLayoutT = ObjectSlotLayout<BlockQ6Kx8>())
            val lhsPtr = lhsB.asPtr().toInt()
            val pool = barrierPool()
            pool.executeStatic(total, { range ->
                val lhsRow: List<BlockQ8K> = fromRawParts(ConstPtr(lhsPtr.toLong()), kInBlocks)
                for (gg in range) {
                    val (pptr, ng, dptr, off) = (metas.indexOf({ (_, ng, _, off) -> gg < off + ng }).takeIf { it >= 0 })!!
                    val g = gg - off
                    val xs = fromRawParts(ConstPtr((pptr + g * kInBlocks * blkBytes).toLong()), kInBlocks)
                    val r = vecDot8Q6kx8Q8k(k, xs, lhsRow)
                    run {
                        copyNonoverlapping(r.asPtr(), MutPtr(((mutPtrFromRustCast(dptr))).rawAddress + ((g * 8).toLong() * (PrimitiveLayouts.F32).strideBytes)), 8)
                    }
                }
            })
        },
    )
}

public fun gemvFusedQ80(kind: PackedKind, k: Int, lhs: List<Float>, parts: List<Pair<QuantizedType, PackedCache>>, dsts: List<List<Float>>) {
    val kInBlocks = k / QK8_0
    threadLocal(FUSED_LHS_SCRATCH, std, cell, RefCell, Vec, std, cell, RefCell, new, Vec, new)
    FUSED_LHS_SCRATCH.with(
        closure2@{ cell ->
            var scratch = cell.borrowMut()
            val elemSize = sizeOf<BlockQ80>(typeLayoutT = ObjectSlotLayout<BlockQ80>())
            val requiredLen = (kInBlocks * elemSize).divCeil(8)
            if (scratch.len() < requiredLen) {
                scratch.resize(requiredLen, 0)
            }
            val lhsB: List<BlockQ80> = fromRawPartsMut(mutPtrFromRustCast(scratch.asMutPtr()), kInBlocks)
            BlockQ80.fromFloat(lhs, lhsB)
            var metas: List<List<Any?>> = ArrayList(parts.size)
            var total = 0
            for ((i, storage, cache) in parts.enumerate()) {
                val n = dsts[i].size
                val ps = cache.getOrInit(kind, { kind.pack(storage, n) })
                val v = ps ?: return@closure2
                metas.add(listOf(v.asPtr().toInt(), n / 4, dsts[i].asMutPtr().toInt(), total))
                total += n / 4
            }
            val blkBytes = sizeOf<BlockQ80x4>(typeLayoutT = ObjectSlotLayout<BlockQ80x4>())
            val lhsPtr = lhsB.asPtr().toInt()
            val pool = barrierPool()
            pool.executeStatic(total, { range ->
                val lhsRow: List<BlockQ80> = fromRawParts(ConstPtr(lhsPtr.toLong()), kInBlocks)
                for (gg in range) {
                    val (pptr, ng, dptr, off) = (metas.indexOf({ (_, ng, _, off) -> gg < off + ng }).takeIf { it >= 0 })!!
                    val g = gg - off
                    val xs = fromRawParts(ConstPtr((pptr + g * kInBlocks * blkBytes).toLong()), kInBlocks)
                    val r = vecDot4Q80x4Q80(k, xs, lhsRow)
                    run {
                        copyNonoverlapping(r.asPtr(), MutPtr(((mutPtrFromRustCast(dptr))).rawAddress + ((g * 4).toLong() * (PrimitiveLayouts.F32).strideBytes)), 4)
                    }
                }
            })
        },
    )
}

internal fun tryGemvFused(k: Int, lhs: List<Float>, parts: List<Pair<QuantizedType, PackedCache>>, dsts: List<List<Float>>): Result<Boolean> {
    return run {
        if (parts.isEmpty() || parts.size != dsts.size) {
            return Ok(false)
        }
        val features = get()
        if (!features.dotprod) {
            return Ok(false)
        }
        val dtype = parts[0].first.dtype()
        if (parts.any({ (s, _) -> s.dtype() != dtype })) {
            return Ok(false)
        }
        val kind = PackedKind.select(dtype, Triple(1, k, dsts[0].size)) ?: return Ok(false)
        for (d in dsts.iter()) {
            if (PackedKind.select(dtype, Triple(1, k, d.len())) != kind) {
                return Ok(false)
            }
        }
        when {
            kind == PackedKind.Q4Kx8 && k.isMultipleOf(QK_K) -> {
                gemvFusedQ4k(kind, k, lhs, parts, dsts)
                Ok(true)
            }
            kind == PackedKind.Q6Kx8 && k.isMultipleOf(QK_K) -> {
                gemvFusedQ6k(kind, k, lhs, parts, dsts)
                Ok(true)
            }
            kind == PackedKind.Q80x4 && k.isMultipleOf(QK8_0) -> {
                gemvFusedQ80(kind, k, lhs, parts, dsts)
                Ok(true)
            }
            else -> Ok(false)
        }
    }
}

public fun indexedGemvQ4k(kind: PackedKind, storage: QuantizedType, cache: PackedCache, nExperts: Int, nOut: Int, k: Int, lhs: List<Float>, nRows: Int, ids: List<UInt>, topk: Int, dst: List<Float>) {
    val kInBlocks = k / QK_K
    val totalRows = nExperts * nOut
    val ps = cache.getOrInit(kind, { kind.pack(storage, totalRows) })
    val packed = ps ?: return
    val lhsQ: List<BlockQ8K> = ArrayList(nRows * kInBlocks)
    lhsQ.setLen(nRows * kInBlocks)
    val lhsQPtr = lhsQ.asMutPtr().toInt()
    val pool = barrierPool()
    pool.executeChunked(nRows, { range ->
        val lhsQPtr1 = MutPtr(lhsQPtr.toLong())
        for (r in range) {
            val row = ((lhs).slice(r * k until (r + 1) * k))
            val out = fromRawPartsMut(MutPtr((lhsQPtr1).rawAddress + ((r * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ8K>()).strideBytes)), kInBlocks)
            BlockQ8K.fromFloat(row, out)
        }
    })
    val groupsPerExpert = nOut / 8
    val nPairs = ids.size
    val totalUnits = nPairs * groupsPerExpert
    val packedPtr = packed.asPtr().toInt()
    val lhsQPtr1 = lhsQ.asPtr().toInt()
    val dstPtr = dst.asMutPtr().toInt()
    val blkBytes = sizeOf<BlockQ4Kx8>(typeLayoutT = ObjectSlotLayout<BlockQ4Kx8>())
    pool.executeChunked(totalUnits, { range ->
        val packedPtr1 = ConstPtr(packedPtr.toLong())
        val lhsQPtr2 = ConstPtr(lhsQPtr1.toLong())
        val dstPtr1 = MutPtr(dstPtr.toLong())
        for (unit in range) {
            val pair = unit / groupsPerExpert
            val g = unit % groupsPerExpert
            val expert = ids[pair].toInt()
            val row =
                if (nRows == nPairs) {
                    pair
                } else {
                    pair / topk
                }
            val lhsRow = fromRawParts(ConstPtr((lhsQPtr2).rawAddress + ((row * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ8K>()).strideBytes)), kInBlocks)
            val groupGlobal = expert * groupsPerExpert + g
            val xs = fromRawParts(ConstPtr(ConstPtr((packedPtr1).rawAddress + ((groupGlobal * kInBlocks * blkBytes).toLong() * (PrimitiveLayouts.U8).strideBytes)).rawAddress, ConstPtr((packedPtr1).rawAddress + ((groupGlobal * kInBlocks * blkBytes).toLong() * (PrimitiveLayouts.U8).strideBytes)).rawMeta), kInBlocks)
            val r = vecDot8Q4kQ8k(k, xs, lhsRow)
            run {
                copyNonoverlapping(r.asPtr(), MutPtr((dstPtr1).rawAddress + ((pair * nOut + g * 8).toLong() * (PrimitiveLayouts.F32).strideBytes)), 8)
            }
        }
    })
}

public fun indexedGemvQ6k(kind: PackedKind, storage: QuantizedType, cache: PackedCache, nExperts: Int, nOut: Int, k: Int, lhs: List<Float>, nRows: Int, ids: List<UInt>, topk: Int, dst: List<Float>) {
    val kInBlocks = k / QK_K
    val totalRows = nExperts * nOut
    val ps = cache.getOrInit(kind, { kind.pack(storage, totalRows) })
    val packed = ps ?: return
    val lhsQ: List<BlockQ8K> = ArrayList(nRows * kInBlocks)
    lhsQ.setLen(nRows * kInBlocks)
    val lhsQPtr = lhsQ.asMutPtr().toInt()
    val pool = barrierPool()
    pool.executeChunked(nRows, { range ->
        val lhsQPtr1 = MutPtr(lhsQPtr.toLong())
        for (r in range) {
            val row = ((lhs).slice(r * k until (r + 1) * k))
            val out = fromRawPartsMut(MutPtr((lhsQPtr1).rawAddress + ((r * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ8K>()).strideBytes)), kInBlocks)
            BlockQ8K.fromFloat(row, out)
        }
    })
    val groupsPerExpert = nOut / 8
    val nPairs = ids.size
    val totalUnits = nPairs * groupsPerExpert
    val packedPtr = packed.asPtr().toInt()
    val lhsQPtr1 = lhsQ.asPtr().toInt()
    val dstPtr = dst.asMutPtr().toInt()
    val blkBytes = sizeOf<BlockQ6Kx8>(typeLayoutT = ObjectSlotLayout<BlockQ6Kx8>())
    pool.executeChunked(totalUnits, { range ->
        val packedPtr1 = ConstPtr(packedPtr.toLong())
        val lhsQPtr2 = ConstPtr(lhsQPtr1.toLong())
        val dstPtr1 = MutPtr(dstPtr.toLong())
        for (unit in range) {
            val pair = unit / groupsPerExpert
            val g = unit % groupsPerExpert
            val expert = ids[pair].toInt()
            val row =
                if (nRows == nPairs) {
                    pair
                } else {
                    pair / topk
                }
            val lhsRow = fromRawParts(ConstPtr((lhsQPtr2).rawAddress + ((row * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ8K>()).strideBytes)), kInBlocks)
            val groupGlobal = expert * groupsPerExpert + g
            val xs = fromRawParts(ConstPtr(ConstPtr((packedPtr1).rawAddress + ((groupGlobal * kInBlocks * blkBytes).toLong() * (PrimitiveLayouts.U8).strideBytes)).rawAddress, ConstPtr((packedPtr1).rawAddress + ((groupGlobal * kInBlocks * blkBytes).toLong() * (PrimitiveLayouts.U8).strideBytes)).rawMeta), kInBlocks)
            val r = vecDot8Q6kx8Q8k(k, xs, lhsRow)
            run {
                copyNonoverlapping(r.asPtr(), MutPtr((dstPtr1).rawAddress + ((pair * nOut + g * 8).toLong() * (PrimitiveLayouts.F32).strideBytes)), 8)
            }
        }
    })
}

public fun indexedGemvQ80(kind: PackedKind, storage: QuantizedType, cache: PackedCache, nExperts: Int, nOut: Int, k: Int, lhs: List<Float>, nRows: Int, ids: List<UInt>, topk: Int, dst: List<Float>) {
    val kInBlocks = k / QK8_0
    val totalRows = nExperts * nOut
    val ps = cache.getOrInit(kind, { kind.pack(storage, totalRows) })
    val packed = ps ?: return
    val lhsQ: List<BlockQ80> = ArrayList(nRows * kInBlocks)
    lhsQ.setLen(nRows * kInBlocks)
    val lhsQPtr = lhsQ.asMutPtr().toInt()
    val pool = barrierPool()
    pool.executeChunked(nRows, { range ->
        val lhsQPtr1 = MutPtr(lhsQPtr.toLong())
        for (r in range) {
            val row = ((lhs).slice(r * k until (r + 1) * k))
            val out = fromRawPartsMut(MutPtr((lhsQPtr1).rawAddress + ((r * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ80>()).strideBytes)), kInBlocks)
            BlockQ80.fromFloat(row, out)
        }
    })
    val groupsPerExpert = nOut / 4
    val nPairs = ids.size
    val totalUnits = nPairs * groupsPerExpert
    val packedPtr = packed.asPtr().toInt()
    val lhsQPtr1 = lhsQ.asPtr().toInt()
    val dstPtr = dst.asMutPtr().toInt()
    val blkBytes = sizeOf<BlockQ80x4>(typeLayoutT = ObjectSlotLayout<BlockQ80x4>())
    pool.executeChunked(totalUnits, { range ->
        val packedPtr1 = ConstPtr(packedPtr.toLong())
        val lhsQPtr2 = ConstPtr(lhsQPtr1.toLong())
        val dstPtr1 = MutPtr(dstPtr.toLong())
        for (unit in range) {
            val pair = unit / groupsPerExpert
            val g = unit % groupsPerExpert
            val expert = ids[pair].toInt()
            val row =
                if (nRows == nPairs) {
                    pair
                } else {
                    pair / topk
                }
            val lhsRow = fromRawParts(ConstPtr((lhsQPtr2).rawAddress + ((row * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ80>()).strideBytes)), kInBlocks)
            val groupGlobal = expert * groupsPerExpert + g
            val xs = fromRawParts(ConstPtr(ConstPtr((packedPtr1).rawAddress + ((groupGlobal * kInBlocks * blkBytes).toLong() * (PrimitiveLayouts.U8).strideBytes)).rawAddress, ConstPtr((packedPtr1).rawAddress + ((groupGlobal * kInBlocks * blkBytes).toLong() * (PrimitiveLayouts.U8).strideBytes)).rawMeta), kInBlocks)
            val r = vecDot4Q80x4Q80(k, xs, lhsRow)
            run {
                copyNonoverlapping(r.asPtr(), MutPtr((dstPtr1).rawAddress + ((pair * nOut + g * 4).toLong() * (PrimitiveLayouts.F32).strideBytes)), 4)
            }
        }
    })
}

public val INDEXED_BUCKET_MIN_PAIRS: Int = 32

internal fun tryIndexedGemv(storage: QuantizedType, cache: PackedCache, nExperts: Int, nOut: Int, k: Int, lhs: List<Float>, nRows: Int, ids: List<UInt>, topk: Int, dst: List<Float>): Result<Boolean> {
    return run {
        val features = get()
        if (!features.dotprod) {
            return Ok(false)
        }
        val kind = PackedKind.select(storage.dtype(), Triple(1, k, nExperts * nOut)) ?: return Ok(false)
        // prefill: bucket pairs by expert and run one tiled matmul per expert instead
        // of a gemv per (token, expert) pair
        if (ids.size >= INDEXED_BUCKET_MIN_PAIRS &&
            when (kind) {
                PackedKind.Q4Kx8, PackedKind.Q6Kx8, PackedKind.Q80x4 -> true
                else -> false
            }
        ) {
            val ps = cache.getOrInit(kind, { kind.pack(storage, nExperts * nOut) })
            val nPairs = ids.size
            val buckets: List<List<Int>> = mutableListOf(mutableListOf(), nExperts)
            for ((pair, e) in ids.enumerate()) {
                buckets[e.toInt()].push(pair)
            }
            val gathered: List<Float> = mutableListOf()
            val scratch: List<Float> = mutableListOf()
            for ((e1, bucket) in buckets.enumerate()) {
                if (bucket.isEmpty()) {
                    continue
                }
                val cnt = bucket.len()
                val cntPad = cnt.nextMultipleOf(4)
                gathered.clear()
                gathered.resize(cntPad * k, 0.0)
                for ((i, pair1) in bucket.iter().enumerate()) {
                    val row =
                        if (nRows == nPairs) {
                            pair1
                        } else {
                            pair1 / topk
                        }
                    ((gathered).slice(i * k until (i + 1) * k)).copyFromSlice(((lhs).slice(row * k until (row + 1) * k)))
                }
                scratch.clear()
                scratch.resize(cntPad * nOut, 0.0)
                when (val tryResult = kind.matmulExpertRange(Triple(cntPad, k, nOut), gathered, ps, e1, scratch)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
                for ((i1, pair2) in bucket.iter().enumerate()) {
                    ((dst).slice(pair2 * nOut until (pair2 + 1) * nOut)).copyFromSlice(((scratch).slice(i1 * nOut until (i1 + 1) * nOut)))
                }
            }
            return Ok(true)
        }
        when {
            kind == PackedKind.Q4Kx8 && k.isMultipleOf(QK_K) && nOut.isMultipleOf(8) -> {
                indexedGemvQ4k(kind, storage, cache, nExperts, nOut, k, lhs, nRows, ids, topk, dst)
                Ok(true)
            }
            kind == PackedKind.Q6Kx8 && k.isMultipleOf(QK_K) && nOut.isMultipleOf(8) -> {
                indexedGemvQ6k(kind, storage, cache, nExperts, nOut, k, lhs, nRows, ids, topk, dst)
                Ok(true)
            }
            kind == PackedKind.Q80x4 && k.isMultipleOf(QK8_0) && nOut.isMultipleOf(4) -> {
                indexedGemvQ80(kind, storage, cache, nExperts, nOut, k, lhs, nRows, ids, topk, dst)
                Ok(true)
            }
            else -> Ok(false)
        }
    }
}
