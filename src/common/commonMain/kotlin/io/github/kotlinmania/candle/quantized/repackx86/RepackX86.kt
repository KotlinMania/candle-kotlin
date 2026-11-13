// port-lint: source quantized/repack_x86.rs
@file:OptIn(ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.quantized.repackx86

import io.github.kotlinmania.candle.quantized.GgmlDType
import io.github.kotlinmania.candle.quantized.QuantizedType
import io.github.kotlinmania.candle.quantized.kquants.BlockQ4K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ6K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ80
import io.github.kotlinmania.candle.quantized.kquants.BlockQ8K
import io.github.kotlinmania.candle.quantized.kquants.QK8_0
import io.github.kotlinmania.candle.quantized.kquants.QK_K
import io.github.kotlinmania.candle.utils.barrierPool
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.libc.SysArchPrctl
import io.github.kotlinmania.libc.syscall
import io.github.kotlinmania.rustlang.core.arch.x8664.*
import io.github.kotlinmania.rustlang.core.mem.AddressableArrayLocal
import io.github.kotlinmania.rustlang.core.mem.FixedArrayTypeLayout
import io.github.kotlinmania.rustlang.core.mem.ObjectSlotLayout
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.mem.constPtrFromRustCast
import io.github.kotlinmania.rustlang.core.mem.mutPtrFromRustCast
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr
import io.github.kotlinmania.rustlang.core.result.Result
import io.github.kotlinmania.candle.quantized.GgmlDType as D

// x86 repacked quantized kernels: weights tiled [n/16][k-blocks][16 rows] so one
// vpdpbusd (AVX512-VNNI) produces 16 output columns for 4 k-values per instruction.
// Activations are BlockQ8K rows; q4k/q6k quants are unsigned so vpdpbusd applies
// directly with the min/offset terms corrected through the q8k bsums, and q80
// weights are stored offset by +128 with the same correction.

internal val TILE_N: Int = 16

// One k-superblock (256 values) for 16 output rows.
internal data class TileQ4K(
    // per-subblock (8) per-row (16) effective scales/mins, pre-unpacked from the 6-bit fields
    internal var scales: List<FloatArray> = List(8) { FloatArray(0) },
    internal var mins: List<FloatArray> = List(8) { FloatArray(0) },
    // nibble-packed: [k8-group (32)][row (16)][4 bytes]; byte j = q[8g+j] | q[8g+j+4] << 4
    internal var qs: UByteArray = UByteArray(QK_K / 8 * TILE_N * 4),
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TileQ4K) return false
        if (this.scales != other.scales) return false
        if (this.mins != other.mins) return false
        if (!this.qs.contentEquals(other.qs)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = this.scales.hashCode()
        result = 31 * result + this.mins.hashCode()
        result = 31 * result + this.qs.contentHashCode()
        return result
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<TileQ4K> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<TileQ4K> {
                override val alignBytes: Long = 4L
                private val off1: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes(
                            (
                                (8).toLong() *
                                    (
                                        io.github.kotlinmania.rustlang.core.mem
                                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32, (TILE_N).toInt())
                                    ).strideBytes
                            ),
                            4L,
                        )
                private val off2: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes(
                            (
                                off1 + (
                                    (8).toLong() *
                                        (
                                            io.github.kotlinmania.rustlang.core.mem
                                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32, (TILE_N).toInt())
                                        ).strideBytes
                                )
                            ),
                            1L,
                        )
                override val sizeBytes: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes((off2 + ((QK_K / 8 * TILE_N * 4).toLong() * (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U8).strideBytes)), 4L)
                override fun readNative(addr: Long): TileQ4K =
                    TileQ4K(
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(
                                    io.github.kotlinmania.rustlang.core.mem
                                        .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32, (TILE_N).toInt()),
                                    (8).toInt(),
                                )
                        ).readNative(addr).map { it.toFloatArray() },
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(
                                    io.github.kotlinmania.rustlang.core.mem
                                        .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32, (TILE_N).toInt()),
                                    (8).toInt(),
                                )
                        ).readNative(addr + off1).map { it.toFloatArray() },
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U8, (QK_K / 8 * TILE_N * 4).toInt())
                        ).readNative(addr + off2).toUByteArray(),
                    )
                override fun writeNative(addr: Long, value: TileQ4K) {
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(
                                io.github.kotlinmania.rustlang.core.mem
                                    .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32, (TILE_N).toInt()),
                                (8).toInt(),
                            )
                    ).writeNative(addr, value.scales.map { it.toList() })
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(
                                io.github.kotlinmania.rustlang.core.mem
                                    .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32, (TILE_N).toInt()),
                                (8).toInt(),
                            )
                    ).writeNative(addr + off1, value.mins.map { it.toList() })
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U8, (QK_K / 8 * TILE_N * 4).toInt())
                    ).writeNative(addr + off2, value.qs.toList())
                }
            }
    }
}

internal data class TileQ6K(
    internal var scales: List<FloatArray> = List(16) { FloatArray(0) },
    // raw 6-bit values as u8 (0..63; real value is v - 32): [k4-group (64)][row (16)][4 bytes]
    internal var qs: UByteArray = UByteArray(QK_K / 4 * TILE_N * 4),
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TileQ6K) return false
        if (this.scales != other.scales) return false
        if (!this.qs.contentEquals(other.qs)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = this.scales.hashCode()
        result = 31 * result + this.qs.contentHashCode()
        return result
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<TileQ6K> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<TileQ6K> {
                override val alignBytes: Long = 4L
                private val off1: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes(
                            (
                                (16).toLong() *
                                    (
                                        io.github.kotlinmania.rustlang.core.mem
                                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32, (TILE_N).toInt())
                                    ).strideBytes
                            ),
                            1L,
                        )
                override val sizeBytes: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes((off1 + ((QK_K / 4 * TILE_N * 4).toLong() * (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U8).strideBytes)), 4L)
                override fun readNative(addr: Long): TileQ6K =
                    TileQ6K(
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(
                                    io.github.kotlinmania.rustlang.core.mem
                                        .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32, (TILE_N).toInt()),
                                    (16).toInt(),
                                )
                        ).readNative(addr).map { it.toFloatArray() },
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U8, (QK_K / 4 * TILE_N * 4).toInt())
                        ).readNative(addr + off1).toUByteArray(),
                    )
                override fun writeNative(addr: Long, value: TileQ6K) {
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(
                                io.github.kotlinmania.rustlang.core.mem
                                    .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32, (TILE_N).toInt()),
                                (16).toInt(),
                            )
                    ).writeNative(addr, value.scales.map { it.toList() })
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U8, (QK_K / 4 * TILE_N * 4).toInt())
                    ).writeNative(addr + off1, value.qs.toList())
                }
            }
    }
}

// One 32-value block group for 16 rows of q80 weights, stored offset by +128.
internal data class TileQ80(
    internal var d: List<F16> = List(TILE_N) { F16() },
    internal var qs: UByteArray = UByteArray(QK8_0 * TILE_N),
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TileQ80) return false
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

internal sealed class PackedX86 {
    public data class Q4k(
        public var value: List<TileQ4K>,
    ) : PackedX86()
    public data class Q6k(
        public var value: List<TileQ6K>,
    ) : PackedX86()
    public data class Q80(
        public var value: List<TileQ80>,
    ) : PackedX86()
}

internal fun select(dtype: GgmlDType, m: Int, n: Int, k: Int): Boolean {
    val lv = level() ?: return false
    if (k == 0 || !n.isMultipleOf(TILE_N) || !k.isMultipleOf(QK_K)) {
        return false
    }
    return when (dtype) {
        D.q4k -> true
        // Keep Q6K GEMV on compact weights; expanded tiles cost bandwidth
        // without reuse across multiple activation rows.
        D.q6k -> m > 1
        // pure avx2 maddubs overflows on the +128 u8 range; q80 needs real dpbusd
        D.q80 -> lv != X86Level.Avx2
        else -> false
    }
}

public fun q4kBlocks(storage: QuantizedType): List<BlockQ4K> {
    val data = storage.storageSizeInBytes()
    val ptr = constPtrFromRustCast(storage.asPtr())
    return fromRawParts(ptr, data / sizeOf<BlockQ4K>(typeLayoutT = ObjectSlotLayout<BlockQ4K>()))
}

public fun q6kBlocks(storage: QuantizedType): List<BlockQ6K> {
    val data = storage.storageSizeInBytes()
    val ptr = constPtrFromRustCast(storage.asPtr())
    return fromRawParts(ptr, data / sizeOf<BlockQ6K>(typeLayoutT = ObjectSlotLayout<BlockQ6K>()))
}

public fun q80Blocks(storage: QuantizedType): List<BlockQ80> {
    val data = storage.storageSizeInBytes()
    val ptr = constPtrFromRustCast(storage.asPtr())
    return fromRawParts(ptr, data / sizeOf<BlockQ80>(typeLayoutT = ObjectSlotLayout<BlockQ80>()))
}

// ggml's 6-bit scale/min unpack for q4k/q5k superblocks.
public fun getScaleMinK4(j: Int, scales: UByteArray): Pair<UByte, UByte> =
    if (j < 4) {
        Pair(scales[j] and 63u.toUByte(), scales[j + 4] and 63u.toUByte())
    } else {
        Pair((scales[j + 4] and 0x0Fu.toUByte()) or ((((scales[j - 4].toUInt() shr 6).toUByte()).toUInt() shl 4).toUByte()), ((scales[j + 4].toUInt() shr 4).toUByte()) or ((((scales[j].toUInt() shr 6).toUByte()).toUInt() shl 4).toUByte()))
    }

internal fun packQ4k(storage: QuantizedType, n: Int, k: Int): PackedX86 {
    val blocks = q4kBlocks(storage)
    val kb = k / QK_K
    val nTiles = n / TILE_N
    val out: List<TileQ4K> = ArrayList(nTiles * kb)
    for (t in 0 until nTiles) {
        for (b in 0 until kb) {
            val tile = TileQ4K(scales = List(8) { FloatArray(TILE_N) { 0.0f } }, mins = List(8) { FloatArray(TILE_N) { 0.0f } }, qs = UByteArray(QK_K / 8 * TILE_N * 4) { 0u.toUByte() })
            for (r in 0 until TILE_N) {
                val blk = blocks[(t * TILE_N + r) * kb + b]
                val d = blk.d.toF32()
                val dmin = blk.dmin.toF32()
                for (sub in 0 until 8) {
                    val (sc, mn) = getScaleMinK4(sub, blk.scales)
                    tile.scales[sub][r] = d * sc.toFloat()
                    tile.mins[sub][r] = dmin * mn.toFloat()
                }
                // q4k qs: 32-byte chunks c hold values [c*64, c*64+32) in the low nibbles
                // and [c*64+32, c*64+64) in the high nibbles
                val vals = List(QK_K) { 0u.toUByte() }
                for (c in 0 until 4) {
                    for (j in 0 until 32) {
                        val byte = blk.qs[c * 32 + j]
                        vals[c * 64 + j] = byte and 0x0Fu.toUByte()
                        vals[c * 64 + 32 + j] = (byte.toUInt() shr 4).toUByte()
                    }
                }
                for (g in 0 until QK_K / 8) {
                    for (j1 in 0 until 4) {
                        tile.qs[(g * TILE_N + r) * 4 + j1] = (vals[g * 8 + j1] or ((vals[g * 8 + j1 + 4].toUInt() shl 4).toUByte())).toUByte()
                    }
                }
            }
            out.add(tile)
        }
    }
    return PackedX86.q4k(out)
}

internal fun packQ6k(storage: QuantizedType, n: Int, k: Int): PackedX86 {
    val blocks = q6kBlocks(storage)
    val kb = k / QK_K
    val nTiles = n / TILE_N
    val out: List<TileQ6K> = ArrayList(nTiles * kb)
    for (t in 0 until nTiles) {
        for (b in 0 until kb) {
            val tile = TileQ6K(scales = List(16) { FloatArray(TILE_N) { 0.0f } }, qs = UByteArray(QK_K / 4 * TILE_N * 4) { 0u.toUByte() })
            for (r in 0 until TILE_N) {
                val blk = blocks[(t * TILE_N + r) * kb + b]
                val d = blk.d.toF32()
                for (sub in 0 until 16) {
                    tile.scales[sub][r] = d * blk.scales[sub].toFloat()
                }
                // q6k: ql 128 bytes (low 4 bits), qh 64 bytes (high 2 bits), in 128-value halves
                val vals = List(QK_K) { 0u.toUByte() }
                for (h in 0 until 2) {
                    val ql = ((blk.ql).slice(h * 64 until blk.ql.size))
                    val qh = ((blk.qh).slice(h * 32 until blk.qh.size))
                    for (j in 0 until 32) {
                        vals[h * 128 + j] = (ql[j] and 0x0Fu.toUByte()) or (((qh[j] and 3u.toUByte()).toUInt() shl 4).toUByte())
                        vals[h * 128 + j + 32] = (ql[j + 32] and 0x0Fu.toUByte()) or (((((qh[j].toUInt() shr 2).toUByte()) and 3u.toUByte()).toUInt() shl 4).toUByte())
                        vals[h * 128 + j + 64] = ((ql[j].toUInt() shr 4).toUByte()) or (((((qh[j].toUInt() shr 4).toUByte()) and 3u.toUByte()).toUInt() shl 4).toUByte())
                        vals[h * 128 + j + 96] = ((ql[j + 32].toUInt() shr 4).toUByte()) or (((((qh[j].toUInt() shr 6).toUByte()) and 3u.toUByte()).toUInt() shl 4).toUByte())
                    }
                }
                for (g in 0 until QK_K / 4) {
                    for (j1 in 0 until 4) {
                        tile.qs[(g * TILE_N + r) * 4 + j1] = vals[g * 4 + j1]
                    }
                }
            }
            out.add(tile)
        }
    }
    return PackedX86.q6k(out)
}

internal fun packQ80(storage: QuantizedType, n: Int, k: Int): PackedX86 {
    val blocks = q80Blocks(storage)
    val kb = k / QK8_0
    val nTiles = n / TILE_N
    val out: List<TileQ80> = ArrayList(nTiles * kb)
    for (t in 0 until nTiles) {
        for (b in 0 until kb) {
            val tile = TileQ80(d = List(TILE_N) { F16.ZERO }, qs = UByteArray(QK8_0 * TILE_N) { 0u.toUByte() })
            for (r in 0 until TILE_N) {
                val blk = blocks[(t * TILE_N + r) * kb + b]
                tile.d[r] = blk.d
                for (g in 0 until QK8_0 / 4) {
                    for (j in 0 until 4) {
                        tile.qs[(g * TILE_N + r) * 4 + j] = (blk.qs[g * 4 + j].toShort() + 128).toUByte()
                    }
                }
            }
            out.add(tile)
        }
    }
    return PackedX86.q80(out)
}

// acc[m][16] f32 for up to MT activation rows against one 16-row weight tile
internal val MAX_MT: Int = 4

public fun bcast4(act: List<Byte>, off: ULong): M512i = mm512Set1Epi32(readUnaligned(constPtrFromRustCast(act.asPtr().add(off)), typeLayoutT = PrimitiveLayouts.I32))

// q4k: one superblock tile against `mt` q8k activation blocks. Subblocks are
// processed in pairs with independent integer accumulators to break the dpbusd
// latency chain when mt is small (decode gemv).
internal fun q4kTile(tile: TileQ4K, acts: List<BlockQ8K>, acc: MutableList<List<M512>>) {
    val mask = mm512Set1Epi8((0x0Fu.toUByte()).toByte())
    for (sub2 in 0 until 4) {
        val sa = sub2 * 2
        val sb = sa + 1
        val isumA = List(MAX_MT) { mm512SetzeroSi512() }
        val isumB = List(MAX_MT) { mm512SetzeroSi512() }
        for (g4 in 0 until 4) {
            val ga = sa * 4 + g4
            val gb = sb * 4 + g4
            val wa = mm512LoaduSi512(constPtrFromRustCast(tile.qs.asPtr().add(ga * TILE_N * 4)))
            val wb = mm512LoaduSi512(constPtrFromRustCast(tile.qs.asPtr().add(gb * TILE_N * 4)))
            val loA = mm512AndSi512(wa, mask)
            val hiA = mm512AndSi512(mm512SrliEpi16(wa, 4), mask)
            val loB = mm512AndSi512(wb, mask)
            val hiB = mm512AndSi512(mm512SrliEpi16(wb, 4), mask)
            for ((m, act) in acts.mapIndexed { index, item -> Pair(index, item) }) {
                isumA[m] = mm512DpbusdEpi32(isumA[m], loA, bcast4(act.qs, sa * 32 + g4 * 8))
                isumA[m] = mm512DpbusdEpi32(isumA[m], hiA, bcast4(act.qs, sa * 32 + g4 * 8 + 4))
                isumB[m] = mm512DpbusdEpi32(isumB[m], loB, bcast4(act.qs, sb * 32 + g4 * 8))
                isumB[m] = mm512DpbusdEpi32(isumB[m], hiB, bcast4(act.qs, sb * 32 + g4 * 8 + 4))
            }
        }
        for ((sub, isum) in listOf(Pair(sa, isumA), Pair(sb, isumB))) {
            val scale = mm512LoaduPs(tile.scales[sub].asPtr())
            val min = mm512LoaduPs(tile.mins[sub].asPtr())
            for ((m1, act1) in acts.mapIndexed { index, item -> Pair(index, item) }) {
                val dall = act1.d
                val f = mm512Cvtepi32Ps(isum[m1])
                acc[m1][0] = mm512FmaddPs(f, mm512MulPs(scale, mm512Set1Ps(dall)), acc[m1][0])
                val bsum = (act1.bsums[sub * 2].toInt() + act1.bsums[sub * 2 + 1].toInt()).toFloat()
                acc[m1][0] = mm512FnmaddPs(min, mm512Set1Ps(dall * bsum), acc[m1][0])
            }
        }
    }
}

// q6k: values are v-32; vpdpbusd on raw u6 then subtract 32 * bsum per 16-value subblock.
internal fun q6kTile(tile: TileQ6K, acts: List<BlockQ8K>, acc: MutableList<List<M512>>) {
    for (sub in 0 until 16) {
        val isum = List(MAX_MT) { mm512SetzeroSi512() }
        // subblock = 16 k-values = 4 k4-groups
        for (g4 in 0 until 4) {
            val g = sub * 4 + g4
            val w = mm512LoaduSi512(constPtrFromRustCast(tile.qs.asPtr().add(g * TILE_N * 4)))
            for ((m, act) in acts.mapIndexed { index, item -> Pair(index, item) }) {
                val a = bcast4(act.qs, sub * 16 + g4 * 4)
                isum[m] = mm512DpbusdEpi32(isum[m], w, a)
            }
        }
        val scale = mm512LoaduPs(tile.scales[sub].asPtr())
        for ((m1, act1) in acts.mapIndexed { index, item -> Pair(index, item) }) {
            val dall = act1.d
            val bsum = act1.bsums[sub].toFloat()
            val f = mm512SubPs(mm512Cvtepi32Ps(isum[m1]), mm512Set1Ps(32.0f * bsum))
            acc[m1][0] = mm512FmaddPs(f, mm512MulPs(scale, mm512Set1Ps(dall)), acc[m1][0])
        }
    }
}

// q80 (+128 offset weights), two 32-value block groups per call with independent
// integer accumulators so the dpbusd latency chain never serializes at small m.
internal fun q80Tile2(t0: TileQ80, t1: TileQ80, acts: List<BlockQ8K>, sub32: Int, acc: MutableList<List<M512>>) {
    val isumA = List(MAX_MT) { mm512SetzeroSi512() }
    val isumB = List(MAX_MT) { mm512SetzeroSi512() }
    for (g4 in 0 until 8) {
        val wa = mm512LoaduSi512(constPtrFromRustCast(t0.qs.asPtr().add(g4 * TILE_N * 4)))
        val wb = mm512LoaduSi512(constPtrFromRustCast(t1.qs.asPtr().add(g4 * TILE_N * 4)))
        for ((m, act) in acts.mapIndexed { index, item -> Pair(index, item) }) {
            isumA[m] = mm512DpbusdEpi32(isumA[m], wa, bcast4(act.qs, sub32 * 32 + g4 * 4))
            isumB[m] = mm512DpbusdEpi32(isumB[m], wb, bcast4(act.qs, (sub32 + 1) * 32 + g4 * 4))
        }
    }
    for ((blk, isum, tile) in listOf(Triple(sub32, isumA, t0), Triple(sub32 + 1, isumB, t1))) {
        val dw = mm512CvtphPs(mm256LoaduSi256(constPtrFromRustCast(tile.d.asPtr())))
        for ((m1, act1) in acts.mapIndexed { index, item -> Pair(index, item) }) {
            val bsum = (act1.bsums[blk * 2].toInt() + act1.bsums[blk * 2 + 1].toInt()).toFloat()
            val f = mm512SubPs(mm512Cvtepi32Ps(isum[m1]), mm512Set1Ps(128.0f * bsum))
            acc[m1][0] = mm512FmaddPs(f, mm512MulPs(dw, mm512Set1Ps(act1.d)), acc[m1][0])
        }
    }
}

// Full matmul over the packed tiles: lhs is m rows already quantized to BlockQ8K,
// parallelized over (m-tiles x n-tiles) on the barrier pool by the caller's chunker.
// Per-unit bodies live in @targetFeature fns so intrinsics inline under portable
// (target-cpu=generic) release builds; closures cannot carry the attribute.
public fun matmulUnit512(packed: PackedX86, lhsQ: List<BlockQ8K>, kb: Int, nt: Int, m0: Int, mt: Int, dstPtr: MutPtr, n: Int) {
    val actRefs: List<BlockQ8K> = List((MAX_MT).toInt()) { lhsQ[m0 * kb] }
    val acc = List((MAX_MT).toInt()) { List(1) { mm512SetzeroPs() } }
    for (b in 0 until kb) {
        for ((i, a) in actRefs.mapIndexed { index, item -> Pair(index, item) }.take(mt)) {
            a = lhsQ[(m0 + i) * kb + b]
        }
        when (packed) {
            is PackedX86.Q4k -> {
                val (tiles) = packed
                q4kTile(tiles[nt * kb + b], ((actRefs).slice(0 until mt)), acc)
            }
            is PackedX86.Q6k -> {
                val (tiles1) = packed
                q6kTile(tiles1[nt * kb + b], ((actRefs).slice(0 until mt)), acc)
            }
            is PackedX86.Q80 -> {
                val (tiles2) = packed
                // q80 tiles cover 32 k-values each; 8 tiles per q8k superblock
                for (s in (0 until 8).stepBy(2)) {
                    q80Tile2(tiles2[nt * (kb * 8) + b * 8 + s], tiles2[nt * (kb * 8) + b * 8 + s + 1], ((actRefs).slice(0 until mt)), s, acc)
                }
            }
        }
    }
    for ((i1, row) in ((acc).slice(0 until mt)).mapIndexed { index, item -> Pair(index, item) }) {
        mm512StoreuPs(MutPtr((dstPtr).rawAddress + (((m0 + i1) * n + nt * TILE_N).toLong() * (PrimitiveLayouts.F32).strideBytes)), row[0])
    }
}

public fun matmulUnit256(packed: PackedX86, lhsQ: List<BlockQ8K>, kb: Int, nt: Int, m0: Int, mt: Int, dstPtr: MutPtr, n: Int, vnni: Boolean) {
    val actRefs: List<BlockQ8K> = List((MAX_MT).toInt()) { lhsQ[m0 * kb] }
    val acc = List((MAX_MT).toInt()) { List(2) { mm256SetzeroPs() } }
    for (b in 0 until kb) {
        for ((i, a) in actRefs.mapIndexed { index, item -> Pair(index, item) }.take(mt)) {
            a = lhsQ[(m0 + i) * kb + b]
        }
        when (packed) {
            is PackedX86.Q4k -> {
                val (tiles) = packed
                val t = tiles[nt * kb + b]
                if (vnni) {
                    q4kTileAvxvnni(t, ((actRefs).slice(0 until mt)), acc)
                } else {
                    q4kTileAvx2(t, ((actRefs).slice(0 until mt)), acc)
                }
            }
            is PackedX86.Q6k -> {
                val (tiles1) = packed
                val t1 = tiles1[nt * kb + b]
                if (vnni) {
                    q6kTileAvxvnni(t1, ((actRefs).slice(0 until mt)), acc)
                } else {
                    q6kTileAvx2(t1, ((actRefs).slice(0 until mt)), acc)
                }
            }
            is PackedX86.Q80 -> {
                val (tiles2) = packed
                for (s in (0 until 8).stepBy(2)) {
                    q80Tile2Avxvnni(tiles2[nt * (kb * 8) + b * 8 + s], tiles2[nt * (kb * 8) + b * 8 + s + 1], ((actRefs).slice(0 until mt)), s, acc)
                }
            }
        }
    }
    for ((i1, row) in ((acc).slice(0 until mt)).mapIndexed { index, item -> Pair(index, item) }) {
        mm256StoreuPs(MutPtr((dstPtr).rawAddress + (((m0 + i1) * n + nt * TILE_N).toLong() * (PrimitiveLayouts.F32).strideBytes)), row[0])
        mm256StoreuPs(MutPtr((dstPtr).rawAddress + (((m0 + i1) * n + nt * TILE_N + 8).toLong() * (PrimitiveLayouts.F32).strideBytes)), row[1])
    }
}

internal fun matmulTiles(packed: PackedX86, lhsQ: List<BlockQ8K>, m: Int, k: Int, n: Int, dst: MutableList<Float>) {
    run {
        val lv = (level() ?: error("select() gated"))
        val kb = k / QK_K
        val nTiles = n / TILE_N
        val mtStep = MAX_MT
        val dstPtr = dst.asMutPtr().toInt()
        val totalUnits = nTiles * m.divCeil(mtStep)
        val mUnits = m.divCeil(mtStep)
        barrierPool().executeChunked(totalUnits, { range ->
            val dstPtr1 = MutPtr(dstPtr.toLong())
            for (unit in range) {
                val nt = unit / mUnits
                val m0 = (unit % mUnits) * mtStep
                val mt = (m - m0).min(mtStep)
                if (lv == X86Level.Avx512Vnni) {
                    matmulUnit512(packed, lhsQ, kb, nt, m0, mt, dstPtr1, n)
                } else {
                    matmulUnit256(packed, lhsQ, kb, nt, m0, mt, dstPtr1, n, lv == X86Level.AvxVnni)
                }
            }
        })
    }
}

internal fun pack(dtype: GgmlDType, storage: QuantizedType, n: Int, k: Int): PackedX86 =
    when (dtype) {
        D.q4k -> packQ4k(storage, n, k)
        D.q6k -> packQ6k(storage, n, k)
        D.q80 -> packQ80(storage, n, k)
        else -> error("select() gates dtypes")
    }

// Quantize lhs rows to BlockQ8K, parallelizing only when there are multiple rows.
internal fun quantizeLhs(lhs: List<Float>, m: Int, k: Int): List<BlockQ8K> {
    val kb = k / QK_K
    val out = mutableListOf(BlockQ8K.zeros(), m * kb)
    if (m == 1) {
        BlockQ8K.fromFloat(((lhs).slice(0 until k)), out)
        return out
    }
    val outPtr = out.asMutPtr().toInt()
    barrierPool().executeChunked(m, { range ->
        val outPtr1 = MutPtr(outPtr.toLong())
        for (r in range) {
            val row = ((lhs).slice(r * k until (r + 1) * k))
            val dst = fromRawPartsMut(MutPtr((outPtr1).rawAddress + ((r * kb).toLong() * (ObjectSlotLayout<BlockQ8K>()).strideBytes)), kb)
            BlockQ8K.fromFloat(row, dst)
        }
    })
    return out
}

// Fused multi-projection gemv: one lhs quantization and one barrier region across
// all projections sharing the same activations (qkv, gate/up). 512-bit only; the
// per-unit body reuses matmulUnit512 so intrinsics inline in portable builds.
internal fun gemvFused(parts: List<Pair<PackedX86, Int>>, lhsQ: List<BlockQ8K>, m: Int, k: Int, dsts: MutableList<List<Float>>) {
    run {
        val kb = k / QK_K
        val mtStep = MAX_MT
        val mUnits = m.divCeil(mtStep)
        // (n, dst ptr, unit offset)
        val metas: List<Triple<Int, Int, Int>> = ArrayList(parts.size)
        var total = 0
        for ((i, p, n) in parts.mapIndexed { index, item -> Pair(index, item) }) {
            metas.add(Triple(n, dsts[i].asMutPtr().toInt(), total))
            total += (n / TILE_N) * mUnits
        }
        val partsRef = parts
        barrierPool().executeChunked(total, { range ->
            for (unit in range) {
                val pi = (ArrayDeque(metas).indexOfLast({ (n, d, off) -> unit >= off }).takeIf { it >= 0 })!!
                val (n1, dstPtr, off) = metas[pi]
                val local = unit - off
                val nt = local / mUnits
                val m0 = (local % mUnits) * mtStep
                val mt = (m - m0).min(mtStep)
                run {
                    matmulUnit512(partsRef[pi].first, lhsQ, kb, nt, m0, mt, mutPtrFromRustCast(dstPtr), n1)
                }
            }
        })
    }
}

// AMX-int8 path for large-m (prefill) q4k matmuls: weights unpacked to i8 in tile
// layout (B tile = 8 rows x 64 bytes = 32k x 16n VNNI-paired), activations repacked
// per 16-row block, tdpbssd accumulating 16x16 i32 per 32-k subblock so the q4k
// per-subblock scales apply exactly.
internal data class TileQ4KAmx(
    internal var scales: List<FloatArray> = List(8) { FloatArray(0) },
    internal var mins: List<FloatArray> = List(8) { FloatArray(0) },
    // [subblock (8)][k4-row (8)][16 cols * 4 k] unsigned 4-bit values as i8
    internal var qs: UByteArray = UByteArray(QK_K * TILE_N),
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TileQ4KAmx) return false
        if (this.scales != other.scales) return false
        if (this.mins != other.mins) return false
        if (!this.qs.contentEquals(other.qs)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = this.scales.hashCode()
        result = 31 * result + this.mins.hashCode()
        result = 31 * result + this.qs.contentHashCode()
        return result
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<TileQ4KAmx> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<TileQ4KAmx> {
                override val alignBytes: Long = 4L
                private val off1: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes(
                            (
                                (8).toLong() *
                                    (
                                        io.github.kotlinmania.rustlang.core.mem
                                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32, (TILE_N).toInt())
                                    ).strideBytes
                            ),
                            4L,
                        )
                private val off2: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes(
                            (
                                off1 + (
                                    (8).toLong() *
                                        (
                                            io.github.kotlinmania.rustlang.core.mem
                                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32, (TILE_N).toInt())
                                        ).strideBytes
                                )
                            ),
                            1L,
                        )
                override val sizeBytes: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes((off2 + ((QK_K * TILE_N).toLong() * (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U8).strideBytes)), 4L)
                override fun readNative(addr: Long): TileQ4KAmx =
                    TileQ4KAmx(
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(
                                    io.github.kotlinmania.rustlang.core.mem
                                        .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32, (TILE_N).toInt()),
                                    (8).toInt(),
                                )
                        ).readNative(addr).map { it.toFloatArray() },
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(
                                    io.github.kotlinmania.rustlang.core.mem
                                        .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32, (TILE_N).toInt()),
                                    (8).toInt(),
                                )
                        ).readNative(addr + off1).map { it.toFloatArray() },
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U8, (QK_K * TILE_N).toInt())
                        ).readNative(addr + off2).toUByteArray(),
                    )
                override fun writeNative(addr: Long, value: TileQ4KAmx) {
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(
                                io.github.kotlinmania.rustlang.core.mem
                                    .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32, (TILE_N).toInt()),
                                (8).toInt(),
                            )
                    ).writeNative(addr, value.scales.map { it.toList() })
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(
                                io.github.kotlinmania.rustlang.core.mem
                                    .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32, (TILE_N).toInt()),
                                (8).toInt(),
                            )
                    ).writeNative(addr + off1, value.mins.map { it.toList() })
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U8, (QK_K * TILE_N).toInt())
                    ).writeNative(addr + off2, value.qs.toList())
                }
            }
    }
}

internal fun amxAvailable(): Boolean {
    val OK: OnceLock<Boolean> = OnceLock.new()
    return OK.getOrInit(
        closure@{
            // forced downlevel testing must exercise the 256-bit paths end to end
            if (level() != X86Level.Avx512Vnni) {
                return@closure false
            }
            // isX86FeatureDetected("amx-int8") is unstable; read CPUID leaf 7 directly
            val hasAmx =
                run {
                    val info = cpuidCount(7, 0)
                    info.edx and (1 shl 25) != 0 && info.edx and (1 shl 24) != 0
                }
            val hasAmx1 = false
            if (!hasAmx1) {
                return@closure false
            }
            // Linux gates AMX tile state behind a per-process permission request; other
            // OSes would need their own enablement, so AMX stays off there.
            run {
                val ARCH_REQ_XCOMP_PERM: Int = 0x1023
                val XFEATURE_XTILEDATA: ULong = 18uL
                val r = syscall(SysArchPrctl, ARCH_REQ_XCOMP_PERM.toLong(), XFEATURE_XTILEDATA.toLong())
                r == 0
            }
            false
        },
    )
}

internal fun packQ4kAmx(storage: QuantizedType, n: Int, k: Int): List<TileQ4KAmx> {
    val blocks = q4kBlocks(storage)
    val kb = k / QK_K
    val nTiles = n / TILE_N
    val out: List<TileQ4KAmx> = ArrayList(nTiles * kb)
    for (t in 0 until nTiles) {
        for (b in 0 until kb) {
            val tile = TileQ4KAmx(scales = List(8) { FloatArray(TILE_N) { 0.0f } }, mins = List(8) { FloatArray(TILE_N) { 0.0f } }, qs = UByteArray(QK_K * TILE_N) { 0u.toUByte() })
            for (r in 0 until TILE_N) {
                val blk = blocks[(t * TILE_N + r) * kb + b]
                val d = blk.d.toF32()
                val dmin = blk.dmin.toF32()
                for (sub in 0 until 8) {
                    val (sc, mn) = getScaleMinK4(sub, blk.scales)
                    tile.scales[sub][r] = d * sc.toFloat()
                    tile.mins[sub][r] = dmin * mn.toFloat()
                }
                val vals = List(QK_K) { 0u.toUByte() }
                for (c in 0 until 4) {
                    for (j in 0 until 32) {
                        val byte = blk.qs[c * 32 + j]
                        vals[c * 64 + j] = byte and 0x0Fu.toUByte()
                        vals[c * 64 + 32 + j] = (byte.toUInt() shr 4).toUByte()
                    }
                }
                for (sub1 in 0 until 8) {
                    for (krow in 0 until 8) {
                        for (j1 in 0 until 4) {
                            tile.qs[((sub1 * 8 + krow) * TILE_N + r) * 4 + j1] = vals[sub1 * 32 + krow * 4 + j1]
                        }
                    }
                }
            }
            out.add(tile)
        }
    }
    return out
}

public data class TileCfg(
    public var palette: UByte = 0u,
    public var startRow: UByte = 0u,
    public var rsvd: UByteArray = UByteArray(14),
    public var colsb: UShortArray = UShortArray(16),
    public var rows: UByteArray = UByteArray(16),
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TileCfg) return false
        if (this.palette != other.palette) return false
        if (this.startRow != other.startRow) return false
        if (!this.rsvd.contentEquals(other.rsvd)) return false
        if (!this.colsb.contentEquals(other.colsb)) return false
        if (!this.rows.contentEquals(other.rows)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = this.palette.hashCode()
        result = 31 * result + this.startRow.hashCode()
        result = 31 * result + this.rsvd.contentHashCode()
        result = 31 * result + this.colsb.contentHashCode()
        result = 31 * result + this.rows.contentHashCode()
        return result
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<TileCfg> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<TileCfg> {
                override val alignBytes: Long = 64L
                override val sizeBytes: Long = 64L
                override fun readNative(addr: Long): TileCfg =
                    TileCfg(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U8).readNative(addr),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U8).readNative(addr + 1L),
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U8, (14).toInt())
                        ).readNative(addr + 2L).toUByteArray(),
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U16, (16).toInt())
                        ).readNative(addr + 16L).toUShortArray(),
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U8, (16).toInt())
                        ).readNative(addr + 48L).toUByteArray(),
                    )
                override fun writeNative(addr: Long, value: TileCfg) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U8).writeNative(addr, value.palette)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U8).writeNative(addr + 1L, value.startRow)
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U8, (14).toInt())
                    ).writeNative(addr + 2L, value.rsvd.toList())
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U16, (16).toInt())
                    ).writeNative(addr + 16L, value.colsb.toList())
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U8, (16).toInt())
                    ).writeNative(addr + 48L, value.rows.toList())
                }
            }
    }
}

// tmm0 = C (16x16 i32), tmm1 = A (16 x 32 i8), tmm2 = B (8 x 64 i8)
public fun configure() {
    val cfg = TileCfg(palette = 1u, startRow = 0u, rsvd = UByteArray(14) { 0u.toUByte() }, colsb = UShortArray(16) { 0u.toUShort() }, rows = UByteArray(16) { 0u.toUByte() })
    // tmm0-3: C (16x16 i32); tmm4-5: A (16x32 i8); tmm6-7: B (8x64 i8)
    for (t in 0 until 4) {
        cfg.colsb[t] = 64u
        cfg.rows[t] = 16u
    }
    for (t1 in 4 until 6) {
        cfg.colsb[t1] = 32u
        cfg.rows[t1] = 16u
    }
    for (t2 in 6 until 8) {
        cfg.colsb[t2] = 64u
        cfg.rows[t2] = 8u
    }
    asm("ldtilecfg [{cfg}]", cfg, reg, cfg, options(nostack))
}

public fun ensureConfigured() {
    AMX_READY.with({ f ->
        if (!f.get()) {
            configure()
            f.set(true)
        }
    })
}

// One (m16-block, n-tile) product over the full k range.
// aPack: [kb][8 subs][16 rows * 32 bytes]; returns via dst rows of n floats.
// 32m x 32n macro-tile: 4 C tiles per (2 A, 2 B) loads so tdp throughput is not
// gated on tile loads. aPack holds 32 m-rows (two 16-row halves interleaved by sub).
internal fun q4kM32(tiles: List<TileQ4KAmx>, nt0: Int, nTiles: Int, kb: Int, aPack: List<Byte>, acts: List<List<BlockQ8K>>, mt: Int, dst: MutPtr, n: Int) {
    ensureConfigured()
    val twoN = nt0 + 1 < nTiles
    val localStorageL737C8 = AddressableArrayLocal(IntArray((4 * 16 * 16).toInt()) { 0 }, FixedArrayTypeLayout(PrimitiveLayouts.I32, (4 * 16 * 16).toInt()))
    var cScratch by localStorageL737C8
    val facc = List((32 * 32).toInt()) { 0.0f }
    val aHalf = kb * 8 * 16 * 32
    run {
        for (b in 0 until kb) {
            val t0 = tiles[nt0 * kb + b]
            val t1 = tiles[(nt0 + twoN.toInt()) * kb + b]
            for (sub in 0 until 8) {
                val a0 = aPack.asPtr().add((b * 8 + sub) * 16 * 32)
                val a1 = aPack.asPtr().add(aHalf + (b * 8 + sub) * 16 * 32)
                val b0 = t0.qs.asPtr().add(sub * 8 * TILE_N * 4)
                val b1 = t1.qs.asPtr().add(sub * 8 * TILE_N * 4)
                asm("tilezero tmm0", "tilezero tmm1", "tilezero tmm2", "tilezero tmm3", "tileloadd tmm4, [{a0} + {sa}*1]", "tileloadd tmm6, [{b0} + {sb}*1]", "tdpbssd tmm0, tmm4, tmm6", "tileloadd tmm7, [{b1} + {sb}*1]", "tdpbssd tmm1, tmm4, tmm7", "tileloadd tmm5, [{a1} + {sa}*1]", "tdpbssd tmm2, tmm5, tmm6", "tdpbssd tmm3, tmm5, tmm7", "tilestored [{c} + {sc}*1], tmm0", "tilestored [{c} + {sc}*1 + 1024], tmm1", "tilestored [{c} + {sc}*1 + 2048], tmm2", "tilestored [{c} + {sc}*1 + 3072], tmm3", a0, reg, a0, a1, reg, a1, b0, reg, b0, b1, reg, b1, c, reg, cScratch, asMutPtr, sa, reg, 32, sb, reg, 64, sc, reg, 64, options(nostack))
                for (half in 0 until 2) {
                    val rows =
                        if (half == 0) {
                            mt.min(16)
                        } else {
                            mt.saturatingSub(16)
                        }
                    if (rows == 0) {
                        continue
                    }
                    for (ntl in 0 until 1 + twoN.toInt()) {
                        val tile =
                            if (ntl == 0) {
                                t0
                            } else {
                                t1
                            }
                        val scale = mm512LoaduPs(tile.scales[sub].asPtr())
                        val min = mm512LoaduPs(tile.mins[sub].asPtr())
                        val cbase = (half * 2 + ntl) * 256
                        for (mi in 0 until rows) {
                            val act = acts[half * 16 + mi][b]
                            val dall = act.d
                            val ci = mm512LoaduSi512(ConstPtr((localStorageL737C8.elementConstPtr(0)).rawAddress + ((cbase + mi * 16).toLong() * (PrimitiveLayouts.I32).strideBytes)))
                            val cf = mm512Cvtepi32Ps(ci)
                            val fo = (half * 16 + mi) * 32 + ntl * 16
                            val f = mm512LoaduPs(facc.asPtr().add(fo))
                            val bsum = (act.bsums[sub * 2].toInt() + act.bsums[sub * 2 + 1].toInt()).toFloat()
                            val f1 = mm512FmaddPs(cf, mm512MulPs(scale, mm512Set1Ps(dall)), f)
                            val f2 = mm512FnmaddPs(min, mm512Set1Ps(dall * bsum), f1)
                            mm512StoreuPs(facc.asMutPtr().add(fo), f2)
                        }
                    }
                }
            }
        }
        val nCols = 16 * (1 + twoN.toInt())
        for (mi1 in 0 until mt) {
            copyNonoverlapping(facc.asPtr().add(mi1 * 32), MutPtr((dst).rawAddress + ((mi1 * n + nt0 * TILE_N).toLong() * (PrimitiveLayouts.F32).strideBytes)), nCols)
        }
    }
}

// AMX matmul: m >= 32, q4k only. acts are per-row BlockQ8K slices.
internal fun matmulAmxQ4k(tiles: List<TileQ4KAmx>, lhsQ: List<BlockQ8K>, m: Int, k: Int, n: Int, dst: MutableList<Float>) {
    run {
        val kb = k / QK_K
        val nTiles = n / TILE_N
        val mBlocks = m.divCeil(32)
        val dstPtr = dst.asMutPtr().toInt()
        val lhsRef = lhsQ
        barrierPool().executeChunked(mBlocks, { range ->
            val dstPtr1 = MutPtr(dstPtr.toLong())
            // per-thread activation repack buffer (two 16-row halves), reused across n
            val aHalf = kb * 8 * 16 * 32
            var aPack = mutableListOf(0, 2 * aHalf)
            var actRows: List<List<BlockQ8K>> = ArrayList(32)
            for (mb in range) {
                val m0 = mb * 32
                val mt = (m - m0).min(32)
                actRows.clear()
                for (i in 0 until mt) {
                    actRows.add(((lhsRef).slice((m0 + i) * kb until (m0 + i + 1) * kb)))
                }
                aPack.fill(0)
                for ((i1, row) in actRows.mapIndexed { index, item -> Pair(index, item) }) {
                    val base = (i1 / 16) * aHalf
                    val r = i1 % 16
                    for (b in 0 until kb) {
                        val qs = row[b].qs
                        for (sub in 0 until 8) {
                            val dstOff = base + ((b * 8 + sub) * 16 + r) * 32
                            ((aPack).slice(dstOff until dstOff + 32)).copyFromSlice(((qs).slice(sub * 32 until sub * 32 + 32)))
                        }
                    }
                }
                var nt = 0
                while (nt < nTiles) {
                    q4kM32(tiles, nt, nTiles, kb, aPack, actRows, mt, MutPtr((dstPtr1).rawAddress + ((m0 * n).toLong() * (PrimitiveLayouts.F32).strideBytes)), n)
                    nt += 2
                }
            }
        })
    }
}

// 256-bit kernel tier for CPUs without AVX512: AVX-VNNI (vpdpbusd ymm) or pure AVX2
// (maddubs+madd, safe for the u4/u6 x i8 operand ranges). Same tile layout, each
// 16-column group processed as two ymm halves.
internal enum class X86Level(
    public val value: UInt,
) {
    Avx512Vnni(0u),
    AvxVnni(1u),
    Avx2(2u),
}

internal fun level(): X86Level? {
    val LEVEL: OnceLock<X86Level?> = OnceLock.new()
    return LEVEL.getOrInit(
        closure1@{
            val forceAvx2 = (variable("MISTRALRS_FORCE_AVX2").asDeref()).rustEquals(Result.Ok("1"))
            val forceVnni = (variable("MISTRALRS_FORCE_AVXVNNI").asDeref()).rustEquals(Result.Ok("1"))
            if (!forceAvx2 && !forceVnni && isX86FeatureDetected("avx512f") && isX86FeatureDetected("avx512vnni")) {
                return@closure1 X86Level.Avx512Vnni
            }
            if (!forceAvx2 && (forceVnni || isX86FeatureDetected("avxvnni")) && isX86FeatureDetected("avx2")) {
                return@closure1 X86Level.AvxVnni
            }
            if (isX86FeatureDetected("avx2") && isX86FeatureDetected("fma")) {
                return@closure1 X86Level.Avx2
            }
            null
        },
    )
}

public fun bcast4256(act: List<Byte>, off: ULong): M256i = mm256Set1Epi32(readUnaligned(constPtrFromRustCast(act.asPtr().add(off)), typeLayoutT = PrimitiveLayouts.I32))

public fun dpVnni(acc: M256i, wU8: M256i, aI8: M256i): M256i = mm256DpbusdAvxEpi32(acc, wU8, aI8)

// dpbusd stand-in for pure AVX2: maddubs (u8 x i8 -> i16 pairs) then madd against ones.
public fun dpAvx2(acc: M256i, wU8: M256i, aI8: M256i): M256i {
    val prod16 = mm256MaddubsEpi16(wU8, aI8)
    return mm256AddEpi32(acc, mm256MaddEpi16(prod16, mm256Set1Epi16(1)))
}

internal fun q4kTileAvxvnni(tile: TileQ4K, acts: List<BlockQ8K>, acc: MutableList<List<M256>>) {
    val mask = mm256Set1Epi8((0x0Fu.toUByte()).toByte())
    for (sub in 0 until 8) {
        val isum = List(MAX_MT) { List(2) { mm256SetzeroSi256() } }
        for (g4 in 0 until 4) {
            val g = sub * 4 + g4
            for (h in 0 until 2) {
                val w = mm256LoaduSi256(constPtrFromRustCast(tile.qs.asPtr().add(g * TILE_N * 4 + h * 32)))
                val lo = mm256AndSi256(w, mask)
                val hi = mm256AndSi256(mm256SrliEpi16(w, 4), mask)
                for ((m, act) in acts.mapIndexed { index, item -> Pair(index, item) }) {
                    val aLo = bcast4256(act.qs, sub * 32 + g4 * 8)
                    val aHi = bcast4256(act.qs, sub * 32 + g4 * 8 + 4)
                    isum[m][h] = dpVnni(isum[m][h], lo, aLo)
                    isum[m][h] = dpVnni(isum[m][h], hi, aHi)
                }
            }
        }
        for (h1 in 0 until 2) {
            val scale = mm256LoaduPs(tile.scales[sub].asPtr().add(h1 * 8))
            val min = mm256LoaduPs(tile.mins[sub].asPtr().add(h1 * 8))
            for ((m1, act1) in acts.mapIndexed { index, item -> Pair(index, item) }) {
                val dall = act1.d
                val f = mm256Cvtepi32Ps(isum[m1][h1])
                acc[m1][h1] = mm256FmaddPs(f, mm256MulPs(scale, mm256Set1Ps(dall)), acc[m1][h1])
                val bsum = (act1.bsums[sub * 2].toInt() + act1.bsums[sub * 2 + 1].toInt()).toFloat()
                acc[m1][h1] = mm256FnmaddPs(min, mm256Set1Ps(dall * bsum), acc[m1][h1])
            }
        }
    }
}

internal fun q4kTileAvx2(tile: TileQ4K, acts: List<BlockQ8K>, acc: MutableList<List<M256>>) {
    val mask = mm256Set1Epi8((0x0Fu.toUByte()).toByte())
    for (sub in 0 until 8) {
        val isum = List(MAX_MT) { List(2) { mm256SetzeroSi256() } }
        for (g4 in 0 until 4) {
            val g = sub * 4 + g4
            for (h in 0 until 2) {
                val w = mm256LoaduSi256(constPtrFromRustCast(tile.qs.asPtr().add(g * TILE_N * 4 + h * 32)))
                val lo = mm256AndSi256(w, mask)
                val hi = mm256AndSi256(mm256SrliEpi16(w, 4), mask)
                for ((m, act) in acts.mapIndexed { index, item -> Pair(index, item) }) {
                    val aLo = bcast4256(act.qs, sub * 32 + g4 * 8)
                    val aHi = bcast4256(act.qs, sub * 32 + g4 * 8 + 4)
                    isum[m][h] = dpAvx2(isum[m][h], lo, aLo)
                    isum[m][h] = dpAvx2(isum[m][h], hi, aHi)
                }
            }
        }
        for (h1 in 0 until 2) {
            val scale = mm256LoaduPs(tile.scales[sub].asPtr().add(h1 * 8))
            val min = mm256LoaduPs(tile.mins[sub].asPtr().add(h1 * 8))
            for ((m1, act1) in acts.mapIndexed { index, item -> Pair(index, item) }) {
                val dall = act1.d
                val f = mm256Cvtepi32Ps(isum[m1][h1])
                acc[m1][h1] = mm256FmaddPs(f, mm256MulPs(scale, mm256Set1Ps(dall)), acc[m1][h1])
                val bsum = (act1.bsums[sub * 2].toInt() + act1.bsums[sub * 2 + 1].toInt()).toFloat()
                acc[m1][h1] = mm256FnmaddPs(min, mm256Set1Ps(dall * bsum), acc[m1][h1])
            }
        }
    }
}

internal fun q6kTileAvxvnni(tile: TileQ6K, acts: List<BlockQ8K>, acc: MutableList<List<M256>>) {
    for (sub in 0 until 16) {
        val isum = List(MAX_MT) { List(2) { mm256SetzeroSi256() } }
        for (g4 in 0 until 4) {
            val g = sub * 4 + g4
            for (h in 0 until 2) {
                val w = mm256LoaduSi256(constPtrFromRustCast(tile.qs.asPtr().add(g * TILE_N * 4 + h * 32)))
                for ((m, act) in acts.mapIndexed { index, item -> Pair(index, item) }) {
                    val a = bcast4256(act.qs, sub * 16 + g4 * 4)
                    isum[m][h] = dpVnni(isum[m][h], w, a)
                }
            }
        }
        for (h1 in 0 until 2) {
            val scale = mm256LoaduPs(tile.scales[sub].asPtr().add(h1 * 8))
            for ((m1, act1) in acts.mapIndexed { index, item -> Pair(index, item) }) {
                val dall = act1.d
                val bsum = act1.bsums[sub].toFloat()
                val f = mm256SubPs(mm256Cvtepi32Ps(isum[m1][h1]), mm256Set1Ps(32.0f * bsum))
                acc[m1][h1] = mm256FmaddPs(f, mm256MulPs(scale, mm256Set1Ps(dall)), acc[m1][h1])
            }
        }
    }
}

internal fun q6kTileAvx2(tile: TileQ6K, acts: List<BlockQ8K>, acc: MutableList<List<M256>>) {
    for (sub in 0 until 16) {
        val isum = List(MAX_MT) { List(2) { mm256SetzeroSi256() } }
        for (g4 in 0 until 4) {
            val g = sub * 4 + g4
            for (h in 0 until 2) {
                val w = mm256LoaduSi256(constPtrFromRustCast(tile.qs.asPtr().add(g * TILE_N * 4 + h * 32)))
                for ((m, act) in acts.mapIndexed { index, item -> Pair(index, item) }) {
                    val a = bcast4256(act.qs, sub * 16 + g4 * 4)
                    isum[m][h] = dpAvx2(isum[m][h], w, a)
                }
            }
        }
        for (h1 in 0 until 2) {
            val scale = mm256LoaduPs(tile.scales[sub].asPtr().add(h1 * 8))
            for ((m1, act1) in acts.mapIndexed { index, item -> Pair(index, item) }) {
                val dall = act1.d
                val bsum = act1.bsums[sub].toFloat()
                val f = mm256SubPs(mm256Cvtepi32Ps(isum[m1][h1]), mm256Set1Ps(32.0f * bsum))
                acc[m1][h1] = mm256FmaddPs(f, mm256MulPs(scale, mm256Set1Ps(dall)), acc[m1][h1])
            }
        }
    }
}

// q80 needs true dpbusd (u8 range overflows maddubs' i16 pairs), so vnni only.
internal fun q80Tile2Avxvnni(t0: TileQ80, t1: TileQ80, acts: List<BlockQ8K>, sub32: Int, acc: MutableList<List<M256>>) {
    val isumA = List(MAX_MT) { List(2) { mm256SetzeroSi256() } }
    val isumB = List(MAX_MT) { List(2) { mm256SetzeroSi256() } }
    for (g4 in 0 until 8) {
        for (h in 0 until 2) {
            val wa = mm256LoaduSi256(constPtrFromRustCast(t0.qs.asPtr().add(g4 * TILE_N * 4 + h * 32)))
            val wb = mm256LoaduSi256(constPtrFromRustCast(t1.qs.asPtr().add(g4 * TILE_N * 4 + h * 32)))
            for ((m, act) in acts.mapIndexed { index, item -> Pair(index, item) }) {
                isumA[m][h] = mm256DpbusdAvxEpi32(isumA[m][h], wa, bcast4256(act.qs, sub32 * 32 + g4 * 4))
                isumB[m][h] = mm256DpbusdAvxEpi32(isumB[m][h], wb, bcast4256(act.qs, (sub32 + 1) * 32 + g4 * 4))
            }
        }
    }
    for ((blk, isum, tile) in listOf(Triple(sub32, isumA, t0), Triple(sub32 + 1, isumB, t1))) {
        for (h1 in 0 until 2) {
            val dw = mm256CvtphPs(mmLoaduSi128(constPtrFromRustCast(tile.d.asPtr().add(h1 * 8))))
            for ((m1, act1) in acts.mapIndexed { index, item -> Pair(index, item) }) {
                val bsum = (act1.bsums[blk * 2].toInt() + act1.bsums[blk * 2 + 1].toInt()).toFloat()
                val f = mm256SubPs(mm256Cvtepi32Ps(isum[m1][h1]), mm256Set1Ps(128.0f * bsum))
                acc[m1][h1] = mm256FmaddPs(f, mm256MulPs(dw, mm256Set1Ps(act1.d)), acc[m1][h1])
            }
        }
    }
}
