package io.github.kotlinmania.candle

import io.github.kotlinmania.half.BF16
import io.github.kotlinmania.half.F16

private fun <T> MutableList<T>.asPtr(): ByteArray = ByteArray(0)

private fun GgmlDType.zeros(): Any =
    if (this == GgmlDType.F16) io.github.kotlinmania.half.F16.ZERO else io.github.kotlinmania.half.BF16.ZERO

enum class GgmlDType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    ;

    private val self: GgmlDType get() = this

    internal fun toU32(): UInt =
        when (self) {
            F32 -> 0u

            F16 -> 1u

            Q4_0 -> 2u

            Q4_1 -> 3u

            Q5_0 -> 6u

            Q5_1 -> 7u

            Q8_0 -> 8u

            Q8_1 -> 9u

            Q2K -> 10u

            Q3K -> 11u

            Q4K -> 12u

            Q5K -> 13u

            Q6K -> 14u

            Q8K -> 15u

            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            BF16 -> 30u
        }

    /**
     * The block dtype
     */
    fun cpuZeros(elemCount: Int): QuantizedType =
        when (self) {
            F32 -> QuantizedVec.new(MutableList(elemCount) { Float.zeros() }, F32GgmlType)
            F16 -> QuantizedVec.new(MutableList(elemCount) { F16.zeros() }, F16)
            Q4_0 -> QuantizedVec.new(MutableList(elemCount / BlockQ40.BLCK_SIZE) { BlockQ40.zeros() }, BlockQ40)
            Q4_1 -> QuantizedVec.new(MutableList(elemCount / BlockQ41.BLCK_SIZE) { BlockQ41.zeros() }, BlockQ41)
            Q5_0 -> QuantizedVec.new(MutableList(elemCount / BlockQ50.BLCK_SIZE) { BlockQ50.zeros() }, BlockQ50)
            Q5_1 -> QuantizedVec.new(MutableList(elemCount / BlockQ51.BLCK_SIZE) { BlockQ51.zeros() }, BlockQ51)
            Q8_0 -> QuantizedVec.new(MutableList(elemCount / BlockQ80.BLCK_SIZE) { BlockQ80.zeros() }, BlockQ80)
            Q8_1 -> QuantizedVec.new(MutableList(elemCount / BlockQ81.BLCK_SIZE) { BlockQ81.zeros() }, BlockQ81)
            Q2K -> QuantizedVec.new(MutableList(elemCount / BlockQ2K.BLCK_SIZE) { BlockQ2K.zeros() }, BlockQ2K)
            Q3K -> QuantizedVec.new(MutableList(elemCount / BlockQ3K.BLCK_SIZE) { BlockQ3K.zeros() }, BlockQ3K)
            Q4K -> QuantizedVec.new(MutableList(elemCount / BlockQ4K.BLCK_SIZE) { BlockQ4K.zeros() }, BlockQ4K)
            Q5K -> QuantizedVec.new(MutableList(elemCount / BlockQ5K.BLCK_SIZE) { BlockQ5K.zeros() }, BlockQ5K)
            Q6K -> QuantizedVec.new(MutableList(elemCount / BlockQ6K.BLCK_SIZE) { BlockQ6K.zeros() }, BlockQ6K)
            Q8K -> QuantizedVec.new(MutableList(elemCount / BlockQ8K.BLCK_SIZE) { BlockQ8K.zeros() }, BlockQ8K)
            BF16 -> QuantizedVec.new(MutableList(elemCount) { BF16.zeros() }, BF16)
        }

    fun fromData(data: ByteArray): QuantizedType {
        val rawData = data
        return when (self) {
            F32 -> QuantizedVec.new(asTSlice(rawData, F32GgmlType).toVec(), F32GgmlType)
            F16 -> QuantizedVec.new(asTSlice(rawData, F16).toVec(), F16)
            Q4_0 -> QuantizedVec.new(asTSlice(rawData, BlockQ40).toVec(), BlockQ40)
            Q4_1 -> QuantizedVec.new(asTSlice(rawData, BlockQ41).toVec(), BlockQ41)
            Q5_0 -> QuantizedVec.new(asTSlice(rawData, BlockQ50).toVec(), BlockQ50)
            Q5_1 -> QuantizedVec.new(asTSlice(rawData, BlockQ51).toVec(), BlockQ51)
            Q8_0 -> QuantizedVec.new(asTSlice(rawData, BlockQ80).toVec(), BlockQ80)
            Q8_1 -> QuantizedVec.new(asTSlice(rawData, BlockQ81).toVec(), BlockQ81)
            Q2K -> QuantizedVec.new(asTSlice(rawData, BlockQ2K).toVec(), BlockQ2K)
            Q3K -> QuantizedVec.new(asTSlice(rawData, BlockQ3K).toVec(), BlockQ3K)
            Q4K -> QuantizedVec.new(asTSlice(rawData, BlockQ4K).toVec(), BlockQ4K)
            Q5K -> QuantizedVec.new(asTSlice(rawData, BlockQ5K).toVec(), BlockQ5K)
            Q6K -> QuantizedVec.new(asTSlice(rawData, BlockQ6K).toVec(), BlockQ6K)
            Q8K -> QuantizedVec.new(asTSlice(rawData, BlockQ8K).toVec(), BlockQ8K)
            BF16 -> QuantizedVec.new(asTSlice(rawData, BF16).toVec(), BF16)
        }
    }

    /**
     * The type size for blocks in bytes.
     */
    fun typeSize(): Int {
        // Block sizes in bytes for each quantized GgmlDType variant.
        return when (self) {
            F32 -> 4

            F16, BF16 -> 2

            Q4_0 -> QuantizedHelper.sizeOf(BlockQ40)

            Q4_1 -> QuantizedHelper.sizeOf(BlockQ41)

            Q5_0 -> QuantizedHelper.sizeOf(BlockQ50)

            Q5_1 -> QuantizedHelper.sizeOf(BlockQ51)

            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Q8_0 -> QuantizedHelper.sizeOf(BlockQ80)

            Q8_1 -> QuantizedHelper.sizeOf(BlockQ81)

            Q2K -> QuantizedHelper.sizeOf(BlockQ2K)

            Q3K -> QuantizedHelper.sizeOf(BlockQ3K)

            Q4K -> QuantizedHelper.sizeOf(BlockQ4K)

            Q5K -> QuantizedHelper.sizeOf(BlockQ5K)

            Q6K -> QuantizedHelper.sizeOf(BlockQ6K)

            Q8K -> QuantizedHelper.sizeOf(BlockQ8K)
        }
    }

    /**
     * The block size, i.e. the number of elements stored in each block.
     */
    fun blockSize(): Int =
        when (self) {
            F32 -> 1
            F16, BF16 -> 1
            Q4_0 -> QK4_0
            Q4_1 -> QK4_1
            Q5_0 -> QK5_0
            Q5_1 -> QK5_1
            Q8_0 -> QK8_0
            Q8_1 -> QK8_1
            Q2K, Q3K, Q4K, Q5K, Q6K, Q8K -> QK_K
        }

    companion object {
        val Q40: GgmlDType = Q4_0
        val Q41: GgmlDType = Q4_1
        val Q50: GgmlDType = Q5_0
        val Q51: GgmlDType = Q5_1
        val Q80: GgmlDType = Q8_0
        val Q81: GgmlDType = Q8_1

        internal fun fromU32(u: UInt): CandleResult<GgmlDType> {
            val dtype =
                when (u) {
                    0u -> F32

                    1u -> F16

                    2u -> Q4_0

                    3u -> Q4_1

                    6u -> Q5_0

                    7u -> Q5_1

                    8u -> Q8_0

                    9u -> Q8_1

                    10u -> Q2K

                    11u -> Q3K

                    12u -> Q4K

                    13u -> Q5K

                    14u -> Q6K

                    15u -> Q8K

                    // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
                    30u -> BF16

                    else -> return CandleResult.Err(
                        CandleException
                            .Msg("unknown dtype for tensor $u")
                            .bt(),
                    )
                }
            return CandleResult.Ok(dtype)
        }
    }
}

// A type-erased version of GgmlType for polymorphic quantized storage.
interface QuantizedType {
    fun dtype(): GgmlDType

    fun matmulT(
        mkn: Triple<Int, Int, Int>,
        lhs: FloatArray,
        dst: FloatArray,
    ): CandleResult<Unit>

    fun matmulTF16(
        mkn: Triple<Int, Int, Int>,
        lhs: Array<F16>,
        dst: Array<F16>,
    ): CandleResult<Unit>

    fun dequantize(elemCount: Int): CandleResult<CpuStorage>

    fun storageSizeInBytes(): Int

    fun asPtr(): ByteArray

    fun blockSize(): Int

    fun fromFloat(xs: FloatArray)

    fun fromFloatImatrix(
        xs: FloatArray,
        imatrixWeights: FloatArray,
        nPerRow: Int,
    )

    fun size(): Int
}

internal class QuantizedVec<E : Any> internal constructor(
    internal val blocks: MutableList<E>,
    private val T: GgmlType<E, *>,
) : QuantizedType {
    private val self: MutableList<E> get() = blocks

    override fun matmulT(
        mkn: Triple<Int, Int, Int>,
        lhs: FloatArray,
        dst: FloatArray,
    ): CandleResult<Unit> = matmul(mkn, lhs, self.asSlice(), dst, T)

    override fun matmulTF16(
        mkn: Triple<Int, Int, Int>,
        lhs: Array<F16>,
        dst: Array<F16>,
    ): CandleResult<Unit> = matmulF16(mkn, lhs, self.asSlice(), dst, T)

    override fun size(): Int = self.len() * QuantizedHelper.sizeOf(T)

    override fun fromFloat(xs: FloatArray) {
        T.fromFloat(xs, self)
    }

    override fun fromFloatImatrix(
        xs: FloatArray,
        imatrixWeights: FloatArray,
        nPerRow: Int,
    ) {
        T.fromFloatImatrix(xs, self, imatrixWeights, nPerRow)
    }

    override fun dtype(): GgmlDType = T.DTYPE

    override fun blockSize(): Int = T.BLCK_SIZE

    override fun dequantize(elemCount: Int): CandleResult<CpuStorage> {
        val ys = FloatArray(elemCount)
        T.toFloat(self.asSlice(), ys)
        return CandleResult.Ok(CpuStorage.F32(ys))
    }

    override fun storageSizeInBytes(): Int = self.len() * QuantizedHelper.sizeOf(T)

    override fun asPtr(): ByteArray = QuantizedHelper.asPtrBytes(self.asPtr(), self, T)

    companion object {
        val new: (MutableList<out Any>, Any) -> QuantizedType = { list, comp ->
            QuantizedHelper.createQuantizedVec(list, comp)
        }
    }
}
