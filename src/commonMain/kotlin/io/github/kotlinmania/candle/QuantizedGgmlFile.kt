// port-lint: source quantized/ggml_file.rs
/**
 * Support for the GGML file format.
 */
package io.github.kotlinmania.candle.ggml

import io.github.kotlinmania.candle.BF16
import io.github.kotlinmania.candle.BlockQ2K
import io.github.kotlinmania.candle.BlockQ3K
import io.github.kotlinmania.candle.BlockQ4K
import io.github.kotlinmania.candle.BlockQ40
import io.github.kotlinmania.candle.BlockQ41
import io.github.kotlinmania.candle.BlockQ5K
import io.github.kotlinmania.candle.BlockQ50
import io.github.kotlinmania.candle.BlockQ51
import io.github.kotlinmania.candle.BlockQ6K
import io.github.kotlinmania.candle.BlockQ80
import io.github.kotlinmania.candle.CandleError
import io.github.kotlinmania.candle.CandleResult
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.F16
import io.github.kotlinmania.candle.F32Scalar
import io.github.kotlinmania.candle.GgmlDType
import io.github.kotlinmania.candle.GgmlType
import io.github.kotlinmania.candle.NpyByteReader
import io.github.kotlinmania.candle.QStorage
import io.github.kotlinmania.candle.QTensor
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.loadQuantized

// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.h#L37
private enum class Magic {
    Ggjt,
    Ggla,
    Ggmf,
    Ggml,
    Ggsn;

    interface Error

    companion object {
        fun tryFrom(value: UInt): CandleResult<Magic> {
            val magic = when (value) {
                0x67676a74u -> Ggjt
                0x67676c61u -> Ggla
                0x67676d66u -> Ggmf
                0x67676d6cu -> Ggml
                0x6767736eu -> Ggsn
                else -> return CandleResult.Err(CandleError.Msg("unknown magic ${value.toString(16)}").bt())
            }
            return CandleResult.Ok(magic)
        }
    }
}

enum class VersionedMagic {
    GgmlUnversioned,
    GgmfV1,
    GgjtV1,
    GgjtV2,
    GgjtV3;

    fun align32(): Boolean = when (this) {
        GgmlUnversioned, GgmfV1 -> false
        GgjtV1, GgjtV2, GgjtV3 -> true
    }

    companion object {
        internal fun read(reader: NpyByteReader): CandleResult<VersionedMagic> {
            val b4 = ByteArray(4)
            when (val res = reader.readExact(b4)) {
                is CandleResult.Ok -> {}
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val rawMagic = ((b4[0].toInt() and 0xFF) or
                ((b4[1].toInt() and 0xFF) shl 8) or
                ((b4[2].toInt() and 0xFF) shl 16) or
                ((b4[3].toInt() and 0xFF) shl 24)).toUInt()
            val magic = when (val res = Magic.tryFrom(rawMagic)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            if (magic == Magic.Ggml) {
                return CandleResult.Ok(GgmlUnversioned)
            }
            when (val res = reader.readExact(b4)) {
                is CandleResult.Ok -> {}
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val version = ((b4[0].toInt() and 0xFF) or
                ((b4[1].toInt() and 0xFF) shl 8) or
                ((b4[2].toInt() and 0xFF) shl 16) or
                ((b4[3].toInt() and 0xFF) shl 24)).toUInt()
            val versionedMagic = when {
                magic == Magic.Ggmf && version == 1u -> GgmfV1
                magic == Magic.Ggjt && version == 1u -> GgjtV1
                magic == Magic.Ggjt && version == 2u -> GgjtV2
                magic == Magic.Ggjt && version == 3u -> GgjtV3
                else -> return CandleResult.Err(
                    CandleError.Msg("ggml: unsupported magic/version $magic/$version").bt()
                )
            }
            return CandleResult.Ok(versionedMagic)
        }
    }
}

data class HParams(
    val nVocab: UInt,
    val nEmbd: UInt,
    val nMult: UInt,
    val nHead: UInt,
    val nLayer: UInt,
    val nRot: UInt,
    val ftype: UInt,
) {
    companion object {
        internal fun read(reader: NpyByteReader): CandleResult<HParams> {
            val readU32 = {
                val b = ByteArray(4)
                when (val res = reader.readExact(b)) {
                    is CandleResult.Ok -> {
                        val v = (b[0].toInt() and 0xFF) or
                            ((b[1].toInt() and 0xFF) shl 8) or
                            ((b[2].toInt() and 0xFF) shl 16) or
                            ((b[3].toInt() and 0xFF) shl 24)
                        CandleResult.Ok(v.toUInt())
                    }
                    is CandleResult.Err -> CandleResult.Err(res.error)
                }
            }
            val nVocab = when (val r = readU32()) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
            val nEmbd = when (val r = readU32()) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
            val nMult = when (val r = readU32()) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
            val nHead = when (val r = readU32()) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
            val nLayer = when (val r = readU32()) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
            val nRot = when (val r = readU32()) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
            val ftype = when (val r = readU32()) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
            return CandleResult.Ok(
                HParams(
                    nVocab = nVocab,
                    nEmbd = nEmbd,
                    nMult = nMult,
                    nHead = nHead,
                    nLayer = nLayer,
                    nRot = nRot,
                    ftype = ftype,
                )
            )
        }
    }
}

data class Vocab(
    val tokenScorePairs: List<Pair<ByteArray, Float>>,
) {
    companion object {
        internal fun read(reader: NpyByteReader, nVocab: Int): CandleResult<Vocab> {
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.cpp#L556
            val tokenScorePairs = ArrayList<Pair<ByteArray, Float>>(nVocab)
            val b4 = ByteArray(4)
            for (index in 0 until nVocab) {
                when (val res = reader.readExact(b4)) {
                    is CandleResult.Ok -> {}
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val len = (b4[0].toInt() and 0xFF) or
                    ((b4[1].toInt() and 0xFF) shl 8) or
                    ((b4[2].toInt() and 0xFF) shl 16) or
                    ((b4[3].toInt() and 0xFF) shl 24)
                val word = ByteArray(len)
                when (val res = reader.readExact(word)) {
                    is CandleResult.Ok -> {}
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                when (val res = reader.readExact(b4)) {
                    is CandleResult.Ok -> {}
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val scoreBits = (b4[0].toInt() and 0xFF) or
                    ((b4[1].toInt() and 0xFF) shl 8) or
                    ((b4[2].toInt() and 0xFF) shl 16) or
                    ((b4[3].toInt() and 0xFF) shl 24)
                val score = Float.fromBits(scoreBits)
                tokenScorePairs.add(word to score)
            }
            return CandleResult.Ok(Vocab(tokenScorePairs))
        }
    }
}

private fun <T : GgmlType> fromRawData(
    rawData: ByteArray,
    sizeInBytes: Int,
    dims: List<Int>,
    device: Device,
    dtype: GgmlDType,
): CandleResult<QTensor> {
    val sliced = if (rawData.size == sizeInBytes) rawData else rawData.copyOf(sizeInBytes)
    val qStorage: QStorage = when (device) {
        is Device.Cpu -> QStorage.Cpu(dtype.fromData(sliced))
        is Device.Metal -> {
            when (val res = loadQuantized<T>(device.device, emptyList())) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
        }
        is Device.Cuda -> {
            when (val res = loadQuantized<T>(device.device, emptyList())) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
        }
    }
    return QTensor.new(qStorage, Shape.fromDims(dims.toIntArray()))
}

/**
 * Creates a [QTensor] from a raw GGML tensor.
 */
fun qtensorFromGgml(
    ggmlDtype: GgmlDType,
    rawData: ByteArray,
    dims: List<Int>,
    device: Device,
): CandleResult<QTensor> {
    var tensorElems = 1
    for (d in dims) {
        tensorElems *= d
    }
    val blockSize = ggmlDtype.blockSize()
    if (tensorElems % blockSize != 0) {
        return CandleResult.Err(
            CandleError.Msg(
                "the number of elements $tensorElems is not divisible by the block size $blockSize"
            ).bt()
        )
    }
    val sizeInBytes = tensorElems / blockSize * ggmlDtype.typeSize()

    return when (ggmlDtype) {
        GgmlDType.F32 -> fromRawData<F32Scalar>(rawData, sizeInBytes, dims, device, ggmlDtype)
        GgmlDType.F16 -> fromRawData<F16>(rawData, sizeInBytes, dims, device, ggmlDtype)
        GgmlDType.BF16 -> fromRawData<BF16>(rawData, sizeInBytes, dims, device, ggmlDtype)
        GgmlDType.Q4_0 -> fromRawData<BlockQ40>(rawData, sizeInBytes, dims, device, ggmlDtype)
        GgmlDType.Q4_1 -> fromRawData<BlockQ41>(rawData, sizeInBytes, dims, device, ggmlDtype)
        GgmlDType.Q5_0 -> fromRawData<BlockQ50>(rawData, sizeInBytes, dims, device, ggmlDtype)
        GgmlDType.Q5_1 -> fromRawData<BlockQ51>(rawData, sizeInBytes, dims, device, ggmlDtype)
        GgmlDType.Q8_0 -> fromRawData<BlockQ80>(rawData, sizeInBytes, dims, device, ggmlDtype)
        GgmlDType.Q2K -> fromRawData<BlockQ2K>(rawData, sizeInBytes, dims, device, ggmlDtype)
        GgmlDType.Q3K -> fromRawData<BlockQ3K>(rawData, sizeInBytes, dims, device, ggmlDtype)
        GgmlDType.Q4K -> fromRawData<BlockQ4K>(rawData, sizeInBytes, dims, device, ggmlDtype)
        GgmlDType.Q5K -> fromRawData<BlockQ5K>(rawData, sizeInBytes, dims, device, ggmlDtype)
        GgmlDType.Q6K -> fromRawData<BlockQ6K>(rawData, sizeInBytes, dims, device, ggmlDtype)
        else -> CandleResult.Err(CandleError.Msg("quantized type $ggmlDtype is not supported yet").bt())
    }
}

private fun readOneTensor(
    reader: NpyByteReader,
    magic: VersionedMagic,
    device: Device,
): CandleResult<Pair<String, QTensor>> {
    val readU32 = {
        val b = ByteArray(4)
        when (val res = reader.readExact(b)) {
            is CandleResult.Ok -> {
                val v = (b[0].toInt() and 0xFF) or
                    ((b[1].toInt() and 0xFF) shl 8) or
                    ((b[2].toInt() and 0xFF) shl 16) or
                    ((b[3].toInt() and 0xFF) shl 24)
                CandleResult.Ok(v.toUInt())
            }
            is CandleResult.Err -> CandleResult.Err(res.error)
        }
    }
    val nDims = when (val r = readU32()) { is CandleResult.Ok -> r.value.toInt(); is CandleResult.Err -> return CandleResult.Err(r.error) }
    val nameLen = when (val r = readU32()) { is CandleResult.Ok -> r.value.toInt(); is CandleResult.Err -> return CandleResult.Err(r.error) }
    val rawDtype = when (val r = readU32()) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
    val ggmlDtype = when (val r = GgmlDType.fromU32(rawDtype)) {
        is CandleResult.Ok -> r.value
        is CandleResult.Err -> return CandleResult.Err(r.error)
    }
    val dims = ArrayList<Int>(nDims)
    for (i in 0 until nDims) {
        val d = when (val r = readU32()) { is CandleResult.Ok -> r.value.toInt(); is CandleResult.Err -> return CandleResult.Err(r.error) }
        dims.add(d)
    }
    // The dimensions are stored in reverse order, see for example:
    // https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/convert.py#L969
    dims.reverse()
    val nameBytes = ByteArray(nameLen)
    when (val res = reader.readExact(nameBytes)) {
        is CandleResult.Ok -> {}
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    val name = nameBytes.decodeToString()

    if (magic.align32()) {
        val pos = reader.position
        reader.position += (32 - pos % 32) % 32
    }
    var tensorElems = 1
    for (d in dims) {
        tensorElems *= d
    }
    val sizeInBytes = tensorElems * ggmlDtype.typeSize() / ggmlDtype.blockSize()
    val rawData = ByteArray(sizeInBytes)
    when (val res = reader.readExact(rawData)) {
        is CandleResult.Ok -> {}
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    return when (val res = qtensorFromGgml(ggmlDtype, rawData, dims, device)) {
        is CandleResult.Ok -> CandleResult.Ok(name to res.value)
        is CandleResult.Err -> CandleResult.Err(
            CandleError.Msg("Error creating tensor $name: ${res.error}").bt()
        )
    }
}

class Content(
    val magic: VersionedMagic,
    val hparams: HParams,
    val vocab: Vocab,
    val tensors: MutableMap<String, QTensor>,
    val device: Device,
) {
    fun remove(name: String): CandleResult<QTensor> {
        val tensor = this.tensors.remove(name)
            ?: return CandleResult.Err(CandleError.Msg("cannot find tensor with name '$name'").bt())
        return CandleResult.Ok(tensor)
    }

    companion object {
        fun read(
            reader: NpyByteReader,
            lastPosition: Int,
            device: Device,
        ): CandleResult<Content> {
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.cpp#L505
            reader.position = 0
            val magic = when (val r = VersionedMagic.read(reader)) {
                is CandleResult.Ok -> r.value
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }
            val hparams = when (val r = HParams.read(reader)) {
                is CandleResult.Ok -> r.value
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }
            val vocab = when (val r = Vocab.read(reader, hparams.nVocab.toInt())) {
                is CandleResult.Ok -> r.value
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }
            val tensors = HashMap<String, QTensor>()

            while (reader.position < lastPosition) {
                val (name, tensor) = when (val r = readOneTensor(reader, magic, device)) {
                    is CandleResult.Ok -> r.value
                    is CandleResult.Err -> return CandleResult.Err(r.error)
                }
                tensors[name] = tensor
            }
            return CandleResult.Ok(
                Content(
                    magic = magic,
                    hparams = hparams,
                    vocab = vocab,
                    tensors = tensors,
                    device = device,
                )
            )
        }
    }
}
