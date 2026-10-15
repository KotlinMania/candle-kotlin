// port-lint: source quantized/ggml_file.rs
/**
 * Support for the GGML file format.
 */
package io.github.kotlinmania.candle.ggml

import io.github.kotlinmania.anyhow.Err
import io.github.kotlinmania.anyhow.Ok
import io.github.kotlinmania.anyhow.Result
import io.github.kotlinmania.candle.BF16
import io.github.kotlinmania.candle.BlockQ2K
import io.github.kotlinmania.candle.BlockQ3K
import io.github.kotlinmania.candle.BlockQ40
import io.github.kotlinmania.candle.BlockQ41
import io.github.kotlinmania.candle.BlockQ4K
import io.github.kotlinmania.candle.BlockQ50
import io.github.kotlinmania.candle.BlockQ51
import io.github.kotlinmania.candle.BlockQ5K
import io.github.kotlinmania.candle.BlockQ6K
import io.github.kotlinmania.candle.BlockQ80
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.F16
import io.github.kotlinmania.candle.F32Scalar
import io.github.kotlinmania.candle.GgmlDType
import io.github.kotlinmania.candle.GgmlType
import io.github.kotlinmania.candle.NpyByteReader
import io.github.kotlinmania.candle.QStorage
import io.github.kotlinmania.candle.QTensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.loadQuantized

// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.h#L37
private enum class Magic {
    Ggjt,
    Ggla,
    Ggmf,
    Ggml,
    Ggsn,
    ;

    interface Error

    companion object {
        fun tryFrom(value: UInt): Result<Magic> {
            val magic = when (value) {
                0x67676a74u -> Ggjt
                0x67676c61u -> Ggla
                0x67676d66u -> Ggmf
                0x67676d6cu -> Ggml
                0x6767736eu -> Ggsn
                else -> bail("unknown magic ${value.toString(16)}")
            }
            return Ok(magic)
        }
    }
}

public enum class VersionedMagic {
    GgmlUnversioned,
    GgmfV1,
    GgjtV1,
    GgjtV2,
    GgjtV3,
    ;

    internal fun align32(): Boolean = when (this) {
        GgmlUnversioned, GgmfV1 -> false
        GgjtV1, GgjtV2, GgjtV3 -> true
    }

    public companion object {
        internal fun <R : NpyByteReader> read(reader: R): Result<VersionedMagic> {
            val rawMagic = reader.readU32<LittleEndian>().unwrap()
            val magic = Magic.tryFrom(rawMagic).unwrap()
            if (magic == Magic.Ggml) {
                return Ok(GgmlUnversioned)
            }
            val version = reader.readU32<LittleEndian>().unwrap()
            val versionedMagic = when (Pair(magic, version)) {
                Pair(Magic.Ggmf, 1u) -> GgmfV1
                Pair(Magic.Ggjt, 1u) -> GgjtV1
                Pair(Magic.Ggjt, 2u) -> GgjtV2
                Pair(Magic.Ggjt, 3u) -> GgjtV3
                else -> bail("ggml: unsupported magic/version $magic/$version")
            }
            return Ok(versionedMagic)
        }
    }
}

public data class HParams(
    public val nVocab: UInt,
    public val nEmbd: UInt,
    public val nMult: UInt,
    public val nHead: UInt,
    public val nLayer: UInt,
    public val nRot: UInt,
    public val ftype: UInt,
) {
    public companion object {
        internal fun <R : NpyByteReader> read(reader: R): Result<HParams> {
            val nVocab = reader.readU32<LittleEndian>().unwrap()
            val nEmbd = reader.readU32<LittleEndian>().unwrap()
            val nMult = reader.readU32<LittleEndian>().unwrap()
            val nHead = reader.readU32<LittleEndian>().unwrap()
            val nLayer = reader.readU32<LittleEndian>().unwrap()
            val nRot = reader.readU32<LittleEndian>().unwrap()
            val ftype = reader.readU32<LittleEndian>().unwrap()
            return Ok(
                HParams(
                    nVocab = nVocab,
                    nEmbd = nEmbd,
                    nMult = nMult,
                    nHead = nHead,
                    nLayer = nLayer,
                    nRot = nRot,
                    ftype = ftype,
                ),
            )
        }
    }
}

public data class Vocab(
    public val tokenScorePairs: List<Pair<ByteArray, Float>>,
) {
    public companion object {
        internal fun <R : NpyByteReader> read(reader: R, nVocab: Int): Result<Vocab> {
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.cpp#L556
            val tokenScorePairs = Vec.withCapacity<Pair<ByteArray, Float>>(nVocab)
            for (index in 0 until nVocab) {
                val len = reader.readU32<LittleEndian>().unwrap().toInt()
                val word = ByteArray(len)
                reader.readExact(word).unwrap()
                val score = reader.readF32<LittleEndian>().unwrap()
                tokenScorePairs.push(Pair(word, score))
            }
            return Ok(Vocab(tokenScorePairs = tokenScorePairs))
        }
    }
}

private fun <T : GgmlType> fromRawData(
    rawData: ByteArray,
    sizeInBytes: Int,
    dims: List<Int>,
    device: Device,
): Result<QTensor> {
    val rawDataPtr = rawData.asPtr()
    val nBlocks = sizeInBytes / sizeOf<T>()
    val data = sliceFromRawParts<T>(rawDataPtr, nBlocks)
    val qStorage: QStorage = when (device) {
        is Device.Cpu -> QStorage.Cpu(data.toVec())
        is Device.Metal -> loadQuantized<T>(device.device, data).unwrap()
        is Device.Cuda -> loadQuantized<T>(device.device, data).unwrap()
    }
    return QTensor.new(qStorage, dims)
}

/**
 * Creates a [Tensor] from a raw GGML tensor.
 */
public fun qtensorFromGgml(
    ggmlDtype: GgmlDType,
    rawData: ByteArray,
    dims: List<Int>,
    device: Device,
): Result<QTensor> {
    val tensorElems = dims.iter().product()
    val blockSize = ggmlDtype.blockSize()
    if (tensorElems % blockSize != 0) {
        bail(
            "the number of elements $tensorElems is not divisible by the block size $blockSize",
        )
    }
    val sizeInBytes = tensorElems / blockSize * ggmlDtype.typeSize()

    return when (ggmlDtype) {
        GgmlDType.F32 -> fromRawData<F32Scalar>(rawData, sizeInBytes, dims, device)
        GgmlDType.F16 -> fromRawData<F16>(rawData, sizeInBytes, dims, device)
        GgmlDType.BF16 -> fromRawData<BF16>(rawData, sizeInBytes, dims, device)
        GgmlDType.Q4_0 -> {
            fromRawData<BlockQ40>(rawData, sizeInBytes, dims, device)
        }
        GgmlDType.Q4_1 -> {
            fromRawData<BlockQ41>(rawData, sizeInBytes, dims, device)
        }
        GgmlDType.Q5_0 -> {
            fromRawData<BlockQ50>(rawData, sizeInBytes, dims, device)
        }
        GgmlDType.Q5_1 -> {
            fromRawData<BlockQ51>(rawData, sizeInBytes, dims, device)
        }
        GgmlDType.Q8_0 -> {
            fromRawData<BlockQ80>(rawData, sizeInBytes, dims, device)
        }
        GgmlDType.Q2K -> {
            fromRawData<BlockQ2K>(rawData, sizeInBytes, dims, device)
        }
        GgmlDType.Q3K -> {
            fromRawData<BlockQ3K>(rawData, sizeInBytes, dims, device)
        }
        GgmlDType.Q4K -> {
            fromRawData<BlockQ4K>(rawData, sizeInBytes, dims, device)
        }
        GgmlDType.Q5K -> {
            fromRawData<BlockQ5K>(rawData, sizeInBytes, dims, device)
        }
        GgmlDType.Q6K -> {
            fromRawData<BlockQ6K>(rawData, sizeInBytes, dims, device)
        }
        else -> bail("quantized type $ggmlDtype is not supported yet")
    }
}

private fun <R : NpyByteReader> readOneTensor(
    reader: R,
    magic: VersionedMagic,
    device: Device,
): Result<Pair<String, QTensor>> {
    val nDims = reader.readU32<LittleEndian>().unwrap()
    val nameLen = reader.readU32<LittleEndian>().unwrap()
    val rawDtype = reader.readU32<LittleEndian>().unwrap()
    val ggmlDtype = GgmlDType.fromU32(rawDtype).unwrap()
    val rawDims = UIntArray(nDims.toInt())
    reader.readU32Into<LittleEndian>(rawDims).unwrap()
    // The dimensions are stored in reverse order, see for example:
    // https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/convert.py#L969
    rawDims.reverse()
    val rawName = ByteArray(nameLen.toInt())
    reader.readExact(rawName).unwrap()
    val name = String.fromUtf8Lossy(rawName).intoOwned()

    if (magic.align32()) {
        val pos = reader.streamPosition().unwrap()
        reader.seek(SeekFrom.Current(((32uL - pos % 32uL) % 32uL).toLong())).unwrap()
    }
    val dims = rawDims.iter().map { u -> u.toInt() }.collect()
    val tensorElems = dims.iter().product()
    val sizeInBytes = tensorElems * ggmlDtype.typeSize() / ggmlDtype.blockSize()
    // Mmap version to avoid copying the data around?
    val rawData = ByteArray(sizeInBytes)
    reader.readExact(rawData).unwrap()
    val res = qtensorFromGgml(ggmlDtype, rawData, dims, device)
    return when (res) {
        is Ok -> Ok(Pair(name, res.value))
        is Err -> bail("Error creating tensor $name: ${res.error}")
    }
}

public class Content(
    public val magic: VersionedMagic,
    public val hparams: HParams,
    public val vocab: Vocab,
    public val tensors: MutableMap<String, QTensor>,
    public val device: Device,
) {
    public fun remove(name: String): Result<QTensor> {
        val tensor = this.tensors.remove(name)
        return when (tensor) {
            null -> bail("cannot find tensor with name '$name'")
            else -> Ok(tensor)
        }
    }

    public companion object {
        public fun <R : NpyByteReader> read(
            reader: R,
            device: Device,
        ): Result<Content> {
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.cpp#L505
            val lastPosition = reader.seek(SeekFrom.End(0)).unwrap()
            reader.seek(SeekFrom.Start(0uL)).unwrap()
            val magic = VersionedMagic.read(reader).unwrap()
            val hparams = HParams.read(reader).unwrap()
            val vocab = Vocab.read(reader, hparams.nVocab.toInt()).unwrap()
            val tensors = HashMap.new<String, QTensor>()

            while (reader.streamPosition().unwrap() != lastPosition) {
                val pair = readOneTensor(reader, magic, device).unwrap()
                val name = pair.first
                val tensor = pair.second
                tensors.insert(name, tensor)
            }
            val clonedDevice = device.clone()
            return Ok(
                Content(
                    magic = magic,
                    hparams = hparams,
                    vocab = vocab,
                    tensors = tensors,
                    device = clonedDevice,
                ),
            )
        }
    }
}

