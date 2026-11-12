// port-lint: source quantized/ggml_file.rs
package io.github.kotlinmania.candle.quantized.ggmlfile

import io.github.kotlinmania.byteorder.LittleEndian
import io.github.kotlinmania.candle.device.Device
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.quantized.GgmlDType
import io.github.kotlinmania.candle.quantized.QStorage
import io.github.kotlinmania.candle.quantized.QTensor
import io.github.kotlinmania.candle.quantized.kquants.BlockQ2K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ3K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ40
import io.github.kotlinmania.candle.quantized.kquants.BlockQ41
import io.github.kotlinmania.candle.quantized.kquants.BlockQ4K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ50
import io.github.kotlinmania.candle.quantized.kquants.BlockQ51
import io.github.kotlinmania.candle.quantized.kquants.BlockQ5K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ6K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ80
import io.github.kotlinmania.candle.quantized.kquants.GgmlType
import io.github.kotlinmania.candle.quantized.metal.loadQuantized
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.rustlang.core.mem.ObjectSlotLayout
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.mem.constPtrFromRustCast
import io.github.kotlinmania.rustlang.core.result.fromResidual

// Support for the GGML file format.

// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.h#L37
public sealed class Magic : TryFrom<UInt> {
    public data object Ggjt : Magic()
    public data object Ggla : Magic()
    public data object Ggmf : Magic()
    public data object Ggml : Magic()
    public data object Ggsn : Magic()

    public fun rustEquals(other: Magic): Boolean = this == other

    public companion object {
        public fun tryFrom(value: UInt): Result<Magic> {
            val magic =
                when (value) {
                    0x67676a74 -> Magic.Ggjt
                    0x67676c61 -> Magic.Ggla
                    0x67676d66 -> Magic.Ggmf
                    0x67676d6c -> Magic.Ggml
                    0x6767736e -> Magic.Ggsn
                    else -> bail("unknown magic {value:08x}")
                }
            return Ok(magic)
        }
    }
}

public enum class VersionedMagic(
    public val value: UInt,
) {
    GgmlUnversioned(0u),
    GgmfV1(1u),
    GgjtV1(2u),
    GgjtV2(3u),
    GgjtV3(4u),
    ;

    public fun align32(): Boolean =
        when (this) {
            VersionedMagic.GgmlUnversioned, VersionedMagic.GgmfV1 -> false
            VersionedMagic.GgjtV1, VersionedMagic.GgjtV2, VersionedMagic.GgjtV3 -> true
        }

    public companion object {
        public fun <R : Read> read(reader: R): Result<VersionedMagic> {
            val magic =
                when (val tryResult = reader.readU32<LittleEndian>()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val magic1 =
                when (val tryResult = Magic.tryFrom(magic)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            if (magic1 == Magic.Ggml) {
                return Ok(VersionedMagic.GgmlUnversioned)
            }
            val version =
                when (val tryResult = reader.readU32<LittleEndian>()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val versionedMagic =
                run {
                    val matchVal = Pair(magic1, version)
                    when {
                        matchVal.first == Magic.Ggmf && matchVal.second == 1 -> VersionedMagic.GgmfV1
                        matchVal.first == Magic.Ggjt && matchVal.second == 1 -> VersionedMagic.GgjtV1
                        matchVal.first == Magic.Ggjt && matchVal.second == 2 -> VersionedMagic.GgjtV2
                        matchVal.first == Magic.Ggjt && matchVal.second == 3 -> VersionedMagic.GgjtV3
                        else -> bail("ggml: unsupported magic/version {magic:?}/{version}")
                    }
                }
            return Ok(versionedMagic)
        }
    }
}

public data class HParams(
    public var nVocab: UInt = 0u,
    public var nEmbd: UInt = 0u,
    public var nMult: UInt = 0u,
    public var nHead: UInt = 0u,
    public var nLayer: UInt = 0u,
    public var nRot: UInt = 0u,
    public var ftype: UInt = 0u,
) {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<HParams> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<HParams> {
                override val alignBytes: Long = 4L
                override val sizeBytes: Long = 28L
                override fun readNative(addr: Long): HParams =
                    HParams(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U32).readNative(addr),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U32).readNative(addr + 4L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U32).readNative(addr + 8L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U32).readNative(addr + 12L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U32).readNative(addr + 16L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U32).readNative(addr + 20L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U32).readNative(addr + 24L),
                    )
                override fun writeNative(addr: Long, value: HParams) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U32).writeNative(addr, value.nVocab)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U32).writeNative(addr + 4L, value.nEmbd)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U32).writeNative(addr + 8L, value.nMult)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U32).writeNative(addr + 12L, value.nHead)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U32).writeNative(addr + 16L, value.nLayer)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U32).writeNative(addr + 20L, value.nRot)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.U32).writeNative(addr + 24L, value.ftype)
                }
            }

        public fun <R : Read> read(reader: R): Result<HParams> {
            val nVocab =
                when (val tryResult = reader.readU32<LittleEndian>()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val nEmbd =
                when (val tryResult = reader.readU32<LittleEndian>()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val nMult =
                when (val tryResult = reader.readU32<LittleEndian>()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val nHead =
                when (val tryResult = reader.readU32<LittleEndian>()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val nLayer =
                when (val tryResult = reader.readU32<LittleEndian>()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val nRot =
                when (val tryResult = reader.readU32<LittleEndian>()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val ftype =
                when (val tryResult = reader.readU32<LittleEndian>()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            return Ok(HParams(nVocab = nVocab, nEmbd = nEmbd, nMult = nMult, nHead = nHead, nLayer = nLayer, nRot = nRot, ftype = ftype))
        }
    }
}

public data class Vocab(
    public var tokenScorePairs: List<Pair<List<UByte>, Float>> = emptyList(),
) {
    public companion object {
        public fun <R : Read> read(reader: R, nVocab: Int): Result<Vocab> {
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.cpp#L556
            var tokenScorePairs = ArrayList(nVocab)
            for (index in 0 until nVocab) {
                val len =
                    (
                        when (val tryResult = reader.readU32<LittleEndian>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    ).toInt()
                val word = mutableListOf(0u.toUByte(), len)
                when (val tryResult = reader.readExact(word)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
                val score =
                    when (val tryResult = reader.readF32<LittleEndian>()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                tokenScorePairs.add(Pair(word, score))
            }
            return Ok(Vocab(tokenScorePairs = tokenScorePairs))
        }
    }
}

public fun <T : GgmlType> fromRawData(rawData: List<UByte>, sizeInBytes: Int, dims: List<Int>, device: Device, typeLayoutT: TypeLayout<T>): Result<QTensor> {
    val rawDataPtr = rawData.asPtr()
    val nBlocks = sizeInBytes / sizeOf<T>(typeLayoutT = typeLayoutT)
    val data = fromRawParts(constPtrFromRustCast(rawDataPtr), nBlocks)
    val data1: QStorage =
        when (device) {
            Device.Cpu -> QStorage.Cpu(data.toVec())
            is Device.Metal -> {
                val (metal) = device
                when (val tryResult = loadQuantized(metal, data)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
            is Device.Cuda -> {
                val (cuda) = device
                when (
                    val tryResult =
                        io.github.kotlinmania.candle.quantized.cuda
                            .loadQuantized(cuda, data)
                ) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        }
    return QTensor.new(data1, dims)
}

/**
 * Creates a [Tensor] from a raw GGML tensor.
 */
public fun qtensorFromGgml(ggmlDtype: GgmlDType, rawData: List<UByte>, dims: List<Int>, device: Device): Result<QTensor> {
    val tensorElems = dims.iter().product<Int>()
    val blockSize = ggmlDtype.blockSize()
    if (tensorElems % blockSize != 0) {
        bail("the number of elements {tensor_elems} is not divisible by the block size {block_size}")
    }
    val sizeInBytes = tensorElems / blockSize * ggmlDtype.typeSize()
    return when (ggmlDtype) {
        GgmlDType.f32 -> fromRawData<Float>(rawData, sizeInBytes, dims, device, typeLayoutT = PrimitiveLayouts.F32)
        GgmlDType.f16 -> fromRawData<F16>(rawData, sizeInBytes, dims, device, typeLayoutT = ObjectSlotLayout<F16>())
        GgmlDType.bf16 -> fromRawData<Bf16>(rawData, sizeInBytes, dims, device, typeLayoutT = ObjectSlotLayout<Bf16>())
        GgmlDType.q40 -> {
            fromRawData<BlockQ40>(rawData, sizeInBytes, dims, device, typeLayoutT = ObjectSlotLayout<BlockQ40>())
        }
        GgmlDType.q41 -> {
            fromRawData<BlockQ41>(rawData, sizeInBytes, dims, device, typeLayoutT = ObjectSlotLayout<BlockQ41>())
        }
        GgmlDType.q50 -> {
            fromRawData<BlockQ50>(rawData, sizeInBytes, dims, device, typeLayoutT = ObjectSlotLayout<BlockQ50>())
        }
        GgmlDType.q51 -> {
            fromRawData<BlockQ51>(rawData, sizeInBytes, dims, device, typeLayoutT = ObjectSlotLayout<BlockQ51>())
        }
        GgmlDType.q80 -> {
            fromRawData<BlockQ80>(rawData, sizeInBytes, dims, device, typeLayoutT = ObjectSlotLayout<BlockQ80>())
        }
        GgmlDType.q2k -> {
            fromRawData<BlockQ2K>(rawData, sizeInBytes, dims, device, typeLayoutT = ObjectSlotLayout<BlockQ2K>())
        }
        GgmlDType.q3k -> {
            fromRawData<BlockQ3K>(rawData, sizeInBytes, dims, device, typeLayoutT = ObjectSlotLayout<BlockQ3K>())
        }
        GgmlDType.q4k -> {
            fromRawData<BlockQ4K>(rawData, sizeInBytes, dims, device, typeLayoutT = ObjectSlotLayout<BlockQ4K>())
        }
        GgmlDType.q5k -> {
            fromRawData<BlockQ5K>(rawData, sizeInBytes, dims, device, typeLayoutT = ObjectSlotLayout<BlockQ5K>())
        }
        GgmlDType.q6k -> {
            fromRawData<BlockQ6K>(rawData, sizeInBytes, dims, device, typeLayoutT = ObjectSlotLayout<BlockQ6K>())
        }
        else -> bail("quantized type {ggml_dtype:?} is not supported yet")
    }
}

public fun <R> readOneTensor(reader: R, magic: VersionedMagic, device: Device): Result<Pair<String, QTensor>> where R : Seek, R : Read {
    val nDims =
        when (val tryResult = reader.readU32<LittleEndian>()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val nameLen =
        when (val tryResult = reader.readU32<LittleEndian>()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val ggmlDtype =
        when (val tryResult = reader.readU32<LittleEndian>()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val ggmlDtype1 =
        when (val tryResult = GgmlDType.fromU32(ggmlDtype)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val dims = mutableListOf(0u, nDims.toInt())
    when (val tryResult = reader.readU32Into<LittleEndian>(dims)) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    // The dimensions are stored in reverse order, see for example:
    // https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/convert.py#L969
    dims.reverse()
    val name = mutableListOf(0u.toUByte(), nameLen.toInt())
    when (val tryResult = reader.readExact(name)) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    val name1 = String.fromUtf8Lossy(name).intoOwned()
    if (magic.align32()) {
        val pos =
            when (val tryResult = reader.streamPosition()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        when (val tryResult = reader.seek(SeekFrom.Current(((32 - pos % 32) % 32).toLong()))) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    }
    val dims1 = ArrayDeque((ArrayDeque(dims)).map({ u -> u.toInt() })).collect<Vec>()
    val tensorElems = dims1.iter().product<Int>()
    val sizeInBytes = tensorElems * ggmlDtype1.typeSize() / ggmlDtype1.blockSize()
    // Note: Mmap version to avoid copying the data around?
    val rawData = mutableListOf(0u.toUByte(), sizeInBytes)
    when (val tryResult = reader.readExact(rawData)) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    return run {
        val matchVal = qtensorFromGgml(ggmlDtype1, rawData, dims1, device)
        when (matchVal) {
            is Result.Ok -> {
                val tensor = matchVal.value
                Ok(Pair(name1, tensor))
            }
            is Result.Err -> {
                val e = matchVal.value
                bail("Error creating tensor {name}: {e}")
            }
        }
    }
}

public data class Content(
    public var magic: VersionedMagic = VersionedMagic.entries.first(),
    public var hparams: HParams = HParams(),
    public var vocab: Vocab = Vocab(),
    public var tensors: Map<String, QTensor> = emptyMap(),
    public var device: Device = Device(),
) {
    public fun remove(name: String): Result<QTensor> =
        when (val matchVal = this.tensors.remove(name)) {
            null -> bail("cannot find tensor with name '{name}'")
            else -> {
                val tensor = matchVal
                Ok(tensor)
            }
        }

    public companion object {
        public fun <R> read(reader: R, device: Device): Result<Content> where R : Seek, R : Read {
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/llama.cpp#L505
            val lastPosition =
                when (val tryResult = reader.seek(SeekFrom.End(0))) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            when (val tryResult = reader.seek(SeekFrom.Start(0))) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            val magic =
                when (val tryResult = VersionedMagic.read(reader)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val hparams =
                when (val tryResult = HParams.read(reader)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val vocab =
                when (val tryResult = Vocab.read(reader, hparams.nVocab.toInt())) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            var tensors = HashMap.new()
            while (when (val tryResult = reader.streamPosition()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                } != lastPosition
            ) {
                val (name, tensor) =
                    when (val tryResult = readOneTensor(reader, magic, device)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                tensors.insert(name, tensor)
            }
            val device1 = device.clone()
            return Ok(Content(magic = magic, hparams = hparams, vocab = vocab, tensors = tensors, device = device1))
        }
    }
}
