// port-lint: source quantized/tokenizer.rs
package io.github.kotlinmania.candle.quantized.tokenizer

import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.quantized.gguffile.Content
import io.github.kotlinmania.candle.quantized.gguffile.Value
import io.github.kotlinmania.gguffile.Value.*
import io.github.kotlinmania.tokenizers.AddedToken
import io.github.kotlinmania.tokenizers.Tokenizer
import io.github.kotlinmania.tokenizers.decoders.DecoderWrapper
import io.github.kotlinmania.tokenizers.models.bpe.Bpe
import io.github.kotlinmania.tokenizers.models.bpe.Vocab
import io.github.kotlinmania.tokenizers.normalizers.NormalizerWrapper
import io.github.kotlinmania.tokenizers.normalizers.unicode.NFC
import io.github.kotlinmania.tokenizers.pretokenizers.PreTokenizerWrapper
import io.github.kotlinmania.tokenizers.pretokenizers.sequence.Sequence
import io.github.kotlinmania.tokenizers.pretokenizers.split.Split
import io.github.kotlinmania.tokenizers.pretokenizers.split.SplitPattern
import io.github.kotlinmania.tokenizers.processors.PostProcessorWrapper
import io.github.kotlinmania.tokenizers.tokenizer.SplitDelimiterBehavior
import io.github.kotlinmania.tokenizers.decoders.bytelevel.ByteLevel as ByteLevelDecoder
import io.github.kotlinmania.tokenizers.pretokenizers.bytelevel.ByteLevel as ByteLevelPre
import io.github.kotlinmania.tokenizers.processors.bytelevel.ByteLevel as ByteLevelProcessor
import io.github.kotlinmania.tokenizers.processors.sequence.Sequence as ProcessorSequence

public interface TokenizerFromGguf<T> {
    public fun fromGguf(ct: Content): Result<T>
}

public fun metadataValue(ct: Content, key: String): Result<Value> = ct.metadata.get(key).withContext({ "missing GGUF metadata key `$key`" })

public fun ggufValueToU32(v: Value): Result<UInt> =
    when (v) {
        is U8 -> {
            val (v1) = v
            Result.success(v1.toUInt())
        }
        is I8 -> {
            val (v2) = v
            Result.success(v2.toUInt())
        }
        is U16 -> {
            val (v3) = v
            Result.success(v3.toUInt())
        }
        is I16 -> {
            val (v4) = v
            Result.success(v4.toUInt())
        }
        is U32 -> {
            val (v5) = v
            Result.success(v5)
        }
        is I32 -> {
            val (v6) = v
            Result.success(v6.toUInt())
        }
        is U64 -> {
            val (v7) = v
            Result.success(v7.toUInt())
        }
        is I64 -> {
            val (v8) = v
            Result.success(v8.toUInt())
        }
        else -> bail("expected numeric value for token type/id, got {v:?}")
    }

public fun valueToStringArray(v: Value, name: String): Result<List<String>> {
    val arr = v.toVec().withContext({ "`$name` is not an array" }).getOrElse { return Result.failure(it) }
    return arr.iter().map({ v1 ->
        v1.toString().map({ s -> s.toString() }).withContext({ "`$name` element is not a string: $v" })
    }).collect()
}

public fun mergesFromValue(v: Value): Result<List<Pair<String, String>>> {
    return valueToStringArray(v, "tokenizer.ggml.merges").getOrElse { return Result.failure(it) }.map({ m ->
        m.splitOnce(' ').map({ (a, b) -> Pair(a.toString(), b.toString()) }).okOrElse({ Error.msg("invalid merge entry `$m`") })
    }).collect()
}

public open class Pipeline(
    public var normalizer: NormalizerWrapper?,
    public var pretokenizer: PreTokenizerWrapper?,
    public var decoder: DecoderWrapper?,
    public var postProcessor: PostProcessorWrapper?,
) {
    public open fun apply(tokenizer: Tokenizer): Result<Unit> {
        when (val matchVal = this.normalizer) {
            null -> {}
            else -> {
                val norm = matchVal
                tokenizer.withNormalizer(norm).mapErr(Error::wrap).getOrElse { return Result.failure(it) }
            }
        }
        when (val matchVal1 = this.pretokenizer) {
            null -> {}
            else -> {
                val pt = matchVal1
                tokenizer.withPreTokenizer(pt)
            }
        }
        when (val matchVal2 = this.decoder) {
            null -> {}
            else -> {
                val dec = matchVal2
                tokenizer.withDecoder(dec)
            }
        }
        when (val matchVal3 = this.postProcessor) {
            null -> {}
            else -> {
                val pp = matchVal3
                tokenizer.withPostProcessor(pp)
            }
        }
        return Result.success(Unit)
    }
}

public fun preTokenizerSequence(regex: String, byteLevel: ByteLevelPre): Result<PreTokenizerWrapper> {
    val split = Split.new(SplitPattern.Regex(regex.toString()), SplitDelimiterBehavior.Isolated, false).mapErr(Error::wrap).getOrElse { return Result.failure(it) }
    return Result.success(Sequence.new(mutableListOf(split.into(), byteLevel.into())).into())
}

public fun pipelineFromPre(pre: String): Result<Pipeline> {
    val REGEX_QWEN2: String = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    val REGEX_LLAMA3: String = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    return Result.success(when (pre) {
        // Matches Qwen2 tokenizer.json settings
        "qwen2" -> Pipeline(normalizer = NFC.into(), pretokenizer = preTokenizerSequence(REGEX_QWEN2, ByteLevelPre.new(false, false, false)).getOrElse { return Result.failure(it) }, decoder = ByteLevelDecoder.new(false, false, false).into(), postProcessor = ByteLevelProcessor.new(false, false, false).into())
        // Matches Smaug/Llama3 style byte-level BPE
        "smaug-bpe", "lfm2", "llama3" -> Pipeline(normalizer = null, pretokenizer = preTokenizerSequence(REGEX_LLAMA3, ByteLevelPre.new(false, true, false)).getOrElse { return Result.failure(it) }, decoder = ByteLevelDecoder.new(true, true, true).into(), postProcessor = ByteLevelProcessor.new(true, false, true).into())
        // Default GPT-2 style BPE
        else -> Pipeline(normalizer = null, pretokenizer = ByteLevelPre.default().into(), decoder = ByteLevelDecoder.default().into(), postProcessor = ByteLevelProcessor.default().into())
    })
}

public fun templateProcessor(tokens: List<String>, bosId: UInt?, eosId: UInt?, addBos: Boolean, addEos: Boolean): PostProcessorWrapper? {
    if ((!addBos && !addEos) || tokens.isEmpty()) {
        return null
    }
    val bos = bosId?.let({ id -> tokens.get(id.toInt()) }).cloned()
    val eos = eosId?.let({ id -> tokens.get(id.toInt()) }).cloned()
    val specials = mutableListOf()
    if (addBos) {
        val bosId1 = (bosId ?: return null)
        val bosTok = (bos.clone() ?: return null)
        specials.add(Pair(bosTok.clone(), bosId1))
    }
    if (addEos) {
        val eosId1 = (eosId ?: return null)
        val eosTok = (eos.clone() ?: return null)
        specials.add(Pair(eosTok.clone(), eosId1))
    }
    val single = mutableListOf()
    if (addBos) {
        single.add((bos.clone() ?: return null))
    }
    single.add("\$0".toString())
    if (addEos) {
        single.add((eos.clone() ?: return null))
    }
    val pair = mutableListOf()
    if (addBos) {
        pair.add("${(bos.clone() ?: return null)}:0")
    }
    pair.add("\$A:0".toString())
    if (addEos) {
        pair.add("${(eos.clone() ?: return null)}:0")
    }
    if (addBos) {
        pair.add("${(bos.clone() ?: return null)}:1")
    }
    pair.add("\$B:1".toString())
    if (addEos) {
        pair.add("${(eos.clone() ?: return null)}:1")
    }
    val proc = (((TemplateProcessing.builder().trySingle(single).ok() ?: return null).tryPair(pair).ok() ?: return null).specialTokens(specials).build().ok() ?: return null)
    return PostProcessorWrapper.Template(proc)
}

public fun tokenizerFromGgufTokenizer(ct: Content): Result<Tokenizer> {
    val modelKind = metadataValue(ct, "tokenizer.ggml.model").getOrElse { return Result.failure(it) }.toString().getOrElse { return Result.failure(it) }.toLowercase()
    if (modelKind != "gpt2") {
        bail("unsupported tokenizer model `{model_kind}`")
    }
    val tokens = valueToStringArray(metadataValue(ct, "tokenizer.ggml.tokens").getOrElse { return Result.failure(it) }, "tokenizer.ggml.tokens").getOrElse { return Result.failure(it) }
    val vocab: Vocab = tokens.enumerate().map({ (i, t) -> Pair(t.clone(), i.toUInt()) }).collect()
    val merges = mergesFromValue(metadataValue(ct, "tokenizer.ggml.merges").getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
    var builder = Bpe.builder().vocabAndMerges(vocab, merges)
    when (val matchVal = metadataValue(ct, "tokenizer.ggml.unk_token_id")) {
        is Ok -> {
            val (value) = matchVal
            val tokenId = ggufValueToU32(value).getOrElse { return Result.failure(it) }
            when (val matchVal1 = tokens.get(tokenId.toInt())) {
                null -> {}
                else -> {
                    val token = matchVal1
                    builder = builder.unkToken(token.clone())
                }
            }
        }
        else -> {}
    }
    when (val matchVal2 = metadataValue(ct, "tokenizer.ggml.byte_fallback")) {
        is Ok -> {
            val (value1) = matchVal2
            builder = builder.byteFallback(value1.toBool().getOrElse { return Result.failure(it) })
        }
        else -> {}
    }
    when (val matchVal3 = metadataValue(ct, "tokenizer.ggml.ignore_merges")) {
        is Ok -> {
            val (value2) = matchVal3
            builder = builder.ignoreMerges(value2.toBool().getOrElse { return Result.failure(it) })
        }
        else -> {}
    }
    val bpe = builder.build().mapErr(Error::wrap).getOrElse { return Result.failure(it) }
    val tokenizer = Tokenizer.new(bpe)
    val pre = metadataValue(ct, "tokenizer.ggml.pre").andThen({ v -> v.toString() }).map({ s -> s.toString() }).unwrapOrElse({ "gpt2".toString() })
    val pipeline = pipelineFromPre(pre.asStr()).getOrElse { return Result.failure(it) }
    val postProcessorBase = pipeline.postProcessor
    val addBos = metadataValue(ct, "tokenizer.ggml.add_bos_token").andThen({ v -> v.toBool() }).unwrapOr(false)
    val addEos = metadataValue(ct, "tokenizer.ggml.add_eos_token").andThen({ v -> v.toBool() }).unwrapOr(false)
    val bosId = metadataValue(ct, "tokenizer.ggml.bos_token_id").andThen(::ggufValueToU32).ok()
    val eosId = metadataValue(ct, "tokenizer.ggml.eos_token_id").andThen(::ggufValueToU32).ok()
    pipeline.apply(tokenizer).getOrElse { return Result.failure(it) }
    // Compose existing post-processor with a template-based one if needed
    val templatePp = templateProcessor(tokens, bosId, eosId, addBos, addEos)
    if (templatePp != null || postProcessorBase != null) {
        val steps = mutableListOf()
        if (postProcessorBase != null) {
            val pp = postProcessorBase
            steps.add(pp)
        }
        if (templatePp != null) {
            val tp = templatePp
            steps.add(tp)
        }
        val pp1 =
            if (steps.size == 1) {
                steps.removeLastOrNull()!!
            } else {
                ProcessorSequence.new(steps).into()
            }
        tokenizer.withPostProcessor(pp1)
    }
    // Mark special tokens so decode(skipSpecialTokens = true) behaves as expected
    run {
        val matchVal4 = metadataValue(ct, "tokenizer.ggml.token_type")
        if (matchVal4 is Ok && matchVal4.value is Value.Array) {
            val (arr) = matchVal4.value as Value.Array
            val specials = mutableListOf()
            for ((idx, v) in arr.iter().enumerate()) {
                val ty = ggufValueToU32(v).getOrElse { return Result.failure(it) }
                // Aligns with llamaTokenType: treat non-normal/non-byte tokens as special.
                val isSpecial =
                    when (ty) {
                        in 2..5 -> true
                        else -> false
                    }
                if (isSpecial) {
                    when (val matchVal5 = tokens.get(idx)) {
                        null -> {}
                        else -> {
                            val tok = matchVal5
                            specials.add(AddedToken.from(tok.clone(), true))
                        }
                    }
                }
            }
            if (!specials.isEmpty()) {
                tokenizer.addSpecialTokens(specials).mapErr(Error::wrap).getOrElse { return Result.failure(it) }
            }
        }
    }
    val explicitSpecials = HashSet.new()
    for (key in listOf("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.pad_token_id", "tokenizer.ggml.sep_token_id", "tokenizer.ggml.unk_token_id")) {
        when (val matchVal6 = metadataValue(ct, key)) {
            is Ok -> {
                val (value3) = matchVal6
                explicitSpecials.insert(ggufValueToU32(value3).getOrElse { return Result.failure(it) })
            }
            else -> {}
        }
    }
    if (!explicitSpecials.isEmpty()) {
        val specials1: Vec = explicitSpecials.intoIter().filterMap({ id -> tokens.get(id.toInt()) }).map({ tok1 -> AddedToken.from(tok1.clone(), true) }).collect()
        if (!specials1.isEmpty()) {
            tokenizer.addSpecialTokens(specials1).mapErr(Error::wrap).getOrElse { return Result.failure(it) }
        }
    }
    return Result.success(tokenizer)
}
