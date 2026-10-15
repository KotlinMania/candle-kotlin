// port-lint: source quantized/tokenizer.rs
package io.github.kotlinmania.candle.tokenizer

import io.github.kotlinmania.candle.CandleException
import io.github.kotlinmania.candle.CandleResult
import io.github.kotlinmania.candle.ErrorContext
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.gguf.Content
import io.github.kotlinmania.candle.gguf.Value

data class NormalizerWrapper(val kind: String)

sealed class PreTokenizerWrapper {
    data class ByteLevel(
        val addPrefixSpace: Boolean = true,
        val trimOffsets: Boolean = true,
        val useRegex: Boolean = true,
    ) : PreTokenizerWrapper()

    data class Split(
        val regex: String,
        val isolated: Boolean = true,
        val invert: Boolean = false,
    ) : PreTokenizerWrapper()

    data class Sequence(val steps: List<PreTokenizerWrapper>) : PreTokenizerWrapper()
}

data class DecoderWrapper(
    val addPrefixSpace: Boolean = true,
    val trimOffsets: Boolean = true,
    val useRegex: Boolean = true,
)

sealed class PostProcessorWrapper {
    data class ByteLevel(
        val addPrefixSpace: Boolean = true,
        val trimOffsets: Boolean = true,
        val useRegex: Boolean = true,
    ) : PostProcessorWrapper()

    data class Template(
        val single: List<String>,
        val pair: List<String>,
        val specialTokens: List<Pair<String, UInt>>,
    ) : PostProcessorWrapper()

    data class Sequence(val steps: List<PostProcessorWrapper>) : PostProcessorWrapper()
}

data class AddedToken(
    val content: String,
    val special: Boolean,
)

class BpeModel(
    val vocab: Map<String, UInt>,
    val merges: List<Pair<String, String>>,
    var unkToken: String? = null,
    var byteFallback: Boolean = false,
    var ignoreMerges: Boolean = false,
)

private val withCandleContext: (CandleResult<*>, () -> String) -> Result<Any?> = { res, msg ->
    when (res) {
        is CandleResult.Ok -> Result.success(res.value)
        is CandleResult.Err -> Result.failure(CandleException.Msg(msg()))
    }
}

public interface TokenizerFromGguf {
    public fun fromGguf(ct: Content): Result<Tokenizer>
}

private fun metadataValue(ct: Content, key: String): Result<Value> {
    return ErrorContext.ForOption(ct.metadata[key])
        .withContext { "missing GGUF metadata key `$key`" }
}

private fun ggufValueToU32(v: Value): Result<UInt> {
    return when (v) {
        is Value.U8 -> Result.success(v.v.toUInt())
        is Value.I8 -> Result.success(v.v.toUInt())
        is Value.U16 -> Result.success(v.v.toUInt())
        is Value.I16 -> Result.success(v.v.toUInt())
        is Value.U32 -> Result.success(v.v)
        is Value.I32 -> Result.success(v.v.toUInt())
        is Value.U64 -> Result.success(v.v.toUInt())
        is Value.I64 -> Result.success(v.v.toUInt())
        else -> bail("expected numeric value for token type/id, got $v")
    }
}

private fun valueToStringArray(v: Value, name: String): Result<List<String>> {
    val arr = when (val r = v.toVec()) {
        is CandleResult.Ok -> r.value
        is CandleResult.Err -> bail("`$name` is not an array")
    }
    return Result.success(
        arr.map { elem ->
            when (val r = elem.toString(true)) {
                is CandleResult.Ok -> r.value.toString()
                is CandleResult.Err -> bail("`$name` element is not a string: $elem")
            }
        }
    )
}

private fun mergesFromValue(v: Value): Result<List<Pair<String, String>>> {
    return Result.success(
        valueToStringArray(v, "tokenizer.ggml.merges")
            .getOrThrow()
            .map { m ->
                val idx = m.indexOf(' ')
                if (idx >= 0) {
                    Pair(m.substring(0, idx), m.substring(idx + 1))
                } else {
                    throw CandleException.msg("invalid merge entry `$m`")
                }
            }
    )
}

private class Pipeline(
    val normalizer: NormalizerWrapper?,
    val pretokenizer: PreTokenizerWrapper?,
    val decoder: DecoderWrapper?,
    val postProcessor: PostProcessorWrapper?,
) {
    fun apply(tokenizer: Tokenizer) {
        if (this.normalizer != null) {
            tokenizer.normalizer = this.normalizer
        }
        if (this.pretokenizer != null) {
            tokenizer.preTokenizer = this.pretokenizer
        }
        if (this.decoder != null) {
            tokenizer.decoder = this.decoder
        }
        if (this.postProcessor != null) {
            tokenizer.postProcessor = this.postProcessor
        }
    }
}

private fun preTokenizerSequence(regex: String, byteLevel: PreTokenizerWrapper.ByteLevel): Result<PreTokenizerWrapper> {
    val split = PreTokenizerWrapper.Split(
        regex = regex.toString(),
        isolated = true,
        invert = false,
    )
    return Result.success(PreTokenizerWrapper.Sequence(listOf(split, byteLevel)))
}

private fun pipelineFromPre(pre: String): Result<Pipeline> {
    val regexQwen2 = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    val regexLlama3 = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

    return Result.success(
        when (pre) {
            // Matches Qwen2 tokenizer.json settings
            "qwen2" -> Pipeline(
                normalizer = NormalizerWrapper("NFC"),
                pretokenizer = preTokenizerSequence(
                    regexQwen2,
                    PreTokenizerWrapper.ByteLevel(false, false, false),
                ).getOrThrow(),
                decoder = DecoderWrapper(false, false, false),
                postProcessor = PostProcessorWrapper.ByteLevel(false, false, false),
            )
            // Matches Smaug/Llama3 style byte-level BPE
            "smaug-bpe", "lfm2", "llama3" -> Pipeline(
                normalizer = null,
                pretokenizer = preTokenizerSequence(
                    regexLlama3,
                    PreTokenizerWrapper.ByteLevel(false, true, false),
                ).getOrThrow(),
                decoder = DecoderWrapper(true, true, true),
                postProcessor = PostProcessorWrapper.ByteLevel(true, false, true),
            )
            // Default GPT-2 style BPE
            else -> Pipeline(
                normalizer = null,
                pretokenizer = PreTokenizerWrapper.ByteLevel(),
                decoder = DecoderWrapper(),
                postProcessor = PostProcessorWrapper.ByteLevel(),
            )
        }
    )
}

private fun templateProcessor(
    tokens: List<String>,
    bosId: UInt?,
    eosId: UInt?,
    addBos: Boolean,
    addEos: Boolean,
): PostProcessorWrapper? {
    if ((!addBos && !addEos) || tokens.isEmpty()) {
        return null
    }

    val bos = bosId?.let { id -> tokens.getOrNull(id.toInt()) }
    val eos = eosId?.let { id -> tokens.getOrNull(id.toInt()) }

    val specials = ArrayList<Pair<String, UInt>>()
    if (addBos) {
        val bosIdVal = bosId ?: return null
        val bosTok = bos ?: return null
        specials.add(Pair(bosTok, bosIdVal))
    }
    if (addEos) {
        val eosIdVal = eosId ?: return null
        val eosTok = eos ?: return null
        specials.add(Pair(eosTok, eosIdVal))
    }

    val single = ArrayList<String>()
    if (addBos) {
        single.add(bos ?: return null)
    }
    single.add("\$0")
    if (addEos) {
        single.add(eos ?: return null)
    }

    val pair = ArrayList<String>()
    if (addBos) {
        pair.add("${bos ?: return null}:0")
    }
    pair.add("\$A:0")
    if (addEos) {
        pair.add("${eos ?: return null}:0")
    }
    if (addBos) {
        pair.add("${bos ?: return null}:1")
    }
    pair.add("\$B:1")
    if (addEos) {
        pair.add("${eos ?: return null}:1")
    }

    return PostProcessorWrapper.Template(
        single = single,
        pair = pair,
        specialTokens = specials,
    )
}

class Tokenizer(
    val model: BpeModel,
    var normalizer: NormalizerWrapper? = null,
    var preTokenizer: PreTokenizerWrapper? = null,
    var decoder: DecoderWrapper? = null,
    var postProcessor: PostProcessorWrapper? = null,
    val specialTokens: MutableList<AddedToken> = ArrayList(),
) {
    companion object : TokenizerFromGguf {
        override fun fromGguf(ct: Content): Result<Tokenizer> {
            val modelKind = withCandleContext(metadataValue(ct, "tokenizer.ggml.model").getOrThrow().toString(true)) { "model" }
                .getOrThrow()
                .toString()
                .lowercase()
            if (modelKind != "gpt2") {
                bail("unsupported tokenizer model `$modelKind`")
            }

            val tokens = valueToStringArray(
                metadataValue(ct, "tokenizer.ggml.tokens").getOrThrow(),
                "tokenizer.ggml.tokens",
            ).getOrThrow()
            val vocab: Map<String, UInt> = tokens
                .withIndex()
                .associate { (i, t) -> Pair(t, i.toUInt()) }
            val merges = mergesFromValue(metadataValue(ct, "tokenizer.ggml.merges").getOrThrow()).getOrThrow()

            val builder = BpeModel(vocab, merges)

            val unkVal = metadataValue(ct, "tokenizer.ggml.unk_token_id").getOrNull()
            if (unkVal != null) {
                val tokenId = ggufValueToU32(unkVal).getOrThrow()
                val token = tokens.getOrNull(tokenId.toInt())
                if (token != null) {
                    builder.unkToken = token
                }
            }

            val byteFallbackVal = metadataValue(ct, "tokenizer.ggml.byte_fallback").getOrNull()
            if (byteFallbackVal != null) {
                builder.byteFallback = withCandleContext(byteFallbackVal.toBool()) { "byte_fallback" }.getOrThrow() as Boolean
            }

            val ignoreMergesVal = metadataValue(ct, "tokenizer.ggml.ignore_merges").getOrNull()
            if (ignoreMergesVal != null) {
                builder.ignoreMerges = withCandleContext(ignoreMergesVal.toBool()) { "ignore_merges" }.getOrThrow() as Boolean
            }

            val bpe = builder
            val tokenizer = Tokenizer(bpe)

            val pre = metadataValue(ct, "tokenizer.ggml.pre")
                .mapCatching { v -> (withCandleContext(v.toString(true)) { "pre" }.getOrThrow() as String) }
                .getOrElse { "gpt2" }
            val pipeline = pipelineFromPre(pre).getOrThrow()
            val postProcessorBase = pipeline.postProcessor

            val addBos = metadataValue(ct, "tokenizer.ggml.add_bos_token")
                .mapCatching { v -> withCandleContext(v.toBool()) { "add_bos" }.getOrThrow() as Boolean }
                .getOrDefault(false)
            val addEos = metadataValue(ct, "tokenizer.ggml.add_eos_token")
                .mapCatching { v -> withCandleContext(v.toBool()) { "add_eos" }.getOrThrow() as Boolean }
                .getOrDefault(false)
            val bosId = metadataValue(ct, "tokenizer.ggml.bos_token_id")
                .mapCatching { v -> ggufValueToU32(v).getOrThrow() }
                .getOrNull()
            val eosId = metadataValue(ct, "tokenizer.ggml.eos_token_id")
                .mapCatching { v -> ggufValueToU32(v).getOrThrow() }
                .getOrNull()

            pipeline.apply(tokenizer)

            // Compose existing post-processor with a template-based one if needed
            val templatePp = templateProcessor(tokens, bosId, eosId, addBos, addEos)
            if (templatePp != null || postProcessorBase != null) {
                val steps = ArrayList<PostProcessorWrapper>()
                if (postProcessorBase != null) {
                    steps.add(postProcessorBase)
                }
                if (templatePp != null) {
                    steps.add(templatePp)
                }
                val pp = if (steps.size == 1) {
                    steps.removeAt(steps.size - 1)
                } else {
                    PostProcessorWrapper.Sequence(steps)
                }
                tokenizer.postProcessor = pp
            }

            // Mark special tokens so decode(skipSpecialTokens = true) behaves as expected
            val tokenTypeVal = metadataValue(ct, "tokenizer.ggml.token_type").getOrNull()
            if (tokenTypeVal is Value.Array) {
                val specials = ArrayList<AddedToken>()
                for ((idx, v) in tokenTypeVal.v.withIndex()) {
                    val ty = ggufValueToU32(v).getOrThrow()
                    // Aligns with llamaTokenType: treat non-normal/non-byte tokens as special.
                    val isSpecial = ty in 2u..5u
                    if (isSpecial) {
                        val tok = tokens.getOrNull(idx)
                        if (tok != null) {
                            specials.add(AddedToken(tok, true))
                        }
                    }
                }
                if (specials.isNotEmpty()) {
                    tokenizer.specialTokens.addAll(specials)
                }
            }

            val explicitSpecials = HashSet<UInt>()
            for (key in listOf(
                "tokenizer.ggml.bos_token_id",
                "tokenizer.ggml.eos_token_id",
                "tokenizer.ggml.pad_token_id",
                "tokenizer.ggml.sep_token_id",
                "tokenizer.ggml.unk_token_id",
            )) {
                val v = metadataValue(ct, key).getOrNull()
                if (v != null) {
                    explicitSpecials.add(ggufValueToU32(v).getOrThrow())
                }
            }
            if (explicitSpecials.isNotEmpty()) {
                val specials = explicitSpecials
                    .mapNotNull { id -> tokens.getOrNull(id.toInt()) }
                    .map { tok -> AddedToken(tok, true) }
                if (specials.isNotEmpty()) {
                    tokenizer.specialTokens.addAll(specials)
                }
            }

            return Result.success(tokenizer)
        }
    }
}
