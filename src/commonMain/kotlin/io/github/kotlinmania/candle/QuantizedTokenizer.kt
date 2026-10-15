// port-lint: source quantized/tokenizer.rs
package io.github.kotlinmania.candle.tokenizer

import io.github.kotlinmania.candle.CandleError
import io.github.kotlinmania.candle.CandleResult
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
    val unkToken: String? = null,
    val byteFallback: Boolean = false,
    val ignoreMerges: Boolean = false,
)

class Tokenizer(
    val model: BpeModel,
    var normalizer: NormalizerWrapper? = null,
    var preTokenizer: PreTokenizerWrapper? = null,
    var decoder: DecoderWrapper? = null,
    val postProcessor: PostProcessorWrapper? = null,
    val specialTokens: MutableList<AddedToken> = ArrayList(),
) {
    companion object : TokenizerFromGguf {
        override fun fromGguf(ct: Content): CandleResult<Tokenizer> {
            val modelVal = when (val r = metadataValue(ct, "tokenizer.ggml.model")) {
                is CandleResult.Ok -> r.value
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }
            val modelStr = when (val r = modelVal.toString(true)) {
                is CandleResult.Ok -> r.value.lowercase()
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }
            if (modelStr != "gpt2") {
                return CandleResult.Err(CandleError.Msg("unsupported tokenizer model `$modelStr`").bt())
            }

            val tokensVal = when (val r = metadataValue(ct, "tokenizer.ggml.tokens")) {
                is CandleResult.Ok -> r.value
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }
            val tokens = when (val r = valueToStringArray(tokensVal, "tokenizer.ggml.tokens")) {
                is CandleResult.Ok -> r.value
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }
            val vocab = LinkedHashMap<String, UInt>(tokens.size)
            for ((i, t) in tokens.withIndex()) {
                vocab[t] = i.toUInt()
            }
            val mergesVal = when (val r = metadataValue(ct, "tokenizer.ggml.merges")) {
                is CandleResult.Ok -> r.value
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }
            val merges = when (val r = mergesFromValue(mergesVal)) {
                is CandleResult.Ok -> r.value
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }

            var unkToken: String? = null
            when (val r = metadataValue(ct, "tokenizer.ggml.unk_token_id")) {
                is CandleResult.Ok -> {
                    val tokenId = when (val idRes = ggufValueToU32(r.value)) {
                        is CandleResult.Ok -> idRes.value
                        is CandleResult.Err -> return CandleResult.Err(idRes.error)
                    }
                    if (tokenId.toInt() in tokens.indices) {
                        unkToken = tokens[tokenId.toInt()]
                    }
                }
                is CandleResult.Err -> {}
            }

            var byteFallback = false
            when (val r = metadataValue(ct, "tokenizer.ggml.byte_fallback")) {
                is CandleResult.Ok -> {
                    byteFallback = when (val bRes = r.value.toBool()) {
                        is CandleResult.Ok -> bRes.value
                        is CandleResult.Err -> return CandleResult.Err(bRes.error)
                    }
                }
                is CandleResult.Err -> {}
            }

            var ignoreMerges = false
            when (val r = metadataValue(ct, "tokenizer.ggml.ignore_merges")) {
                is CandleResult.Ok -> {
                    ignoreMerges = when (val bRes = r.value.toBool()) {
                        is CandleResult.Ok -> bRes.value
                        is CandleResult.Err -> return CandleResult.Err(bRes.error)
                    }
                }
                is CandleResult.Err -> {}
            }

            val bpe = BpeModel(
                vocab = vocab,
                merges = merges,
                unkToken = unkToken,
                byteFallback = byteFallback,
                ignoreMerges = ignoreMerges,
            )
            val tokenizer = Tokenizer(bpe)

            val pre = when (val r = metadataValue(ct, "tokenizer.ggml.pre")) {
                is CandleResult.Ok -> when (val sRes = r.value.toString(true)) {
                    is CandleResult.Ok -> sRes.value
                    is CandleResult.Err -> "gpt2"
                }
                is CandleResult.Err -> "gpt2"
            }
            val pipeline = when (val r = pipelineFromPre(pre)) {
                is CandleResult.Ok -> r.value
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }
            val postProcessorBase = pipeline.postProcessor

            val addBos = when (val r = metadataValue(ct, "tokenizer.ggml.add_bos_token")) {
                is CandleResult.Ok -> when (val bRes = r.value.toBool()) {
                    is CandleResult.Ok -> bRes.value
                    is CandleResult.Err -> false
                }
                is CandleResult.Err -> false
            }
            val addEos = when (val r = metadataValue(ct, "tokenizer.ggml.add_eos_token")) {
                is CandleResult.Ok -> when (val bRes = r.value.toBool()) {
                    is CandleResult.Ok -> bRes.value
                    is CandleResult.Err -> false
                }
                is CandleResult.Err -> false
            }
            val bosId = when (val r = metadataValue(ct, "tokenizer.ggml.bos_token_id")) {
                is CandleResult.Ok -> when (val uRes = ggufValueToU32(r.value)) {
                    is CandleResult.Ok -> uRes.value
                    is CandleResult.Err -> null
                }
                is CandleResult.Err -> null
            }
            val eosId = when (val r = metadataValue(ct, "tokenizer.ggml.eos_token_id")) {
                is CandleResult.Ok -> when (val uRes = ggufValueToU32(r.value)) {
                    is CandleResult.Ok -> uRes.value
                    is CandleResult.Err -> null
                }
                is CandleResult.Err -> null
            }

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
            when (val r = metadataValue(ct, "tokenizer.ggml.token_type")) {
                is CandleResult.Ok -> {
                    val v = r.value
                    if (v is Value.Array) {
                        val specials = ArrayList<AddedToken>()
                        for ((idx, elem) in v.v.withIndex()) {
                            val ty = when (val tyRes = ggufValueToU32(elem)) {
                                is CandleResult.Ok -> tyRes.value
                                is CandleResult.Err -> return CandleResult.Err(tyRes.error)
                            }
                            // Aligns with llamaTokenType: treat non-normal/non-byte tokens as special.
                            val isSpecial = ty in 2u..5u
                            if (isSpecial && idx in tokens.indices) {
                                specials.add(AddedToken(tokens[idx], true))
                            }
                        }
                        if (specials.isNotEmpty()) {
                            tokenizer.specialTokens.addAll(specials)
                        }
                    }
                }
                is CandleResult.Err -> {}
            }

            val explicitSpecials = HashSet<UInt>()
            for (key in listOf(
                "tokenizer.ggml.bos_token_id",
                "tokenizer.ggml.eos_token_id",
                "tokenizer.ggml.pad_token_id",
                "tokenizer.ggml.sep_token_id",
                "tokenizer.ggml.unk_token_id",
            )) {
                when (val r = metadataValue(ct, key)) {
                    is CandleResult.Ok -> {
                        val id = when (val idRes = ggufValueToU32(r.value)) {
                            is CandleResult.Ok -> idRes.value
                            is CandleResult.Err -> return CandleResult.Err(idRes.error)
                        }
                        explicitSpecials.add(id)
                    }
                    is CandleResult.Err -> {}
                }
            }
            if (explicitSpecials.isNotEmpty()) {
                val specials = ArrayList<AddedToken>()
                for (id in explicitSpecials) {
                    if (id.toInt() in tokens.indices) {
                        specials.add(AddedToken(tokens[id.toInt()], true))
                    }
                }
                if (specials.isNotEmpty()) {
                    tokenizer.specialTokens.addAll(specials)
                }
            }

            return CandleResult.Ok(tokenizer)
        }
    }
}

interface TokenizerFromGguf {
    fun fromGguf(ct: Content): CandleResult<Tokenizer>
}

private fun metadataValue(ct: Content, key: String): CandleResult<Value> {
    val v = ct.metadata[key]
        ?: return CandleResult.Err(CandleError.Msg("missing GGUF metadata key `$key`"))
    return CandleResult.Ok(v)
}

private fun ggufValueToU32(v: Value): CandleResult<UInt> = when (v) {
    is Value.U8 -> CandleResult.Ok(v.v.toUInt())
    is Value.I8 -> CandleResult.Ok(v.v.toUInt())
    is Value.U16 -> CandleResult.Ok(v.v.toUInt())
    is Value.I16 -> CandleResult.Ok(v.v.toUInt())
    is Value.U32 -> CandleResult.Ok(v.v)
    is Value.I32 -> CandleResult.Ok(v.v.toUInt())
    is Value.U64 -> CandleResult.Ok(v.v.toUInt())
    is Value.I64 -> CandleResult.Ok(v.v.toUInt())
    else -> CandleResult.Err(CandleError.Msg("expected numeric value for token type/id, got $v").bt())
}

private fun valueToStringArray(v: Value, name: String): CandleResult<List<String>> {
    val arr = when (val r = v.toVec()) {
        is CandleResult.Ok -> r.value
        is CandleResult.Err -> return CandleResult.Err(CandleError.Msg("`$name` is not an array"))
    }
    val out = ArrayList<String>(arr.size)
    for (elem in arr) {
        val s = when (val r = elem.toString(true)) {
            is CandleResult.Ok -> r.value
            is CandleResult.Err -> return CandleResult.Err(
                CandleError.Msg("`$name` element is not a string: $elem")
            )
        }
        out.add(s)
    }
    return CandleResult.Ok(out)
}

private fun mergesFromValue(v: Value): CandleResult<List<Pair<String, String>>> {
    val mergeStrings = when (val r = valueToStringArray(v, "tokenizer.ggml.merges")) {
        is CandleResult.Ok -> r.value
        is CandleResult.Err -> return CandleResult.Err(r.error)
    }
    val out = ArrayList<Pair<String, String>>(mergeStrings.size)
    for (m in mergeStrings) {
        val sp = m.indexOf(' ')
        if (sp < 0) {
            return CandleResult.Err(CandleError.Msg("invalid merge entry `$m`"))
        }
        out.add(m.substring(0, sp) to m.substring(sp + 1))
    }
    return CandleResult.Ok(out)
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

private fun preTokenizerSequence(regex: String, byteLevel: PreTokenizerWrapper.ByteLevel): CandleResult<PreTokenizerWrapper> {
    val split = PreTokenizerWrapper.Split(regex = regex, isolated = true, invert = false)
    return CandleResult.Ok(PreTokenizerWrapper.Sequence(listOf(split, byteLevel)))
}

private fun pipelineFromPre(pre: String): CandleResult<Pipeline> {
    val regexQwen2 = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    val regexLlama3 = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

    val pipeline = when (pre) {
        // Matches Qwen2 tokenizer.json settings
        "qwen2" -> {
            val preSeq = when (val r = preTokenizerSequence(regexQwen2, PreTokenizerWrapper.ByteLevel(false, false, false))) {
                is CandleResult.Ok -> r.value
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }
            Pipeline(
                normalizer = NormalizerWrapper("NFC"),
                pretokenizer = preSeq,
                decoder = DecoderWrapper(false, false, false),
                postProcessor = PostProcessorWrapper.ByteLevel(false, false, false),
            )
        }
        // Matches Smaug/Llama3 style byte-level BPE
        "smaug-bpe", "lfm2", "llama3" -> {
            val preSeq = when (val r = preTokenizerSequence(regexLlama3, PreTokenizerWrapper.ByteLevel(false, true, false))) {
                is CandleResult.Ok -> r.value
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }
            Pipeline(
                normalizer = null,
                pretokenizer = preSeq,
                decoder = DecoderWrapper(true, true, true),
                postProcessor = PostProcessorWrapper.ByteLevel(true, false, true),
            )
        }
        // Default GPT-2 style BPE
        else -> Pipeline(
            normalizer = null,
            pretokenizer = PreTokenizerWrapper.ByteLevel(),
            decoder = DecoderWrapper(),
            postProcessor = PostProcessorWrapper.ByteLevel(),
        )
    }
    return CandleResult.Ok(pipeline)
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
        val bid = bosId ?: return null
        val btok = bos ?: return null
        specials.add(btok to bid)
    }
    if (addEos) {
        val eid = eosId ?: return null
        val etok = eos ?: return null
        specials.add(etok to eid)
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
