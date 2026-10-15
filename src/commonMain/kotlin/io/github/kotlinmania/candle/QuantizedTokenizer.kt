// port-lint: source quantized/tokenizer.rs
package io.github.kotlinmania.candle.tokenizer

import io.github.kotlinmania.candle.CandleError
import io.github.kotlinmania.candle.CandleResult
import io.github.kotlinmania.candle.ErrorContext
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.gguf.Content
import io.github.kotlinmania.candle.gguf.Value

public data class NormalizerWrapper(val kind: String)

private object NFC {
    val into: () -> NormalizerWrapper = { NormalizerWrapper("NFC") }
}

public enum class SplitDelimiterBehavior {
    Isolated,
}

public sealed class SplitPattern {
    public data class Regex(val pattern: String) : SplitPattern()
}

public sealed class PreTokenizerWrapper {
    public data class ByteLevel(
        val addPrefixSpace: Boolean = true,
        val trimOffsets: Boolean = true,
        val useRegex: Boolean = true,
    ) : PreTokenizerWrapper() {
        public val into: () -> PreTokenizerWrapper = { this }
    }

    public data class Split(
        val regex: String,
        val isolated: Boolean = true,
        val invert: Boolean = false,
    ) : PreTokenizerWrapper() {
        public val into: () -> PreTokenizerWrapper = { this }

        public companion object {
            public val new: (SplitPattern, SplitDelimiterBehavior, Boolean) -> CandleResult<Split> = { pat, _, inv ->
                val rx = when (pat) {
                    is SplitPattern.Regex -> pat.pattern
                }
                CandleResult.Ok(Split(rx, true, inv))
            }
        }
    }

    public data class Sequence(val steps: List<PreTokenizerWrapper>) : PreTokenizerWrapper() {
        public val into: () -> PreTokenizerWrapper = { this }

        public companion object {
            public val new: (List<PreTokenizerWrapper>) -> Sequence = { s -> Sequence(s) }
        }
    }
}

private object ByteLevelPre {
    val new: (Boolean, Boolean, Boolean) -> PreTokenizerWrapper.ByteLevel = { a, b, c ->
        PreTokenizerWrapper.ByteLevel(a, b, c)
    }
    val default: () -> PreTokenizerWrapper.ByteLevel = {
        PreTokenizerWrapper.ByteLevel()
    }
}

public data class DecoderWrapper(
    val addPrefixSpace: Boolean = true,
    val trimOffsets: Boolean = true,
    val useRegex: Boolean = true,
) {
    public val into: () -> DecoderWrapper = { this }
}

private object ByteLevelDecoder {
    val new: (Boolean, Boolean, Boolean) -> DecoderWrapper = { a, b, c ->
        DecoderWrapper(a, b, c)
    }
    val default: () -> DecoderWrapper = {
        DecoderWrapper()
    }
}

public sealed class PostProcessorWrapper {
    public data class ByteLevel(
        val addPrefixSpace: Boolean = true,
        val trimOffsets: Boolean = true,
        val useRegex: Boolean = true,
    ) : PostProcessorWrapper() {
        public val into: () -> PostProcessorWrapper = { this }
    }

    public data class Template(
        val single: List<String>,
        val pair: List<String>,
        val specialTokens: List<Pair<String, UInt>>,
    ) : PostProcessorWrapper()

    public data class Sequence(val steps: List<PostProcessorWrapper>) : PostProcessorWrapper() {
        public val into: () -> PostProcessorWrapper = { this }
    }
}

private object ByteLevelProcessor {
    val new: (Boolean, Boolean, Boolean) -> PostProcessorWrapper.ByteLevel = { a, b, c ->
        PostProcessorWrapper.ByteLevel(a, b, c)
    }
    val default: () -> PostProcessorWrapper.ByteLevel = {
        PostProcessorWrapper.ByteLevel()
    }
}

private object ProcessorSequence {
    val new: (List<PostProcessorWrapper>) -> PostProcessorWrapper.Sequence = { steps ->
        PostProcessorWrapper.Sequence(steps)
    }
}

private class TemplateProcessingBuilder {
    private var singleList: List<String> = emptyList()
    private var pairList: List<String> = emptyList()
    private var specialsList: List<Pair<String, UInt>> = emptyList()

    val trySingle: (List<String>) -> CandleResult<TemplateProcessingBuilder> = { s ->
        singleList = s
        CandleResult.Ok(this)
    }
    val tryPair: (List<String>) -> CandleResult<TemplateProcessingBuilder> = { p ->
        pairList = p
        CandleResult.Ok(this)
    }
    val specialTokens: (List<Pair<String, UInt>>) -> TemplateProcessingBuilder = { sp ->
        specialsList = sp
        this
    }
    val build: () -> CandleResult<PostProcessorWrapper.Template> = {
        CandleResult.Ok(PostProcessorWrapper.Template(singleList, pairList, specialsList))
    }
}

private object tokenizers {
    object processors {
        object template {
            object TemplateProcessing {
                val builder: () -> TemplateProcessingBuilder = { TemplateProcessingBuilder() }
            }
        }
    }
}

public data class AddedToken(
    val content: String,
    val special: Boolean,
) {
    public companion object {
        public val from: (String, Boolean) -> AddedToken = { content, special ->
            AddedToken(content, special)
        }
    }
}

public class BpeModel(
    public val vocab: Map<String, UInt>,
    public val merges: List<Pair<String, String>>,
    public var unkToken: String? = null,
    public var byteFallback: Boolean = false,
    public var ignoreMerges: Boolean = false,
)

private class BpeBuilder {
    private var vocabMap: Map<String, UInt> = emptyMap()
    private var mergesList: List<Pair<String, String>> = emptyList()
    private var unk: String? = null
    private var byteFb: Boolean = false
    private var ignoreM: Boolean = false

    val vocabAndMerges: (Map<String, UInt>, List<Pair<String, String>>) -> BpeBuilder = { v, m ->
        vocabMap = v
        mergesList = m
        this
    }
    val unkToken: (String) -> BpeBuilder = { u ->
        unk = u
        this
    }
    val byteFallback: (Boolean) -> BpeBuilder = { b ->
        byteFb = b
        this
    }
    val ignoreMerges: (Boolean) -> BpeBuilder = { i ->
        ignoreM = i
        this
    }
    val build: () -> CandleResult<BpeModel> = {
        CandleResult.Ok(BpeModel(vocabMap, mergesList, unk, byteFb, ignoreM))
    }
}

private object BPE {
    val builder: () -> BpeBuilder = { BpeBuilder() }
}

private object Error {
    val msg: (String) -> Throwable = { m -> CandleError.Msg(m) }
    val wrap: (Throwable) -> Throwable = { e -> CandleError.wrap(e) }
}

private fun <T> CandleResult<T>.ok(): T? = (this as? CandleResult.Ok)?.value
private fun <T> CandleResult<T>.unwrapOr(default: T): T = if (this is CandleResult.Ok) this.value else default

private val format: (String, Any?) -> String = { fmt, a ->
    fmt.replace("{key}", a.toString())
        .replace("{v:?}", a.toString())
        .replace("{name}", a.toString())
        .replace("{m}", a.toString())
        .replace("{model_kind}", a.toString())
        .replace("{}", a.toString())
}

private fun <T> T?.withContext(f: () -> String): CandleResult<T> =
    if (this != null) CandleResult.Ok(this) else CandleResult.Err(CandleError.Msg(f()))

private fun <T> CandleResult<T>.withContext(f: () -> String): CandleResult<T> =
    when (this) {
        is CandleResult.Ok -> this
        is CandleResult.Err -> CandleResult.Err((this.error as? CandleError ?: CandleError.Wrapped(this.error.toString())).context(f()))
    }

private val String.splitOnce: (Char) -> Pair<String, String>?
    get() = { sep ->
        val idx = this.indexOf(sep)
        if (idx >= 0) Pair(this.substring(0, idx), this.substring(idx + 1)) else null
    }

private fun <T> T?.okOrElse(errFn: () -> Throwable): CandleResult<T> =
    if (this != null) CandleResult.Ok(this) else CandleResult.Err(errFn())

public interface TokenizerFromGguf {
    public fun fromGguf(ct: Content): Result<Tokenizer>
}

public fun metadataValue(ct: Content, key: String): CandleResult<Value> {
    return ct.metadata[key]
        .withContext { format("missing GGUF metadata key `{key}`", key) }
}

public fun ggufValueToU32(v: Value): CandleResult<UInt> {
    return when (v) {
        is Value.U8 -> CandleResult.Ok(v.v.toUInt())
        is Value.I8 -> CandleResult.Ok(v.v.toUInt())
        is Value.U16 -> CandleResult.Ok(v.v.toUInt())
        is Value.I16 -> CandleResult.Ok(v.v.toUInt())
        is Value.U32 -> CandleResult.Ok(v.v)
        is Value.I32 -> CandleResult.Ok(v.v.toUInt())
        is Value.U64 -> CandleResult.Ok(v.v.toUInt())
        is Value.I64 -> CandleResult.Ok(v.v.toUInt())
        else -> bail("expected numeric value for token type/id, got $v")
    }
}

public fun valueToStringArray(v: Value, name: String): CandleResult<List<String>> {
    val arr = v
        .toVec()
        .withContext { format("`{name}` is not an array", name) }
        .unwrap()
    val result = arr
        .iterator()
        .asSequence()
        .map { elem ->
            elem.toString(true)
                .map { s -> s.toString() }
                .withContext { format("`{name}` element is not a string: {v:?}", name) }
                .unwrap()
        }
        .toList()
    return CandleResult.Ok(result)
}

public fun mergesFromValue(v: Value): CandleResult<List<Pair<String, String>>> {
    val list = valueToStringArray(v, "tokenizer.ggml.merges")
        .unwrap()
        .map { m ->
            m.splitOnce(' ')
                ?.let { pair -> Pair(pair.first.toString(), pair.second.toString()) }
                .okOrElse { Error.msg(format("invalid merge entry `{m}`", m)) }
                .unwrap()
        }
    return CandleResult.Ok(list)
}

public class Pipeline(
    public val normalizer: NormalizerWrapper?,
    public val pretokenizer: PreTokenizerWrapper?,
    public val decoder: DecoderWrapper?,
    public val postProcessor: PostProcessorWrapper?,
) {
    public fun apply(tokenizer: Tokenizer, self: Pipeline = this) {
        val norm = self.normalizer
        if (norm != null) {
            tokenizer.withNormalizer(norm)
        }
        val pt = self.pretokenizer
        if (pt != null) {
            tokenizer.withPreTokenizer(pt)
        }
        val dec = self.decoder
        if (dec != null) {
            tokenizer.withDecoder(dec)
        }
        val pp = self.postProcessor
        if (pp != null) {
            tokenizer.withPostProcessor(pp)
        }
    }
}

public fun preTokenizerSequence(regex: String, byteLevel: PreTokenizerWrapper.ByteLevel): CandleResult<PreTokenizerWrapper> {
    val split = PreTokenizerWrapper.Split.new(
        SplitPattern.Regex(regex.toString()),
        SplitDelimiterBehavior.Isolated,
        false,
    ).mapErr(Error.wrap).unwrap()
    return CandleResult.Ok(PreTokenizerWrapper.Sequence.new(listOf(split.into(), byteLevel.into())).into())
}

public fun pipelineFromPre(pre: String): CandleResult<Pipeline> {
    val regexQwen2 = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    val regexLlama3 = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

    return CandleResult.Ok(
        when (pre) {
            // Matches Qwen2 tokenizer.json settings
            "qwen2" -> Pipeline(
                normalizer = NFC.into(),
                pretokenizer = preTokenizerSequence(
                    regexQwen2,
                    ByteLevelPre.new(false, false, false),
                ).unwrap(),
                decoder = ByteLevelDecoder.new(false, false, false).into(),
                postProcessor = ByteLevelProcessor.new(false, false, false).into(),
            )
            // Matches Smaug/Llama3 style byte-level BPE
            "smaug-bpe", "lfm2", "llama3" -> Pipeline(
                normalizer = null,
                pretokenizer = preTokenizerSequence(
                    regexLlama3,
                    ByteLevelPre.new(false, true, false),
                ).unwrap(),
                decoder = ByteLevelDecoder.new(true, true, true).into(),
                postProcessor = ByteLevelProcessor.new(true, false, true).into(),
            )
            // Default GPT-2 style BPE
            else -> Pipeline(
                normalizer = null,
                pretokenizer = ByteLevelPre.default().into(),
                decoder = ByteLevelDecoder.default().into(),
                postProcessor = ByteLevelProcessor.default().into(),
            )
        },
    )
}

public fun templateProcessor(
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
    single.add("\$0".toString())
    if (addEos) {
        single.add(eos ?: return null)
    }

    val pair = ArrayList<String>()
    if (addBos) {
        pair.add(format("{}:0", bos ?: return null))
    }
    pair.add("\$A:0".toString())
    if (addEos) {
        pair.add(format("{}:0", eos ?: return null))
    }
    if (addBos) {
        pair.add(format("{}:1", bos ?: return null))
    }
    pair.add("\$B:1".toString())
    if (addEos) {
        pair.add(format("{}:1", eos ?: return null))
    }

    val proc = tokenizers.processors.template.TemplateProcessing.builder()
        .trySingle(single)
        .ok()
        ?.tryPair(pair)
        ?.ok()
        ?.specialTokens(specials)
        ?.build()
        ?.ok() ?: return null

    return proc
}

public class Tokenizer(
    public val model: BpeModel,
    public var normalizer: NormalizerWrapper? = null,
    public var preTokenizer: PreTokenizerWrapper? = null,
    public var decoder: DecoderWrapper? = null,
    public var postProcessor: PostProcessorWrapper? = null,
    public val specialTokens: MutableList<AddedToken> = ArrayList(),
) {
    public val withNormalizer: (NormalizerWrapper?) -> Unit = { n -> normalizer = n }
    public val withPreTokenizer: (PreTokenizerWrapper?) -> Unit = { pt -> preTokenizer = pt }
    public val withDecoder: (DecoderWrapper?) -> Unit = { dec -> decoder = dec }
    public val withPostProcessor: (PostProcessorWrapper?) -> Unit = { pp -> postProcessor = pp }
    public val addSpecialTokens: (List<AddedToken>) -> Unit = { sp -> specialTokens.addAll(sp) }

    public companion object Self : TokenizerFromGguf {
        public val new: (BpeModel) -> Tokenizer = { bpe -> Tokenizer(bpe) }

        override fun fromGguf(ct: Content): Result<Tokenizer> {
            val modelKind = metadataValue(ct, "tokenizer.ggml.model")
                .unwrap()
                .toString(true)
                .unwrap()
                .lowercase()
            if (modelKind != "gpt2") {
                return Result.failure(CandleError.Msg(format("unsupported tokenizer model `{model_kind}`", modelKind)))
            }

            val tokens = valueToStringArray(
                metadataValue(ct, "tokenizer.ggml.tokens").unwrap(),
                "tokenizer.ggml.tokens",
            ).unwrap()
            val vocab: Map<String, UInt> = tokens
                .withIndex()
                .map { indexed -> Pair(indexed.value, indexed.index.toUInt()) }
                .toMap()
            val merges = mergesFromValue(metadataValue(ct, "tokenizer.ggml.merges").unwrap()).unwrap()

            var builder = BPE.builder().vocabAndMerges(vocab, merges)

            val unkRes = metadataValue(ct, "tokenizer.ggml.unk_token_id")
            if (unkRes is CandleResult.Ok) {
                val tokenId = ggufValueToU32(unkRes.value).unwrap()
                val token = tokens.getOrNull(tokenId.toInt())
                if (token != null) {
                    builder = builder.unkToken(token)
                }
            }

            val byteFallbackRes = metadataValue(ct, "tokenizer.ggml.byte_fallback")
            if (byteFallbackRes is CandleResult.Ok) {
                builder = builder.byteFallback(byteFallbackRes.value.toBool().unwrap())
            }

            val ignoreMergesRes = metadataValue(ct, "tokenizer.ggml.ignore_merges")
            if (ignoreMergesRes is CandleResult.Ok) {
                builder = builder.ignoreMerges(ignoreMergesRes.value.toBool().unwrap())
            }

            val bpe = builder.build().mapErr(Error.wrap).unwrap()
            val tokenizer = Tokenizer.new(bpe)

            val pre = metadataValue(ct, "tokenizer.ggml.pre")
                .andThen { v -> v.toString(true) }
                .map { s -> s.toString() }
                .unwrapOrElse { "gpt2".toString() }
            val pipeline = pipelineFromPre(pre).unwrap()
            val postProcessorBase = pipeline.postProcessor

            val addBos = metadataValue(ct, "tokenizer.ggml.add_bos_token")
                .andThen { v -> v.toBool() }
                .unwrapOr(false)
            val addEos = metadataValue(ct, "tokenizer.ggml.add_eos_token")
                .andThen { v -> v.toBool() }
                .unwrapOr(false)
            val bosId = metadataValue(ct, "tokenizer.ggml.bos_token_id")
                .andThen { v -> ggufValueToU32(v) }
                .ok()
            val eosId = metadataValue(ct, "tokenizer.ggml.eos_token_id")
                .andThen { v -> ggufValueToU32(v) }
                .ok()

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
                    ProcessorSequence.new(steps).into()
                }
                tokenizer.withPostProcessor(pp)
            }

            // Mark special tokens so decode(skipSpecialTokens = true) behaves as expected
            val tokenTypeRes = metadataValue(ct, "tokenizer.ggml.token_type")
            if (tokenTypeRes is CandleResult.Ok && tokenTypeRes.value is Value.Array) {
                val arr = (tokenTypeRes.value as Value.Array).v
                val specials = ArrayList<AddedToken>()
                for (indexed in arr.withIndex()) {
                    val idx = indexed.index
                    val v = indexed.value
                    val ty = ggufValueToU32(v).unwrap()
                    // Aligns with llamaTokenType: treat non-normal/non-byte tokens as special.
                    val isSpecial = ty in 2u..5u
                    if (isSpecial) {
                        val tok = tokens.getOrNull(idx)
                        if (tok != null) {
                            specials.add(AddedToken.from(tok, true))
                        }
                    }
                }
                if (specials.isNotEmpty()) {
                    tokenizer.addSpecialTokens(specials)
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
                val valRes = metadataValue(ct, key)
                if (valRes is CandleResult.Ok) {
                    explicitSpecials.add(ggufValueToU32(valRes.value).unwrap())
                }
            }
            if (explicitSpecials.isNotEmpty()) {
                val specials = explicitSpecials
                    .mapNotNull { id -> tokens.getOrNull(id.toInt()) }
                    .map { tok -> AddedToken.from(tok, true) }
                if (specials.isNotEmpty()) {
                    tokenizer.addSpecialTokens(specials)
                }
            }

            return Result.success(tokenizer)
        }
    }
}
