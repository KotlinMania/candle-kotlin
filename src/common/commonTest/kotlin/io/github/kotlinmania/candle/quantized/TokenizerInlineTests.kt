// port-lint: tests quantized/tokenizer.rs
package io.github.kotlinmania.candle.quantized.tokenizer

import io.github.kotlinmania.gguffile.Value.*
import io.github.kotlinmania.tokenizers.tokenizer.OffsetReferential
import io.github.kotlinmania.tokenizers.tokenizer.OffsetType
import io.github.kotlinmania.tokenizers.tokenizer.PreTokenizedString

public fun splits(pre: String, input: String): List<String> {
    val pipeline = pipelineFromPre(pre).unwrap()
    val preTokenized = PreTokenizedString.from(input)
    pipeline.pretokenizer
        .unwrap()
        .preTokenize(preTokenized)
        .unwrap()
    return preTokenized
        .getSplits(OffsetReferential.Original, OffsetType.Byte)
        .intoIter()
        .map({ (s, _, _) -> s.toString() })
        .toMutableList()
}

// Sample made to test negative lookahead like (`\s+(?!\S)`).
// If someone changes the backend to one that does not support backtracking regex the tests below will fail.
public val SAMPLE: String = "Hello'Ve world 123\n\n  foo"

@kotlin.test.Test
public fun qwen2PretokenizerSplits() {
    check(splits("qwen2", SAMPLE) == listOf("Hello", "'Ve", "Ġworld", "Ġ", "1", "2", "3", "ĊĊ", "Ġ", "Ġfoo"))
}

@kotlin.test.Test
public fun llama3PretokenizerSplits() {
    // Differs from qwen2 only in digit grouping: `\p{N}{1,3}` vs `\p{N}`.
    check(splits("llama3", SAMPLE) == listOf("Hello", "'Ve", "Ġworld", "Ġ", "123", "ĊĊ", "Ġ", "Ġfoo"))
}
