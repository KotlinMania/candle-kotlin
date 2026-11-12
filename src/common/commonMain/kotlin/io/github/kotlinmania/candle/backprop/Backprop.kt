// port-lint: source backprop.rs
package io.github.kotlinmania.candle.backprop

import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.op.BinaryOp
import io.github.kotlinmania.candle.op.Op
import io.github.kotlinmania.candle.op.ReduceOp
import io.github.kotlinmania.candle.op.UnaryOp
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.candle.tensor.TensorId
import io.github.kotlinmania.rustlang.core.result.fromResidual
import io.github.kotlinmania.rustlang.std.collections.hashmap.Entry

// Methods for backpropagation of gradients.

// arg has been reduced to node via reduceDims, expand it back to arg.
// This has to handle keepdims.
public fun broadcastBack(arg: Tensor, node: Tensor, reducedDims: List<Int>): Result<Tensor> {
    return if (arg.rank() == node.rank()) {
        // keepdim = true
        node.broadcastAs(arg.shape())
    } else {
        // keepdim = false
        // first expand the reduced dims.
        when (val tryResult = node.reshape(reducedDims)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.broadcastAs(arg.shape())
    }
}

/**
 * Return all the nodes that lead to this value in a topologically sorted vec, the first
 * elements having dependencies on the latter ones, e.g. the first element if any is the
 * argument.
 * This assumes that the op graph is a DAG.
 */
public fun Tensor.sortedNodes(): List<Tensor> {
    // The vec of sorted nodes is passed as an owned value rather than a mutable reference
    // to get around some scope limitations.
    fun walk(node: Tensor, nodes: List<Tensor>, alreadySeen: Map<TensorId, Boolean>): Pair<Boolean, List<Tensor>> {
        when (val matchVal = alreadySeen.get(node.id())) {
            null -> {}
            else -> {
                val tg = matchVal
                return Pair(tg, nodes)
            }
        }
        var trackGrad = false
        var nodes56 =
            if (node.isVariable()) {
                // Do not call recursively on the "leaf" nodes.
                trackGrad = true
                nodes
            } else if (node.dtype().isInt()) {
                nodes
            } else {
                when (val matchVal1 = node.op()) {
                    null -> {
                        nodes
                    }
                    else -> {
                        val op = matchVal1
                        when {
                            op is Op.IndexAdd -> {
                                val (t1, t2, t3, _) = op
                                val (tg1, nodes1) = walk(t1, nodes, alreadySeen)
                                trackGrad = trackGrad or tg1
                                val (tg2, nodes2) = walk(t2, nodes1, alreadySeen)
                                trackGrad = trackGrad or tg2
                                val (tg3, nodes3) = walk(t3, nodes2, alreadySeen)
                                trackGrad = trackGrad or tg3
                                nodes3
                            }
                            op is Op.Scatter -> {
                                val (t11, t21, t31, _) = op
                                val (tg4, nodes4) = walk(t11, nodes, alreadySeen)
                                trackGrad = trackGrad or tg4
                                val (tg5, nodes5) = walk(t21, nodes4, alreadySeen)
                                trackGrad = trackGrad or tg5
                                val (tg6, nodes6) = walk(t31, nodes5, alreadySeen)
                                trackGrad = trackGrad or tg6
                                nodes6
                            }
                            op is Op.ScatterAdd -> {
                                val (t12, t22, t32, _) = op
                                val (tg7, nodes7) = walk(t12, nodes, alreadySeen)
                                trackGrad = trackGrad or tg7
                                val (tg8, nodes8) = walk(t22, nodes7, alreadySeen)
                                trackGrad = trackGrad or tg8
                                val (tg9, nodes9) = walk(t32, nodes8, alreadySeen)
                                trackGrad = trackGrad or tg9
                                nodes9
                            }
                            op is Op.CustomOp3 -> {
                                val (t13, t23, t33, _) = op
                                val (tg10, nodes10) = walk(t13, nodes, alreadySeen)
                                trackGrad = trackGrad or tg10
                                val (tg11, nodes11) = walk(t23, nodes10, alreadySeen)
                                trackGrad = trackGrad or tg11
                                val (tg12, nodes12) = walk(t33, nodes11, alreadySeen)
                                trackGrad = trackGrad or tg12
                                nodes12
                            }
                            op is Op.WhereCond -> {
                                val (t14, t24, t34) = op
                                val (tg13, nodes13) = walk(t14, nodes, alreadySeen)
                                trackGrad = trackGrad or tg13
                                val (tg14, nodes14) = walk(t24, nodes13, alreadySeen)
                                trackGrad = trackGrad or tg14
                                val (tg15, nodes15) = walk(t34, nodes14, alreadySeen)
                                trackGrad = trackGrad or tg15
                                nodes15
                            }
                            op is Op.Conv1D -> {
                                val lhs = op.arg
                                val rhs = op.kernel
                                val (tg16, nodes16) = walk(lhs, nodes, alreadySeen)
                                trackGrad = trackGrad or tg16
                                val (tg17, nodes17) = walk(rhs, nodes16, alreadySeen)
                                trackGrad = trackGrad or tg17
                                nodes17
                            }
                            op is Op.ConvTranspose1D -> {
                                val lhs1 = op.arg
                                val rhs1 = op.kernel
                                val (tg18, nodes18) = walk(lhs1, nodes, alreadySeen)
                                trackGrad = trackGrad or tg18
                                val (tg19, nodes19) = walk(rhs1, nodes18, alreadySeen)
                                trackGrad = trackGrad or tg19
                                nodes19
                            }
                            op is Op.Conv2D -> {
                                val lhs2 = op.arg
                                val rhs2 = op.kernel
                                val (tg20, nodes20) = walk(lhs2, nodes, alreadySeen)
                                trackGrad = trackGrad or tg20
                                val (tg21, nodes21) = walk(rhs2, nodes20, alreadySeen)
                                trackGrad = trackGrad or tg21
                                nodes21
                            }
                            op is Op.ConvTranspose2D -> {
                                val lhs3 = op.arg
                                val rhs3 = op.kernel
                                val (tg22, nodes22) = walk(lhs3, nodes, alreadySeen)
                                trackGrad = trackGrad or tg22
                                val (tg23, nodes23) = walk(rhs3, nodes22, alreadySeen)
                                trackGrad = trackGrad or tg23
                                nodes23
                            }
                            op is Op.CustomOp2 -> {
                                val (lhs4, rhs4, _) = op
                                val (tg24, nodes24) = walk(lhs4, nodes, alreadySeen)
                                trackGrad = trackGrad or tg24
                                val (tg25, nodes25) = walk(rhs4, nodes24, alreadySeen)
                                trackGrad = trackGrad or tg25
                                nodes25
                            }
                            op is Op.Binary -> {
                                val (lhs5, rhs5, _) = op
                                val (tg26, nodes26) = walk(lhs5, nodes, alreadySeen)
                                trackGrad = trackGrad or tg26
                                val (tg27, nodes27) = walk(rhs5, nodes26, alreadySeen)
                                trackGrad = trackGrad or tg27
                                nodes27
                            }
                            op is Op.Gather -> {
                                val (lhs6, rhs6, _) = op
                                val (tg28, nodes28) = walk(lhs6, nodes, alreadySeen)
                                trackGrad = trackGrad or tg28
                                val (tg29, nodes29) = walk(rhs6, nodes28, alreadySeen)
                                trackGrad = trackGrad or tg29
                                nodes29
                            }
                            op is Op.IndexSelect -> {
                                val (lhs7, rhs7, _) = op
                                val (tg30, nodes30) = walk(lhs7, nodes, alreadySeen)
                                trackGrad = trackGrad or tg30
                                val (tg31, nodes31) = walk(rhs7, nodes30, alreadySeen)
                                trackGrad = trackGrad or tg31
                                nodes31
                            }
                            op is Op.Matmul -> {
                                val (lhs8, rhs8) = op
                                val (tg32, nodes32) = walk(lhs8, nodes, alreadySeen)
                                trackGrad = trackGrad or tg32
                                val (tg33, nodes33) = walk(rhs8, nodes32, alreadySeen)
                                trackGrad = trackGrad or tg33
                                nodes33
                            }
                            op is Op.SliceScatter0 -> {
                                val (lhs9, rhs9, _) = op
                                val (tg34, nodes34) = walk(lhs9, nodes, alreadySeen)
                                trackGrad = trackGrad or tg34
                                val (tg35, nodes35) = walk(rhs9, nodes34, alreadySeen)
                                trackGrad = trackGrad or tg35
                                nodes35
                            }
                            op is Op.Cat -> {
                                val (args, _) = op
                                args.iter().fold(nodes, { nodes36, arg ->
                                    val (tg36, nodes37) = walk(arg, nodes36, alreadySeen)
                                    trackGrad = trackGrad or tg36
                                    nodes37
                                })
                            }
                            op is Op.Affine -> {
                                val arg = op.arg
                                val mul = op.mul
                                if (mul == 0.0) {
                                    nodes
                                } else {
                                    val (tg36, nodes36) = walk(arg, nodes, alreadySeen)
                                    trackGrad = trackGrad or tg36
                                    nodes36
                                }
                            }
                            op is Op.Unary && op.field1 == UnaryOp.Ceil -> nodes
                            op is Op.Unary && op.field1 == UnaryOp.Floor -> nodes
                            op is Op.Unary && op.field1 == UnaryOp.Round -> nodes
                            op is Op.Unary && op.field1 == UnaryOp.Sign -> nodes
                            op is Op.Reshape -> {
                                val (node1) = op
                                val (tg37, nodes37) = walk(node1, nodes, alreadySeen)
                                trackGrad = trackGrad or tg37
                                nodes37
                            }
                            op is Op.UpsampleNearest1D -> {
                                val node2 = op.arg
                                val (tg38, nodes38) = walk(node2, nodes, alreadySeen)
                                trackGrad = trackGrad or tg38
                                nodes38
                            }
                            op is Op.UpsampleNearest2D -> {
                                val node3 = op.arg
                                val (tg39, nodes39) = walk(node3, nodes, alreadySeen)
                                trackGrad = trackGrad or tg39
                                nodes39
                            }
                            op is Op.UpsampleBilinear2D -> {
                                val node4 = op.arg
                                val (tg40, nodes40) = walk(node4, nodes, alreadySeen)
                                trackGrad = trackGrad or tg40
                                nodes40
                            }
                            op is Op.AvgPool2D -> {
                                val node5 = op.arg
                                val (tg41, nodes41) = walk(node5, nodes, alreadySeen)
                                trackGrad = trackGrad or tg41
                                nodes41
                            }
                            op is Op.MaxPool2D -> {
                                val node6 = op.arg
                                val (tg42, nodes42) = walk(node6, nodes, alreadySeen)
                                trackGrad = trackGrad or tg42
                                nodes42
                            }
                            op is Op.Copy -> {
                                val (node7) = op
                                val (tg43, nodes43) = walk(node7, nodes, alreadySeen)
                                trackGrad = trackGrad or tg43
                                nodes43
                            }
                            op is Op.Broadcast -> {
                                val (node8) = op
                                val (tg44, nodes44) = walk(node8, nodes, alreadySeen)
                                trackGrad = trackGrad or tg44
                                nodes44
                            }
                            op is Op.Cmp -> {
                                val (node9, _) = op
                                val (tg45, nodes45) = walk(node9, nodes, alreadySeen)
                                trackGrad = trackGrad or tg45
                                nodes45
                            }
                            op is Op.Reduce && (op.field1 == ReduceOp.Min || op.field1 == ReduceOp.Sum || op.field1 == ReduceOp.Max) -> {
                                val (node10, _, _) = op as Op.Reduce
                                val (tg46, nodes46) = walk(node10, nodes, alreadySeen)
                                trackGrad = trackGrad or tg46
                                nodes46
                            }
                            op is Op.ToDevice -> {
                                val (node11) = op
                                val (tg47, nodes47) = walk(node11, nodes, alreadySeen)
                                trackGrad = trackGrad or tg47
                                nodes47
                            }
                            op is Op.Transpose -> {
                                val (node12, _, _) = op
                                val (tg48, nodes48) = walk(node12, nodes, alreadySeen)
                                trackGrad = trackGrad or tg48
                                nodes48
                            }
                            op is Op.Permute -> {
                                val (node13, _) = op
                                val (tg49, nodes49) = walk(node13, nodes, alreadySeen)
                                trackGrad = trackGrad or tg49
                                nodes49
                            }
                            op is Op.Narrow -> {
                                val (node14, _, _, _) = op
                                val (tg50, nodes50) = walk(node14, nodes, alreadySeen)
                                trackGrad = trackGrad or tg50
                                nodes50
                            }
                            op is Op.Unary -> {
                                val (node15, _) = op
                                val (tg51, nodes51) = walk(node15, nodes, alreadySeen)
                                trackGrad = trackGrad or tg51
                                nodes51
                            }
                            op is Op.Elu -> {
                                val (node16, _) = op
                                val (tg52, nodes52) = walk(node16, nodes, alreadySeen)
                                trackGrad = trackGrad or tg52
                                nodes52
                            }
                            op is Op.Powf -> {
                                val (node17, _) = op
                                val (tg53, nodes53) = walk(node17, nodes, alreadySeen)
                                trackGrad = trackGrad or tg53
                                nodes53
                            }
                            op is Op.CustomOp1 -> {
                                val (node18, _) = op
                                val (tg54, nodes54) = walk(node18, nodes, alreadySeen)
                                trackGrad = trackGrad or tg54
                                nodes54
                            }
                            op is Op.ToDType -> {
                                val (node19) = op
                                if (node19.dtype().isFloat()) {
                                    val (tg55, nodes55) = walk(node19, nodes, alreadySeen)
                                    trackGrad = trackGrad or tg55
                                    nodes55
                                } else {
                                    nodes
                                }
                            }
                            else -> nodes
                        }
                    }
                }
            }
        alreadySeen.insert(node.id(), trackGrad)
        if (trackGrad) {
            nodes56.push(node)
        }
        return Pair(trackGrad, nodes56)
    }
    var (tg, nodes) = walk(this, mutableListOf(), HashMap.new())
    nodes.reverse()
    return nodes
}

public fun Tensor.backward(): Result<GradStore> {
    val sortedNodes = this.sortedNodes()
    val grads = GradStore.new()
    grads.insert(
        this,
        when (
            val tryResult =
                when (val tryResult = this.onesLike()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }.contiguous()
        ) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        },
    )
    for (node in sortedNodes.iter()) {
        if (node.isVariable()) {
            continue
        }
        val grad = grads.remove(node).expect("candle internal error - grad not populated")
        // https://github.com/huggingface/candle/issues/1241
        // Ideally, we would make these operations in place where possible to ensure that we
        // do not have to allocate too often. Here we just call `.detach` to avoid computing
        // the backprop graph of the backprop itself. This would be an issue for second order
        // derivatives but these are out of scope at the moment.
        val doNotDetach = CANDLE_GRAD_DO_NOT_DETACH.with({ b -> b })
        val grad1 =
            if (doNotDetach) {
                grad
            } else {
                grad.detach()
            }
        when (val matchVal = node.op()) {
            null -> {}
            else -> {
                val op = matchVal
                when {
                    op is Op.Binary && op.field2 == BinaryOp.Add -> {
                        val (lhs, rhs, _) = op
                        val lhsSumGrad =
                            when (val tryResult = grads.orInsert(lhs)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        lhsSumGrad =
                            when (val tryResult = lhsSumGrad.add(grad1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val rhsSumGrad =
                            when (val tryResult = grads.orInsert(rhs)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        rhsSumGrad =
                            when (val tryResult = rhsSumGrad.add(grad1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Binary && op.field2 == BinaryOp.Sub -> {
                        val (lhs1, rhs1, _) = op
                        val lhsSumGrad1 =
                            when (val tryResult = grads.orInsert(lhs1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        lhsSumGrad1 =
                            when (val tryResult = lhsSumGrad1.add(grad1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val rhsSumGrad1 =
                            when (val tryResult = grads.orInsert(rhs1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        rhsSumGrad1 =
                            when (val tryResult = rhsSumGrad1.sub(grad1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Binary && op.field2 == BinaryOp.Mul -> {
                        val (lhs2, rhs2, _) = op
                        val lhsGrad =
                            when (val tryResult = grad1.mul(rhs2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val lhsSumGrad2 =
                            when (val tryResult = grads.orInsert(lhs2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        lhsSumGrad2 =
                            when (val tryResult = lhsSumGrad2.add(lhsGrad)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val rhsGrad =
                            when (val tryResult = grad1.mul(lhs2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val rhsSumGrad2 =
                            when (val tryResult = grads.orInsert(rhs2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        rhsSumGrad2 =
                            when (val tryResult = rhsSumGrad2.add(rhsGrad)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Binary && op.field2 == BinaryOp.Div -> {
                        val (lhs3, rhs3, _) = op
                        val lhsGrad1 =
                            when (val tryResult = grad1.div(rhs3)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val lhsSumGrad3 =
                            when (val tryResult = grads.orInsert(lhs3)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        lhsSumGrad3 =
                            when (val tryResult = lhsSumGrad3.add(lhsGrad1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val rhsGrad1 =
                            when (
                                val tryResult =
                                    when (val tryResult = grad1.mul(lhs3)) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.div(
                                        when (val tryResult = rhs3.sqr()) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val rhsSumGrad3 =
                            when (val tryResult = grads.orInsert(rhs3)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        rhsSumGrad3 =
                            when (val tryResult = rhsSumGrad3.sub(rhsGrad1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Binary && op.field2 == BinaryOp.Minimum -> {
                        val (lhs4, rhs4, _) = op
                        val maskLhs =
                            when (
                                val tryResult =
                                    when (val tryResult = node.eq(lhs4)) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.toDtype(grad1.dtype())
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val maskRhs =
                            when (
                                val tryResult =
                                    when (val tryResult = node.eq(rhs4)) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.toDtype(grad1.dtype())
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        // If both masks are 1 one the same point, we want to scale the
                        // gradient by 0.5 rather than 1.
                        val lhsGrad2 =
                            when (
                                val tryResult =
                                    when (val tryResult = maskLhs.mul(grad1)) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.div(
                                        when (val tryResult = (maskRhs + 1.0)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val lhsSumGrad4 =
                            when (val tryResult = grads.orInsert(lhs4)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        lhsSumGrad4 =
                            when (val tryResult = lhsSumGrad4.add(lhsGrad2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val rhsGrad2 =
                            when (
                                val tryResult =
                                    when (val tryResult = maskRhs.mul(grad1)) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.div(
                                        when (val tryResult = (maskLhs + 1.0)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val rhsSumGrad4 =
                            when (val tryResult = grads.orInsert(rhs4)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        rhsSumGrad4 =
                            when (val tryResult = rhsSumGrad4.add(rhsGrad2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Binary && op.field2 == BinaryOp.Maximum -> {
                        val (lhs5, rhs5, _) = op
                        val maskLhs1 =
                            when (
                                val tryResult =
                                    when (val tryResult = node.eq(lhs5)) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.toDtype(grad1.dtype())
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val maskRhs1 =
                            when (
                                val tryResult =
                                    when (val tryResult = node.eq(rhs5)) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.toDtype(grad1.dtype())
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val lhsGrad3 =
                            when (
                                val tryResult =
                                    when (val tryResult = maskLhs1.mul(grad1)) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.div(
                                        when (val tryResult = (maskRhs1 + 1.0)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val lhsSumGrad5 =
                            when (val tryResult = grads.orInsert(lhs5)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        lhsSumGrad5 =
                            when (val tryResult = lhsSumGrad5.add(lhsGrad3)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val rhsGrad3 =
                            when (
                                val tryResult =
                                    when (val tryResult = maskRhs1.mul(grad1)) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.div(
                                        when (val tryResult = (maskLhs1 + 1.0)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val rhsSumGrad5 =
                            when (val tryResult = grads.orInsert(rhs5)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        rhsSumGrad5 =
                            when (val tryResult = rhsSumGrad5.add(rhsGrad3)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.WhereCond -> {
                        val (pred, t, f) = op
                        val zeros =
                            when (val tryResult = grad1.zerosLike()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val tSumGrad =
                            when (val tryResult = grads.orInsert(t)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val tGrad =
                            when (val tryResult = pred.whereCond(grad1, zeros)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        tSumGrad =
                            when (val tryResult = tSumGrad.add(tGrad)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val fSumGrad =
                            when (val tryResult = grads.orInsert(f)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val fGrad =
                            when (val tryResult = pred.whereCond(zeros, grad1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        fSumGrad =
                            when (val tryResult = fSumGrad.add(fGrad)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Conv1D -> {
                        val arg = op.arg
                        val kernel = op.kernel
                        val padding = op.padding
                        val stride = op.stride
                        val dilation = op.dilation
                        // The output height for convTranspose1d is:
                        // (lIn - 1) * stride - 2 * padding + dilation * (kSize - 1) + outPadding + 1
                        val gradLIn =
                            when (val tryResult = grad1.dim(2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val kSize =
                            when (val tryResult = kernel.dim(2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val outSize = (gradLIn - 1) * stride + dilation * (kSize - 1) + 1 - 2 * padding
                        val outPadding =
                            when (val tryResult = arg.dim(2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            } - outSize
                        val gradArg =
                            when (val tryResult = grad1.convTranspose1d(kernel, padding, outPadding, stride, dilation, 1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad =
                            when (val tryResult = grads.orInsert(arg)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad =
                            when (val tryResult = sumGrad.add(gradArg)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val gradKernel =
                            when (
                                val tryResult =
                                    when (
                                        val tryResult =
                                            when (val tryResult = arg.transpose(0, 1)) {
                                                is Result.Ok -> tryResult.value
                                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                            }.conv1d(
                                                when (val tryResult = grad1.transpose(0, 1)) {
                                                    is Result.Ok -> tryResult.value
                                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                                },
                                                padding,
                                                dilation,
                                                stride,
                                                1,
                                            )
                                    ) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.transpose(0, 1)
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad1 =
                            when (val tryResult = grads.orInsert(kernel)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val (_, _, k0) =
                            when (val tryResult = kernel.dims3()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val (_, _, gK0) =
                            when (val tryResult = gradKernel.dims3()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val gradKernel1 =
                            if (gK0 != k0) {
                                when (val tryResult = gradKernel.narrow(2, 0, k0)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                            } else {
                                gradKernel
                            }
                        sumGrad1 =
                            when (val tryResult = sumGrad1.add(gradKernel1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Conv2D -> {
                        val arg1 = op.arg
                        val kernel1 = op.kernel
                        val padding1 = op.padding
                        val stride1 = op.stride
                        val dilation1 = op.dilation
                        // The output height for convTranspose2d is:
                        // (iH - 1) * stride - 2 * padding + dilation * (kH - 1) + outPadding + 1
                        val gradH =
                            when (val tryResult = grad1.dim(2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val kH =
                            when (val tryResult = kernel1.dim(2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val outSize1 = (gradH - 1) * stride1 + dilation1 * (kH - 1) + 1 - 2 * padding1
                        val outPadding1 =
                            when (val tryResult = arg1.dim(2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            } - outSize1
                        val gradArg1 =
                            when (val tryResult = grad1.convTranspose2d(kernel1, padding1, outPadding1, stride1, dilation1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad2 =
                            when (val tryResult = grads.orInsert(arg1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad2 =
                            when (val tryResult = sumGrad2.add(gradArg1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val gradKernel2 =
                            when (
                                val tryResult =
                                    when (
                                        val tryResult =
                                            when (val tryResult = arg1.transpose(0, 1)) {
                                                is Result.Ok -> tryResult.value
                                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                            }.conv2d(
                                                when (val tryResult = grad1.transpose(0, 1)) {
                                                    is Result.Ok -> tryResult.value
                                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                                },
                                                padding1,
                                                dilation1,
                                                stride1,
                                                1,
                                            )
                                    ) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.transpose(0, 1)
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad3 =
                            when (val tryResult = grads.orInsert(kernel1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val (_, _, k01, k1) =
                            when (val tryResult = kernel1.dims4()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val (_, _, gK01, gK1) =
                            when (val tryResult = gradKernel2.dims4()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val gradKernel3 =
                            if (gK01 != k01 || gK1 != k1) {
                                when (
                                    val tryResult =
                                        when (val tryResult = gradKernel2.narrow(2, 0, k01)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        }.narrow(3, 0, k1)
                                ) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                            } else {
                                gradKernel2
                            }
                        sumGrad3 =
                            when (val tryResult = sumGrad3.add(gradKernel3)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.ConvTranspose1D ->
                        when (val tryResult = Err(Error.BackwardNotSupported(op = "conv-transpose1d"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    op is Op.ConvTranspose2D -> {
                        val arg2 = op.arg
                        val kernel2 = op.kernel
                        val padding2 = op.padding
                        val stride2 = op.stride
                        val dilation2 = op.dilation
                        val gradArg2 =
                            when (val tryResult = grad1.conv2d(kernel2, padding2, stride2, dilation2, 1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad4 =
                            when (val tryResult = grads.orInsert(arg2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad4 =
                            when (val tryResult = sumGrad4.add(gradArg2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val gradKernel4 =
                            when (
                                val tryResult =
                                    when (
                                        val tryResult =
                                            when (val tryResult = grad1.transpose(0, 1)) {
                                                is Result.Ok -> tryResult.value
                                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                            }.conv2d(
                                                when (val tryResult = arg2.transpose(0, 1)) {
                                                    is Result.Ok -> tryResult.value
                                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                                },
                                                padding2,
                                                dilation2,
                                                stride2,
                                                1,
                                            )
                                    ) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.transpose(0, 1)
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad5 =
                            when (val tryResult = grads.orInsert(kernel2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val (_, _, k02, k11) =
                            when (val tryResult = kernel2.dims4()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val (_, _, gK02, gK11) =
                            when (val tryResult = gradKernel4.dims4()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val gradKernel5 =
                            if (gK02 != k02 || gK11 != k11) {
                                when (
                                    val tryResult =
                                        when (val tryResult = gradKernel4.narrow(2, 0, k02)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        }.narrow(3, 0, k11)
                                ) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                            } else {
                                gradKernel4
                            }
                        sumGrad5 =
                            when (val tryResult = sumGrad5.add(gradKernel5)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.AvgPool2D -> {
                        val arg3 = op.arg
                        val kernelSize = op.kernelSize
                        val stride3 = op.stride
                        if (kernelSize != stride3) {
                            bail("backward not supported for avgpool2d if ksize {kernel_size:?} != stride {stride:?}")
                        }
                        val (n, c, h, w) =
                            when (val tryResult = arg3.dims4()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val gradArg3 =
                            when (val tryResult = grad1.upsampleNearest2d(h, w)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val gradArg4 =
                            when (val tryResult = (gradArg3 * (1.0 / (kernelSize.first * kernelSize.second).toDouble()))) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad6 =
                            when (val tryResult = grads.orInsert(arg3)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad6 =
                            when (val tryResult = sumGrad6.add(gradArg4)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.MaxPool2D -> {
                        val arg4 = op.arg
                        val kernelSize1 = op.kernelSize
                        val stride4 = op.stride
                        if (kernelSize1 != stride4) {
                            bail("backward not supported for maxpool2d if ksize {kernel_size:?} != stride {stride:?}")
                        }
                        val (n1, c1, h1, w1) =
                            when (val tryResult = arg4.dims4()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        // For computing the max-pool gradient, we compute a mask where a 1 means
                        // that the element is the maximum, then we apply this mask to the
                        // upsampled gradient (taking into account that multiple max may exist so
                        // we scale the gradient for this case).
                        val nodeUpsampled =
                            when (val tryResult = node.upsampleNearest2d(h1, w1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val mask =
                            when (
                                val tryResult =
                                    when (val tryResult = arg4.eq(nodeUpsampled)) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.toDtype(arg4.dtype())
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val avg =
                            when (val tryResult = mask.avgPool2dWithStride(kernelSize1, stride4)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val gradArg5 =
                            when (
                                val tryResult = (
                                    when (
                                        val tryResult =
                                            when (val tryResult = (grad1 * avg)) {
                                                is Result.Ok -> tryResult.value
                                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                            }.upsampleNearest2d(h1, w1)
                                    ) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    } * mask
                                )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad7 =
                            when (val tryResult = grads.orInsert(arg4)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad7 =
                            when (val tryResult = sumGrad7.add(gradArg5)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.UpsampleNearest1D -> {
                        val arg5 = op.arg
                        val targetSize = op.targetSize
                        val (n2, c, size) =
                            when (val tryResult = arg5.dims3()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        if (targetSize % size != 0) {
                            bail("backward not supported for non integer upscaling factors")
                        }
                        val scale = targetSize / size
                        val kernel3 =
                            when (val tryResult = Tensor.ones(Triple(c, 1, scale), arg5.dtype(), arg5.device())) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val convSum =
                            when (val tryResult = grad1.conv1d(kernel3, 0, scale, 1, c)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad8 =
                            when (val tryResult = grads.orInsert(arg5)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad8 = convSum
                    }
                    op is Op.UpsampleNearest2D -> {
                        val arg6 = op.arg
                        val targetH = op.targetH
                        val targetW = op.targetW
                        val (n3, c1, h2, w2) =
                            when (val tryResult = arg6.dims4()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        if (targetH % h2 != 0 || targetW % w2 != 0) {
                            bail("backward not supported for non integer upscaling factors")
                        }
                        val scaleH = targetH / h2
                        val scaleW = targetW / w2
                        if (scaleH != scaleW) {
                            bail("backward not supported for non uniform upscaling factors")
                        }
                        val kernel4 =
                            when (val tryResult = Tensor.ones(listOf(c1, 1, scaleH, scaleW), arg6.dtype(), arg6.device())) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val convSum1 =
                            when (val tryResult = grad1.conv2d(kernel4, 0, scaleH, 1, c1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad9 =
                            when (val tryResult = grads.orInsert(arg6)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad9 = convSum1
                    }
                    op is Op.UpsampleBilinear2D -> {
                        bail("backward not supported for upsample_bilinear2d")
                    }
                    op is Op.SliceScatter0 -> {
                        val (lhs6, rhs6, startRhs) = op
                        val rhsSumGrad6 =
                            when (val tryResult = grads.orInsert(rhs6)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val rhsGrad4 =
                            when (
                                val tryResult =
                                    grad1.narrow(
                                        0,
                                        startRhs,
                                        when (val tryResult = rhs6.dim(0)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        rhsSumGrad6 =
                            when (val tryResult = rhsSumGrad6.add(rhsGrad4)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val lhsSumGrad6 =
                            when (val tryResult = grads.orInsert(lhs6)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val lhsGrad4 =
                            when (
                                val tryResult =
                                    grad1.sliceScatter0(
                                        when (val tryResult = rhs6.zerosLike()) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                        startRhs,
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        lhsSumGrad6 =
                            when (val tryResult = lhsSumGrad6.add(lhsGrad4)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Gather -> {
                        val (arg7, indexes, dim) = op
                        val sumGrad10 =
                            when (val tryResult = grads.orInsert(arg7)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad10 =
                            when (val tryResult = sumGrad10.scatterAdd(indexes, grad1, dim)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Scatter -> {
                        val (init, indexes1, src, dim1) = op
                        val initSumGrad =
                            when (val tryResult = grads.orInsert(init)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        initSumGrad =
                            when (val tryResult = initSumGrad.add(grad1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val srcGrad =
                            when (val tryResult = grad1.gather(indexes1, dim1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val srcSumGrad =
                            when (val tryResult = grads.orInsert(src)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        srcSumGrad =
                            when (val tryResult = srcSumGrad.add(srcGrad)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.ScatterAdd -> {
                        val (init1, indexes2, src1, dim2) = op
                        val initSumGrad1 =
                            when (val tryResult = grads.orInsert(init1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val mask1 =
                            when (val tryResult = init1.onesLike()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val mask2 =
                            when (
                                val tryResult =
                                    mask1.scatter(
                                        indexes2,
                                        when (val tryResult = mask1.zerosLike()) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                        dim2,
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        initSumGrad1 =
                            when (
                                val tryResult =
                                    initSumGrad1.add(
                                        when (val tryResult = grad1.mul(mask2)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val srcGrad1 =
                            when (val tryResult = grad1.gather(indexes2, dim2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val srcSumGrad1 =
                            when (val tryResult = grads.orInsert(src1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        srcSumGrad1 =
                            when (val tryResult = srcSumGrad1.add(srcGrad1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.IndexAdd -> {
                        val (init2, indexes3, src2, dim3) = op
                        val initSumGrad2 =
                            when (val tryResult = grads.orInsert(init2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        initSumGrad2 =
                            when (val tryResult = initSumGrad2.add(grad1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val srcGrad2 =
                            when (val tryResult = grad1.indexSelect(indexes3, dim3)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val srcSumGrad2 =
                            when (val tryResult = grads.orInsert(src2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        srcSumGrad2 =
                            when (val tryResult = srcSumGrad2.add(srcGrad2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.IndexSelect -> {
                        val (arg8, indexes4, dim4) = op
                        val sumGrad11 =
                            when (val tryResult = grads.orInsert(arg8)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        // indexSelect accepts strided ids but indexAdd requires contiguous ones.
                        val indexes5 =
                            when (val tryResult = indexes4.contiguous()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad11 =
                            when (val tryResult = sumGrad11.indexAdd(indexes5, grad1, dim4)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Matmul -> {
                        val (lhs7, rhs7) = op
                        // Skipping checks, the op went ok, we can skip
                        // the matmul size checks for now.
                        val lhsGrad5 =
                            when (
                                val tryResult =
                                    grad1.matmul(
                                        when (val tryResult = rhs7.t()) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val lhsSumGrad7 =
                            when (val tryResult = grads.orInsert(lhs7)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        lhsSumGrad7 =
                            when (val tryResult = lhsSumGrad7.add(lhsGrad5)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val rhsGrad5 =
                            when (
                                val tryResult =
                                    when (val tryResult = lhs7.t()) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.matmul(grad1)
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val rhsSumGrad7 =
                            when (val tryResult = grads.orInsert(rhs7)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        rhsSumGrad7 =
                            when (val tryResult = rhsSumGrad7.add(rhsGrad5)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Cat -> {
                        val (args, dim5) = op
                        var startIdx = 0
                        for (arg9 in args) {
                            val len = arg9.dims()[dim5]
                            val argGrad =
                                when (val tryResult = grad1.narrow(dim5, startIdx, len)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                            val sumGrad12 =
                                when (val tryResult = grads.orInsert(arg9)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                            sumGrad12 =
                                when (val tryResult = sumGrad12.add(argGrad)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                            startIdx += len
                        }
                    }
                    op is Op.Broadcast -> {
                        val (arg10) = op
                        val argDims = arg10.dims()
                        val nodeDims = node.dims()
                        // The number of dims that have been inserted on the left.
                        val leftDims = nodeDims.len() - argDims.len()
                        var sumDims: List<Int> = (0 until leftDims).collect()
                        for ((dim6, nodeDim, argDim) in (((nodeDims).slice(leftDims until nodeDims.size)).zip(argDims.iter())).enumerate()) {
                            if (nodeDim != argDim) {
                                sumDims.push(dim6 + leftDims)
                            }
                        }
                        var argGrad1 =
                            when (val tryResult = grad1.sumKeepdim(sumDims.asSlice())) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        for (i in 0 until leftDims) {
                            argGrad1 =
                                when (val tryResult = argGrad1.squeeze(0)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                        }
                        val sumGrad13 =
                            when (val tryResult = grads.orInsert(arg10)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad13 =
                            when (
                                val tryResult =
                                    sumGrad13.add(
                                        when (val tryResult = argGrad1.broadcastAs(sumGrad13.dims())) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Reduce && op.field1 == ReduceOp.Sum -> {
                        val (arg11, _, reducedDims) = op
                        val grad2 =
                            when (val tryResult = broadcastBack(arg11, grad1, reducedDims)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad14 =
                            when (val tryResult = grads.orInsert(arg11)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad14 =
                            when (val tryResult = sumGrad14.add(grad2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Reduce && op.field1 == ReduceOp.Max -> {
                        val (arg12, _, reducedDims1) = op
                        val node1 =
                            when (val tryResult = broadcastBack(arg12, node, reducedDims1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val grad3 =
                            when (val tryResult = broadcastBack(arg12, grad1, reducedDims1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val grad4 =
                            when (
                                val tryResult =
                                    when (
                                        val tryResult =
                                            when (val tryResult = node1.eq(arg12)) {
                                                is Result.Ok -> tryResult.value
                                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                            }.toDtype(grad3.dtype())
                                    ) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.mul(grad3)
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad15 =
                            when (val tryResult = grads.orInsert(arg12)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad15 =
                            when (
                                val tryResult =
                                    sumGrad15.add(
                                        when (val tryResult = grad4.broadcastAs(sumGrad15.dims())) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Reduce && op.field1 == ReduceOp.Min -> {
                        val (arg13, _, reducedDims2) = op
                        val node2 =
                            when (val tryResult = broadcastBack(arg13, node, reducedDims2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val grad5 =
                            when (val tryResult = broadcastBack(arg13, grad1, reducedDims2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val grad6 =
                            when (
                                val tryResult =
                                    when (
                                        val tryResult =
                                            when (val tryResult = node2.eq(arg13)) {
                                                is Result.Ok -> tryResult.value
                                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                            }.toDtype(grad5.dtype())
                                    ) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.mul(grad5)
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad16 =
                            when (val tryResult = grads.orInsert(arg13)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad16 =
                            when (
                                val tryResult =
                                    sumGrad16.add(
                                        when (val tryResult = grad6.broadcastAs(sumGrad16.dims())) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.ToDType -> {
                        val (arg14) = op
                        val sumGrad17 =
                            when (val tryResult = grads.orInsert(arg14)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad17 =
                            when (
                                val tryResult =
                                    sumGrad17.add(
                                        when (val tryResult = grad1.toDtype(arg14.dtype())) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Copy -> {
                        val (arg15) = op
                        val sumGrad18 =
                            when (val tryResult = grads.orInsert(arg15)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad18 =
                            when (val tryResult = sumGrad18.add(grad1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Affine -> {
                        val arg16 = op.arg
                        val mul = op.mul
                        val argGrad2 =
                            when (val tryResult = grad1.affine(mul, 0.0)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad19 =
                            when (val tryResult = grads.orInsert(arg16)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad19 =
                            when (val tryResult = sumGrad19.add(argGrad2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Log -> {
                        val (arg17, _) = op
                        val sumGrad20 =
                            when (val tryResult = grads.orInsert(arg17)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad20 =
                            when (
                                val tryResult =
                                    sumGrad20.add(
                                        when (val tryResult = (grad1 / arg17)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Sin -> {
                        val (arg18, _) = op
                        val sumGrad21 =
                            when (val tryResult = grads.orInsert(arg18)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad21 =
                            when (
                                val tryResult =
                                    sumGrad21.add(
                                        when (val tryResult = (grad1 * arg18.cos())) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Cos -> {
                        val (arg19, _) = op
                        val sumGrad22 =
                            when (val tryResult = grads.orInsert(arg19)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad22 =
                            when (
                                val tryResult =
                                    sumGrad22.sub(
                                        when (val tryResult = (grad1 * arg19.sin())) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Tanh -> {
                        val (arg20, _) = op
                        val sumGrad23 =
                            when (val tryResult = grads.orInsert(arg20)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val minusDtanh =
                            when (
                                val tryResult = (
                                    when (val tryResult = node.sqr()) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    } - 1.0
                                )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad23 =
                            when (
                                val tryResult =
                                    sumGrad23.sub(
                                        when (val tryResult = (grad1 * minusDtanh)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Abs -> {
                        val (arg21, _) = op
                        val sumGrad24 =
                            when (val tryResult = grads.orInsert(arg21)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val ones =
                            when (val tryResult = arg21.onesLike()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val absGrad =
                            when (
                                val tryResult =
                                    arg21.ge(
                                        when (val tryResult = arg21.zerosLike()) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.whereCond(
                                ones,
                                when (val tryResult = ones.neg()) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                },
                            )
                        sumGrad24 =
                            when (
                                val tryResult =
                                    sumGrad24.add(
                                        when (val tryResult = (grad1 * absGrad)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Exp -> {
                        val (arg22, _) = op
                        val sumGrad25 =
                            when (val tryResult = grads.orInsert(arg22)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad25 =
                            when (
                                val tryResult =
                                    sumGrad25.add(
                                        when (val tryResult = (grad1 * node)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Neg -> {
                        val (arg23, _) = op
                        val sumGrad26 =
                            when (val tryResult = grads.orInsert(arg23)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad26 =
                            when (val tryResult = sumGrad26.sub(grad1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Recip -> {
                        val (arg24, _) = op
                        val sumGrad27 =
                            when (val tryResult = grads.orInsert(arg24)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val grad7 =
                            when (
                                val tryResult = (
                                    grad1 /
                                        when (val tryResult = arg24.sqr()) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        }
                                )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad27 =
                            when (val tryResult = sumGrad27.sub(grad7)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Narrow -> {
                        val (arg25, dim7, startIdx1, len1) = op
                        val argDims1 = arg25.dims()
                        val leftPad =
                            if (startIdx1 == 0) {
                                null
                            } else {
                                val dims = argDims1.toVec()
                                dims[dim7] = startIdx1
                                when (val tryResult = Tensor.zeros(dims, grad1.dtype(), grad1.device())) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                            }
                        val rightPad = argDims1[dim7] - startIdx1 - len1
                        val rightPad1 =
                            if (rightPad == 0) {
                                null
                            } else {
                                val dims1 = argDims1.toVec()
                                dims1[dim7] = rightPad
                                when (val tryResult = Tensor.zeros(dims1, grad1.dtype(), grad1.device())) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                            }
                        val argGrad3 =
                            run {
                                val matchVal1 = Pair(leftPad, rightPad1)
                                when {
                                    matchVal1.first == null && matchVal1.second == null -> grad1
                                    matchVal1.first != null && matchVal1.second == null -> {
                                        val l = matchVal1.first
                                        when (val tryResult = Tensor.cat(listOf(l, grad1), dim7)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        }
                                    }
                                    matchVal1.first == null && matchVal1.second != null -> {
                                        val r = matchVal1.second
                                        when (val tryResult = Tensor.cat(listOf(grad1, r), dim7)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        }
                                    }
                                    else -> {
                                        val l1 = matchVal1.first
                                        val r1 = matchVal1.second
                                        when (val tryResult = Tensor.cat(listOf(l1, grad1, r1), dim7)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        }
                                    }
                                }
                            }
                        val sumGrad28 =
                            when (val tryResult = grads.orInsert(arg25)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad28 =
                            when (val tryResult = sumGrad28.add(argGrad3)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Floor -> {}
                    op is Op.Unary && op.field1 == UnaryOp.Round -> {}
                    op is Op.Reduce && op.field1 == ReduceOp.ArgMin -> {}
                    op is Op.Reduce && op.field1 == ReduceOp.ArgMax -> {}
                    op is Op.Unary && op.field1 == UnaryOp.Sign -> {}
                    op is Op.Cmp -> {}
                    op is Op.Reshape -> {
                        val (arg26) = op
                        val argGrad4 =
                            when (val tryResult = grad1.reshape(arg26.dims())) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad29 =
                            when (val tryResult = grads.orInsert(arg26)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad29 =
                            when (val tryResult = sumGrad29.add(argGrad4)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Ceil ->
                        when (val tryResult = Err(Error.BackwardNotSupported(op = "ceil"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    op is Op.Unary && op.field1 == UnaryOp.Gelu -> {
                        val (arg27, _) = op
                        val sumGrad30 =
                            when (val tryResult = grads.orInsert(arg27)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val cube =
                            when (val tryResult = arg27.powf(3.0)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val tanh =
                            when (
                                val tryResult =
                                    when (
                                        val tryResult = (
                                            0.0356774 * cube +
                                                when (val tryResult = (0.797885 * arg27)) {
                                                    is Result.Ok -> tryResult.value
                                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                                }
                                        )
                                    ) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.tanh()
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val geluGrad =
                            when (
                                val tryResult = (
                                    when (
                                        val tryResult = (
                                            when (val tryResult = (0.5 * tanh)) {
                                                is Result.Ok -> tryResult.value
                                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                            } + when (
                                                val tryResult = (
                                                    0.0535161 * cube +
                                                        when (val tryResult = (0.398942 * arg27)) {
                                                            is Result.Ok -> tryResult.value
                                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                                        }
                                                )
                                            ) {
                                                is Result.Ok -> tryResult.value
                                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                            } * (
                                                1.0 -
                                                    when (val tryResult = tanh.powf(2.0)) {
                                                        is Result.Ok -> tryResult.value
                                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                                    }
                                            )
                                        )
                                    ) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    } + 0.5
                                )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad30 =
                            when (
                                val tryResult =
                                    sumGrad30.add(
                                        when (val tryResult = (grad1 * geluGrad)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Erf -> {
                        val (arg28, _) = op
                        val sumGrad31 =
                            when (val tryResult = grads.orInsert(arg28)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        // d/dx erf(x) = 2/sqrt(pi) * e^(-x^2)
                        val erfGrad =
                            (2.0 / Double.PI.sqrt()) *
                                when (
                                    val tryResult =
                                        (
                                            when (
                                                val tryResult =
                                                    when (val tryResult = arg28.sqr()) {
                                                        is Result.Ok -> tryResult.value
                                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                                    }.neg()
                                            ) {
                                                is Result.Ok -> tryResult.value
                                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                            }
                                        ).exp()
                                ) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                        sumGrad31 =
                            when (
                                val tryResult =
                                    sumGrad31.add(
                                        when (val tryResult = (grad1 * erfGrad)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.GeluErf -> {
                        val (arg29, _) = op
                        val sumGrad32 =
                            when (val tryResult = grads.orInsert(arg29)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        // d/dx geluErf(x) = 0.5 + 0.398942 e^(-x^2/2) x + 0.5 erf(x/sqrt(2))
                        val negHalfSquare =
                            when (
                                val tryResult = (
                                    when (
                                        val tryResult =
                                            when (val tryResult = arg29.sqr()) {
                                                is Result.Ok -> tryResult.value
                                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                            }.neg()
                                    ) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    } / 2.0
                                )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val scaledExpArg =
                            when (
                                val tryResult = (
                                    0.398942 *
                                        when (val tryResult = negHalfSquare.exp()) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        } * arg29
                                )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val argScaledSqrt =
                            when (val tryResult = (arg29 / 2.0.sqrt())) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val erfScaledSqrt =
                            when (
                                val tryResult = (
                                    0.5 *
                                        when (val tryResult = argScaledSqrt.erf()) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        }
                                )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val geluErfGrad =
                            when (val tryResult = (0.5 + scaledExpArg + erfScaledSqrt)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad32 =
                            when (
                                val tryResult =
                                    sumGrad32.add(
                                        when (val tryResult = (grad1 * geluErfGrad)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Relu -> {
                        val (arg30, _) = op
                        val sumGrad33 =
                            when (val tryResult = grads.orInsert(arg30)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val reluGrad =
                            when (
                                val tryResult =
                                    when (
                                        val tryResult =
                                            arg30.ge(
                                                when (val tryResult = arg30.zerosLike()) {
                                                    is Result.Ok -> tryResult.value
                                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                                },
                                            )
                                    ) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.toDtype(arg30.dtype())
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad33 =
                            when (
                                val tryResult =
                                    sumGrad33.add(
                                        when (val tryResult = (grad1 * reluGrad)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Silu -> {
                        val (arg31, _) = op
                        val sumGrad34 =
                            when (val tryResult = grads.orInsert(arg31)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        // d/dx silu = sigmoid(x) * (1 + x * (1 - sigmoid(x))) = sigmoid(x) * (1 - node) + node
                        val sigmoidArg =
                            when (
                                val tryResult =
                                    when (
                                        val tryResult = (
                                            when (
                                                val tryResult =
                                                    when (val tryResult = arg31.neg()) {
                                                        is Result.Ok -> tryResult.value
                                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                                    }.exp()
                                            ) {
                                                is Result.Ok -> tryResult.value
                                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                            } + 1.0
                                        )
                                    ) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.recip()
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val siluGrad = sigmoidArg * (1.0 - node) + node
                        sumGrad34 =
                            when (
                                val tryResult =
                                    sumGrad34.add(
                                        when (val tryResult = (grad1 * siluGrad)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Elu -> {
                        val (arg32, alpha) = op
                        // d/dx elu(x) = 1 for x > 0, alpha * e^x for x <= 0
                        val sumGrad35 =
                            when (val tryResult = grads.orInsert(arg32)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val zeros1 =
                            when (val tryResult = arg32.zerosLike()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val positiveMask =
                            when (
                                val tryResult =
                                    when (val tryResult = arg32.gt(zeros1)) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.toDtype(arg32.dtype())
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val negativeMask =
                            when (
                                val tryResult =
                                    when (val tryResult = arg32.le(zeros1)) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.toDtype(arg32.dtype())
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        // node == alpha * (e^x - 1) for x <= 0, reuse it
                        val negativeExpMask =
                            when (val tryResult = (negativeMask * (node + alpha))) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val combinedMask =
                            when (val tryResult = (positiveMask + negativeExpMask)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad35 =
                            when (
                                val tryResult =
                                    sumGrad35.add(
                                        when (val tryResult = (grad1 * combinedMask)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        },
                                    )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Powf -> {
                        val (arg33, e) = op
                        val argGrad5 =
                            when (
                                val tryResult = (
                                    when (
                                        val tryResult = (
                                            grad1 *
                                                when (val tryResult = arg33.powf(e - 1.0)) {
                                                    is Result.Ok -> tryResult.value
                                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                                }
                                        )
                                    ) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    } * e
                                )
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad36 =
                            when (val tryResult = grads.orInsert(arg33)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad36 =
                            when (val tryResult = sumGrad36.add(argGrad5)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.CustomOp1 -> {
                        val (arg34, c2) = op
                        when (
                            val matchVal2 =
                                when (val tryResult = c2.bwd(arg34, node, grad1)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                        ) {
                            null -> {}
                            else -> {
                                val argGrad6 = matchVal2
                                val sumGrad37 =
                                    when (val tryResult = grads.orInsert(arg34)) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }
                                sumGrad37 =
                                    when (val tryResult = sumGrad37.add(argGrad6)) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }
                            }
                        }
                    }
                    op is Op.CustomOp2 -> {
                        val (arg1, arg2, c3) = op
                        val (argGrad1, argGrad2) =
                            when (val tryResult = c3.bwd(arg1, arg2, node, grad1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        if (argGrad1 != null) {
                            val argGrad11 = argGrad1
                            val sumGrad38 =
                                when (val tryResult = grads.orInsert(arg1)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                            sumGrad38 =
                                when (val tryResult = sumGrad38.add(argGrad11)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                        }
                        if (argGrad2 != null) {
                            val argGrad21 = argGrad2
                            val sumGrad39 =
                                when (val tryResult = grads.orInsert(arg2)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                            sumGrad39 =
                                when (val tryResult = sumGrad39.add(argGrad21)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                        }
                    }
                    op is Op.CustomOp3 -> {
                        val (arg11, arg21, arg3, c4) = op
                        val (argGrad12, argGrad22, argGrad3) =
                            when (val tryResult = c4.bwd(arg11, arg21, arg3, node, grad1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        if (argGrad12 != null) {
                            val argGrad13 = argGrad12
                            val sumGrad40 =
                                when (val tryResult = grads.orInsert(arg11)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                            sumGrad40 =
                                when (val tryResult = sumGrad40.add(argGrad13)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                        }
                        if (argGrad22 != null) {
                            val argGrad23 = argGrad22
                            val sumGrad41 =
                                when (val tryResult = grads.orInsert(arg21)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                            sumGrad41 =
                                when (val tryResult = sumGrad41.add(argGrad23)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                        }
                        if (argGrad3 != null) {
                            val argGrad31 = argGrad3
                            val sumGrad42 =
                                when (val tryResult = grads.orInsert(arg3)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                            sumGrad42 =
                                when (val tryResult = sumGrad42.add(argGrad31)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                        }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Sqr -> {
                        val (arg35, _) = op
                        val argGrad7 =
                            when (
                                val tryResult =
                                    when (val tryResult = arg35.mul(grad1)) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.affine(2.0, 0.0)
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad43 =
                            when (val tryResult = grads.orInsert(arg35)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad43 =
                            when (val tryResult = sumGrad43.add(argGrad7)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Sqrt -> {
                        val (arg36, _) = op
                        val argGrad8 =
                            when (
                                val tryResult =
                                    when (val tryResult = grad1.div(node)) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.affine(0.5, 0.0)
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad44 =
                            when (val tryResult = grads.orInsert(arg36)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad44 =
                            when (val tryResult = sumGrad44.add(argGrad8)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.ToDevice -> {
                        val (arg37) = op
                        val sumGrad45 =
                            when (val tryResult = grads.orInsert(arg37)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val argGrad9 =
                            when (val tryResult = grad1.toDevice(sumGrad45.device())) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad45 =
                            when (val tryResult = sumGrad45.add(argGrad9)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    op is Op.Transpose -> {
                        val (arg38, dim1, dim2) = op
                        val argGrad10 =
                            when (val tryResult = grad1.transpose(dim1, dim2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad46 =
                            when (val tryResult = grads.orInsert(arg38)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad46 =
                            when (val tryResult = sumGrad46.add(argGrad10)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                    else -> {
                        val (arg39, dims2) = op
                        var invDims = mutableListOf(0, dims2.len())
                        for ((i, dimIdx) in dims2.iter().enumerate()) {
                            invDims[dimIdx] = i
                        }
                        val argGrad11 =
                            when (val tryResult = grad1.permute(invDims)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val sumGrad47 =
                            when (val tryResult = grads.orInsert(arg39)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        sumGrad47 =
                            when (val tryResult = sumGrad47.add(argGrad11)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                    }
                }
            }
        }
    }
    return Ok(grads)
}

/**
 * A store for gradients, associating a tensor id to the corresponding gradient tensor, used for back propagation.
 */
public data class GradStore(
    public var value: Map<TensorId, Tensor> = emptyMap(),
) {
    public var field0: Map<TensorId, Tensor>
        get() = value
        set(v) {
            value = v
        }

    /**
     * Get the gradient tensor corresponding to the given tensor id
     */
    public fun getId(id: TensorId): Tensor? = this.value.get(id)

    /**
     * Get the gradient tensor associated with the given tensor
     */
    public fun get(tensor: Tensor): Tensor? = this.value.get(tensor.id())

    /**
     * Remove the gradient tensor associated with the given tensor, returning it if it exists
     */
    public fun remove(tensor: Tensor): Tensor? = this.value.remove(tensor.id())

    /**
     * Insert a gradient tensor associated with the given tensor, returning the previous gradient tensor if it existed
     */
    public fun insert(tensor: Tensor, grad: Tensor): Tensor? = this.value.insert(tensor.id(), grad)

    /**
     * Insert a gradient tensor associated with the given tensor id, returning the previous gradient tensor if it existed
     */
    public fun insertId(id: TensorId, grad: Tensor): Tensor? = this.value.insert(id, grad)

    /**
     * Get the gradient tensor associated with the given tensor, or, if it does not exist,
     * insert a tensor of zeroes, with the same shape and type as the given tensors and return it
     */
    public fun orInsert(tensor: Tensor): Result<Tensor> {
        val grad1 =
            when (val matchVal = this.value.entry(tensor.id())) {
                is Entry.Occupied -> {
                    val (entry) = matchVal
                    entry.intoMut()
                }
                is Entry.Vacant -> {
                    val (entry1) = matchVal
                    val grad =
                        when (val tryResult = tensor.zerosLike()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    entry1.insert(grad)
                }
            }
        return Ok(grad1)
    }

    /**
     * Extend this gradient store with the contents of another.
     * If an entry is already occupied, updates and ensures new tensor follows correct detach semantics.
     * Otherwise simply inserts.
     */
    public fun extend(other: GradStore): Result<Unit> {
        for ((id, grad) in other.value) {
            when (val matchVal = this.value.entry(id)) {
                is Entry.Occupied -> {
                    val (entry) = matchVal
                    val newGrad =
                        when (val tryResult = entry.get().add(grad)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val doNotDetach = CANDLE_GRAD_DO_NOT_DETACH.with({ b -> b })
                    val newGrad1 =
                        if (doNotDetach) {
                            newGrad
                        } else {
                            newGrad.detach()
                        }
                    entry.getMut() = newGrad1
                }
                is Entry.Vacant -> {
                    val (entry1) = matchVal
                    entry1.insert(grad)
                }
            }
        }
        return Ok(Unit)
    }

    /**
     * Get the tensor ids of the stored gradient tensors
     */
    public fun getIds(): Any = this.value.keys()

    public companion object {
        /**
         * Create a new gradient store
         */
        public fun new(): GradStore = GradStore.default()
    }
}
