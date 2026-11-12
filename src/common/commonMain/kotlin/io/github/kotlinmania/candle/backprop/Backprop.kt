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
        node.reshape(reducedDims).getOrElse { return Result.failure(it) }.broadcastAs(arg.shape())
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
            } else when (val matchVal1 = node.op()) {
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
    grads.insert(this, this.onesLike().getOrElse { return Result.failure(it) }.contiguous().getOrElse { return Result.failure(it) })
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
                        val lhsSumGrad = grads.orInsert(lhs).getOrElse { return Result.failure(it) }
                        lhsSumGrad = lhsSumGrad.add(grad1).getOrElse { return Result.failure(it) }
                        val rhsSumGrad = grads.orInsert(rhs).getOrElse { return Result.failure(it) }
                        rhsSumGrad = rhsSumGrad.add(grad1).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Binary && op.field2 == BinaryOp.Sub -> {
                        val (lhs1, rhs1, _) = op
                        val lhsSumGrad1 = grads.orInsert(lhs1).getOrElse { return Result.failure(it) }
                        lhsSumGrad1 = lhsSumGrad1.add(grad1).getOrElse { return Result.failure(it) }
                        val rhsSumGrad1 = grads.orInsert(rhs1).getOrElse { return Result.failure(it) }
                        rhsSumGrad1 = rhsSumGrad1.sub(grad1).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Binary && op.field2 == BinaryOp.Mul -> {
                        val (lhs2, rhs2, _) = op
                        val lhsGrad = grad1.mul(rhs2).getOrElse { return Result.failure(it) }
                        val lhsSumGrad2 = grads.orInsert(lhs2).getOrElse { return Result.failure(it) }
                        lhsSumGrad2 = lhsSumGrad2.add(lhsGrad).getOrElse { return Result.failure(it) }
                        val rhsGrad = grad1.mul(lhs2).getOrElse { return Result.failure(it) }
                        val rhsSumGrad2 = grads.orInsert(rhs2).getOrElse { return Result.failure(it) }
                        rhsSumGrad2 = rhsSumGrad2.add(rhsGrad).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Binary && op.field2 == BinaryOp.Div -> {
                        val (lhs3, rhs3, _) = op
                        val lhsGrad1 = grad1.div(rhs3).getOrElse { return Result.failure(it) }
                        val lhsSumGrad3 = grads.orInsert(lhs3).getOrElse { return Result.failure(it) }
                        lhsSumGrad3 = lhsSumGrad3.add(lhsGrad1).getOrElse { return Result.failure(it) }
                        val rhsGrad1 = grad1.mul(lhs3).getOrElse { return Result.failure(it) }.div(rhs3.sqr().getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                        val rhsSumGrad3 = grads.orInsert(rhs3).getOrElse { return Result.failure(it) }
                        rhsSumGrad3 = rhsSumGrad3.sub(rhsGrad1).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Binary && op.field2 == BinaryOp.Minimum -> {
                        val (lhs4, rhs4, _) = op
                        val maskLhs = node.eq(lhs4).getOrElse { return Result.failure(it) }.toDtype(grad1.dtype()).getOrElse { return Result.failure(it) }
                        val maskRhs = node.eq(rhs4).getOrElse { return Result.failure(it) }.toDtype(grad1.dtype()).getOrElse { return Result.failure(it) }
                        // If both masks are 1 one the same point, we want to scale the
                        // gradient by 0.5 rather than 1.
                        val lhsGrad2 = maskLhs.mul(grad1).getOrElse { return Result.failure(it) }.div((maskRhs + 1.0).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                        val lhsSumGrad4 = grads.orInsert(lhs4).getOrElse { return Result.failure(it) }
                        lhsSumGrad4 = lhsSumGrad4.add(lhsGrad2).getOrElse { return Result.failure(it) }
                        val rhsGrad2 = maskRhs.mul(grad1).getOrElse { return Result.failure(it) }.div((maskLhs + 1.0).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                        val rhsSumGrad4 = grads.orInsert(rhs4).getOrElse { return Result.failure(it) }
                        rhsSumGrad4 = rhsSumGrad4.add(rhsGrad2).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Binary && op.field2 == BinaryOp.Maximum -> {
                        val (lhs5, rhs5, _) = op
                        val maskLhs1 = node.eq(lhs5).getOrElse { return Result.failure(it) }.toDtype(grad1.dtype()).getOrElse { return Result.failure(it) }
                        val maskRhs1 = node.eq(rhs5).getOrElse { return Result.failure(it) }.toDtype(grad1.dtype()).getOrElse { return Result.failure(it) }
                        val lhsGrad3 = maskLhs1.mul(grad1).getOrElse { return Result.failure(it) }.div((maskRhs1 + 1.0).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                        val lhsSumGrad5 = grads.orInsert(lhs5).getOrElse { return Result.failure(it) }
                        lhsSumGrad5 = lhsSumGrad5.add(lhsGrad3).getOrElse { return Result.failure(it) }
                        val rhsGrad3 = maskRhs1.mul(grad1).getOrElse { return Result.failure(it) }.div((maskLhs1 + 1.0).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                        val rhsSumGrad5 = grads.orInsert(rhs5).getOrElse { return Result.failure(it) }
                        rhsSumGrad5 = rhsSumGrad5.add(rhsGrad3).getOrElse { return Result.failure(it) }
                    }
                    op is Op.WhereCond -> {
                        val (pred, t, f) = op
                        val zeros = grad1.zerosLike().getOrElse { return Result.failure(it) }
                        val tSumGrad = grads.orInsert(t).getOrElse { return Result.failure(it) }
                        val tGrad = pred.whereCond(grad1, zeros).getOrElse { return Result.failure(it) }
                        tSumGrad = tSumGrad.add(tGrad).getOrElse { return Result.failure(it) }
                        val fSumGrad = grads.orInsert(f).getOrElse { return Result.failure(it) }
                        val fGrad = pred.whereCond(zeros, grad1).getOrElse { return Result.failure(it) }
                        fSumGrad = fSumGrad.add(fGrad).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Conv1D -> {
                        val arg = op.arg
                        val kernel = op.kernel
                        val padding = op.padding
                        val stride = op.stride
                        val dilation = op.dilation
                        // The output height for convTranspose1d is:
                        // (lIn - 1) * stride - 2 * padding + dilation * (kSize - 1) + outPadding + 1
                        val gradLIn = grad1.dim(2).getOrElse { return Result.failure(it) }
                        val kSize = kernel.dim(2).getOrElse { return Result.failure(it) }
                        val outSize = (gradLIn - 1) * stride + dilation * (kSize - 1) + 1 - 2 * padding
                        val outPadding = arg.dim(2).getOrElse { return Result.failure(it) } - outSize
                        val gradArg = grad1.convTranspose1d(kernel, padding, outPadding, stride, dilation, 1).getOrElse { return Result.failure(it) }
                        val sumGrad = grads.orInsert(arg).getOrElse { return Result.failure(it) }
                        sumGrad = sumGrad.add(gradArg).getOrElse { return Result.failure(it) }
                        val gradKernel = arg.transpose(0, 1).getOrElse { return Result.failure(it) }.conv1d(grad1.transpose(0, 1).getOrElse { return Result.failure(it) }, padding, dilation, stride, 1).getOrElse { return Result.failure(it) }.transpose(0, 1).getOrElse { return Result.failure(it) }
                        val sumGrad1 = grads.orInsert(kernel).getOrElse { return Result.failure(it) }
                        val (_, _, k0) = kernel.dims3().getOrElse { return Result.failure(it) }
                        val (_, _, gK0) = gradKernel.dims3().getOrElse { return Result.failure(it) }
                        val gradKernel1 =
                            if (gK0 != k0) {
                                gradKernel.narrow(2, 0, k0).getOrElse { return Result.failure(it) }
                            } else {
                                gradKernel
                            }
                        sumGrad1 = sumGrad1.add(gradKernel1).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Conv2D -> {
                        val arg1 = op.arg
                        val kernel1 = op.kernel
                        val padding1 = op.padding
                        val stride1 = op.stride
                        val dilation1 = op.dilation
                        // The output height for convTranspose2d is:
                        // (iH - 1) * stride - 2 * padding + dilation * (kH - 1) + outPadding + 1
                        val gradH = grad1.dim(2).getOrElse { return Result.failure(it) }
                        val kH = kernel1.dim(2).getOrElse { return Result.failure(it) }
                        val outSize1 = (gradH - 1) * stride1 + dilation1 * (kH - 1) + 1 - 2 * padding1
                        val outPadding1 = arg1.dim(2).getOrElse { return Result.failure(it) } - outSize1
                        val gradArg1 = grad1.convTranspose2d(kernel1, padding1, outPadding1, stride1, dilation1).getOrElse { return Result.failure(it) }
                        val sumGrad2 = grads.orInsert(arg1).getOrElse { return Result.failure(it) }
                        sumGrad2 = sumGrad2.add(gradArg1).getOrElse { return Result.failure(it) }
                        val gradKernel2 = arg1.transpose(0, 1).getOrElse { return Result.failure(it) }.conv2d(grad1.transpose(0, 1).getOrElse { return Result.failure(it) }, padding1, dilation1, stride1, 1).getOrElse { return Result.failure(it) }.transpose(0, 1).getOrElse { return Result.failure(it) }
                        val sumGrad3 = grads.orInsert(kernel1).getOrElse { return Result.failure(it) }
                        val (_, _, k01, k1) = kernel1.dims4().getOrElse { return Result.failure(it) }
                        val (_, _, gK01, gK1) = gradKernel2.dims4().getOrElse { return Result.failure(it) }
                        val gradKernel3 =
                            if (gK01 != k01 || gK1 != k1) {
                                gradKernel2.narrow(2, 0, k01).getOrElse { return Result.failure(it) }.narrow(3, 0, k1).getOrElse { return Result.failure(it) }
                            } else {
                                gradKernel2
                            }
                        sumGrad3 = sumGrad3.add(gradKernel3).getOrElse { return Result.failure(it) }
                    }
                    op is Op.ConvTranspose1D -> Result.failure(Error.BackwardNotSupported(op = "conv-transpose1d")).getOrElse { return Result.failure(it) }
                    op is Op.ConvTranspose2D -> {
                        val arg2 = op.arg
                        val kernel2 = op.kernel
                        val padding2 = op.padding
                        val stride2 = op.stride
                        val dilation2 = op.dilation
                        val gradArg2 = grad1.conv2d(kernel2, padding2, stride2, dilation2, 1).getOrElse { return Result.failure(it) }
                        val sumGrad4 = grads.orInsert(arg2).getOrElse { return Result.failure(it) }
                        sumGrad4 = sumGrad4.add(gradArg2).getOrElse { return Result.failure(it) }
                        val gradKernel4 = grad1.transpose(0, 1).getOrElse { return Result.failure(it) }.conv2d(arg2.transpose(0, 1).getOrElse { return Result.failure(it) }, padding2, dilation2, stride2, 1).getOrElse { return Result.failure(it) }.transpose(0, 1).getOrElse { return Result.failure(it) }
                        val sumGrad5 = grads.orInsert(kernel2).getOrElse { return Result.failure(it) }
                        val (_, _, k02, k11) = kernel2.dims4().getOrElse { return Result.failure(it) }
                        val (_, _, gK02, gK11) = gradKernel4.dims4().getOrElse { return Result.failure(it) }
                        val gradKernel5 =
                            if (gK02 != k02 || gK11 != k11) {
                                gradKernel4.narrow(2, 0, k02).getOrElse { return Result.failure(it) }.narrow(3, 0, k11).getOrElse { return Result.failure(it) }
                            } else {
                                gradKernel4
                            }
                        sumGrad5 = sumGrad5.add(gradKernel5).getOrElse { return Result.failure(it) }
                    }
                    op is Op.AvgPool2D -> {
                        val arg3 = op.arg
                        val kernelSize = op.kernelSize
                        val stride3 = op.stride
                        if (kernelSize != stride3) {
                            bail("backward not supported for avgpool2d if ksize {kernel_size:?} != stride {stride:?}")
                        }
                        val (n, c, h, w) = arg3.dims4().getOrElse { return Result.failure(it) }
                        val gradArg3 = grad1.upsampleNearest2d(h, w).getOrElse { return Result.failure(it) }
                        val gradArg4 = (gradArg3 * (1.0 / (kernelSize.first * kernelSize.second).toDouble())).getOrElse { return Result.failure(it) }
                        val sumGrad6 = grads.orInsert(arg3).getOrElse { return Result.failure(it) }
                        sumGrad6 = sumGrad6.add(gradArg4).getOrElse { return Result.failure(it) }
                    }
                    op is Op.MaxPool2D -> {
                        val arg4 = op.arg
                        val kernelSize1 = op.kernelSize
                        val stride4 = op.stride
                        if (kernelSize1 != stride4) {
                            bail("backward not supported for maxpool2d if ksize {kernel_size:?} != stride {stride:?}")
                        }
                        val (n1, c1, h1, w1) = arg4.dims4().getOrElse { return Result.failure(it) }
                        // For computing the max-pool gradient, we compute a mask where a 1 means
                        // that the element is the maximum, then we apply this mask to the
                        // upsampled gradient (taking into account that multiple max may exist so
                        // we scale the gradient for this case).
                        val nodeUpsampled = node.upsampleNearest2d(h1, w1).getOrElse { return Result.failure(it) }
                        val mask = arg4.eq(nodeUpsampled).getOrElse { return Result.failure(it) }.toDtype(arg4.dtype()).getOrElse { return Result.failure(it) }
                        val avg = mask.avgPool2dWithStride(kernelSize1, stride4).getOrElse { return Result.failure(it) }
                        val gradArg5 = ((grad1 * avg).getOrElse { return Result.failure(it) }.upsampleNearest2d(h1, w1).getOrElse { return Result.failure(it) } * mask).getOrElse { return Result.failure(it) }
                        val sumGrad7 = grads.orInsert(arg4).getOrElse { return Result.failure(it) }
                        sumGrad7 = sumGrad7.add(gradArg5).getOrElse { return Result.failure(it) }
                    }
                    op is Op.UpsampleNearest1D -> {
                        val arg5 = op.arg
                        val targetSize = op.targetSize
                        val (n2, c, size) = arg5.dims3().getOrElse { return Result.failure(it) }
                        if (targetSize % size != 0) {
                            bail("backward not supported for non integer upscaling factors")
                        }
                        val scale = targetSize / size
                        val kernel3 = Tensor.ones(Triple(c, 1, scale), arg5.dtype(), arg5.device()).getOrElse { return Result.failure(it) }
                        val convSum = grad1.conv1d(kernel3, 0, scale, 1, c).getOrElse { return Result.failure(it) }
                        val sumGrad8 = grads.orInsert(arg5).getOrElse { return Result.failure(it) }
                        sumGrad8 = convSum
                    }
                    op is Op.UpsampleNearest2D -> {
                        val arg6 = op.arg
                        val targetH = op.targetH
                        val targetW = op.targetW
                        val (n3, c1, h2, w2) = arg6.dims4().getOrElse { return Result.failure(it) }
                        if (targetH % h2 != 0 || targetW % w2 != 0) {
                            bail("backward not supported for non integer upscaling factors")
                        }
                        val scaleH = targetH / h2
                        val scaleW = targetW / w2
                        if (scaleH != scaleW) {
                            bail("backward not supported for non uniform upscaling factors")
                        }
                        val kernel4 = Tensor.ones(listOf(c1, 1, scaleH, scaleW), arg6.dtype(), arg6.device()).getOrElse { return Result.failure(it) }
                        val convSum1 = grad1.conv2d(kernel4, 0, scaleH, 1, c1).getOrElse { return Result.failure(it) }
                        val sumGrad9 = grads.orInsert(arg6).getOrElse { return Result.failure(it) }
                        sumGrad9 = convSum1
                    }
                    op is Op.UpsampleBilinear2D -> {
                        bail("backward not supported for upsample_bilinear2d")
                    }
                    op is Op.SliceScatter0 -> {
                        val (lhs6, rhs6, startRhs) = op
                        val rhsSumGrad6 = grads.orInsert(rhs6).getOrElse { return Result.failure(it) }
                        val rhsGrad4 = grad1.narrow(0, startRhs, rhs6.dim(0).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                        rhsSumGrad6 = rhsSumGrad6.add(rhsGrad4).getOrElse { return Result.failure(it) }
                        val lhsSumGrad6 = grads.orInsert(lhs6).getOrElse { return Result.failure(it) }
                        val lhsGrad4 = grad1.sliceScatter0(rhs6.zerosLike().getOrElse { return Result.failure(it) }, startRhs).getOrElse { return Result.failure(it) }
                        lhsSumGrad6 = lhsSumGrad6.add(lhsGrad4).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Gather -> {
                        val (arg7, indexes, dim) = op
                        val sumGrad10 = grads.orInsert(arg7).getOrElse { return Result.failure(it) }
                        sumGrad10 = sumGrad10.scatterAdd(indexes, grad1, dim).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Scatter -> {
                        val (init, indexes1, src, dim1) = op
                        val initSumGrad = grads.orInsert(init).getOrElse { return Result.failure(it) }
                        initSumGrad = initSumGrad.add(grad1).getOrElse { return Result.failure(it) }
                        val srcGrad = grad1.gather(indexes1, dim1).getOrElse { return Result.failure(it) }
                        val srcSumGrad = grads.orInsert(src).getOrElse { return Result.failure(it) }
                        srcSumGrad = srcSumGrad.add(srcGrad).getOrElse { return Result.failure(it) }
                    }
                    op is Op.ScatterAdd -> {
                        val (init1, indexes2, src1, dim2) = op
                        val initSumGrad1 = grads.orInsert(init1).getOrElse { return Result.failure(it) }
                        val mask1 = init1.onesLike().getOrElse { return Result.failure(it) }
                        val mask2 = mask1.scatter(indexes2, mask1.zerosLike().getOrElse { return Result.failure(it) }, dim2).getOrElse { return Result.failure(it) }
                        initSumGrad1 = initSumGrad1.add(grad1.mul(mask2).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                        val srcGrad1 = grad1.gather(indexes2, dim2).getOrElse { return Result.failure(it) }
                        val srcSumGrad1 = grads.orInsert(src1).getOrElse { return Result.failure(it) }
                        srcSumGrad1 = srcSumGrad1.add(srcGrad1).getOrElse { return Result.failure(it) }
                    }
                    op is Op.IndexAdd -> {
                        val (init2, indexes3, src2, dim3) = op
                        val initSumGrad2 = grads.orInsert(init2).getOrElse { return Result.failure(it) }
                        initSumGrad2 = initSumGrad2.add(grad1).getOrElse { return Result.failure(it) }
                        val srcGrad2 = grad1.indexSelect(indexes3, dim3).getOrElse { return Result.failure(it) }
                        val srcSumGrad2 = grads.orInsert(src2).getOrElse { return Result.failure(it) }
                        srcSumGrad2 = srcSumGrad2.add(srcGrad2).getOrElse { return Result.failure(it) }
                    }
                    op is Op.IndexSelect -> {
                        val (arg8, indexes4, dim4) = op
                        val sumGrad11 = grads.orInsert(arg8).getOrElse { return Result.failure(it) }
                        // indexSelect accepts strided ids but indexAdd requires contiguous ones.
                        val indexes5 = indexes4.contiguous().getOrElse { return Result.failure(it) }
                        sumGrad11 = sumGrad11.indexAdd(indexes5, grad1, dim4).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Matmul -> {
                        val (lhs7, rhs7) = op
                        // Skipping checks, the op went ok, we can skip
                        // the matmul size checks for now.
                        val lhsGrad5 = grad1.matmul(rhs7.t().getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                        val lhsSumGrad7 = grads.orInsert(lhs7).getOrElse { return Result.failure(it) }
                        lhsSumGrad7 = lhsSumGrad7.add(lhsGrad5).getOrElse { return Result.failure(it) }
                        val rhsGrad5 = lhs7.t().getOrElse { return Result.failure(it) }.matmul(grad1).getOrElse { return Result.failure(it) }
                        val rhsSumGrad7 = grads.orInsert(rhs7).getOrElse { return Result.failure(it) }
                        rhsSumGrad7 = rhsSumGrad7.add(rhsGrad5).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Cat -> {
                        val (args, dim5) = op
                        var startIdx = 0
                        for (arg9 in args) {
                            val len = arg9.dims()[dim5]
                            val argGrad = grad1.narrow(dim5, startIdx, len).getOrElse { return Result.failure(it) }
                            val sumGrad12 = grads.orInsert(arg9).getOrElse { return Result.failure(it) }
                            sumGrad12 = sumGrad12.add(argGrad).getOrElse { return Result.failure(it) }
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
                        for ((dim6, nodeDim, argDim) in (nodeDims[leftDims until nodeDims.size].zip(argDims.iter())).enumerate()) {
                            if (nodeDim != argDim) {
                                sumDims.push(dim6 + leftDims)
                            }
                        }
                        var argGrad1 = grad1.sumKeepdim(sumDims.asSlice()).getOrElse { return Result.failure(it) }
                        for (i in 0 until leftDims) {
                            argGrad1 = argGrad1.squeeze(0).getOrElse { return Result.failure(it) }
                        }
                        val sumGrad13 = grads.orInsert(arg10).getOrElse { return Result.failure(it) }
                        sumGrad13 = sumGrad13.add(argGrad1.broadcastAs(sumGrad13.dims()).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Reduce && op.field1 == ReduceOp.Sum -> {
                        val (arg11, _, reducedDims) = op
                        val grad2 = broadcastBack(arg11, grad1, reducedDims).getOrElse { return Result.failure(it) }
                        val sumGrad14 = grads.orInsert(arg11).getOrElse { return Result.failure(it) }
                        sumGrad14 = sumGrad14.add(grad2).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Reduce && op.field1 == ReduceOp.Max -> {
                        val (arg12, _, reducedDims1) = op
                        val node1 = broadcastBack(arg12, node, reducedDims1).getOrElse { return Result.failure(it) }
                        val grad3 = broadcastBack(arg12, grad1, reducedDims1).getOrElse { return Result.failure(it) }
                        val grad4 = node1.eq(arg12).getOrElse { return Result.failure(it) }.toDtype(grad3.dtype()).getOrElse { return Result.failure(it) }.mul(grad3).getOrElse { return Result.failure(it) }
                        val sumGrad15 = grads.orInsert(arg12).getOrElse { return Result.failure(it) }
                        sumGrad15 = sumGrad15.add(grad4.broadcastAs(sumGrad15.dims()).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Reduce && op.field1 == ReduceOp.Min -> {
                        val (arg13, _, reducedDims2) = op
                        val node2 = broadcastBack(arg13, node, reducedDims2).getOrElse { return Result.failure(it) }
                        val grad5 = broadcastBack(arg13, grad1, reducedDims2).getOrElse { return Result.failure(it) }
                        val grad6 = node2.eq(arg13).getOrElse { return Result.failure(it) }.toDtype(grad5.dtype()).getOrElse { return Result.failure(it) }.mul(grad5).getOrElse { return Result.failure(it) }
                        val sumGrad16 = grads.orInsert(arg13).getOrElse { return Result.failure(it) }
                        sumGrad16 = sumGrad16.add(grad6.broadcastAs(sumGrad16.dims()).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                    }
                    op is Op.ToDType -> {
                        val (arg14) = op
                        val sumGrad17 = grads.orInsert(arg14).getOrElse { return Result.failure(it) }
                        sumGrad17 = sumGrad17.add(grad1.toDtype(arg14.dtype()).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Copy -> {
                        val (arg15) = op
                        val sumGrad18 = grads.orInsert(arg15).getOrElse { return Result.failure(it) }
                        sumGrad18 = sumGrad18.add(grad1).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Affine -> {
                        val arg16 = op.arg
                        val mul = op.mul
                        val argGrad2 = grad1.affine(mul, 0.0).getOrElse { return Result.failure(it) }
                        val sumGrad19 = grads.orInsert(arg16).getOrElse { return Result.failure(it) }
                        sumGrad19 = sumGrad19.add(argGrad2).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Log -> {
                        val (arg17, _) = op
                        val sumGrad20 = grads.orInsert(arg17).getOrElse { return Result.failure(it) }
                        sumGrad20 = sumGrad20.add((grad1 / arg17).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Sin -> {
                        val (arg18, _) = op
                        val sumGrad21 = grads.orInsert(arg18).getOrElse { return Result.failure(it) }
                        sumGrad21 = sumGrad21.add((grad1 * arg18.cos()).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Cos -> {
                        val (arg19, _) = op
                        val sumGrad22 = grads.orInsert(arg19).getOrElse { return Result.failure(it) }
                        sumGrad22 = sumGrad22.sub((grad1 * arg19.sin()).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Tanh -> {
                        val (arg20, _) = op
                        val sumGrad23 = grads.orInsert(arg20).getOrElse { return Result.failure(it) }
                        val minusDtanh = (node.sqr().getOrElse { return Result.failure(it) } - 1.0).getOrElse { return Result.failure(it) }
                        sumGrad23 = sumGrad23.sub((grad1 * minusDtanh).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Abs -> {
                        val (arg21, _) = op
                        val sumGrad24 = grads.orInsert(arg21).getOrElse { return Result.failure(it) }
                        val ones = arg21.onesLike().getOrElse { return Result.failure(it) }
                        val absGrad = arg21.ge(arg21.zerosLike().getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }.whereCond(ones, ones.neg().getOrElse { return Result.failure(it) })
                        sumGrad24 = sumGrad24.add((grad1 * absGrad).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Exp -> {
                        val (arg22, _) = op
                        val sumGrad25 = grads.orInsert(arg22).getOrElse { return Result.failure(it) }
                        sumGrad25 = sumGrad25.add((grad1 * node).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Neg -> {
                        val (arg23, _) = op
                        val sumGrad26 = grads.orInsert(arg23).getOrElse { return Result.failure(it) }
                        sumGrad26 = sumGrad26.sub(grad1).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Recip -> {
                        val (arg24, _) = op
                        val sumGrad27 = grads.orInsert(arg24).getOrElse { return Result.failure(it) }
                        val grad7 = (grad1 / arg24.sqr().getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                        sumGrad27 = sumGrad27.sub(grad7).getOrElse { return Result.failure(it) }
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
                                Tensor.zeros(dims, grad1.dtype(), grad1.device()).getOrElse { return Result.failure(it) }
                            }
                        val rightPad = argDims1[dim7] - startIdx1 - len1
                        val rightPad1 =
                            if (rightPad == 0) {
                                null
                            } else {
                                val dims1 = argDims1.toVec()
                                dims1[dim7] = rightPad
                                Tensor.zeros(dims1, grad1.dtype(), grad1.device()).getOrElse { return Result.failure(it) }
                            }
                        val argGrad3 =
                            run {
                                val matchVal1 = Pair(leftPad, rightPad1)
                                when {
                                    matchVal1.first == null && matchVal1.second == null -> grad1
                                    matchVal1.first != null && matchVal1.second == null -> {
                                        val l = matchVal1.first
                                        Tensor.cat(listOf(l, grad1), dim7).getOrElse { return Result.failure(it) }
                                    }
                                    matchVal1.first == null && matchVal1.second != null -> {
                                        val r = matchVal1.second
                                        Tensor.cat(listOf(grad1, r), dim7).getOrElse { return Result.failure(it) }
                                    }
                                    else -> {
                                        val l1 = matchVal1.first
                                        val r1 = matchVal1.second
                                        Tensor.cat(listOf(l1, grad1, r1), dim7).getOrElse { return Result.failure(it) }
                                    }
                                }
                            }
                        val sumGrad28 = grads.orInsert(arg25).getOrElse { return Result.failure(it) }
                        sumGrad28 = sumGrad28.add(argGrad3).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Floor -> {}
                    op is Op.Unary && op.field1 == UnaryOp.Round -> {}
                    op is Op.Reduce && op.field1 == ReduceOp.ArgMin -> {}
                    op is Op.Reduce && op.field1 == ReduceOp.ArgMax -> {}
                    op is Op.Unary && op.field1 == UnaryOp.Sign -> {}
                    op is Op.Cmp -> {}
                    op is Op.Reshape -> {
                        val (arg26) = op
                        val argGrad4 = grad1.reshape(arg26.dims()).getOrElse { return Result.failure(it) }
                        val sumGrad29 = grads.orInsert(arg26).getOrElse { return Result.failure(it) }
                        sumGrad29 = sumGrad29.add(argGrad4).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Ceil -> Result.failure(Error.BackwardNotSupported(op = "ceil")).getOrElse { return Result.failure(it) }
                    op is Op.Unary && op.field1 == UnaryOp.Gelu -> {
                        val (arg27, _) = op
                        val sumGrad30 = grads.orInsert(arg27).getOrElse { return Result.failure(it) }
                        val cube = arg27.powf(3.0).getOrElse { return Result.failure(it) }
                        val tanh = (0.0356774 * cube + (0.797885 * arg27).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }.tanh().getOrElse { return Result.failure(it) }
                        val geluGrad = (((0.5 * tanh).getOrElse { return Result.failure(it) } + (0.0535161 * cube + (0.398942 * arg27).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) } * (1.0 - tanh.powf(2.0).getOrElse { return Result.failure(it) })).getOrElse { return Result.failure(it) } + 0.5).getOrElse { return Result.failure(it) }
                        sumGrad30 = sumGrad30.add((grad1 * geluGrad).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Erf -> {
                        val (arg28, _) = op
                        val sumGrad31 = grads.orInsert(arg28).getOrElse { return Result.failure(it) }
                        // d/dx erf(x) = 2/sqrt(pi) * e^(-x^2)
                        val erfGrad = (2.0 / Double.PI.sqrt()) * (arg28.sqr().getOrElse { return Result.failure(it) }.neg().getOrElse { return Result.failure(it) }).exp().getOrElse { return Result.failure(it) }
                        sumGrad31 = sumGrad31.add((grad1 * erfGrad).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.GeluErf -> {
                        val (arg29, _) = op
                        val sumGrad32 = grads.orInsert(arg29).getOrElse { return Result.failure(it) }
                        // d/dx geluErf(x) = 0.5 + 0.398942 e^(-x^2/2) x + 0.5 erf(x/sqrt(2))
                        val negHalfSquare = (arg29.sqr().getOrElse { return Result.failure(it) }.neg().getOrElse { return Result.failure(it) } / 2.0).getOrElse { return Result.failure(it) }
                        val scaledExpArg = (0.398942 * negHalfSquare.exp().getOrElse { return Result.failure(it) } * arg29).getOrElse { return Result.failure(it) }
                        val argScaledSqrt = (arg29 / 2.0.sqrt()).getOrElse { return Result.failure(it) }
                        val erfScaledSqrt = (0.5 * argScaledSqrt.erf().getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                        val geluErfGrad = (0.5 + scaledExpArg + erfScaledSqrt).getOrElse { return Result.failure(it) }
                        sumGrad32 = sumGrad32.add((grad1 * geluErfGrad).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Relu -> {
                        val (arg30, _) = op
                        val sumGrad33 = grads.orInsert(arg30).getOrElse { return Result.failure(it) }
                        val reluGrad = arg30.ge(arg30.zerosLike().getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }.toDtype(arg30.dtype()).getOrElse { return Result.failure(it) }
                        sumGrad33 = sumGrad33.add((grad1 * reluGrad).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Silu -> {
                        val (arg31, _) = op
                        val sumGrad34 = grads.orInsert(arg31).getOrElse { return Result.failure(it) }
                        // d/dx silu = sigmoid(x) * (1 + x * (1 - sigmoid(x))) = sigmoid(x) * (1 - node) + node
                        val sigmoidArg = (arg31.neg().getOrElse { return Result.failure(it) }.exp().getOrElse { return Result.failure(it) } + 1.0).getOrElse { return Result.failure(it) }.recip().getOrElse { return Result.failure(it) }
                        val siluGrad = sigmoidArg * (1.0 - node) + node
                        sumGrad34 = sumGrad34.add((grad1 * siluGrad).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Elu -> {
                        val (arg32, alpha) = op
                        // d/dx elu(x) = 1 for x > 0, alpha * e^x for x <= 0
                        val sumGrad35 = grads.orInsert(arg32).getOrElse { return Result.failure(it) }
                        val zeros1 = arg32.zerosLike().getOrElse { return Result.failure(it) }
                        val positiveMask = arg32.gt(zeros1).getOrElse { return Result.failure(it) }.toDtype(arg32.dtype()).getOrElse { return Result.failure(it) }
                        val negativeMask = arg32.le(zeros1).getOrElse { return Result.failure(it) }.toDtype(arg32.dtype()).getOrElse { return Result.failure(it) }
                        // node == alpha * (e^x - 1) for x <= 0, reuse it
                        val negativeExpMask = (negativeMask * (node + alpha)).getOrElse { return Result.failure(it) }
                        val combinedMask = (positiveMask + negativeExpMask).getOrElse { return Result.failure(it) }
                        sumGrad35 = sumGrad35.add((grad1 * combinedMask).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Powf -> {
                        val (arg33, e) = op
                        val argGrad5 = ((grad1 * arg33.powf(e - 1.0).getOrElse { return Result.failure(it) }).getOrElse { return Result.failure(it) } * e).getOrElse { return Result.failure(it) }
                        val sumGrad36 = grads.orInsert(arg33).getOrElse { return Result.failure(it) }
                        sumGrad36 = sumGrad36.add(argGrad5).getOrElse { return Result.failure(it) }
                    }
                    op is Op.CustomOp1 -> {
                        val (arg34, c2) = op
                        when (val matchVal2 = c2.bwd(arg34, node, grad1).getOrElse { return Result.failure(it) }) {
                            null -> {}
                            else -> {
                                val argGrad6 = matchVal2
                                val sumGrad37 = grads.orInsert(arg34).getOrElse { return Result.failure(it) }
                                sumGrad37 = sumGrad37.add(argGrad6).getOrElse { return Result.failure(it) }
                            }
                        }
                    }
                    op is Op.CustomOp2 -> {
                        val (arg1, arg2, c3) = op
                        val (argGrad1, argGrad2) = c3.bwd(arg1, arg2, node, grad1).getOrElse { return Result.failure(it) }
                        if (argGrad1 != null) {
                            val argGrad11 = argGrad1
                            val sumGrad38 = grads.orInsert(arg1).getOrElse { return Result.failure(it) }
                            sumGrad38 = sumGrad38.add(argGrad11).getOrElse { return Result.failure(it) }
                        }
                        if (argGrad2 != null) {
                            val argGrad21 = argGrad2
                            val sumGrad39 = grads.orInsert(arg2).getOrElse { return Result.failure(it) }
                            sumGrad39 = sumGrad39.add(argGrad21).getOrElse { return Result.failure(it) }
                        }
                    }
                    op is Op.CustomOp3 -> {
                        val (arg11, arg21, arg3, c4) = op
                        val (argGrad12, argGrad22, argGrad3) = c4.bwd(arg11, arg21, arg3, node, grad1).getOrElse { return Result.failure(it) }
                        if (argGrad12 != null) {
                            val argGrad13 = argGrad12
                            val sumGrad40 = grads.orInsert(arg11).getOrElse { return Result.failure(it) }
                            sumGrad40 = sumGrad40.add(argGrad13).getOrElse { return Result.failure(it) }
                        }
                        if (argGrad22 != null) {
                            val argGrad23 = argGrad22
                            val sumGrad41 = grads.orInsert(arg21).getOrElse { return Result.failure(it) }
                            sumGrad41 = sumGrad41.add(argGrad23).getOrElse { return Result.failure(it) }
                        }
                        if (argGrad3 != null) {
                            val argGrad31 = argGrad3
                            val sumGrad42 = grads.orInsert(arg3).getOrElse { return Result.failure(it) }
                            sumGrad42 = sumGrad42.add(argGrad31).getOrElse { return Result.failure(it) }
                        }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Sqr -> {
                        val (arg35, _) = op
                        val argGrad7 = arg35.mul(grad1).getOrElse { return Result.failure(it) }.affine(2.0, 0.0).getOrElse { return Result.failure(it) }
                        val sumGrad43 = grads.orInsert(arg35).getOrElse { return Result.failure(it) }
                        sumGrad43 = sumGrad43.add(argGrad7).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Unary && op.field1 == UnaryOp.Sqrt -> {
                        val (arg36, _) = op
                        val argGrad8 = grad1.div(node).getOrElse { return Result.failure(it) }.affine(0.5, 0.0).getOrElse { return Result.failure(it) }
                        val sumGrad44 = grads.orInsert(arg36).getOrElse { return Result.failure(it) }
                        sumGrad44 = sumGrad44.add(argGrad8).getOrElse { return Result.failure(it) }
                    }
                    op is Op.ToDevice -> {
                        val (arg37) = op
                        val sumGrad45 = grads.orInsert(arg37).getOrElse { return Result.failure(it) }
                        val argGrad9 = grad1.toDevice(sumGrad45.device()).getOrElse { return Result.failure(it) }
                        sumGrad45 = sumGrad45.add(argGrad9).getOrElse { return Result.failure(it) }
                    }
                    op is Op.Transpose -> {
                        val (arg38, dim1, dim2) = op
                        val argGrad10 = grad1.transpose(dim1, dim2).getOrElse { return Result.failure(it) }
                        val sumGrad46 = grads.orInsert(arg38).getOrElse { return Result.failure(it) }
                        sumGrad46 = sumGrad46.add(argGrad10).getOrElse { return Result.failure(it) }
                    }
                    else -> {
                        val (arg39, dims2) = op
                        var invDims = mutableListOf(0, dims2.len())
                        for ((i, dimIdx) in dims2.iter().enumerate()) {
                            invDims[dimIdx] = i
                        }
                        val argGrad11 = grad1.permute(invDims).getOrElse { return Result.failure(it) }
                        val sumGrad47 = grads.orInsert(arg39).getOrElse { return Result.failure(it) }
                        sumGrad47 = sumGrad47.add(argGrad11).getOrElse { return Result.failure(it) }
                    }
                }
            }
        }
    }
    return Result.success(grads)
}

/**
 * A store for gradients, associating a tensor id to the corresponding gradient tensor, used for back propagation.
 */
public open class GradStore(
    public var value: Map<TensorId, Tensor>,
) {
    public var field0: Map<TensorId, Tensor>
        get() = value
        set(v) {
            value = v
        }

    /**
     * Get the gradient tensor corresponding to the given tensor id
     */
    public open fun getId(id: TensorId): Tensor? = this.value.get(id)

    /**
     * Get the gradient tensor associated with the given tensor
     */
    public open fun get(tensor: Tensor): Tensor? = this.value.get(tensor.id())

    /**
     * Remove the gradient tensor associated with the given tensor, returning it if it exists
     */
    public open fun remove(tensor: Tensor): Tensor? = this.value.remove(tensor.id())

    /**
     * Insert a gradient tensor associated with the given tensor, returning the previous gradient tensor if it existed
     */
    public open fun insert(tensor: Tensor, grad: Tensor): Tensor? = this.value.insert(tensor.id(), grad)

    /**
     * Insert a gradient tensor associated with the given tensor id, returning the previous gradient tensor if it existed
     */
    public open fun insertId(id: TensorId, grad: Tensor): Tensor? = this.value.insert(id, grad)

    /**
     * Get the gradient tensor associated with the given tensor, or, if it does not exist,
     * insert a tensor of zeroes, with the same shape and type as the given tensors and return it
     */
    public open fun orInsert(tensor: Tensor): Result<Tensor> {
        val grad1 =
            when (val matchVal = this.value.entry(tensor.id())) {
                is Entry.Occupied -> {
                    val (entry) = matchVal
                    entry.intoMut()
                }
                is Entry.Vacant -> {
                    val (entry1) = matchVal
                    val grad = tensor.zerosLike().getOrElse { return Result.failure(it) }
                    entry1.insert(grad)
                }
            }
        return Result.success(grad1)
    }

    /**
     * Extend this gradient store with the contents of another.
     * If an entry is already occupied, updates and ensures new tensor follows correct detach semantics.
     * Otherwise simply inserts.
     */
    public open fun extend(other: GradStore): Result<Unit> {
        for ((id, grad) in other.value) {
            when (val matchVal = this.value.entry(id)) {
                is Entry.Occupied -> {
                    val (entry) = matchVal
                    val newGrad = entry.get().add(grad).getOrElse { return Result.failure(it) }
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
        return Result.success(Unit)
    }

    /**
     * Get the tensor ids of the stored gradient tensors
     */
    public open fun getIds(): Any = this.value.keys()

    public companion object {
        /**
         * Create a new gradient store
         */
        public fun new(): GradStore = GradStore.default()
    }
}
