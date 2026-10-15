// port-lint: source backprop.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.envie.getenv
import io.github.kotlinmania.threadlocal.ThreadLocal

// arg has been reduced to node via reduceDims, expand it back to arg.
// This has to handle keepdims.
private fun broadcastBack(arg: Tensor, node: Tensor, reducedDims: List<Int>): Tensor {
    return if (arg.rank() == node.rank()) {
        // keepdim = true
        node.broadcastAs(arg.shape())
    } else {
        // keepdim = false
        // first expand the reduced dims.
        node.reshape(Shape(reducedDims)).broadcastAs(arg.shape())
    }
}

private val CANDLE_GRAD_DO_NOT_DETACH: ThreadLocal<Boolean> = ThreadLocal()

private val opTernaryArgs: (Op) -> Triple<Tensor, Tensor, Tensor> = { op ->
    when (op) {
        is Op.IndexAdd -> Triple(op.init, op.indexes, op.src)
        is Op.Scatter -> Triple(op.arg, op.index, op.src)
        is Op.ScatterAdd -> Triple(op.arg, op.index, op.src)
        is Op.CustomOp3 -> Triple(op.t1, op.t2, op.t3)
        is Op.WhereCond -> Triple(op.cond, op.onTrue, op.onFalse)
        else -> throw IllegalStateException()
    }
}

private val opBinaryArgs: (Op) -> Pair<Tensor, Tensor> = { op ->
    when (op) {
        is Op.Conv1D -> Pair(op.arg, op.kernel)
        is Op.ConvTranspose1D -> Pair(op.arg, op.kernel)
        is Op.Conv2D -> Pair(op.arg, op.kernel)
        is Op.ConvTranspose2D -> Pair(op.arg, op.kernel)
        is Op.CustomOp2 -> Pair(op.lhs, op.rhs)
        is Op.Binary -> Pair(op.lhs, op.rhs)
        is Op.Gather -> Pair(op.arg, op.index)
        is Op.IndexSelect -> Pair(op.arg, op.index)
        is Op.Matmul -> Pair(op.lhs, op.rhs)
        is Op.SliceScatter0 -> Pair(op.lhs, op.rhs)
        else -> throw IllegalStateException()
    }
}

private val opUnaryArg: (Op) -> Tensor = { op ->
    when (op) {
        is Op.Reshape -> op.arg
        is Op.UpsampleNearest1D -> op.arg
        is Op.UpsampleNearest2D -> op.arg
        is Op.UpsampleBilinear2D -> op.arg
        is Op.AvgPool2D -> op.arg
        is Op.MaxPool2D -> op.arg
        is Op.Copy -> op.arg
        is Op.Broadcast -> op.arg
        is Op.Cmp -> op.arg
        is Op.Reduce -> op.arg
        is Op.ToDevice -> op.arg
        is Op.Transpose -> op.arg
        is Op.Permute -> op.arg
        is Op.Narrow -> op.arg
        is Op.Unary -> op.arg
        is Op.Elu -> op.arg
        is Op.Powf -> op.arg
        is Op.CustomOp1 -> op.arg
        is Op.ToDType -> op.arg
        else -> throw IllegalStateException()
    }
}

/**
 * Return all the nodes that lead to this value in a topologically sorted vec, the first
 * elements having dependencies on the latter ones, e.g. the first element if any is the
 * argument.
 * This assumes that the op graph is a DAG.
 */
public fun Tensor.sortedNodes(): List<Tensor> {
    // The list of sorted nodes is passed as an owned value rather than a mutable reference.
    fun walk(
        node: Tensor,
        nodes: MutableList<Tensor>,
        alreadySeen: MutableMap<TensorId, Boolean>,
    ): Pair<Boolean, MutableList<Tensor>> {
        val seenTg = alreadySeen[node.id()]
        if (seenTg != null) {
            return Pair(seenTg, nodes)
        }
        var trackGrad = false
        val nextNodes = if (node.isVariable()) {
            // Do not call recursively on the "leaf" nodes.
            trackGrad = true
            nodes
        } else if (node.dtype().isInt()) {
            nodes
        } else {
            val op = node.op()
            if (op != null) {
                when (op) {
                    is Op.IndexAdd,
                    is Op.Scatter,
                    is Op.ScatterAdd,
                    is Op.CustomOp3,
                    is Op.WhereCond -> {
                        val (t1, t2, t3) = opTernaryArgs(op)
                        val (tg1, nodes1) = walk(t1, nodes, alreadySeen)
                        trackGrad = trackGrad || tg1
                        val (tg2, nodes2) = walk(t2, nodes1, alreadySeen)
                        trackGrad = trackGrad || tg2
                        val (tg3, nodes3) = walk(t3, nodes2, alreadySeen)
                        trackGrad = trackGrad || tg3
                        nodes3
                    }
                    is Op.Conv1D,
                    is Op.ConvTranspose1D,
                    is Op.Conv2D,
                    is Op.ConvTranspose2D,
                    is Op.CustomOp2,
                    is Op.Binary,
                    is Op.Gather,
                    is Op.IndexSelect,
                    is Op.Matmul,
                    is Op.SliceScatter0 -> {
                        val (lhs, rhs) = opBinaryArgs(op)
                        val (tg1, nodes1) = walk(lhs, nodes, alreadySeen)
                        trackGrad = trackGrad || tg1
                        val (tg2, nodes2) = walk(rhs, nodes1, alreadySeen)
                        trackGrad = trackGrad || tg2
                        nodes2
                    }
                    is Op.Cat -> op.args.fold(nodes) { accNodes, arg ->
                        val (tg, next) = walk(arg, accNodes, alreadySeen)
                        trackGrad = trackGrad || tg
                        next
                    }
                    is Op.Affine -> {
                        if (op.mul == 0.0) {
                            nodes
                        } else {
                            val (tg, next) = walk(op.arg, nodes, alreadySeen)
                            trackGrad = trackGrad || tg
                            next
                        }
                    }
                    is Op.Unary -> {
                        when (op.op) {
                            UnaryOp.Ceil,
                            UnaryOp.Floor,
                            UnaryOp.Round,
                            UnaryOp.Sign -> nodes
                            else -> {
                                val (tg, next) = walk(op.arg, nodes, alreadySeen)
                                trackGrad = trackGrad || tg
                                next
                            }
                        }
                    }
                    is Op.Reshape,
                    is Op.UpsampleNearest1D,
                    is Op.UpsampleNearest2D,
                    is Op.UpsampleBilinear2D,
                    is Op.AvgPool2D,
                    is Op.MaxPool2D,
                    is Op.Copy,
                    is Op.Broadcast,
                    is Op.Cmp,
                    is Op.ToDevice,
                    is Op.Transpose,
                    is Op.Permute,
                    is Op.Narrow,
                    is Op.Elu,
                    is Op.Powf,
                    is Op.CustomOp1 -> {
                        val child = opUnaryArg(op)
                        val (tg, next) = walk(child, nodes, alreadySeen)
                        trackGrad = trackGrad || tg
                        next
                    }
                    is Op.ToDType -> {
                        if (op.arg.dtype().isFloat()) {
                            val (tg, next) = walk(op.arg, nodes, alreadySeen)
                            trackGrad = trackGrad || tg
                            next
                        } else {
                            nodes
                        }
                    }
                    is Op.Reduce -> {
                        when (op.op) {
                            ReduceOp.Min, ReduceOp.Sum, ReduceOp.Max -> {
                                val (tg, next) = walk(op.arg, nodes, alreadySeen)
                                trackGrad = trackGrad || tg
                                next
                            }
                            ReduceOp.ArgMin, ReduceOp.ArgMax -> nodes
                        }
                    }
                }
            } else {
                nodes
            }
        }
        alreadySeen[node.id()] = trackGrad
        if (trackGrad) {
            nextNodes.add(node)
        }
        return Pair(trackGrad, nextNodes)
    }
    val (_, nodes) = walk(this, mutableListOf(), HashMap())
    nodes.reverse()
    return nodes
}

public fun Tensor.backward(): GradStore {
    val sortedNodes = this.sortedNodes()
    val grads = GradStore.new()
    grads.insert(this, this.onesLike().contiguous())
    for (node in sortedNodes) {
        if (node.isVariable()) {
            continue
        }
        val gradRaw = grads
            .remove(node)
            ?: bail("candle internal error - grad not populated")
        // https://github.com/huggingface/candle/issues/1241
        // Ideally, we would make these operations in place where possible to ensure that we
        // do not have to allocate too often. Here we just call `.detach` to avoid computing
        // the backprop graph of the backprop itself. This would be an issue for second order
        // derivatives but these are out of scope at the moment.
        val doNotDetach = CANDLE_GRAD_DO_NOT_DETACH.getOr {
            val s = getenv("CANDLE_GRAD_DO_NOT_DETACH")
            s != null && s.isNotEmpty() && s != "0"
        }
        val grad = if (doNotDetach) gradRaw else gradRaw.detach()
        val op = node.op()
        if (op != null) {
            when (op) {
                is Op.Binary -> {
                    val lhs = op.lhs
                    val rhs = op.rhs
                    when (op.op) {
                        BinaryOp.Add -> {
                            val lhsSumGrad = grads.orInsert(lhs)
                            grads.insert(lhs, lhsSumGrad.add(grad))
                            val rhsSumGrad = grads.orInsert(rhs)
                            grads.insert(rhs, rhsSumGrad.add(grad))
                        }
                        BinaryOp.Sub -> {
                            val lhsSumGrad = grads.orInsert(lhs)
                            grads.insert(lhs, lhsSumGrad.add(grad))
                            val rhsSumGrad = grads.orInsert(rhs)
                            grads.insert(rhs, rhsSumGrad.sub(grad))
                        }
                        BinaryOp.Mul -> {
                            val lhsGrad = grad.mul(rhs)
                            val lhsSumGrad = grads.orInsert(lhs)
                            grads.insert(lhs, lhsSumGrad.add(lhsGrad))
                            val rhsGrad = grad.mul(lhs)
                            val rhsSumGrad = grads.orInsert(rhs)
                            grads.insert(rhs, rhsSumGrad.add(rhsGrad))
                        }
                        BinaryOp.Div -> {
                            val lhsGrad = grad.div(rhs)
                            val lhsSumGrad = grads.orInsert(lhs)
                            grads.insert(lhs, lhsSumGrad.add(lhsGrad))
                            val rhsGrad = grad.mul(lhs).div(rhs.sqr())
                            val rhsSumGrad = grads.orInsert(rhs)
                            grads.insert(rhs, rhsSumGrad.sub(rhsGrad))
                        }
                        BinaryOp.Minimum,
                        BinaryOp.Maximum -> {
                            val maskLhs = node.eq(lhs).toDtype(grad.dtype())
                            val maskRhs = node.eq(rhs).toDtype(grad.dtype())

                            // If both masks are 1 one the same point, we want to scale the
                            // gradient by 0.5 rather than 1.
                            val lhsGrad = maskLhs.mul(grad).div(maskRhs.affine(1.0, 1.0))
                            val lhsSumGrad = grads.orInsert(lhs)
                            grads.insert(lhs, lhsSumGrad.add(lhsGrad))

                            val rhsGrad = maskRhs.mul(grad).div(maskLhs.affine(1.0, 1.0))
                            val rhsSumGrad = grads.orInsert(rhs)
                            grads.insert(rhs, rhsSumGrad.add(rhsGrad))
                        }
                    }
                }
                is Op.WhereCond -> {
                    val pred = op.cond
                    val t = op.onTrue
                    val f = op.onFalse
                    val zeros = grad.zerosLike()
                    val tSumGrad = grads.orInsert(t)
                    val tGrad = pred.whereCond(grad, zeros)
                    grads.insert(t, tSumGrad.add(tGrad))
                    val fSumGrad = grads.orInsert(f)
                    val fGrad = pred.whereCond(zeros, grad)
                    grads.insert(f, fSumGrad.add(fGrad))
                }
                is Op.Conv1D -> {
                    val arg = op.arg
                    val kernel = op.kernel
                    val padding = op.padding
                    val stride = op.stride
                    val dilation = op.dilation
                    // The output height for convTranspose1d is:
                    // (lIn - 1) * stride - 2 * padding + dilation * (kSize - 1) + outPadding + 1
                    val gradLIn = grad.dim(2)
                    val kSize = kernel.dim(2)
                    val outSize =
                        (gradLIn - 1) * stride + dilation * (kSize - 1) + 1 - 2 * padding
                    val outPadding = arg.dim(2) - outSize
                    val gradArg = grad.convTranspose1d(
                        kernel,
                        padding,
                        outPadding,
                        stride,
                        dilation,
                        /* groups */ 1,
                    )
                    val sumGrad = grads.orInsert(arg)
                    grads.insert(arg, sumGrad.add(gradArg))

                    val gradKernelRaw = arg
                        .transpose(0, 1)
                        .conv1d(grad.transpose(0, 1), padding, dilation, stride, 1)
                        .transpose(0, 1)
                    val sumGradK = grads.orInsert(kernel)
                    val (_, _, k0) = kernel.dims3()
                    val (_, _, gK0) = gradKernelRaw.dims3()
                    val gradKernel = if (gK0 != k0) {
                        gradKernelRaw.narrow(2, 0, k0)
                    } else {
                        gradKernelRaw
                    }
                    grads.insert(kernel, sumGradK.add(gradKernel))
                }
                is Op.Conv2D -> {
                    val arg = op.arg
                    val kernel = op.kernel
                    val padding = op.padding
                    val stride = op.stride
                    val dilation = op.dilation
                    // The output height for convTranspose2d is:
                    // (iH - 1) * stride - 2 * padding + dilation * (kH - 1) + outPadding + 1
                    val gradH = grad.dim(2)
                    val kH = kernel.dim(2)
                    val outSize =
                        (gradH - 1) * stride + dilation * (kH - 1) + 1 - 2 * padding
                    val outPadding = arg.dim(2) - outSize
                    val gradArg = grad.convTranspose2d(
                        kernel,
                        padding,
                        outPadding,
                        stride,
                        dilation,
                    )
                    val sumGrad = grads.orInsert(arg)
                    grads.insert(arg, sumGrad.add(gradArg))

                    val gradKernelRaw = arg
                        .transpose(0, 1)
                        .conv2d(grad.transpose(0, 1), padding, dilation, stride, 1)
                        .transpose(0, 1)
                    val sumGradK = grads.orInsert(kernel)
                    val (_, _, k0, k1) = kernel.dims4()
                    val (_, _, gK0, gK1) = gradKernelRaw.dims4()
                    val gradKernel = if (gK0 != k0 || gK1 != k1) {
                        gradKernelRaw.narrow(2, 0, k0).narrow(3, 0, k1)
                    } else {
                        gradKernelRaw
                    }
                    grads.insert(kernel, sumGradK.add(gradKernel))
                }
                is Op.ConvTranspose1D -> throw CandleException.BackwardNotSupported(
                    op = "conv-transpose1d",
                )
                is Op.ConvTranspose2D -> {
                    val arg = op.arg
                    val kernel = op.kernel
                    val padding = op.padding
                    val stride = op.stride
                    val dilation = op.dilation
                    val gradArg = grad.conv2d(kernel, padding, stride, dilation, 1)
                    val sumGrad = grads.orInsert(arg)
                    grads.insert(arg, sumGrad.add(gradArg))

                    val gradKernelRaw = grad
                        .transpose(0, 1)
                        .conv2d(arg.transpose(0, 1), padding, dilation, stride, 1)
                        .transpose(0, 1)
                    val sumGradK = grads.orInsert(kernel)
                    val (_, _, k0, k1) = kernel.dims4()
                    val (_, _, gK0, gK1) = gradKernelRaw.dims4()
                    val gradKernel = if (gK0 != k0 || gK1 != k1) {
                        gradKernelRaw.narrow(2, 0, k0).narrow(3, 0, k1)
                    } else {
                        gradKernelRaw
                    }
                    grads.insert(kernel, sumGradK.add(gradKernel))
                }
                is Op.AvgPool2D -> {
                    val arg = op.arg
                    val kernelSize = op.kernelSize
                    val stride = op.stride
                    if (kernelSize != stride) {
                        bail("backward not supported for avgpool2d if ksize $kernelSize != stride $stride")
                    }
                    val (_, _, h, w) = arg.dims4()
                    val gradArgUpsampled = grad.upsampleNearest2d(h, w)
                    val gradArg =
                        gradArgUpsampled.affine(1.0 / (kernelSize.first * kernelSize.second).toDouble(), 0.0)
                    val sumGrad = grads.orInsert(arg)
                    grads.insert(arg, sumGrad.add(gradArg))
                }
                is Op.MaxPool2D -> {
                    val arg = op.arg
                    val kernelSize = op.kernelSize
                    val stride = op.stride
                    if (kernelSize != stride) {
                        bail("backward not supported for maxpool2d if ksize $kernelSize != stride $stride")
                    }
                    val (_, _, h, w) = arg.dims4()
                    // For computing the max-pool gradient, we compute a mask where a 1 means
                    // that the element is the maximum, then we apply this mask to the
                    // upsampled gradient (taking into account that multiple max may exist so
                    // we scale the gradient for this case).
                    val nodeUpsampled = node.upsampleNearest2d(h, w)
                    val mask = arg.eq(nodeUpsampled).toDtype(arg.dtype())
                    val avg = mask.avgPool2dWithStride(kernelSize, stride)
                    val gradArg = grad.mul(avg).upsampleNearest2d(h, w).mul(mask)
                    val sumGrad = grads.orInsert(arg)
                    grads.insert(arg, sumGrad.add(gradArg))
                }
                is Op.UpsampleNearest1D -> {
                    val arg = op.arg
                    val targetSize = op.targetSize
                    val (_, c, size) = arg.dims3()
                    if (targetSize % size != 0) {
                        bail("backward not supported for non integer upscaling factors")
                    }
                    val scale = targetSize / size

                    val kernel = Tensor.ones(Shape(c, 1, scale), arg.dtype(), arg.device())
                    val convSum = grad.conv1d(kernel, 0, scale, 1, c)
                    grads.orInsert(arg)
                    grads.insert(arg, convSum)
                }
                is Op.UpsampleNearest2D -> {
                    val arg = op.arg
                    val targetH = op.targetH
                    val targetW = op.targetW
                    val (_, c, h, w) = arg.dims4()
                    if (targetH % h != 0 || targetW % w != 0) {
                        bail("backward not supported for non integer upscaling factors")
                    }
                    val scaleH = targetH / h
                    val scaleW = targetW / w

                    if (scaleH != scaleW) {
                        bail("backward not supported for non uniform upscaling factors")
                    }
                    val kernel =
                        Tensor.ones(Shape(c, 1, scaleH, scaleW), arg.dtype(), arg.device())
                    val convSum = grad.conv2d(kernel, 0, scaleH, 1, c)
                    grads.orInsert(arg)
                    grads.insert(arg, convSum)
                }
                is Op.UpsampleBilinear2D -> {
                    bail("backward not supported for upsample_bilinear2d")
                }
                is Op.SliceScatter0 -> {
                    val lhs = op.lhs
                    val rhs = op.rhs
                    val startRhs = op.start
                    val rhsSumGrad = grads.orInsert(rhs)
                    val rhsGrad = grad.narrow(0, startRhs, rhs.dim(0))
                    grads.insert(rhs, rhsSumGrad.add(rhsGrad))

                    val lhsSumGrad = grads.orInsert(lhs)
                    val lhsGrad = grad.sliceScatter0(rhs.zerosLike(), startRhs)
                    grads.insert(lhs, lhsSumGrad.add(lhsGrad))
                }
                is Op.Gather -> {
                    val arg = op.arg
                    val indexes = op.index
                    val dim = op.dim
                    val sumGrad = grads.orInsert(arg)
                    grads.insert(arg, sumGrad.scatterAdd(indexes, grad, dim))
                }
                is Op.Scatter -> {
                    val init = op.arg
                    val indexes = op.index
                    val src = op.src
                    val dim = op.dim
                    val initSumGrad = grads.orInsert(init)
                    grads.insert(init, initSumGrad.add(grad))

                    val srcGrad = grad.gather(indexes, dim)
                    val srcSumGrad = grads.orInsert(src)
                    grads.insert(src, srcSumGrad.add(srcGrad))
                }
                is Op.ScatterAdd -> {
                    val init = op.arg
                    val indexes = op.index
                    val src = op.src
                    val dim = op.dim
                    val initSumGrad = grads.orInsert(init)
                    val maskInit = init.onesLike()
                    val mask = maskInit.scatter(indexes, maskInit.zerosLike(), dim)
                    grads.insert(init, initSumGrad.add(grad.mul(mask)))

                    val srcGrad = grad.gather(indexes, dim)
                    val srcSumGrad = grads.orInsert(src)
                    grads.insert(src, srcSumGrad.add(srcGrad))
                }
                is Op.IndexAdd -> {
                    val init = op.init
                    val indexes = op.indexes
                    val src = op.src
                    val dim = op.dim
                    val initSumGrad = grads.orInsert(init)
                    grads.insert(init, initSumGrad.add(grad))

                    val srcGrad = grad.indexSelect(indexes, dim)
                    val srcSumGrad = grads.orInsert(src)
                    grads.insert(src, srcSumGrad.add(srcGrad))
                }
                is Op.IndexSelect -> {
                    val arg = op.arg
                    val indexes = op.index
                    val dim = op.dim
                    val sumGrad = grads.orInsert(arg)
                    grads.insert(arg, sumGrad.indexAdd(indexes, grad, dim))
                }
                is Op.Matmul -> {
                    val lhs = op.lhs
                    val rhs = op.rhs
                    // Skipping checks, the op went ok, we can skip
                    // the matmul size checks for now.

                    val lhsGrad = grad.matmul(rhs.t())
                    val lhsSumGrad = grads.orInsert(lhs)
                    grads.insert(lhs, lhsSumGrad.add(lhsGrad))

                    val rhsGrad = lhs.t().matmul(grad)
                    val rhsSumGrad = grads.orInsert(rhs)
                    grads.insert(rhs, rhsSumGrad.add(rhsGrad))
                }
                is Op.Cat -> {
                    var startIdx = 0
                    for (arg in op.args) {
                        val len = arg.dims()[op.dim]
                        val argGrad = grad.narrow(op.dim, startIdx, len)
                        val sumGrad = grads.orInsert(arg)
                        grads.insert(arg, sumGrad.add(argGrad))
                        startIdx += len
                    }
                }
                is Op.Broadcast -> {
                    val arg = op.arg
                    val argDims = arg.dims()
                    val nodeDims = node.dims()
                    // The number of dims that have been inserted on the left.
                    val leftDims = nodeDims.size - argDims.size
                    val sumDims: MutableList<Int> = (0 until leftDims).toMutableList()
                    for (dim in argDims.indices) {
                        val nodeDim = nodeDims[leftDims + dim]
                        val argDim = argDims[dim]
                        if (nodeDim != argDim) {
                            sumDims.add(dim + leftDims)
                        }
                    }

                    var argGrad = grad.sumKeepdim(sumDims)
                    for (i in 0 until leftDims) {
                        argGrad = argGrad.squeeze(0)
                    }
                    val sumGrad = grads.orInsert(arg)
                    grads.insert(arg, sumGrad.add(argGrad.broadcastAs(sumGrad.shape())))
                }
                is Op.Reduce -> {
                    val arg = op.arg
                    val reducedDims = op.dims
                    when (op.op) {
                        ReduceOp.Sum -> {
                            val gradB = broadcastBack(arg, grad, reducedDims)
                            val sumGrad = grads.orInsert(arg)
                            grads.insert(arg, sumGrad.add(gradB))
                        }
                        ReduceOp.Max -> {
                            val nodeB = broadcastBack(arg, node, reducedDims)
                            val gradB = broadcastBack(arg, grad, reducedDims)
                            val gradMasked = nodeB.eq(arg).toDtype(gradB.dtype()).mul(gradB)
                            val sumGrad = grads.orInsert(arg)
                            grads.insert(arg, sumGrad.add(gradMasked.broadcastAs(sumGrad.shape())))
                        }
                        ReduceOp.Min -> {
                            val nodeB = broadcastBack(arg, node, reducedDims)
                            val gradB = broadcastBack(arg, grad, reducedDims)
                            val gradMasked = nodeB.eq(arg).toDtype(gradB.dtype()).mul(gradB)
                            val sumGrad = grads.orInsert(arg)
                            grads.insert(arg, sumGrad.add(gradMasked.broadcastAs(sumGrad.shape())))
                        }
                        ReduceOp.ArgMin,
                        ReduceOp.ArgMax -> Unit
                    }
                }
                is Op.ToDType -> {
                    val arg = op.arg
                    val sumGrad = grads.orInsert(arg)
                    grads.insert(arg, sumGrad.add(grad.toDtype(arg.dtype())))
                }
                is Op.Copy -> {
                    val arg = op.arg
                    val sumGrad = grads.orInsert(arg)
                    grads.insert(arg, sumGrad.add(grad))
                }
                is Op.Affine -> {
                    val arg = op.arg
                    val argGrad = grad.affine(op.mul, 0.0)
                    val sumGrad = grads.orInsert(arg)
                    grads.insert(arg, sumGrad.add(argGrad))
                }
                is Op.Unary -> {
                    val arg = op.arg
                    when (op.op) {
                        UnaryOp.Log -> {
                            val sumGrad = grads.orInsert(arg)
                            grads.insert(arg, sumGrad.add(grad.div(arg)))
                        }
                        UnaryOp.Sin -> {
                            val sumGrad = grads.orInsert(arg)
                            grads.insert(arg, sumGrad.add(grad.mul(arg.cos())))
                        }
                        UnaryOp.Cos -> {
                            val sumGrad = grads.orInsert(arg)
                            grads.insert(arg, sumGrad.sub(grad.mul(arg.sin())))
                        }
                        UnaryOp.Tanh -> {
                            val sumGrad = grads.orInsert(arg)
                            val minusDtanh = node.sqr().affine(1.0, -1.0)
                            grads.insert(arg, sumGrad.sub(grad.mul(minusDtanh)))
                        }
                        UnaryOp.Abs -> {
                            val sumGrad = grads.orInsert(arg)
                            val ones = arg.onesLike()
                            val absGrad = arg.ge(arg.zerosLike()).whereCond(ones, ones.neg())
                            grads.insert(arg, sumGrad.add(grad.mul(absGrad)))
                        }
                        UnaryOp.Exp -> {
                            val sumGrad = grads.orInsert(arg)
                            grads.insert(arg, sumGrad.add(grad.mul(node)))
                        }
                        UnaryOp.Neg -> {
                            val sumGrad = grads.orInsert(arg)
                            grads.insert(arg, sumGrad.sub(grad))
                        }
                        UnaryOp.Recip -> {
                            val sumGrad = grads.orInsert(arg)
                            val gradRecip = grad.div(arg.sqr())
                            grads.insert(arg, sumGrad.sub(gradRecip))
                        }
                        UnaryOp.Floor,
                        UnaryOp.Round,
                        UnaryOp.Sign -> Unit
                        UnaryOp.Ceil -> throw CandleException.BackwardNotSupported(op = "ceil")
                        UnaryOp.Gelu -> {
                            val sumGrad = grads.orInsert(arg)
                            val cube = arg.powf(3.0)
                            val tanh = cube.affine(0.0356774, 0.0).add(arg.affine(0.797885, 0.0)).tanh()
                            val geluGrad = tanh.affine(0.5, 0.0)
                                .add(cube.affine(0.0535161, 0.0).add(arg.affine(0.398942, 0.0)).mul(tanh.powf(2.0).affine(-1.0, 1.0)))
                                .affine(1.0, 0.5)
                            grads.insert(arg, sumGrad.add(grad.mul(geluGrad)))
                        }
                        UnaryOp.Erf -> {
                            val sumGrad = grads.orInsert(arg)
                            // d/dx erf(x) = 2/sqrt(pi) * e^(-x^2)
                            val erfGrad =
                                arg.sqr().neg().exp().affine(2.0 / kotlin.math.sqrt(kotlin.math.PI), 0.0)
                            grads.insert(arg, sumGrad.add(grad.mul(erfGrad)))
                        }
                        UnaryOp.GeluErf -> {
                            val sumGrad = grads.orInsert(arg)
                            // d/dx geluErf(x) = 0.5 + 0.398942 e^(-x^2/2) x + 0.5 erf(x/sqrt(2))
                            val negHalfSquare = arg.sqr().neg().affine(0.5, 0.0)
                            val scaledExpArg = negHalfSquare.exp().affine(0.398942, 0.0).mul(arg)
                            val argScaledSqrt = arg.affine(1.0 / kotlin.math.sqrt(2.0), 0.0)
                            val erfScaledSqrt = argScaledSqrt.erf().affine(0.5, 0.0)
                            val geluErfGrad = scaledExpArg.add(erfScaledSqrt).affine(1.0, 0.5)
                            grads.insert(arg, sumGrad.add(grad.mul(geluErfGrad)))
                        }
                        UnaryOp.Relu -> {
                            val sumGrad = grads.orInsert(arg)
                            val reluGrad = arg.ge(arg.zerosLike()).toDtype(arg.dtype())
                            grads.insert(arg, sumGrad.add(grad.mul(reluGrad)))
                        }
                        UnaryOp.Silu -> {
                            val sumGrad = grads.orInsert(arg)
                            // d/dx silu = sigmoid(x) * (1 + x * (1 - sigmoid(x))) = sigmoid(x) * (1 - node) + node
                            val sigmoidArg = arg.neg().exp().affine(1.0, 1.0).recip()
                            val siluGrad = sigmoidArg.mul(node.affine(-1.0, 1.0)).add(node)
                            grads.insert(arg, sumGrad.add(grad.mul(siluGrad)))
                        }
                        UnaryOp.Sqr -> {
                            val argGrad = arg.mul(grad).affine(2.0, 0.0)
                            val sumGrad = grads.orInsert(arg)
                            grads.insert(arg, sumGrad.add(argGrad))
                        }
                        UnaryOp.Sqrt -> {
                            val argGrad = grad.div(node).affine(0.5, 0.0)
                            val sumGrad = grads.orInsert(arg)
                            grads.insert(arg, sumGrad.add(argGrad))
                        }
                    }
                }
                is Op.Narrow -> {
                    val arg = op.arg
                    val dim = op.dim
                    val startIdx = op.start
                    val len = op.len
                    val argDims = arg.dims()
                    val leftPad = if (startIdx == 0) {
                        null
                    } else {
                        val dims = argDims.toMutableList()
                        dims[dim] = startIdx
                        Tensor.zeros(Shape(dims), grad.dtype(), grad.device())
                    }
                    val rightPadLen = argDims[dim] - startIdx - len
                    val rightPad = if (rightPadLen == 0) {
                        null
                    } else {
                        val dims = argDims.toMutableList()
                        dims[dim] = rightPadLen
                        Tensor.zeros(Shape(dims), grad.dtype(), grad.device())
                    }
                    val argGrad = when {
                        leftPad == null && rightPad == null -> grad
                        leftPad != null && rightPad == null -> Tensor.cat(listOf(leftPad, grad), dim)
                        leftPad == null && rightPad != null -> Tensor.cat(listOf(grad, rightPad), dim)
                        else -> Tensor.cat(listOf(leftPad!!, grad, rightPad!!), dim)
                    }
                    val sumGrad = grads.orInsert(arg)
                    grads.insert(arg, sumGrad.add(argGrad))
                }
                is Op.Cmp -> Unit
                is Op.Reshape -> {
                    val arg = op.arg
                    val argGrad = grad.reshape(arg.shape())
                    val sumGrad = grads.orInsert(arg)
                    grads.insert(arg, sumGrad.add(argGrad))
                }
                is Op.Elu -> {
                    val arg = op.arg
                    val alpha = op.alpha
                    // d/dx elu(x) = 1 for x > 0, alpha * e^x for x <= 0
                    val sumGrad = grads.orInsert(arg)
                    val zeros = arg.zerosLike()
                    val positiveMask = arg.gt(zeros).toDtype(arg.dtype())
                    val negativeMask = arg.le(zeros).toDtype(arg.dtype())
                    // node == alpha * (e^x - 1) for x <= 0, reuse it
                    val negativeExpMask = negativeMask.mul(node.affine(1.0, alpha))
                    val combinedMask = positiveMask.add(negativeExpMask)
                    grads.insert(arg, sumGrad.add(grad.mul(combinedMask)))
                }
                is Op.Powf -> {
                    val arg = op.arg
                    val e = op.e
                    val argGrad = grad.mul(arg.powf(e - 1.0)).affine(e, 0.0)
                    val sumGrad = grads.orInsert(arg)
                    grads.insert(arg, sumGrad.add(argGrad))
                }
                is Op.CustomOp1 -> {
                    val arg = op.arg
                    val argGrad = op.op.bwd(arg, node, grad)
                    if (argGrad != null) {
                        val sumGrad = grads.orInsert(arg)
                        grads.insert(arg, sumGrad.add(argGrad))
                    }
                }
                is Op.CustomOp2 -> {
                    val arg1 = op.lhs
                    val arg2 = op.rhs
                    val (argGrad1, argGrad2) = op.op.bwd(arg1, arg2, node, grad)
                    if (argGrad1 != null) {
                        val sumGrad = grads.orInsert(arg1)
                        grads.insert(arg1, sumGrad.add(argGrad1))
                    }
                    if (argGrad2 != null) {
                        val sumGrad = grads.orInsert(arg2)
                        grads.insert(arg2, sumGrad.add(argGrad2))
                    }
                }
                is Op.CustomOp3 -> {
                    val arg1 = op.t1
                    val arg2 = op.t2
                    val arg3 = op.t3
                    val (argGrad1, argGrad2, argGrad3) =
                        op.op.bwd(arg1, arg2, arg3, node, grad)
                    if (argGrad1 != null) {
                        val sumGrad = grads.orInsert(arg1)
                        grads.insert(arg1, sumGrad.add(argGrad1))
                    }
                    if (argGrad2 != null) {
                        val sumGrad = grads.orInsert(arg2)
                        grads.insert(arg2, sumGrad.add(argGrad2))
                    }
                    if (argGrad3 != null) {
                        val sumGrad = grads.orInsert(arg3)
                        grads.insert(arg3, sumGrad.add(argGrad3))
                    }
                }
                is Op.ToDevice -> {
                    val arg = op.arg
                    val sumGrad = grads.orInsert(arg)
                    val argGrad = grad.toDevice(sumGrad.device())
                    grads.insert(arg, sumGrad.add(argGrad))
                }
                is Op.Transpose -> {
                    val arg = op.arg
                    val argGrad = grad.transpose(op.dim1, op.dim2)
                    val sumGrad = grads.orInsert(arg)
                    grads.insert(arg, sumGrad.add(argGrad))
                }
                is Op.Permute -> {
                    val arg = op.arg
                    val dims = op.dims
                    val invDims = IntArray(dims.size) { 0 }
                    for (entry in dims.withIndex()) {
                        invDims[entry.value] = entry.index
                    }
                    val argGrad = grad.permute(*invDims)
                    val sumGrad = grads.orInsert(arg)
                    grads.insert(arg, sumGrad.add(argGrad))
                }
            }
        }
    }
    return grads
}

/**
 * A store for gradients, associating a tensor id to the corresponding gradient tensor, used for back propagation.
 */
public class GradStore(
    private val grads: MutableMap<TensorId, Tensor> = mutableMapOf(),
) {
    private val self: GradStore get() = this

    public companion object {
        public val default: () -> GradStore = { GradStore() }

        /**
         * Create a new gradient store
         */
        public fun new(): GradStore {
            return default()
        }
    }

    /**
     * Get the gradient tensor corresponding to the given tensor id
     */
    public fun getId(id: TensorId): Tensor? {
        return self.grads[id]
    }

    /**
     * Get the gradient tensor associated with the given tensor
     */
    public fun get(tensor: Tensor): Tensor? {
        return self.grads[tensor.id()]
    }

    /**
     * Remove the gradient tensor associated with the given tensor, returning it if it exists
     */
    public fun remove(tensor: Tensor): Tensor? {
        return self.grads.remove(tensor.id())
    }

    /**
     * Insert a gradient tensor associated with the given tensor, returning the previous gradient tensor if it existed
     */
    public fun insert(tensor: Tensor, grad: Tensor): Tensor? {
        return self.grads.put(tensor.id(), grad)
    }

    /**
     * Insert a gradient tensor associated with the given tensor id, returning the previous gradient tensor if it existed
     */
    public fun insertId(id: TensorId, grad: Tensor): Tensor? {
        return self.grads.put(id, grad)
    }

    /**
     * Get the gradient tensor associated with the given tensor, or, if it does not exist,
     * insert a tensor of zeroes, with the same shape and type as the given tensors and return it
     */
    public fun orInsert(tensor: Tensor): Tensor {
        val existing = self.grads[tensor.id()]
        val grad = if (existing != null) {
            existing
        } else {
            val grad = tensor.zerosLike()
            self.grads[tensor.id()] = grad
            grad
        }
        return grad
    }

    /**
     * Extend this gradient store with the contents of another.
     * If an entry is already occupied, updates and ensures new tensor follows correct detach semantics.
     * Otherwise simply inserts.
     */
    public fun extend(other: GradStore) {
        for (entry in other.grads.entries) {
            val id = entry.key
            val grad = entry.value
            val existing = self.grads[id]
            if (existing != null) {
                val newGradRaw = existing.add(grad)
                val doNotDetach = CANDLE_GRAD_DO_NOT_DETACH.getOr {
                    val s = getenv("CANDLE_GRAD_DO_NOT_DETACH")
                    s != null && s.isNotEmpty() && s != "0"
                }
                val newGrad = if (doNotDetach) {
                    newGradRaw
                } else {
                    newGradRaw.detach()
                }
                self.grads[id] = newGrad
            } else {
                self.grads[id] = grad
            }
        }
    }

    /**
     * Get the tensor ids of the stored gradient tensors
     */
    public fun getIds(): Set<TensorId> {
        return self.grads.keys
    }
}
