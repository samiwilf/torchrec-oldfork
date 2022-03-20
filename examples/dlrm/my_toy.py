
from __future__ import absolute_import, division, print_function, unicode_literals

from typing import Callable, List, Optional, Tuple, TypeVar


import argparse

# miscellaneous
import builtins
import datetime
import json
import sys
import time
import itertools
import traceback
import copy 

import unittest
# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings

# numpy
import numpy as np

# projection
import sklearn.metrics



# pytorch
import torch
import torch.nn as nn
from torch._ops import ops
from torch.autograd.profiler import record_function
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

try:
    import fbgemm_gpu
    from fbgemm_gpu import split_table_batched_embeddings_ops
    from fbgemm_gpu.split_table_batched_embeddings_ops import (
        CacheAlgorithm,
        PoolingMode,
        OptimType,
        SparseType,
        SplitTableBatchedEmbeddingBagsCodegen,
        IntNBitTableBatchedEmbeddingBagsCodegen,
    )
except (ImportError, OSError):
    fbgemm_gpu_import_error_msg = traceback.format_exc()
    fbgemm_gpu = None

from fbgemm_gpu.split_table_batched_embeddings_ops import (
    OptimType,
    SparseType,
    RecordCacheMetrics,
    BoundsCheckMode,
)
#open_source = True
#if open_source:
#    # pyre-ignore[21]
#    from test_utils import gpu_available, gpu_unavailable
#else:
#    from fbgemm_gpu.test.test_utils import gpu_available, gpu_unavailable

from hypothesis import HealthCheck, Verbosity, assume, given, settings
from hypothesis.strategies import composite
from torch import Tensor    



# quantize_fbgemm_gpu_embedding_bag is partially lifted from
# fbgemm_gpu/test/split_embedding_inference_converter.py, def _quantize_split_embs.
# Converts SplitTableBatchedEmbeddingBagsCodegen to IntNBitTableBatchedEmbeddingBagsCodegen
def quantize_fbgemm_gpu_embedding_bag(model, quantize_type, device):
    embedding_specs = []
    if device.type == "cpu":
        emb_location = split_table_batched_embeddings_ops.EmbeddingLocation.HOST
    else:
        emb_location = split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE

    for (E, D, _, _) in model.embedding_specs:
        weights_ty = quantize_type
        if D % weights_ty.align_size() != 0:
            assert D % 4 == 0
            weights_ty = (
                SparseType.FP16
            )  # fall back to FP16 if dimension couldn't be aligned with the required size
        embedding_specs.append(("", E, D, weights_ty, emb_location))

    q_model = (
        split_table_batched_embeddings_ops.IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
            pooling_mode=model.pooling_mode,
            device=device,
        )
    )
    q_model.initialize_weights()
    for t, (_, _, _, weight_ty, _) in enumerate(embedding_specs):
        if weight_ty == SparseType.FP16:
            original_weight = model.split_embedding_weights()[t]
            q_weight = original_weight.half()
            weights = torch.tensor(q_weight.cpu().numpy().view(np.uint8))
            q_model.split_embedding_weights()[t][0].data.copy_(weights)

        elif weight_ty == SparseType.INT8:
            original_weight = model.split_embedding_weights()[t]
            q_weight = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                original_weight
            )
            weights = q_weight[:, :-8]
            scale_shift = torch.tensor(
                q_weight[:, -8:]
                .contiguous()
                .cpu()
                .numpy()
                .view(np.float32)
                .astype(np.float16)
                .view(np.uint8)
            )
            q_model.split_embedding_weights()[t][0].data.copy_(weights)
            q_model.split_embedding_weights()[t][1].data.copy_(scale_shift)

        elif weight_ty == SparseType.INT4 or weight_ty == SparseType.INT2:
            original_weight = model.split_embedding_weights()[t]
            q_weight = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                original_weight,
                bit_rate=quantize_type.bit_rate(),
            )
            weights = q_weight[:, :-4]
            scale_shift = torch.tensor(
                q_weight[:, -4:].contiguous().cpu().numpy().view(np.uint8)
            )
            q_model.split_embedding_weights()[t][0].data.copy_(weights)
            q_model.split_embedding_weights()[t][1].data.copy_(scale_shift)
    return q_model


def create_fbgemm_gpu_emb_bag(
    device,
    emb_l,
    m_spa,
    quantize_bits,
    learning_rate,
    codegen_preference=None,
    requires_grad=True,
):
    if isinstance(emb_l[0], nn.EmbeddingBag):
        emb_l = [e.weight for e in emb_l]
    Es = [e.shape[0] for e in emb_l]

    if isinstance(m_spa, list):
        Ds = m_spa
    else:
        Ds = [m_spa for _ in emb_l]

    if device.type == "cpu":
        emb_location = split_table_batched_embeddings_ops.EmbeddingLocation.HOST
        compute_device = split_table_batched_embeddings_ops.ComputeDevice.CPU
    else:
        emb_location = split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED_CACHING
        compute_device = split_table_batched_embeddings_ops.ComputeDevice.CUDA
    pooling_mode = PoolingMode.SUM
    cache_algorithm = CacheAlgorithm.LRU

    quantize_type = SparseType.FP32

    feature_table_map = list(range(len(Es)))

    fbgemm_gpu_emb_bag = SplitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=[
            (
                E,  # num of rows in the table
                D,  # num of columns in the table
                emb_location,
                compute_device,
            )
            for (E, D) in zip(Es, Ds)
        ],
        weights_precision=quantize_type,
        optimizer=OptimType.EXACT_SGD,
        feature_table_map=feature_table_map,
        learning_rate=0.1,
        cache_algorithm=cache_algorithm,
        pooling_mode=pooling_mode,
    )
    weights = fbgemm_gpu_emb_bag.split_embedding_weights()
    for i, emb in enumerate(weights):
        emb.data.copy_(emb_l[i])

    return fbgemm_gpu_emb_bag


# The purpose of this wrapper is to encapsulate the format conversions to/from fbgemm_gpu
# so parallel_apply() executes the format-in -> fbgemm_gpu op -> format-out instructions
# for each respective GPU in parallel.
class fbgemm_gpu_emb_bag_wrapper(nn.Module):
    def __init__(
        self,
        device,
        emb_l,
        m_spa,
        quantize_bits,
        learning_rate,
        codegen_preference,
        requires_grad,
    ):
        super(fbgemm_gpu_emb_bag_wrapper, self).__init__()
        self.fbgemm_gpu_emb_bag = create_fbgemm_gpu_emb_bag(
            device,
            emb_l,
            m_spa,
            quantize_bits,
            learning_rate,
            codegen_preference,
            requires_grad,
        )
        self.device = device
        self.m_spa = m_spa
        # create cumsum array for mixed dimension support
        if isinstance(m_spa, list):
            self.m_spa_cumsum = np.cumsum([0] + m_spa)
        if not requires_grad:
            torch.no_grad()
            torch.set_grad_enabled(False)

    def forward(self, lS_o, lS_i, v_W_l=None):

        # convert offsets to fbgemm format
        lengths_list = list(map(len, lS_i))
        indices_lengths_cumsum = np.cumsum([0] + lengths_list)
        if isinstance(lS_o, list):
            lS_o = torch.stack(lS_o)
        lS_o = lS_o.to(self.device)
        lS_o += torch.from_numpy(indices_lengths_cumsum[:-1, np.newaxis]).to(
            self.device
        )
        numel = torch.tensor([indices_lengths_cumsum[-1]], dtype=torch.long).to(
            self.device
        )
        lS_o = torch.cat((lS_o.flatten(), numel))

        # create per_sample_weights
        if v_W_l:
            per_sample_weights = torch.cat(
                [a.gather(0, b) for a, b in zip(v_W_l, lS_i)]
            )
        else:
            per_sample_weights = None

        print("\n\n")
        print("fbgemm initialized weights")
        a = self.fbgemm_gpu_emb_bag.split_embedding_weights()[0].flatten().detach().cpu().numpy().tolist()
        for e in range(10):
            print(e, " ", a[e])

        # convert indices to fbgemm_gpu format
        if isinstance(lS_i, torch.Tensor):
            lS_i = [lS_i]
        lS_i = torch.cat(lS_i, dim=0).contiguous().to(self.device)

        # gpu embedding bag op
        ly = self.fbgemm_gpu_emb_bag(lS_i, lS_o, per_sample_weights)
        gos = torch.randn_like(ly).contiguous()
        ly.backward(gos)

        
        self.fbgemm_gpu_emb_bag.flush()

        print("\n\n")
        print("fbgemm updated weights")
        a = self.fbgemm_gpu_emb_bag.split_embedding_weights()[0].flatten().detach().cpu().numpy().tolist()
        for e in range(10):
            print(e, " ", a[e])


        # convert the results to the next layer's input format.
        if isinstance(self.m_spa, list):
            # handle mixed dimensions case.
            ly = [
                ly[:, s:e]
                for (s, e) in zip(self.m_spa_cumsum[:-1], self.m_spa_cumsum[1:])
            ]
        else:
            # handle case in which all tables share the same column dimension.
            cols = self.m_spa
            ntables = len(self.fbgemm_gpu_emb_bag.embedding_specs)
            ly = ly.reshape(-1, ntables, cols).swapaxes(0, 1)
            ly = list(ly)
        
        return ly












def toy1():

    device = torch.device("cuda:" + str(0))
    n = 5 
    m = 8
    ndevices = 0
    k = 0
    torch.manual_seed(0)
    # make emb used to make FBGEMM object!
    emb_l = nn.ModuleList([
        nn.EmbeddingBag(n, m, mode="sum", sparse=False),
        nn.EmbeddingBag(n, m, mode="sum", sparse=False),
        nn.EmbeddingBag(n, m, mode="sum", sparse=False),
    ]).to(device)

    quantize_bits = 32
    learning_rate = 1.0
    fbgemm_gpu_codegen_pref = "Split"
    requires_grad = True
    fbgemm_emb = fbgemm_gpu_emb_bag_wrapper(
        device,
        emb_l,
        m,
        quantize_bits,
        learning_rate,
        fbgemm_gpu_codegen_pref,
        requires_grad
    )

    fbgemm_p = [Parameter(p.data)
    for emb in (
        [e.fbgemm_gpu_emb_bag for e in [fbgemm_emb]]
    )
    for p in emb.split_embedding_weights()]


    torch.manual_seed(0)
    emb_l2 = nn.ModuleList([
        nn.EmbeddingBag(n, m, mode="sum", sparse=False, device = device),
        nn.EmbeddingBag(n, m, mode="sum", sparse=False, device = device),
        nn.EmbeddingBag(n, m, mode="sum", sparse=False, device = device),
    ]).to(device)

    params = []
    params.extend(fbgemm_p)
    params.extend(list(emb_l2.parameters()))
    optimizer = torch.optim.SGD(params, lr=1.0)

    lS_i = [torch.randint(high=n, size=(1,32)).squeeze(0).to(device) for _ in range(3)]
    lS_o = [torch.arange(32).to(device) for _ in range(3)]

    lS_i2=copy.deepcopy(lS_i)
    lS_o2=[torch.arange(32).to(device) for _ in range(3)]



    losseslog = open("/home/ubuntu/repos/torchrec-fork/examples/dlrm/fbgemm_weights", "a")
    E = fbgemm_emb.fbgemm_gpu_emb_bag.split_embedding_weights()[0]
    line = str(E.detach().cpu().numpy().tolist()) + "\n"
    losseslog.write(line)
    losseslog.close()


    ly = torch.cat( 
        fbgemm_emb(lS_o, lS_i, None) 
    )

    ly2 = torch.cat(
        [ e(i,o) for e, i, o in zip(emb_l2, lS_i2, lS_o2)]
    )


    gradient_signal1 = torch.zeros_like(ly)*0.99
    fbgemm_emb.fbgemm_gpu_emb_bag.backward(gradient_signal1)
    gradient_signal2 = torch.zeros_like(ly2)*0.99
    ly2.backward(gradient_signal2)

    losseslog = open("/home/ubuntu/repos/torchrec-fork/examples/dlrm/pytorch_grad", "a")
    E = emb_l2[0].weight.grad
    line = str(E.detach().cpu().numpy().tolist()) + "\n"
    losseslog.write(line)
    losseslog.close()

    losseslog = open("/home/ubuntu/repos/torchrec-fork/examples/dlrm/fbgemm_grad", "a")
    E = fbgemm_emb.fbgemm_gpu_emb_bag.split_embedding_weights()[0]
    line = str(E.detach().cpu().numpy().tolist()) + "\n"
    losseslog.write(line)
    losseslog.close()

    print("done")



################################################################################
################################################################################
################################################################################
################################################################################
################################################################################


def toy2():

    Deviceable = TypeVar("Deviceable", torch.nn.EmbeddingBag, Tensor)


    def round_up(a: int, b: int) -> int:
        return int((a + b - 1) // b) * b

    def to_device(t: Deviceable, use_cpu: bool) -> Deviceable:
        # pyre-fixme[7]: Expected `Deviceable` but got `Union[Tensor,
        #  torch.nn.EmbeddingBag]`.
        return t.cpu() if use_cpu else t.cuda()        

    def b_indices(
        b: Callable[..., torch.Tensor],
        x: torch.Tensor,
        per_sample_weights: Optional[torch.Tensor] = None,
        use_cpu: bool = False,
        do_pooling: bool = True,
    ) -> torch.Tensor:
        (indices, offsets) = get_offsets_from_dense(x)
        if do_pooling:
            return b(
                to_device(indices, use_cpu),
                to_device(offsets, use_cpu),
                per_sample_weights=per_sample_weights,
            )
        else:
            return b(to_device(indices, use_cpu))

    def get_table_batched_offsets_from_dense(
        merged_indices: torch.Tensor, use_cpu: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (T, B, L) = merged_indices.size()
        lengths = np.ones((T, B)) * L
        flat_lengths = lengths.flatten()
        return (
            to_device(merged_indices.contiguous().view(-1), use_cpu),
            to_device(
                torch.tensor(([0] + np.cumsum(flat_lengths).tolist())).long(),
                use_cpu,
            ),
        )

    def get_offsets_from_dense(indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        (B, L) = indices.size()
        return (
            indices.contiguous().view(-1),
            torch.tensor(
                np.cumsum(np.asarray([0] + [L for _ in range(B)])[:-1]).astype(np.int64)
            ),
        )

    class SplitTableBatchedEmbeddingsTest(unittest.TestCase):

        def test_backward_sgd(  # noqa C901
            self,
            T: int = 4,
            D: int = 8,
            B: int = 25,
            log_E: int = 4,
            L: int = 17,
            weights_precision: SparseType = SparseType.FP32,
            weighted: bool = False,
            mixed: bool = False,
            use_cache: bool = True,
            cache_algorithm: split_table_batched_embeddings_ops.CacheAlgorithm = split_table_batched_embeddings_ops.CacheAlgorithm.LRU,
            long_segments: bool = True,
            pooling_mode: split_table_batched_embeddings_ops.PoolingMode = PoolingMode.SUM,
            use_cpu: bool = False,
            exact: bool = True,
        ) -> None:
            # NOTE: cache is not applicable to CPU version.
            assume(not use_cpu or not use_cache)
            # NOTE: limit (T * B * L * D) to avoid timeout for CPU version!
            assume(not use_cpu or T * B * L * D <= 2048)
            assume(not (use_cpu and weights_precision == SparseType.FP16))
            # GPU only does exact sgd
            assume((use_cpu and not long_segments) or exact)
            # No bag ops only work on GPUs, no mixed, no weighted
            emb_op = (
                split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen
            )

            mode = "sum"
            do_pooling = True

            E = int(10 ** log_E)
            if use_cpu:
                D = (D + 15) // 16 * 4
            else:
                D = D * 4
            if not mixed:
                Ds = [D] * T
                Es = [E] * T
            else:
                Ds = [
                    round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
                    for _ in range(T)
                ]
                Es = [
                    np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
                ]
            compute_device = split_table_batched_embeddings_ops.ComputeDevice.CUDA
            if use_cpu:
                managed = [split_table_batched_embeddings_ops.EmbeddingLocation.HOST] * T
                compute_device = split_table_batched_embeddings_ops.ComputeDevice.CPU
            elif use_cache:
                managed = [
                    split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED_CACHING
                ] * T
                if mixed:
                    average_D = sum(Ds) // T
                    for t, d in enumerate(Ds):
                        managed[t] = (
                            split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE
                            if d < average_D
                            else managed[t]
                        )
            else:
                managed = [
                    np.random.choice(
                        [
                            split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                            split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED,
                        ]
                    )
                    for _ in range(T)
                ]
            bs = [
                to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True), use_cpu)
                for (E, D) in zip(Es, Ds)
            ]

            feature_table_map = list(range(T))
            if exact:
                table_to_replicate = T // 2
                bs.insert(table_to_replicate, bs[table_to_replicate])
                feature_table_map.insert(table_to_replicate, table_to_replicate)

            xs = [
                to_device(
                    torch.from_numpy(
                        np.random.choice(range(Es[t]), size=(B, L), replace=exact).astype(
                            np.int64
                        )
                    ),
                    use_cpu,
                )
                for t in feature_table_map
            ]

            if long_segments and L > 0:
                for x in xs:
                    x[:, 0] = 0

            xws = [to_device(torch.randn(size=(B, L)), use_cpu) for _ in range(len(xs))]
            xws_acc_type = copy.deepcopy(xws)

            if weights_precision == SparseType.FP16 and not use_cpu:
                # NOTE: CPU version of torch.nn.EmbeddingBag doesn't support fp16.
                xws = [xw.half() for xw in xws]

            fs = (
                [
                    b_indices(b, x, use_cpu=use_cpu, do_pooling=do_pooling)
                    for (b, x) in zip(bs, xs)
                ]
                if not weighted
                else [
                    b_indices(
                        b,
                        x,
                        per_sample_weights=xw.view(-1),
                        use_cpu=use_cpu,
                        do_pooling=do_pooling,
                    )
                    for (b, x, xw) in zip(bs, xs, xws)
                ]
            )
            gos = [torch.randn_like(f) for f in fs]
            [f.backward(go) for (f, go) in zip(fs, gos)]
            # do SGD update
            lr = 0.05
            if exact:
                # pyre-fixme[61]: `table_to_replicate` may not be initialized here.
                del bs[table_to_replicate]
            new_weights = [(b.weight - b.weight.grad * lr) for b in bs]

            cc = emb_op(
                embedding_specs=[
                    (E, D, M, compute_device) for (E, D, M) in zip(Es, Ds, managed)
                ],
                optimizer=OptimType.EXACT_SGD if exact else OptimType.SGD,
                feature_table_map=feature_table_map,
                learning_rate=lr,
                weights_precision=weights_precision,
                cache_algorithm=cache_algorithm,
                pooling_mode=pooling_mode,
            )

            for t in range(T):
                cc.split_embedding_weights()[t].data.copy_(bs[t].weight)

            x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
            xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)

            """
            passing = True
            a = cc.split_embedding_weights()[0].flatten().detach().cpu().numpy().tolist()
            b = bs[0].weight.flatten().detach().cpu().numpy().tolist()
            for k in range(len(a)):
                if a[k] != b[k]:
                    print(a[k],b[k],"MISMATCH!!")
                    passing = False
            if passing:
                print("Test passed!")
            """

            print("\n\n")
            print("fbgemm initialized weights")
            a = cc.split_embedding_weights()[0].flatten().detach().cpu().numpy().tolist()
            for e in range(10):
                print(e, " ", a[e])

            (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu)
            fc2 = cc(indices, offsets)
            if do_pooling:
                goc = torch.cat([go.view(B, -1) for go in gos], dim=1).contiguous()
            else:
                goc = torch.cat(gos, dim=0).contiguous()
            fc2.backward(goc)
            if use_cache:
                cc.flush()

            """
            passing = True
            a = cc.split_embedding_weights()[0].flatten().detach().cpu().numpy().tolist()
            b = new_weights[0].flatten().detach().cpu().numpy().tolist()
            for k in range(len(a[:100])):
                if a[k] != b[k]:
                    print(a[k],b[k],"MISMATCH!!")
                    passing = False
            if passing:
                print("Test passed!")
            """

            print("\n\n")
            print("fbgemm updated weights")
            a = cc.split_embedding_weights()[0].flatten().detach().cpu().numpy().tolist()
            for e in range(10):
                print(e, " ", a[e])

            print("\n")
            print("nn.EmbeddingBag updated weights")
            a = new_weights[0].flatten().detach().cpu().numpy().tolist()
            for e in range(10):
                print(e, " ", a[e])                

            for t in range(T):
                torch.testing.assert_allclose(
                    cc.split_embedding_weights()[t],
                    # pyre-fixme[16]: `float` has no attribute `half`.
                    new_weights[t].half()
                    if weights_precision == SparseType.FP16 and not use_cpu
                    else new_weights[t],
                    atol=(1.0e-2 if long_segments else 5.0e-3)
                    if weights_precision == SparseType.FP16
                    else 1.0e-5,
                    rtol=2.0e-2 if weights_precision == SparseType.FP16 else 1.0e-5,
                )
    m = SplitTableBatchedEmbeddingsTest()
    m.test_backward_sgd()


if __name__ == "__main__":
    toy1()
