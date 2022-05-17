import torch
import numpy as np
from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

class multihot():
    def __init__(
        self,
        index_split_num,
        ln_emb,
        batch_size
    ):
        self.index_split_num = index_split_num
        self.batch_size = batch_size
        self.ln_emb = ln_emb

        sigma_np = np.array([(rows_count)//20 for rows_count in ln_emb])
        mu_np = np.array([0.0 for _ in ln_emb])
        cache_l = [
            np.random.normal(mu_np, sigma_np, size=(index_split_num - 1, len(mu_np)))
            for _ in range(2048)
        ]
        for i, cache in enumerate(cache_l):
            for k, e in enumerate(ln_emb):
                if e < 200:
                    cache[:,k]=float('-inf')
            cache = np.append(mu_np[np.newaxis,:], cache, axis=0)
            cache_l[i] = np.sort(cache, axis=0)

        self.caches = np.stack(cache_l)


    def make_new_batch(self, lS_o, lS_i):
        ln_emb = self.ln_emb
        batch_size = self.batch_size
        lS_i = lS_i.reshape(-1, batch_size)
        multi_hot_i_l = []
        s = 0
        for k, e in enumerate(self.ln_emb):

            b = lS_i[k,:].numpy()
            b = np.repeat(b[:, np.newaxis], self.index_split_num, axis=-1)

            a = self.caches[:,:,k]
            # a = caches[0,:,k] # Use for single offset vector per table.

            c = a + b

            offsets = np.sum( np.logical_and(c >= 0, c < ln_emb[k]), axis = -1)
            indices = c[ np.logical_and(c>=0, c < ln_emb[k]) ]

            lS_o[s : s + batch_size] = torch.tensor(offsets[:]).int()
            multi_hot_i_l.append(torch.Tensor(indices).int())
            s += batch_size

        lS_o.data.copy_(
            torch.cumsum( torch.concat((torch.tensor([0]), lS_o[:-1])), axis=0))

        lS_i = torch.cat(multi_hot_i_l)
        return lS_o, lS_i

    def convert_to_multi_hot(self, batch: Batch) -> Batch:

        lS_i = batch.sparse_features._values
        lS_o = batch.sparse_features._offsets
        lS_o, lS_i = self.make_new_batch(lS_o, lS_i)

        new_sparse_features=KeyedJaggedTensor.from_offsets_sync(
            keys=batch.sparse_features._keys,
            values=lS_i,
            offsets=lS_o,
        )
        return Batch(
            dense_features=batch.dense_features,
            sparse_features=new_sparse_features,
            labels=batch.labels,
        )
