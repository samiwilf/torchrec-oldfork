import torch
import numpy as np
from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

import multi_hot_hasher

class multihot_uniform():
    def __init__(
        self,
        multi_hot_size,
        multi_hot_min_table_size,
        ln_emb,
        batch_size,
        use_sha = False,
    ):
        self.use_sha = use_sha
        self.multi_hot_min_table_size = multi_hot_min_table_size
        self.multi_hot_size = multi_hot_size
        self.batch_size = batch_size
        self.ln_emb = ln_emb
        self.cache_vectors_count = 100000
        self.cache = self.__make_offsets_cache(multi_hot_size, ln_emb)

    def __make_offsets_cache(self, multi_hot_size, ln_emb):
        cache = np.zeros((len(ln_emb), self.cache_vectors_count, self.multi_hot_size))
        for k, e in enumerate(ln_emb):
            np.random.seed(k)
            cache[k,:,:] = np.random.randint(0,e, size=(self.cache_vectors_count, self.multi_hot_size))
        cache = cache.astype(float)
        # cache axes are [table, batch, offset]
        for k, e in enumerate(ln_emb):
            if e < self.multi_hot_min_table_size:
                cache[k,:,:]=float('-inf')
        # cache[:,0,:] = 0
        cache = torch.from_numpy(cache)
        return cache

    def __make_new_batch(self, lS_o, lS_i):
        batch_size = self.batch_size
        lS_i = lS_i.reshape(-1, batch_size)

        if 1 < self.multi_hot_size:
            multi_hot_i_l = []
            for cf, table_length in enumerate(self.ln_emb):
                if table_length >= self.multi_hot_min_table_size:

                    if self.use_sha:
                        multi_hot_i = multi_hot_hasher.meta_1_hot_hasher(cf, table_length, batch_size, lS_i[cf].numpy(), self.multi_hot_size, self.use_sha )
                        multi_hot_i = torch.tensor(np.array(multi_hot_i).reshape(-1))
                    else:
                        keys = lS_i[cf] % self.cache_vectors_count
                        multi_hot_i = torch.nn.functional.embedding(keys, self.cache[cf])
                        multi_hot_i = (multi_hot_i + lS_i[cf].unsqueeze(-1)) % table_length
                        multi_hot_i = multi_hot_i.reshape(-1).int()

                    multi_hot_i_l.append(multi_hot_i)
                    lS_o[cf*batch_size : (cf+1)*batch_size] = self.multi_hot_size
                else:
                    multi_hot_i_l.append(lS_i[cf])
                    lS_o[cf*batch_size : (cf+1)*batch_size] = 1.0
            lS_o.data.copy_(
                torch.cumsum( torch.concat((torch.tensor([0]), lS_o[:-1])), axis=0))
            lS_i = torch.cat(multi_hot_i_l)
            return lS_o, lS_i
        else:
            return lS_o, torch.cat(lS_i)

    def convert_to_multi_hot(self, batch: Batch) -> Batch:

        lS_i = batch.sparse_features._values
        lS_o = batch.sparse_features._offsets
        lS_o, lS_i = self.__make_new_batch(lS_o, lS_i)

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