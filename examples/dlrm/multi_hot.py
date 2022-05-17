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

        self.sigma_np = sigma_np
        self.mu_np = mu_np

    def __make_offsets_cache(self, sigma_np, mu_np, index_split_num, ln_emb):
        tables_count = len(ln_emb)
        cache = np.random.normal(mu_np, sigma_np, size=(self.batch_size * index_split_num, len(ln_emb)))
        cache = cache.reshape(self.batch_size, index_split_num, len(ln_emb))

        # cache axes are [batch, offset, table]
        for k, e in enumerate(ln_emb):
            if e < 200:
                cache[:,:,k]=float('-inf')
        cache[:,0,:] = 0
        return cache

    def __make_new_batch(self, lS_o, lS_i):
        batch_size = self.batch_size
        lS_i = lS_i.reshape(-1, batch_size)
        multi_hot_i_l = []

        mh_offsets = self.__make_offsets_cache(
            self.sigma_np, self.mu_np, self.index_split_num, self.ln_emb)

        # Transpose axes to be [table, batch, offset]
        mh_offsets = mh_offsets.transpose((2, 0, 1))

        sparse_feats = lS_i.numpy()
        sparse_feats = np.repeat(sparse_feats[:,:,np.newaxis], self.index_split_num, axis=-1)

        multi_hot_vecs = sparse_feats + mh_offsets

        s = 0
        for k, table_length in enumerate(self.ln_emb):
            mhvec = multi_hot_vecs[k,:,:]
            indices_in_bounds = np.logical_and(mhvec >= 0, mhvec < table_length)
            offsets = np.sum(indices_in_bounds, axis = -1)
            indices = mhvec[indices_in_bounds]

            lS_o[s : s + batch_size] = torch.Tensor(offsets).int()
            multi_hot_i_l.append(torch.Tensor(indices).int())
            s += batch_size

        lS_o.data.copy_(
            torch.cumsum( torch.concat((torch.tensor([0]), lS_o[:-1])), axis=0))
        lS_i = torch.cat(multi_hot_i_l)
        return lS_o, lS_i

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





class multihot_uniform():
    def __init__(
        self,
        multi_hot_size,
        multi_hot_min_table_size,
        ln_emb,
        batch_size,
    ):
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
                # cache[:,:,k]=np.iinfo(np.int32).max  #int('-inf')
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