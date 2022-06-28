import torch
import numpy as np
from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class Multihot():
    def __init__(
        self,
        multi_hot_size,
        multi_hot_min_table_size,
        ln_emb,
        batch_size,
        collect_freqs_stats,
        type: str = "uniform",
        weighted_pooling: bool = False,
    ):
        if type != "uniform" and type != "pareto":
            raise ValueError(
                "Multi-hot distribution type {} is not supported."
                "Only \"uniform\" and \"pareto\" are supported.".format(type)
            )
        #self.dist_type = 'pareto'
        self.dist_type = type
        self.weighted_pooling = weighted_pooling

        self.multi_hot_min_table_size = multi_hot_min_table_size
        self.multi_hot_size = multi_hot_size
        self.batch_size = batch_size
        self.ln_emb = ln_emb
        self.cache_vectors_count = 10000
        self.lS_i_offsets_cache = self.__make_indices_offsets_cache(multi_hot_size, ln_emb, self.cache_vectors_count)
        self.lS_o_cache = self.__make_offsets_cache(multi_hot_size, multi_hot_min_table_size, ln_emb, batch_size)



        weights_l = []
        for table_length in ln_emb:
            if table_length < self.multi_hot_min_table_size:
                weights = torch.ones((batch_size, 1), dtype=torch.float32)
            else:
                weights = torch.ones((batch_size, multi_hot_size), dtype=torch.float32)
                weights[:,1:] *= 0.20 / (multi_hot_size-1)
                weights[:,0] *= 0.8
            weights_l.append(weights.reshape(-1))
        #weights = [ torch.ones((multi_hot_size))*0.20/multi_hot_size for rows_count in ln_emb ]
        self.weighted_pooling_tensor = torch.concat(weights_l, axis=0)

        if weighted_pooling == False:
            self.weighted_pooling_tensor = None


        # For plotting frequency access
        self.collect_freqs_stats = collect_freqs_stats
        self.collect_freqs_stats_temp_disable = False
        self.freqs_pre_hash = []
        self.freqs_post_hash = []
        for row_count in ln_emb:
            self.freqs_pre_hash.append(np.zeros((row_count)))
            self.freqs_post_hash.append(np.zeros((row_count)))

    def save_freqs_stats(self, rank):
        pre_dict = {str(k) : e for k, e in enumerate(self.freqs_pre_hash)}
        np.save(f"stats_pre_hash_{self.dist_type}.npy", pre_dict)
        post_dict = {str(k) : e for k, e in enumerate(self.freqs_post_hash)}
        np.save(f"stats_post_hash_{self.dist_type}.npy", post_dict)

    def __make_indices_offsets_cache(self, multi_hot_size, ln_emb, cache_vectors_count):
        cache = []
        for k, e in enumerate(ln_emb):
            np.random.seed(k) # The seed is necessary for all ranks produce the same lookup values.
            if self.dist_type == "pareto":
                cache.append(np.random.pareto(a=0.25, size=(e, multi_hot_size)).astype(np.int32) % e)
            else:
                cache.append(np.random.randint(0, e, size=(cache_vectors_count, multi_hot_size)))
        # cache axes are [table, batch, offset]
        cache = [ torch.from_numpy(table_cache).int() for table_cache in cache ]
        return cache
    def __make_offsets_cache(self, multi_hot_size, multi_hot_min_table_size, ln_emb, batch_size):
        lS_o = torch.ones((len(ln_emb) * self.batch_size), dtype=torch.int32)
        for cf, table_length in enumerate(ln_emb):
            if table_length >= multi_hot_min_table_size:
                lS_o[cf*batch_size : (cf+1)*batch_size] = multi_hot_size
        lS_o = torch.cumsum( torch.concat((torch.tensor([0]), lS_o)), axis=0)
        return lS_o

    def __make_new_batch(self, lS_o, lS_i):
        batch_size = self.batch_size
        lS_i = lS_i.reshape(-1, batch_size)

        if 1 < self.multi_hot_size:
            multi_hot_i_l = []
            for cf, table_length in enumerate(self.ln_emb):
                if table_length < self.multi_hot_min_table_size:
                    multi_hot_i_l.append(lS_i[cf])
                else:
                    if self.dist_type == 'uniform':
                        keys = lS_i[cf] % self.cache_vectors_count
                        multi_hot_i_offsets = torch.nn.functional.embedding(keys, self.lS_i_offsets_cache[cf])
                        multi_hot_i = (multi_hot_i_offsets + lS_i[cf].unsqueeze(-1)) % table_length
                    else:
                        keys = lS_i[cf]
                        multi_hot_i = torch.nn.functional.embedding(keys, self.lS_i_offsets_cache[cf])
                        multi_hot_i[:,0] = keys
                    multi_hot_i = multi_hot_i.reshape(-1)
                    multi_hot_i_l.append(multi_hot_i)
                    if self.collect_freqs_stats:
                        self.freqs_pre_hash[cf][lS_i[cf]] += 1
                        self.freqs_post_hash[cf][multi_hot_i] += 1
            lS_i = torch.cat(multi_hot_i_l)
            return self.lS_o_cache, lS_i
        else:
            lS_i = torch.cat(lS_i)
            return self.lS_o, lS_i

    def convert_to_multi_hot(self, batch: Batch) -> Batch:
        lS_i = batch.sparse_features._values
        lS_o = batch.sparse_features._offsets
        lS_o, lS_i = self.__make_new_batch(lS_o, lS_i)


        new_sparse_features=KeyedJaggedTensor.from_offsets_sync(
            keys=batch.sparse_features._keys,
            values=lS_i,
            weights=self.weighted_pooling_tensor,
            offsets=lS_o,
        )
        return Batch(
            dense_features=batch.dense_features,
            sparse_features=new_sparse_features,
            labels=batch.labels,
        )