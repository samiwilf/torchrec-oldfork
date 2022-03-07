#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Optional, List

import torch
from torch import nn
from torchrec.datasets.utils import Batch
from torchrec.models.dlrm import DLRM
from torchrec.modules.embedding_modules import EmbeddingBagCollection

import sqlite3
import struct
import numpy as np
def setup_db_connection():

    cache_table_name = "IO_DATA"

    cache_file_name = "/home/ubuntu/dlrm_ml_cache.db"
    db_connection = sqlite3.connect(cache_file_name)
    db_connection.execute('pragma journal_mode=wal')
    db_cursor = db_connection.cursor()
    db_cursor.execute("CREATE TABLE IF NOT EXISTS IO_DATA (minibatch_size INT, X blob, lS_o blob, lS_i blob, lS_i_lengths blob, Z blob, T blob, ln_emb blob)")    
    cache_exists = db_cursor.execute("SELECT count(*) FROM IO_DATA").fetchone()[0]
    cache_exists = True if cache_exists > 0 else False
    if not cache_exists:
        print("Creating ",cache_file_name)
    else:
        print("Using ", cache_file_name)    
    return db_connection, cache_exists, cache_table_name
def disk_cache_get(db_connection, cache_table_name = "IO_DATA"):
    db_cursor = db_connection.cursor()

    db_cursor.execute("SELECT * FROM " + cache_table_name)
    records = db_cursor.fetchall()
    minibatch_size = int(records[0][0])

    train_ld = []
    for record in records:
        (X, lS_o, lS_i_flat, lS_i_lengths, Z, T, ln_emb) = [np.frombuffer(rec, dtype=np.double) for rec in record[1:]]
        lS_i_flat = lS_i_flat.copy()
        lS_i_flat[:]=0
        ln_emb = ln_emb.astype(int)
        segment_indices_list = np.cumsum([0] + list(map(int, lS_i_lengths))).tolist()
        lS_i = [torch.tensor(lS_i_flat[s:e].tolist(), dtype=torch.long) for s, e in zip(segment_indices_list[:-1], segment_indices_list[1:])]
        row_size = minibatch_size
        lS_o = [torch.tensor(lS_o[x:x+row_size].tolist(), dtype=torch.long) for x in range(0,len(lS_o),row_size)]
        X = torch.tensor(X.reshape(minibatch_size, -1).tolist(), dtype=torch.float32)
        T = torch.tensor(T.reshape(minibatch_size, -1).tolist(), dtype=torch.float32)
        Z = torch.tensor(Z.reshape(minibatch_size, -1).tolist(), dtype=torch.float32)
        train_ld.append((X, lS_o, lS_i, Z, T))
    test_ld = train_ld
    return (train_ld, test_ld, ln_emb)

class DLRMTrain(nn.Module):
    """
    nn.Module to wrap DLRM model to use with train_pipeline.

    DLRM Recsys model from "Deep Learning Recommendation Model for Personalization and
    Recommendation Systems" (https://arxiv.org/abs/1906.00091). Processes sparse
    features by learning pooled embeddings for each feature. Learns the relationship
    between dense features and sparse features by projecting dense features into the
    same embedding space. Also, learns the pairwise relationships between sparse
    features.

    The module assumes all sparse features have the same embedding dimension
    (i.e, each EmbeddingBagConfig uses the same embedding_dim)

    Constructor Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define SparseArch.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (list[int]): the layer sizes for the DenseArch.
        over_arch_layer_sizes (list[int]): the layer sizes for the OverArch. NOTE: The
            output dimension of the InteractionArch should not be manually specified
            here.
        dense_device: (Optional[torch.device]).

    Call Args:
        batch: batch used with criteo and random data from torchrec.datasets

    Returns:
        Tuple[loss, Tuple[loss, logits, labels]]

    Example::

        ebc = EmbeddingBagCollection(config=ebc_config)
        model = DLRMTrain(
           embedding_bag_collection=ebc,
           dense_in_features=100,
           dense_arch_layer_sizes=[20],
           over_arch_layer_sizes=[5, 1],
        )
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        dense_device: Optional[torch.device] = None,
    ) -> None:
        #self.db_connection, disk_cache_exists, db_cache_table_name = setup_db_connection()
        #self.train_ld, self.test_ld, self.ln_emb = disk_cache_get(self.db_connection)
        #self.nbatches = len(self.train_ld)    
        super().__init__()
        self.model = DLRM(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            over_arch_layer_sizes=over_arch_layer_sizes,
            dense_device=dense_device,
        )
        #self.loss_fn: nn.Module = torch.nn.BCEWithLogitsLoss()
        self.loss_fn: nn.Module = torch.nn.BCELoss(reduction="mean")

        # set all parameters to 1
        #for p in list(self.model.parameters()):
        #    p.data.flatten()[:] = 0.1 #torch.ones(p.data.size(), dtype = p.data.dtype, device=p.data.device)

        # Already set in batched_embeddings_kernel.py param.data.uniform_ call.
        # for name, buffer in self.model.sparse_arch.embedding_bag_collection.named_buffers():
        #     print(name, buffer.shape)
        #     if 't_cat_' in name:
        #         buffer.data.flatten()[:]=1.0

        #print(list(self.model.sparse_arch.embedding_bag_collection.named_buffers())[0][1].data.flatten()[:10])
        #for p in self.model.sparse_arch.embedding_bag_collection.fused_optimizer.param_groups:
        #    p['lr'] = 1.00            

    def forward(
        self, batch: Batch
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        if False:
            d = batch.dense_features.device
            batch.dense_features = self.train_ld[1][0].to(d)            
            lS_o = self.train_ld[1][1]
            lS_i = self.train_ld[1][2]
            lengths_list = list(map(len, lS_i))
            indices_lengths_cumsum = np.cumsum([0] + lengths_list)

            lS_o = torch.stack(lS_o)
            lS_o += torch.from_numpy(indices_lengths_cumsum[:-1, np.newaxis])
            numel = torch.tensor([indices_lengths_cumsum[-1]], dtype=torch.long)
            lS_o = torch.cat((lS_o.flatten(), numel))
            
            batch.sparse_features._values.data = torch.concat(lS_i).to(d)
            batch.sparse_features._offsets = lS_o.to(d)
            batch.sparse_features._lengths = (lS_o[1:] - lS_o[:-1]).to(d) # == list of sizes of each offset group
            batch.sparse_features._length_per_key = lengths_list # == list of lengths of each embedding tables' embedding bag op output
            batch.sparse_features._offset_per_key = indices_lengths_cumsum.tolist()

        
        #try:
        #    batch.dense_features.flatten()[:]=1.0
        #    batch.sparse_features._values.data[:]=0
        #except:
        #    pass
        logits = self.model(batch.dense_features, batch.sparse_features)
        logits = torch.sigmoid(logits)
        #logits = logits.squeeze()
        logits = logits.reshape(-1)
        loss = self.loss_fn(logits, batch.labels.float())        
        return loss, (loss.detach(), logits.detach(), batch.labels.detach())
