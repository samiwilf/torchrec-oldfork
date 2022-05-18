
#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torchrec.distributed as trec_dist
from torchrec.datasets.criteo import (  # noqa
    DEFAULT_INT_NAMES,
    DEFAULT_CAT_NAMES,
    CAT_FEATURE_COUNT,
)
from torchrec.inference.model_packager import load_pickle_config
from torchrec.inference.modules import (
    PredictFactory,
    PredictModule,
    quantize_embeddings,
)
from torchrec.models.dlrm import DLRM
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

# OSS Only


@dataclass
class DLRMModelConfig:
    dense_arch_layer_sizes: List[int]
    dense_in_features: int
    embedding_dim: int
    id_list_features_keys: List[str]
    num_embeddings_per_feature: List[int]
    num_embeddings: int
    over_arch_layer_sizes: List[int]


class DLRMPredictModule(PredictModule):
    """
    nn.Module to wrap DLRM model to use for inference.

    Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define SparseArch.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (List[int]): the layer sizes for the DenseArch.
        over_arch_layer_sizes (List[int]): the layer sizes for the OverArch. NOTE: The
            output dimension of the InteractionArch should not be manually specified
            here.
        id_list_features_keys (List[str]): the names of the sparse features. Used to
            construct a batch for inference.
        dense_device: (Optional[torch.device]).
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        id_list_features_keys: List[str],
        dense_device: Optional[torch.device] = None,
    ) -> None:
        module = DLRM(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            over_arch_layer_sizes=over_arch_layer_sizes,
            dense_device=dense_device,
        )
        super().__init__(module, dense_device)

        self.id_list_features_keys: List[str] = id_list_features_keys

    def predict_forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch (Dict[str, torch.Tensor]): currently expects input dense features
                to be mapped to the key "float_features" and input sparse features
                to be mapped to the key "id_list_features".

        Returns:
            Dict[str, torch.Tensor]: output of inference.
        """
        try:
            predictions = self.predict_module(
                batch["float_features"],
                KeyedJaggedTensor(
                    keys=self.id_list_features_keys,
                    lengths=batch["id_list_features.lengths"],
                    values=batch["id_list_features.values"],
                ),
            )
        except Exception as e:
            logger.info(e)
            raise e
        return {"default": predictions.to(torch.device("cpu")).float()}


class DLRMPredictFactory(PredictFactory):
    def __init__(self) -> None:
        self.model_config: DLRMModelConfig = load_pickle_config(
            "config.pkl", DLRMModelConfig
        )

    def create_predict_module(self, rank: int, world_size: int) -> torch.nn.Module:
        logging.basicConfig(level=logging.INFO)
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        trec_dist.DistributedModelParallel.SHARE_SHARDED = True

        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=self.model_config.embedding_dim,
                num_embeddings=self.model_config.num_embeddings_per_feature[feature_idx]
                if self.model_config.num_embeddings is None
                else self.model_config.num_embeddings,
                feature_names=[feature_name],
            )
            for feature_idx, feature_name in enumerate(
                self.model_config.id_list_features_keys
            )
        ]
        ebc = EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))

        module = DLRMPredictModule(
            embedding_bag_collection=ebc,
            dense_in_features=self.model_config.dense_in_features,
            dense_arch_layer_sizes=self.model_config.dense_arch_layer_sizes,
            over_arch_layer_sizes=self.model_config.over_arch_layer_sizes,
            dense_device=device,
        )
        module = quantize_embeddings(module, dtype=torch.qint8, inplace=True)
        return trec_dist.DistributedModelParallel(
            module=module,
            device=device,
            env=trec_dist.ShardingEnv.from_local(world_size, rank),
            init_data_parallel=False,
        )

    def batching_metadata(self) -> Dict[str, str]:
        return {
            "float_features": "dense",
            "id_list_features": "sparse",
        }
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

    Args:
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
        super().__init__()
        self.model = DLRM(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            over_arch_layer_sizes=over_arch_layer_sizes,
            dense_device=dense_device,
        )
        self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    def forward(
        self, batch: Batch
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        logits = self.model(batch.dense_features, batch.sparse_features)
        logits = logits.squeeze()
        loss = self.loss_fn(logits, batch.labels.float())

        return loss, (loss.detach(), logits.detach(), batch.labels.detach())
load("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")
load("@fbcode_macros//build_defs:python_library.bzl", "python_library")
load("//antlir/fbpkg:fbpkg.bzl", "fbpkg")

python_library(
    name = "dlrm_lib",
    srcs = [
        "dlrm_main.py",
    ],
    deps = [
        "fbsource//third-party/pypi/pyre-extensions:pyre-extensions",
        "fbsource//third-party/pypi/tqdm:tqdm",
        "//ai_codesign/benchmarks/dlrm/torchrec_dlrm/data:dlrm_dataloader",
        "//ai_codesign/benchmarks/dlrm/torchrec_dlrm/modules:dlrm_train",
        "//caffe2:torch",
        "//github/third-party/PyTorchLightning/metrics:lib",
        "//torchrec:torchrec",
        "//torchrec/datasets:criteo",
        "//torchrec/datasets:utils",
        "//torchrec/distributed:distributed",
        "//torchrec/distributed:embeddingbag",
        "//torchrec/distributed:model_parallel",
        "//torchrec/distributed:types",
        "//torchrec/modules:embedding_configs",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--undersampling_rate",
        type=float,
        help="Desired proportion of zero-labeled samples to retain (i.e. undersampling zero-labeled rows)."
        " Ex. 0.3 indicates only 30pct of the rows with label 0 will be kept."
        " All rows with label 1 will be kept. Value should be between 0 and 1."
        " When not supplied, no undersampling occurs.",
    )

    train_model = DLRMTrain(
        embedding_bag_collection=EmbeddingBagCollection(
            tables=eb_configs, device=torch.device("meta")
        ),
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=list(map(int, args.dense_arch_layer_sizes.split(","))),
        over_arch_layer_sizes=list(map(int, args.over_arch_layer_sizes.split(","))),
        dense_device=device,
    )
    fused_params = {
        "learning_rate": args.learning_rate,
    }
    sharders = [
        EmbeddingBagCollectionSharder(fused_params=fused_params),
    ]
#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, NamedTuple, Optional

import torch
import torchrec as trec
from hpc.torchrec.models.examples.sparsenn import PreprocInput
from torchrec.models.dlrm import DLRM
from torchrec import EmbeddingBagCollection
from caffe2.torch.fb.hpc.model_inputs import ModelInputs

@dataclass
class DLRMArchConfig:
    dense_arch_layer_sizes: List[int] = field(default_factory=list)
    over_arch_layer_sizes: List[int] = field(default_factory=list)

@dataclass
class DLRMExampleModelConfig:
    inputs: ModelInputs = field(default_factory=ModelInputs)
    embedding_tables: List[trec.EmbeddingBagConfig] = field(default_factory=list)
    arch: DLRMArchConfig = field(default_factory=DLRMArchConfig)


class ModelInput(NamedTuple):
    supervision_label: torch.Tensor
    supervision_weight: torch.Tensor
    float_features: torch.Tensor
    id_list_features: trec.KeyedJaggedTensor


class DLRMExample(torch.nn.Module):
    def __init__(
        self,
        model_config: DLRMExampleModelConfig,
        dense_device: Optional[torch.device] = None,
        sparse_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        ebc = EmbeddingBagCollection(
            tables=model_config.embedding_tables,
            device=sparse_device
        )
        self.model = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=model_config.inputs.float_features.count,
            dense_arch_layer_sizes=model_config.arch.dense_arch_layer_sizes,
            over_arch_layer_sizes=model_config.arch.over_arch_layer_sizes,
            dense_device=dense_device,
        )

    def _forward_impl(self, inputs: ModelInput) -> torch.Tensor:
        logits = self.model(inputs.float_features, inputs.id_list_features)
        return logits.squeeze()

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return self._forward_impl(inputs)

class DLRMExampleTrain(DLRMExample):
    # pyre-ignore [15]
    def forward(
        self, inputs: ModelInput
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        logits = self._forward_impl(inputs)
        pred = torch.sigmoid(logits)

        label = inputs.supervision_label.squeeze()
        weight = inputs.supervision_weight.squeeze()
        loss = torch.mean(
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits, label, weight=weight
            )
        )
        return (loss, (loss.detach(), pred.detach(), label.detach(), weight.detach()))

class DLRMExampleFull(torch.nn.Module):
    def __init__(
        self,
        preproc: torch.nn.Module,
        main: DLRMExample,
    ) -> None:
        super().__init__()
        self.preproc = preproc
        self.main = main

    def forward(
        self, input: "PreprocInput"
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        preproc_res = self.preproc(input)
        float_features = preproc_res.float_features
        id_list_features = preproc_res.id_list_features
        model_input = ModelInput(
            supervision_label=torch.empty(0),
            supervision_weight=torch.empty(0),
            float_features=float_features
            if float_features is not None
            else torch.empty(0),
            id_list_features=id_list_features
            if id_list_features is not None
            else trec.KeyedJaggedTensor.empty(),
        )
        main_res = self.main(model_input)
        return {"prediction": (main_res, main_res)}
#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
from torch import nn
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.mlp import MLP
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


def choose(n: int, k: int) -> int:
    """
    Simple implementation of math.comb for Python 3.7 compatibility.
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


class SparseArch(nn.Module):
    """
    Processes the sparse features of DLRM. Does embedding lookups for all EmbeddingBag
    and embedding features of each collection.

    Args:
        embedding_bag_collection (EmbeddingBagCollection): represents a collection of
            pooled embeddings.

    Example::

        eb1_config = EmbeddingBagConfig(
           name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
           name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_arch = SparseArch(embedding_bag_collection)

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_offsets_sync(
           keys=["f1", "f2"],
           values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
           offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        sparse_embeddings = sparse_arch(features)
    """

    def __init__(self, embedding_bag_collection: EmbeddingBagCollection) -> None:
        super().__init__()
        self.embedding_bag_collection: EmbeddingBagCollection = embedding_bag_collection
        assert (
            self.embedding_bag_collection.embedding_bag_configs
        ), "Embedding bag collection cannot be empty!"
        self.D: int = self.embedding_bag_collection.embedding_bag_configs[
            0
        ].embedding_dim
        self._sparse_feature_names: List[str] = [
            name
            for conf in embedding_bag_collection.embedding_bag_configs
            for name in conf.feature_names
        ]

        self.F: int = len(self._sparse_feature_names)

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
            features (KeyedJaggedTensor): an input tensor of sparse features.

        Returns:
            torch.Tensor: tensor of shape B X F X D.
        """

        sparse_features: KeyedTensor = self.embedding_bag_collection(features)

        B: int = features.stride()

        sparse: Dict[str, torch.Tensor] = sparse_features.to_dict()
        sparse_values: List[torch.Tensor] = []
        for name in self.sparse_feature_names:
            sparse_values.append(sparse[name])

        return torch.cat(sparse_values, dim=1).reshape(B, self.F, self.D)

    @property
    def sparse_feature_names(self) -> List[str]:
        return self._sparse_feature_names


class DenseArch(nn.Module):
    """
    Processes the dense features of DLRM model.

    Args:
        in_features (int): dimensionality of the dense input features.
        layer_sizes (List[int]): list of layer sizes.
        device (Optional[torch.device]): default compute device.

    Example::

        B = 20
        D = 3
        dense_arch = DenseArch(10, layer_sizes=[15, D])
        dense_embedded = dense_arch(torch.rand((B, 10)))
    """

    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model: nn.Module = MLP(
            in_features, layer_sizes, bias=True, activation="relu", device=device
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): an input tensor of dense features.

        Returns:
            torch.Tensor: an output tensor of size B X D.
        """
        return self.model(features)


class InteractionArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features). Returns the pairwise dot product of each sparse feature pair,
    the dot product of each sparse features with the output of the dense layer,
    and the dense layer itself (all concatenated).

    .. note::
        The dimensionality of the `dense_features` (D) is expected to match the
        dimensionality of the `sparse_features` so that the dot products between them
        can be computed.


    Args:
        num_sparse_features (int): F.

    Example::

        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionArch(num_sparse_features=len(keys))

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        #  B X (D + F + F choose 2)
        concat_dense = inter_arch(dense_features, sparse_features)
    """

    def __init__(self, num_sparse_features: int) -> None:
        super().__init__()
        self.F: int = num_sparse_features
        self.triu_indices: torch.Tensor = torch.triu_indices(
            self.F + 1, self.F + 1, offset=1
        )

    def forward(
        self, dense_features: torch.Tensor, sparse_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): an input tensor of size B X D.
            sparse_features (torch.Tensor): an input tensor of size B X F X D.

        Returns:
            torch.Tensor: an output tensor of size B X (D + F + F choose 2).
        """
        if self.F <= 0:
            return dense_features
        (B, D) = dense_features.shape

        combined_values = torch.cat(
            (dense_features.unsqueeze(1), sparse_features), dim=1
        )

        # dense/sparse + sparse/sparse interaction
        # size B X (F + F choose 2)
        interactions = torch.bmm(
            combined_values, torch.transpose(combined_values, 1, 2)
        )
        interactions_flat = interactions[:, self.triu_indices[0], self.triu_indices[1]]

        return torch.cat((dense_features, interactions_flat), dim=1)


class OverArch(nn.Module):
    """
    Final Arch of DLRM - simple MLP over OverArch.

    Args:
        in_features (int): size of the input.
        layer_sizes (List[int]): sizes of the layers of the `OverArch`.
        device (Optional[torch.device]): default compute device.

    Example::

        B = 20
        D = 3
        over_arch = OverArch(10, [5, 1])
        logits = over_arch(torch.rand((B, 10)))
    """

    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if len(layer_sizes) <= 1:
            raise ValueError("OverArch must have multiple layers.")
        self.model: nn.Module = nn.Sequential(
            MLP(
                in_features,
                layer_sizes[:-1],
                bias=True,
                activation="relu",
                device=device,
            ),
            nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=True, device=device),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor):

        Returns:
            torch.Tensor: size B X layer_sizes[-1]
        """
        return self.model(features)


class DLRM(nn.Module):
    """
    Recsys model from "Deep Learning Recommendation Model for Personalization and
    Recommendation Systems" (https://arxiv.org/abs/1906.00091). Processes sparse
    features by learning pooled embeddings for each feature. Learns the relationship
    between dense features and sparse features by projecting dense features into the
    same embedding space. Also, learns the pairwise relationships between sparse
    features.

    The module assumes all sparse features have the same embedding dimension
    (i.e. each EmbeddingBagConfig uses the same embedding_dim).

    The following notation is used throughout the documentation for the models:

    * F: number of sparse features
    * D: embedding_dimension of sparse features
    * B: batch size
    * num_features: number of dense features

    Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define `SparseArch`.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (List[int]): the layer sizes for the `DenseArch`.
        over_arch_layer_sizes (List[int]): the layer sizes for the `OverArch`.
            The output dimension of the `InteractionArch` should not be manually
            specified here.
        dense_device (Optional[torch.device]): default compute device.

    Example::

        B = 2
        D = 8

        eb1_config = EmbeddingBagConfig(
           name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
           name="t2",
           embedding_dim=D,
           num_embeddings=100,
           feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        model = DLRM(
           embedding_bag_collection=ebc,
           dense_in_features=100,
           dense_arch_layer_sizes=[20],
           over_arch_layer_sizes=[5, 1],
        )

        features = torch.rand((B, 100))

        #     0       1
        # 0   [1,2] [4,5]
        # 1   [4,3] [2,9]
        # ^
        # feature
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
           keys=["f1", "f3"],
           values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
           offsets=torch.tensor([0, 2, 4, 6, 8]),
        )

        logits = model(
           dense_features=features,
           sparse_features=sparse_features,
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
        super().__init__()
        assert (
            len(embedding_bag_collection.embedding_bag_configs) > 0
        ), "At least one embedding bag is required"
        for i in range(1, len(embedding_bag_collection.embedding_bag_configs)):
            conf_prev = embedding_bag_collection.embedding_bag_configs[i - 1]
            conf = embedding_bag_collection.embedding_bag_configs[i]
            assert (
                conf_prev.embedding_dim == conf.embedding_dim
            ), "All EmbeddingBagConfigs must have the same dimension"
        embedding_dim: int = embedding_bag_collection.embedding_bag_configs[
            0
        ].embedding_dim
        if dense_arch_layer_sizes[-1] != embedding_dim:
            raise ValueError(
                f"embedding_bag_collection dimension ({embedding_dim}) and final dense "
                "arch layer size ({dense_arch_layer_sizes[-1]}) must match."
            )

        self.sparse_arch: SparseArch = SparseArch(embedding_bag_collection)
        num_sparse_features: int = len(self.sparse_arch.sparse_feature_names)

        self.dense_arch = DenseArch(
            in_features=dense_in_features,
            layer_sizes=dense_arch_layer_sizes,
            device=dense_device,
        )
        self.inter_arch = InteractionArch(num_sparse_features=num_sparse_features)

        over_in_features: int = (
            embedding_dim + choose(num_sparse_features, 2) + num_sparse_features
        )
        self.over_arch = OverArch(
            in_features=over_in_features,
            layer_sizes=over_arch_layer_sizes,
            device=dense_device,
        )

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): the dense features.
            sparse_features (KeyedJaggedTensor): the sparse features.

        Returns:
            torch.Tensor: logits.
        """
        embedded_dense = self.dense_arch(dense_features)
        embedded_sparse = self.sparse_arch(sparse_features)
        concatenated_dense = self.inter_arch(
            dense_features=embedded_dense, sparse_features=embedded_sparse
        )
        logits = self.over_arch(concatenated_dense)
        return logits