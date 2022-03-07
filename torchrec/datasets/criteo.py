#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from typing import (
    Iterator,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
    Tuple,
)

import torchrec.mysettings as mysettings
from torchrec.mysettings import (
    ARGV,
    INT_FEATURE_COUNT,
    CAT_FEATURE_COUNT,
    DAYS,
    BATCH_SIZE,
    SETTING,
    LOG_FILE,
)

import numpy as np
import torch
import torch.utils.data.datapipes as dp
from iopath.common.file_io import PathManagerFactory, PathManager
from pyre_extensions import none_throws
from torch.utils.data import IterDataPipe, IterableDataset
from torchrec.datasets.utils import (
    LoadFiles,
    ReadLinesFromCSV,
    safe_cast,
    PATH_MANAGER_KEY,
    Batch,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

FREQUENCY_THRESHOLD = 3
INT_FEATURE_COUNT = mysettings.INT_FEATURE_COUNT 
CAT_FEATURE_COUNT = mysettings.CAT_FEATURE_COUNT 
#INT_FEATURE_COUNT = 13
#CAT_FEATURE_COUNT = 26
DAYS = mysettings.DAYS #1#24
DEFAULT_LABEL_NAME = "label"
DEFAULT_INT_NAMES: List[str] = [f"int_{idx}" for idx in range(INT_FEATURE_COUNT)]
DEFAULT_CAT_NAMES: List[str] = [f"cat_{idx}" for idx in range(CAT_FEATURE_COUNT)]
DEFAULT_COLUMN_NAMES: List[str] = [
    DEFAULT_LABEL_NAME,
    *DEFAULT_INT_NAMES,
    *DEFAULT_CAT_NAMES,
]

COLUMN_TYPE_CASTERS: List[Callable[[Union[int, str]], Union[int, str]]] = [
    lambda val: safe_cast(val, int, 0),
    *(lambda val: safe_cast(val, int, 0) for _ in range(INT_FEATURE_COUNT)),
    *(lambda val: safe_cast(val, str, "") for _ in range(CAT_FEATURE_COUNT)),
]


def _default_row_mapper(example: List[str]) -> Dict[str, Union[int, str]]:
    column_names = reversed(DEFAULT_COLUMN_NAMES)
    column_type_casters = reversed(COLUMN_TYPE_CASTERS)
    return {
        next(column_names): next(column_type_casters)(val) for val in reversed(example)
    }

# From OSS
import math
import sys
import pathlib
import os
from os import path
class DataLoader:
    """
    DataLoader dedicated for the Criteo Terabyte Click Logs dataset
    """

    def __init__(
            self,
            rank,
            world_size,
            data_filename,
            data_directory,
            days,
            batch_size,
            max_ind_range=-1,
            split="train",
            drop_last_batch=False
    ):
        self.data_filename = data_filename
        self.data_directory = data_directory
        self.days = days
        self.batch_size = batch_size
        self.max_ind_range = max_ind_range

        total_file = os.path.join(
            data_directory,
            data_filename + "_day_count.npz"
        )

        with np.load(total_file) as data:
            total_per_file = data["total_per_file"][np.array(days)]
        self.total_per_file = total_per_file

        self.length = sum(total_per_file)
        #if split == "test" or split == "val":
        #    self.length = int(np.ceil(self.length / 2.))
        self.split = split
        self.drop_last_batch = drop_last_batch

        # compute offsets per file
        self.offset_per_file = np.array([0] + [x for x in total_per_file])
        for i, d in enumerate(days):
            self.offset_per_file[i + 1] += self.offset_per_file[i]

        self.rank = rank
        self.rows_per_rank = self.length // world_size
        self.start_row = rank * self.rows_per_rank
        self.last_row = self.start_row + self.rows_per_rank - 1

    def __iter__(self):
        return iter(
            self.__batch_generator(
                self.data_filename, self.data_directory, self.days,
                self.batch_size, self.split, self.drop_last_batch, self.max_ind_range
            )
        )

    def __len__(self):
        if self.drop_last_batch:
            return self.length // self.batch_size
        else:
            return math.ceil(self.length / self.batch_size)
    #From OSS
    def __transform_features(self,
            x_int_batch, x_cat_batch, y_batch, max_ind_range, flag_input_torch_tensor=False
    ):
        if max_ind_range > 0:
            x_cat_batch = x_cat_batch % max_ind_range

        if flag_input_torch_tensor:
            x_int_batch = torch.log(x_int_batch.clone().detach().type(torch.float) + 1)
            x_cat_batch = x_cat_batch.clone().detach().type(torch.long)
            y_batch = y_batch.clone().detach().type(torch.float32).view(-1, 1)
        else:
            x_int_batch = torch.log(torch.tensor(x_int_batch, dtype=torch.float) + 1)
            x_cat_batch = torch.tensor(x_cat_batch, dtype=torch.long)
            y_batch = torch.tensor(y_batch, dtype=torch.float32).view(-1, 1)

        batch_size = x_cat_batch.shape[0]
        feature_count = x_cat_batch.shape[1]
        lS_o = torch.arange(batch_size).reshape(1, -1).repeat(feature_count, 1)

        return x_int_batch, lS_o, x_cat_batch.t(), y_batch.view(-1, 1)
    # From OSS
    def __batch_generator(
            self, data_filename, data_directory, days, batch_size, split, drop_last, max_ind_range
    ):
        previous_file = None
        for day in days:
            filepath = os.path.join(
                data_directory,
                data_filename + "_{}_reordered.npz".format(day)
            )

            print('Loading file: ', filepath)
            print(f"Rank {self.rank} Reading day {filepath[:-4]+'***.npy'}")
            x_int = np.load(str(filepath)[:-4] + "_int.npy")
            x_cat = np.load(str(filepath)[:-4] + "_cat.npy")                
            y = np.load(str(filepath)[:-4] + "_cat.npy")

            samples_in_file = y.shape[0]
            batch_start_idx = 0
            if split == "test" or split == "val":
                length = int(np.ceil(samples_in_file / 2.))
                if split == "test":
                    samples_in_file = length
                elif split == "val":
                    batch_start_idx = samples_in_file - length

            while batch_start_idx < samples_in_file - batch_size:

                missing_samples = batch_size
                if previous_file is not None:
                    missing_samples -= previous_file['y'].shape[0]

                current_slice = slice(batch_start_idx, batch_start_idx + missing_samples)

                x_int_batch = x_int[current_slice]
                x_cat_batch = x_cat[current_slice]
                y_batch = y[current_slice]

                if previous_file is not None:
                    x_int_batch = np.concatenate(
                        [previous_file['x_int'], x_int_batch],
                        axis=0
                    )
                    x_cat_batch = np.concatenate(
                        [previous_file['x_cat'], x_cat_batch],
                        axis=0
                    )
                    y_batch = np.concatenate([previous_file['y'], y_batch], axis=0)
                    previous_file = None

                if x_int_batch.shape[0] != batch_size:
                    raise ValueError('should not happen')

                yield self.__transform_features(x_int_batch, x_cat_batch, y_batch, max_ind_range)

                batch_start_idx += missing_samples
            if batch_start_idx != samples_in_file:
                current_slice = slice(batch_start_idx, samples_in_file)
                if previous_file is not None:
                    previous_file = {
                        'x_int' : np.concatenate(
                            [previous_file['x_int'], x_int[current_slice]],
                            axis=0
                        ),
                        'x_cat' : np.concatenate(
                            [previous_file['x_cat'], x_cat[current_slice]],
                            axis=0
                        ),
                        'y' : np.concatenate([previous_file['y'], y[current_slice]], axis=0)
                    }
                else:
                    previous_file = {
                        'x_int' : x_int[current_slice],
                        'x_cat' : x_cat[current_slice],
                        'y' : y[current_slice]
                    }

        if not drop_last:
            yield self.__transform_features(
                previous_file['x_int'],
                previous_file['x_cat'],
                previous_file['y'],
                max_ind_range
            )    
    def __batch_generator2(self,
            data_filename, data_directory, days, batch_size, split, drop_last, max_ind_range
    ):
        track_size = 0
        gid = 0
        while gid < self.length:
            x_int_batch, x_cat_batch, y_batch = [],[],[]
            day_prev = day = np.digitize(gid, self.offset_per_file) - 1
            read_file = self.start_row <= self.offset_per_file[day+1] and \
                self.offset_per_file[day] < self.last_row
            filepath = os.path.join(
                data_directory,
                data_filename + "_{}_reordered.npz".format(day)
            )
            print(f"Rank {self.rank} Reading day {filepath} {read_file} START!")
            #data = np.load(filepath) if read_file else None
            if read_file:
                with np.load(filepath) as data:
                    x_int = data["X_int"]
                    x_cat = data["X_cat"]
                    y = data["y"]
            print(f"Rank {self.rank} Reading day {filepath} {read_file} DONE!")
            while day == day_prev:                
                s = gid - self.offset_per_file[day]
                e = min(gid + batch_size - track_size, self.offset_per_file[day+1]) - self.offset_per_file[day]
                track_size += e - s
                current_slice = slice(s, e)
                if read_file:
                    print("read 1 start")        
                    x_int_batch.append(x_int[current_slice])
                    print("read 1 append done")
                    x_cat_batch.append(x_cat[current_slice])
                    print("read 2 append done")
                    y_batch.append(y[current_slice])
                    print("appended something")
                else:
                    x_int_batch.append(np.empty((e-s,13)))
                    x_cat_batch.append(np.empty((e-s,26)))
                    y_batch.append(np.empty((e-s))) 
                    print("append nothin")                   
                if track_size == batch_size:
                    track_size = 0
                    x_int_batch = np.concatenate(x_int_batch)
                    x_cat_batch = np.concatenate(x_cat_batch)
                    y_batch = np.concatenate(y_batch)
                    yield self.__transform_features(x_int_batch, x_cat_batch, y_batch, max_ind_range)
                    x_int_batch, x_cat_batch, y_batch = [],[],[]
                day_prev = day
                gid_prev = gid
                gid = e + self.offset_per_file[day]
                print(gid)
                day = np.digitize(gid, self.offset_per_file) - 1                        
                if day != day_prev or gid <= gid_prev:
                    break
    def __batch_generator_(self,
            data_filename, data_directory, days, batch_size, split, drop_last, max_ind_range
    ):
        track_size = 0
        # global batch start index across all rows in all days. 
        gid_prev = -1
        gid = 0
        while gid + batch_size < self.length and gid_prev < gid:
            x_int_batch, x_cat_batch, y_batch = [],[],[]
            day = np.digitize(gid, self.offset_per_file) - 1
            read_file = self.start_row <= self.offset_per_file[day+1] and \
                self.offset_per_file[day] < self.last_row
            filepath = os.path.join(
                data_directory,
                data_filename + "_{}_reordered.npz".format(day)
            )
            print(f"Rank {self.rank} Reading day {filepath[:-4]+'***.npy'}")
            X_int = np.load(filepath[:-4] + "_int.npy") if read_file else None
            X_cat = np.load(filepath[:-4] + "_cat.npy") if read_file else None
            y = np.load(filepath[:-4] + "_y.npy") if read_file else None
            while True:
                # local start and end indices within a day
                s = gid - self.offset_per_file[day]
                e = min(gid + batch_size - track_size, self.offset_per_file[day+1]) - self.offset_per_file[day]
                track_size += e - s
                if read_file:
                    x_int_batch.append(X_int[s:e])
                    x_cat_batch.append(X_cat[s:e])
                    y_batch.append(y[s:e])
                else:
                    x_int_batch.append(np.empty((e-s,1)))
                    x_cat_batch.append(np.empty((e-s,1)))
                    y_batch.append(np.empty((e-s,1)))                    
                if track_size == batch_size:
                    track_size = 0
                    x_int_batch = np.concatenate(x_int_batch)
                    x_cat_batch = np.concatenate(x_cat_batch)
                    y_batch = np.concatenate(y_batch)
                    yield self.__transform_features(x_int_batch, x_cat_batch, y_batch, max_ind_range)
                    x_int_batch, x_cat_batch, y_batch = [],[],[]
                gid_prev = gid
                gid = e + self.offset_per_file[day]
                #if self.rank == 7 and day > 6:
                print(self.rank, day, self.offset_per_file[day+1] - gid)
                day_prev = day
                day = np.digitize(gid, self.offset_per_file) - 1  
                day_tail = np.digitize(gid + batch_size, self.offset_per_file)                         
                if day != day_prev or gid <= gid_prev or day_tail == len(self.offset_per_file):
                    break
class CriteoIterDataPipe(IterDataPipe):
    """
    IterDataPipe that can be used to stream either the Criteo 1TB Click Logs Dataset
    (https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) or the
    Kaggle/Criteo Display Advertising Dataset
    (https://www.kaggle.com/c/criteo-display-ad-challenge/) from the source TSV
    files.

    Args:
        paths (Iterable[str]): local paths to TSV files that constitute the Criteo
            dataset.
        row_mapper (Optional[Callable[[List[str]], Any]]): function to apply to each
            split TSV line.
        open_kw: options to pass to underlying invocation of
            iopath.common.file_io.PathManager.open.

    Example::

        datapipe = CriteoIterDataPipe(
            ("/home/datasets/criteo/day_0.tsv", "/home/datasets/criteo/day_1.tsv")
        )
        datapipe = dp.iter.Batcher(datapipe, 100)
        datapipe = dp.iter.Collator(datapipe)
        batch = next(iter(datapipe))
    """

    def __init__(
        self,
        paths: Iterable[str],
        *,
        # pyre-ignore[2]
        row_mapper: Optional[Callable[[List[str]], Any]] = _default_row_mapper,
        # pyre-ignore[2]
        **open_kw,
    ) -> None:
        self.paths = paths
        self.row_mapper = row_mapper
        self.open_kw: Any = open_kw  # pyre-ignore[4]

    # pyre-ignore[3]
    def __iter__(self) -> Iterator[Any]:
        worker_info = torch.utils.data.get_worker_info()
        paths = self.paths
        if worker_info is not None:
            paths = (
                path
                for (idx, path) in enumerate(paths)
                if idx % worker_info.num_workers == worker_info.id
            )
        datapipe = LoadFiles(paths, mode="r", **self.open_kw)
        datapipe = ReadLinesFromCSV(datapipe, delimiter="\t")
        if self.row_mapper:
            datapipe = dp.iter.Mapper(datapipe, self.row_mapper)
        yield from datapipe


def criteo_terabyte(
    paths: Iterable[str],
    *,
    # pyre-ignore[2]
    row_mapper: Optional[Callable[[List[str]], Any]] = _default_row_mapper,
    # pyre-ignore[2]
    **open_kw,
) -> IterDataPipe:
    """`Criteo 1TB Click Logs <https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/>`_ Dataset

    Args:
        paths (Iterable[str]): local paths to TSV files that constitute the Criteo 1TB
            dataset.
        row_mapper (Optional[Callable[[List[str]], Any]]): function to apply to each
            split TSV line.
        open_kw: options to pass to underlying invocation of
            iopath.common.file_io.PathManager.open.

    Example::

        datapipe = criteo_terabyte(
            ("/home/datasets/criteo/day_0.tsv", "/home/datasets/criteo/day_1.tsv")
        )
        datapipe = dp.iter.Batcher(datapipe, 100)
        datapipe = dp.iter.Collator(datapipe)
        batch = next(iter(datapipe))
    """
    return CriteoIterDataPipe(paths, row_mapper=row_mapper, **open_kw)


def criteo_kaggle(
    path: str,
    *,
    # pyre-ignore[2]
    row_mapper: Optional[Callable[[List[str]], Any]] = _default_row_mapper,
    # pyre-ignore[2]
    **open_kw,
) -> IterDataPipe:
    """`Kaggle/Criteo Display Advertising <https://www.kaggle.com/c/criteo-display-ad-challenge/>`_ Dataset

    Args:
        root (str): local path to train or test dataset file.
        row_mapper (Optional[Callable[[List[str]], Any]]): function to apply to each split TSV line.
        open_kw: options to pass to underlying invocation of iopath.common.file_io.PathManager.open.

    Example::

        train_datapipe = criteo_kaggle(
            "/home/datasets/criteo_kaggle/train.txt",
        )
        example = next(iter(train_datapipe))
        test_datapipe = criteo_kaggle(
            "/home/datasets/criteo_kaggle/test.txt",
        )
        example = next(iter(test_datapipe))
    """
    return CriteoIterDataPipe((path,), row_mapper=row_mapper, **open_kw)


class BinaryCriteoUtils:
    """
    Utility functions used to preprocess, save, load, partition, etc. the Criteo
    dataset in a binary (numpy) format.
    """

    @staticmethod
    def tsv_to_npys(
        in_file: str,
        out_dense_file: str,
        out_sparse_file: str,
        out_labels_file: str,
        path_manager_key: str = PATH_MANAGER_KEY,
    ) -> None:
        """
        Convert one Criteo tsv file to three npy files: one for dense (np.float32), one
        for sparse (np.int32), and one for labels (np.int32).

        Args:
            in_file (str): Input tsv file path.
            out_dense_file (str): Output dense npy file path.
            out_sparse_file (str): Output sparse npy file path.
            out_labels_file (str): Output labels npy file path.
            path_manager_key (str): Path manager key used to load from different
                filesystems.

        Returns:
            None.
        """

        def row_mapper(row: List[str]) -> Tuple[List[int], List[int], int]:
            label = safe_cast(row[0], int, 0)
            dense = [safe_cast(row[i], int, 0) for i in range(1, 1 + INT_FEATURE_COUNT)]
            sparse = [
                int(safe_cast(row[i], str, "0") or "0", 16)
                for i in range(
                    1 + INT_FEATURE_COUNT, 1 + INT_FEATURE_COUNT + CAT_FEATURE_COUNT
                )
            ]
            return dense, sparse, label  # pyre-ignore[7]

        dense, sparse, labels = [], [], []
        for (row_dense, row_sparse, row_label) in CriteoIterDataPipe(
            [in_file], row_mapper=row_mapper
        ):
            dense.append(row_dense)
            sparse.append(row_sparse)
            labels.append(row_label)

        # PyTorch tensors can't handle uint32, but we can save space by not
        # using int64. Numpy will automatically handle dense values >= 2 ** 31.
        dense_np = np.array(dense, dtype=np.int32)
        del dense
        sparse_np = np.array(sparse, dtype=np.int32)
        del sparse
        labels_np = np.array(labels, dtype=np.int32)
        del labels

        # Log is expensive to compute at runtime.
        dense_np += 3
        dense_np = np.log(dense_np, dtype=np.float32)

        # To be consistent with dense and sparse.
        labels_np = labels_np.reshape((-1, 1))

        path_manager = PathManagerFactory().get(path_manager_key)
        for (fname, arr) in [
            (out_dense_file, dense_np),
            (out_sparse_file, sparse_np),
            (out_labels_file, labels_np),
        ]:
            with path_manager.open(fname, "wb") as fout:
                np.save(fout, arr)

    @staticmethod
    def get_shape_from_npy(
        path: str, path_manager_key: str = PATH_MANAGER_KEY
    ) -> Tuple[int, ...]:
        """
        Returns the shape of an npy file using only its header.

        Args:
            path (str): Input npy file path.
            path_manager_key (str): Path manager key used to load from different
                filesystems.

        Returns:
            shape (Tuple[int, ...]): Shape tuple.
        """   #pp1 = '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M/day_0_reordered_int.npy'
        # ff1 = path_manager.open(pp1, "rb")
        # np.lib.format.read_magic(ff1)
        # np.lib.format.read_array_header_1_0(ff1)
        path_manager = PathManagerFactory().get(path_manager_key)
        with path_manager.open(path, "rb") as fin:
            np.lib.format.read_magic(fin)
            shape, _order, _dtype = np.lib.format.read_array_header_1_0(fin)
            return shape

    @staticmethod
    def get_file_idx_to_row_range(
        lengths: List[int],
        rank: int,
        world_size: int,
    ) -> Dict[int, Tuple[int, int]]:
        """
        Given a rank, world_size, and the lengths (number of rows) for a list of files,
        return which files and which portions of those files (represented as row ranges
        - all range indices are inclusive) should be handled by the rank. Each rank
        will be assigned the same number of rows.

        The ranges are determined in such a way that each rank deals with large
        continuous ranges of files. This enables each rank to reduce the amount of data
        it needs to read while avoiding seeks.

        Args:
            lengths (List[int]): A list of row counts for each file.
            rank (int): rank.
            world_size (int): world size.

        Returns:
            output (Dict[int, Tuple[int, int]]): Mapping of which files to the range in
                those files to be handled by the rank. The keys of this dict are indices
                of lengths.
        """

        # All ..._g variables are globals indices (meaning they range from 0 to
        # total_length - 1). All ..._l variables are local indices (meaning they range
        # from 0 to lengths[i] - 1 for the ith file).

        total_length = sum(lengths)
        rows_per_rank = total_length // world_size

        # Global indices that rank is responsible for. All ranges (left, right) are
        # inclusive.
        rank_left_g = rank * rows_per_rank
        rank_right_g = (rank + 1) * rows_per_rank - 1

        output = {}

        # Find where range (rank_left_g, rank_right_g) intersects each file's range.
        file_left_g, file_right_g = -1, -1
        for idx, length in enumerate(lengths):
            file_left_g = file_right_g + 1
            file_right_g = file_left_g + length - 1

            # If the ranges overlap.
            if rank_left_g <= file_right_g and rank_right_g >= file_left_g:
                overlap_left_g, overlap_right_g = max(rank_left_g, file_left_g), min(
                    rank_right_g, file_right_g
                )

                # Convert overlap in global numbers to (local) numbers specific to the
                # file.
                overlap_left_l = overlap_left_g - file_left_g
                overlap_right_l = overlap_right_g - file_left_g
                output[idx] = (overlap_left_l, overlap_right_l)

        return output

    @staticmethod
    def load_npy_range(
        fname: str, #'/home/ubuntu/mountpoint/criteo/1tb_numpy/day_0_dense.npy'
        start_row: int,
        num_rows: int, #195841983
        path_manager_key: str = PATH_MANAGER_KEY,
    ) -> np.ndarray:
        """
        Load part of an npy file.

        NOTE: Assumes npy represents a numpy array of ndim 2.

        Args:
            fname (str): path string to npy file.
            start_row (int): starting row from the npy file.
            num_rows (int): number of rows to get from the npy file.
            path_manager_key (str): Path manager key used to load from different
                filesystems.

        Returns:
            output (np.ndarray): numpy array with the desired range of data from the
                supplied npy file.
        """
        # pp1 = '/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M/day_0_reordered_int.npy'
        # ff1 = path_manager.open(pp1, "rb")
        # np.lib.format.read_magic(ff1)
        # s1, o1, d1 = np.lib.format.read_array_header_1_0(ff1)

        path_manager = PathManagerFactory().get(path_manager_key)

        print("fname ",fname)
        # /home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M/day_0_reordered_y.npy
        with path_manager.open(fname, "rb") as fin:
            np.lib.format.read_magic(fin)
            shape, _order, dtype = np.lib.format.read_array_header_1_0(fin)
            shape = list(shape)
            shape[0] = int(min(shape[0],2048*2*500))
            num_rows = int(shape[0]/2)
            if len(shape) == 2:
                total_rows, row_size = shape
            elif len(shape) == 1:
                total_rows, row_size = shape[0], 1
            else:
                raise ValueError("Cannot load range for npy with ndim == 2.")

            if False:
                if not (0 <= start_row < total_rows): #total_rows == 195841983 
                   raise ValueError(
                        f"start_row ({start_row}) is out of bounds. It must be between 0 "
                        f"and {total_rows - 1}, inclusive."
                    )
                if not (start_row + num_rows <= total_rows): # num_rows == 195841983
                    raise ValueError(
                        f"num_rows ({num_rows}) exceeds number of available rows "
                        f"({total_rows}) for the given start_row ({start_row})."
                    )

            offset = start_row * row_size * dtype.itemsize
            fin.seek(offset, os.SEEK_CUR)
            num_entries = num_rows * row_size
            if SETTING == 1:
                if '_cat' in fname:
                    data = np.zeros((1000,1)).astype(np.float32)
                else:
                    data = np.ones((1000,1)).astype(np.float32)
                num_rows, row_size = 1000, 1
            if SETTING == 2:
                fake_file_num_entries = BATCH_SIZE*100
                if '_cat' in fname:
                    data = np.zeros((fake_file_num_entries,1)).astype(np.float32)
                else:
                    data = np.ones((fake_file_num_entries,1)).astype(np.float32)
                num_rows, row_size = fake_file_num_entries, 1
                if False: #'_int' in fname or '_cat' in fname:
                    data1 = np.ones((50,1)).astype(np.float32)
                    #BinaryCriteoUtils.load_npy_range.counter = getattr(BinaryCriteoUtils.load_npy_range, 'counter', 0) + 1
                    #if BinaryCriteoUtils.load_npy_range.counter % 2 == 1:
                    #    data2 = np.zeros((10,1)).astype(np.float32)
                    data2 = np.ones((50,1)).astype(np.float32)
                    data = np.concatenate([data1, data2])

                    data = np.ones((10000,1)).astype(np.float32)
                    num_rows, row_size = 10000, 1
                    if True:
                        if '_cat' in fname:
                            data = np.zeros((10000,26)).astype(np.int32)
                            num_rows, row_size = 10000, 26
                        if '_y' in fname:
                            data = np.ones((10000,1)).astype(np.float32)
                            num_rows, row_size = 10000, 1                        
                #else:
                #    data = np.array([[1]]).astype(np.float32)
                #    num_rows, row_size = 1, 1
                #data = np.array([1]).astype(np.float32)[:, np.newaxis]
                #num_rows, row_size = 1, 1                
            if SETTING == 4:
                data = np.fromfile(fin, dtype=dtype, count=num_entries)
                if dtype == np.float64:
                    data = data.astype(np.float32)
                if '_int' in fname:
                    data = np.log(data + 1)
            return data.reshape((num_rows, row_size))

    @staticmethod
    def sparse_to_contiguous(
        in_files: List[str],
        output_dir: str,
        frequency_threshold: int = FREQUENCY_THRESHOLD,
        columns: int = CAT_FEATURE_COUNT,
        path_manager_key: str = PATH_MANAGER_KEY,
        output_file_suffix: str = "_contig_freq.npy",
    ) -> None:
        """
        Convert all sparse .npy files to have contiguous integers. Store in a separate
        .npy file. All input files must be processed together because columns
        can have matching IDs between files. Hence, they must be transformed
        together. Also, the transformed IDs are not unique between columns. IDs
        that appear less than frequency_threshold amount of times will be remapped
        to have a value of 1.

        Example transformation, frequenchy_threshold of 2:
        day_0_sparse.npy
        | col_0 | col_1 |
        -----------------
        | abc   | xyz   |
        | iop   | xyz   |

        day_1_sparse.npy
        | col_0 | col_1 |
        -----------------
        | iop   | tuv   |
        | lkj   | xyz   |

        day_0_sparse_contig.npy
        | col_0 | col_1 |
        -----------------
        | 1     | 2     |
        | 2     | 2     |

        day_1_sparse_contig.npy
        | col_0 | col_1 |
        -----------------
        | 2     | 1     |
        | 1     | 2     |

        Args:
            in_files List[str]: Input directory of npy files.
            out_dir (str): Output directory of processed npy files.
            frequency_threshold: IDs occuring less than this frequency will be remapped to a value of 1.
            path_manager_key (str): Path manager key used to load from different filesystems.

        Returns:
            None.
        """

        # Load each .npy file of sparse features. Transformations are made along the columns.
        # Thereby, transpose the input to ease operations.
        # E.g. file_to_features = {"day_0_sparse": [array([[3,6,7],[7,9,3]]}
        file_to_features: Dict[str, np.ndarray] = {}
        for f in in_files:
            name = os.path.basename(f).split(".")[0]
            file_to_features[name] = np.load(f).transpose()
            print(f"Successfully loaded file: {f}")

        # Iterate through each column in each file and map the sparse ids to contiguous ids.
        for col in range(columns):
            print(f"Processing column: {col}")

            # Iterate through each row in each file for the current column and determine the
            # frequency of each sparse id.
            sparse_to_frequency: Dict[int, int] = {}
            if frequency_threshold > 1:
                for f in file_to_features:
                    for _, sparse in enumerate(file_to_features[f][col]):
                        if sparse in sparse_to_frequency:
                            sparse_to_frequency[sparse] += 1
                        else:
                            sparse_to_frequency[sparse] = 1

            # Iterate through each row in each file for the current column and remap each
            # sparse id to a contiguous id. The contiguous ints start at a value of 2 so that
            # infrequenct IDs (determined by the frequency_threshold) can be remapped to 1.
            running_sum = 2
            sparse_to_contiguous_int: Dict[int, int] = {}

            for f in file_to_features:
                print(f"Processing file: {f}")

                for i, sparse in enumerate(file_to_features[f][col]):
                    if sparse not in sparse_to_contiguous_int:
                        # If the ID appears less than frequency_threshold amount of times
                        # remap the value to 1.
                        if (
                            frequency_threshold > 1
                            and sparse_to_frequency[sparse] < frequency_threshold
                        ):
                            sparse_to_contiguous_int[sparse] = 1
                        else:
                            sparse_to_contiguous_int[sparse] = running_sum
                            running_sum += 1

                    # Re-map sparse value to contiguous in place.
                    file_to_features[f][col][i] = sparse_to_contiguous_int[sparse]

        path_manager = PathManagerFactory().get(path_manager_key)
        for f, features in file_to_features.items():
            output_file = os.path.join(output_dir, f + output_file_suffix)
            with path_manager.open(output_file, "wb") as fout:
                print(f"Writing file: {output_file}")
                # Transpose back the features when saving, as they were transposed when loading.
                np.save(fout, features.transpose())

    @staticmethod
    def shuffle(
        input_dir_labels_and_dense: str,
        input_dir_sparse: str,
        output_dir_shuffled: str,
        rows_per_day: Dict[int, int],
        output_dir_full_set: Optional[str] = None,
        days: int = DAYS,
        int_columns: int = INT_FEATURE_COUNT,
        sparse_columns: int = CAT_FEATURE_COUNT,
        path_manager_key: str = PATH_MANAGER_KEY,
    ) -> None:
        """
        Shuffle the dataset. Expects the files to be in .npy format and the data
        to be split by day and by dense, sparse and label data.
        Dense data must be in: day_x_dense.npy
        Sparse data must be in: day_x_sparse.npy
        Labels data must be in: day_x_labels.npy

        The dataset will be reconstructed, shuffled and then split back into
        separate dense, sparse and labels files.

        Args:
            input_dir_labels_and_dense (str): Input directory of labels and dense npy files.
            input_dir_sparse (str): Input directory of sparse npy files.
            output_dir_shuffled (str): Output directory for shuffled labels, dense and sparse npy files.
            rows_per_day Dict[int, int]: Number of rows in each file.
            output_dir_full_set (str): Output directory of the full dataset, if desired.
            days (int): Number of day files.
            int_columns (int): Number of columns with dense features.
            columns (int): Total number of columns.
            path_manager_key (str): Path manager key used to load from different filesystems.
        """

        total_rows = sum(rows_per_day.values())
        columns = int_columns + sparse_columns + 1  # add 1 for label column
        full_dataset = np.zeros((total_rows, columns), dtype=np.float32)
        curr_first_row = 0
        curr_last_row = 0
        for d in range(0, days):
            curr_last_row += rows_per_day[d]

            # dense
            path_to_file = os.path.join(
                input_dir_labels_and_dense, f"day_{d}_dense.npy"
            )
            data = np.load(path_to_file)
            print(
                f"Day {d} dense- {curr_first_row}-{curr_last_row} loaded files - {time.time()} - {path_to_file}"
            )

            full_dataset[curr_first_row:curr_last_row, 0:int_columns] = data
            del data

            # sparse
            path_to_file = os.path.join(input_dir_sparse, f"day_{d}_sparse.npy")
            data = np.load(path_to_file)
            print(
                f"Day {d} sparse- {curr_first_row}-{curr_last_row} loaded files - {time.time()} - {path_to_file}"
            )

            full_dataset[curr_first_row:curr_last_row, int_columns : columns - 1] = data
            del data

            # labels
            path_to_file = os.path.join(
                input_dir_labels_and_dense, f"day_{d}_labels.npy"
            )
            data = np.load(path_to_file)
            print(
                f"Day {d} labels- {curr_first_row}-{curr_last_row} loaded files - {time.time()} - {path_to_file}"
            )

            full_dataset[curr_first_row:curr_last_row, columns - 1 :] = data
            del data

            curr_first_row = curr_last_row

        path_manager = PathManagerFactory().get(path_manager_key)

        # Save the full dataset
        if output_dir_full_set is not None:
            full_output_file = os.path.join(output_dir_full_set, "full.npy")
            with path_manager.open(full_output_file, "wb") as fout:
                print(f"Writing full set file: {full_output_file}")
                np.save(fout, full_dataset)

        print("Shuffling dataset")
        np.random.shuffle(full_dataset)

        # Slice and save each portion into dense, sparse and labels
        curr_first_row = 0
        curr_last_row = 0
        for d in range(0, days):
            curr_last_row += rows_per_day[d]

            # write dense columns
            shuffled_dense_file = os.path.join(
                output_dir_shuffled, f"day_{d}_dense.npy"
            )
            with path_manager.open(shuffled_dense_file, "wb") as fout:
                print(
                    f"Writing rows {curr_first_row}-{curr_last_row-1} dense file: {shuffled_dense_file}"
                )
                np.save(fout, full_dataset[curr_first_row:curr_last_row, 0:int_columns])

            # write sparse columns
            shuffled_sparse_file = os.path.join(
                output_dir_shuffled, f"day_{d}_sparse.npy"
            )
            with path_manager.open(shuffled_sparse_file, "wb") as fout:
                print(
                    f"Writing rows {curr_first_row}-{curr_last_row-1} sparse file: {shuffled_sparse_file}"
                )
                np.save(
                    fout,
                    full_dataset[
                        curr_first_row:curr_last_row, int_columns : columns - 1
                    ].astype(np.int32),
                )

            # write labels columns
            shuffled_labels_file = os.path.join(
                output_dir_shuffled, f"day_{d}_labels.npy"
            )
            with path_manager.open(shuffled_labels_file, "wb") as fout:
                print(
                    f"Writing rows {curr_first_row}-{curr_last_row-1} labels file: {shuffled_labels_file}"
                )
                np.save(
                    fout,
                    full_dataset[curr_first_row:curr_last_row, columns - 1 :].astype(
                        np.int32
                    ),
                )

            curr_first_row = curr_last_row


class InMemoryBinaryCriteoIterDataPipe(IterableDataset):
    """
    Datapipe designed to operate over binary (npy) versions of Criteo datasets. Loads
    the entire dataset into memory to prevent disk speed from affecting throughout. Each
    rank reads only the data for the portion of the dataset it is responsible for.

    The torchrec/datasets/scripts/preprocess_criteo.py script can be used to convert
    the Criteo tsv files to the npy files expected by this dataset.

    Args:
        dense_paths (List[str]): List of path strings to dense npy files.
        sparse_paths (List[str]): List of path strings to sparse npy files.
        labels_paths (List[str]): List of path strings to labels npy files.
        batch_size (int): batch size.
        rank (int): rank.
        world_size (int): world size.
        shuffle_batches (bool): Whether to shuffle batches
        hashes (Optional[int]): List of max categorical feature value for each feature.
            Length of this list should be CAT_FEATURE_COUNT.
        path_manager_key (str): Path manager key used to load from different
            filesystems.

    Example::

        template = "/home/datasets/criteo/1tb_binary/day_{}_{}.npy"
        datapipe = InMemoryBinaryCriteoIterDataPipe(
            dense_paths=[template.format(0, "dense"), template.format(1, "dense")],
            sparse_paths=[template.format(0, "sparse"), template.format(1, "sparse")],
            labels_paths=[template.format(0, "labels"), template.format(1, "labels")],
            batch_size=1024,
            rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
        )
        batch = next(iter(datapipe))
    """

    def __init__(
        self,
        dense_paths: List[str],
        sparse_paths: List[str],
        labels_paths: List[str],
        batch_size: int,
        rank: int,
        world_size: int,
        shuffle_batches: bool = False,
        hashes: Optional[List[int]] = None,
        path_manager_key: str = PATH_MANAGER_KEY,

    ) -> None:

        if False:
            # From OSS, dlrm_data_pytorch.py, class CriteoDataset(Dataset):
            dataset = 'terabyte'
            max_ind_range = 40000000
            sub_sample_rate = 0.0
            randomize = 'total'
            split="train" #or train
            raw_path='/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M/day'
            pro_data='/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M/'
            memory_map=True
            dataset_multiprocessing=False
            # dataset
            # tar_fea = 1   # single target
            den_fea = 13  # 13 dense  features
            # spa_fea = 26  # 26 sparse features
            # tad_fea = tar_fea + den_fea
            # tot_fea = tad_fea + spa_fea
            if dataset == "kaggle":
                days = 7
                out_file = "kaggleAdDisplayChallenge_processed"
            elif dataset == "terabyte":
                days = 1 #24
                out_file = "terabyte_processed"
            else:
                raise(ValueError("Data set option is not supported"))
            self.max_ind_range = max_ind_range
            self.memory_map = memory_map

            # split the datafile into path and filename
            lstr = raw_path.split("/")
            self.d_path = "/".join(lstr[0:-1]) + "/"
            self.d_file = lstr[-1].split(".")[0] if dataset == "kaggle" else lstr[-1]
            self.npzfile = self.d_path + (
                (self.d_file + "_day") if dataset == "kaggle" else self.d_file
            )
            self.trafile = self.d_path + (
                (self.d_file + "_fea") if dataset == "kaggle" else "fea"
            )

            # check if pre-processed data is available
            data_ready = True
            if memory_map:
                for i in range(days):
                    reo_data = self.npzfile + "_{0}_reordered.npz".format(i)
                    if not path.exists(str(reo_data)):
                        data_ready = False
            else:
                if not path.exists(str(pro_data)):
                    data_ready = False

            # pre-process data if needed
            # WARNNING: when memory mapping is used we get a collection of files
            if data_ready:
                print("Reading pre-processed data=%s" % (str(pro_data)))
                file = str(pro_data)
            else:
                pass
                # print("Reading raw data=%s" % (str(raw_path)))
                # file = data_utils.getCriteoAdData(
                #     raw_path,
                #     out_file,
                #     max_ind_range,
                #     sub_sample_rate,
                #     days,
                #     split,
                #     randomize,
                #     dataset == "kaggle",
                #     memory_map,
                #     dataset_multiprocessing,
                # )

            # get a number of samples per day
            total_file = self.d_path + self.d_file + "_day_count.npz"
            with np.load(total_file) as data:
                total_per_file = data["total_per_file"]
            # compute offsets per file
            self.offset_per_file = np.array([0] + [x for x in total_per_file])
            for i in range(days):
                self.offset_per_file[i + 1] += self.offset_per_file[i]
            # print(self.offset_per_file)

            # setup data
            if memory_map:
                # setup the training/testing split
                self.split = split
                if split == 'none' or split == 'train':
                    self.day = 0
                    self.max_day_range = days if split == 'none' else days - 1
                elif split == 'test' or split == 'val':
                    self.day = days - 1
                    num_samples = self.offset_per_file[days] - \
                                self.offset_per_file[days - 1]
                    self.test_size = int(np.ceil(num_samples / 2.))
                    self.val_size = num_samples - self.test_size
                else:
                    sys.exit("ERROR: dataset split is neither none, nor train or test.")

                '''
                # text
                print("text")
                for i in range(days):
                    fi = self.npzfile + "_{0}".format(i)
                    with open(fi) as data:
                        ttt = 0; nnn = 0
                        for _j, line in enumerate(data):
                            ttt +=1
                            if np.int32(line[0]) > 0:
                                nnn +=1
                        print("day=" + str(i) + " total=" + str(ttt) + " non-zeros="
                            + str(nnn) + " ratio=" +str((nnn * 100.) / ttt) + "%")
                # processed
                print("processed")
                for i in range(days):
                    fi = self.npzfile + "_{0}_processed.npz".format(i)
                    with np.load(fi) as data:
                        yyy = data["y"]
                    ttt = len(yyy)
                    nnn = np.count_nonzero(yyy)
                    print("day=" + str(i) + " total=" + str(ttt) + " non-zeros="
                        + str(nnn) + " ratio=" +str((nnn * 100.) / ttt) + "%")
                # reordered
                print("reordered")
                for i in range(days):
                    fi = self.npzfile + "_{0}_reordered.npz".format(i)
                    with np.load(fi) as data:
                        yyy = data["y"]
                    ttt = len(yyy)
                    nnn = np.count_nonzero(yyy)
                    print("day=" + str(i) + " total=" + str(ttt) + " non-zeros="
                        + str(nnn) + " ratio=" +str((nnn * 100.) / ttt) + "%")
                '''

                # load unique counts
                with np.load(self.d_path + self.d_file + "_fea_count.npz") as data:
                    self.counts = data["counts"]
                self.m_den = den_fea  # X_int.shape[1]
                self.n_emb = len(self.counts)
                print("Sparse features= %d, Dense features= %d" % (self.n_emb, self.m_den))

                # Load the test data
                # Only a single day is used for testing
                if self.split == 'test' or self.split == 'val':
                    # only a single day is used for testing
                    fi = self.npzfile + "_{0}_reordered.npz".format(
                        self.day
                    )
                    with np.load(fi) as data:
                        self.X_int = data["X_int"]  # continuous  feature
                        self.X_cat = data["X_cat"]  # categorical feature
                        self.y = data["y"]          # target

            else:
                # load and preprocess data
                with np.load(file) as data:
                    X_int = data["X_int"]  # continuous  feature
                    X_cat = data["X_cat"]  # categorical feature
                    y = data["y"]          # target
                    self.counts = data["counts"]
                self.m_den = X_int.shape[1]  # den_fea
                self.n_emb = len(self.counts)
                print("Sparse fea = %d, Dense fea = %d" % (self.n_emb, self.m_den))

                # create reordering
                indices = np.arange(len(y))

                if split == "none":
                    # randomize all data
                    if randomize == "total":
                        indices = np.random.permutation(indices)
                        print("Randomized indices...")

                    X_int[indices] = X_int
                    X_cat[indices] = X_cat
                    y[indices] = y

                else:
                    indices = np.array_split(indices, self.offset_per_file[1:-1])

                    # randomize train data (per day)
                    if randomize == "day":  # or randomize == "total":
                        for i in range(len(indices) - 1):
                            indices[i] = np.random.permutation(indices[i])
                        print("Randomized indices per day ...")

                    train_indices = np.concatenate(indices[:-1])
                    test_indices = indices[-1]
                    test_indices, val_indices = np.array_split(test_indices, 2)

                    print("Defined %s indices..." % (split))

                    # randomize train data (across days)
                    if randomize == "total":
                        train_indices = np.random.permutation(train_indices)
                        print("Randomized indices across days ...")

                    # create training, validation, and test sets
                    if split == 'train':
                        self.X_int = [X_int[i] for i in train_indices]
                        self.X_cat = [X_cat[i] for i in train_indices]
                        self.y = [y[i] for i in train_indices]
                    elif split == 'val':
                        self.X_int = [X_int[i] for i in val_indices]
                        self.X_cat = [X_cat[i] for i in val_indices]
                        self.y = [y[i] for i in val_indices]
                    elif split == 'test':
                        self.X_int = [X_int[i] for i in test_indices]
                        self.X_cat = [X_cat[i] for i in test_indices]
                        self.y = [y[i] for i in test_indices]

                print("Split data according to indices...")

        
        
        if False:
            #print("\n\ncreating dataset iterator\n\n")
            self.train_loader = DataLoader(
                rank,
                world_size,
                data_directory='/home/ubuntu/mountpoint/criteo_terabyte_subsample0.0_maxind40M',
                data_filename='day',
                days=list(range(3)),
                #days=list(range(1)),
                batch_size=2048,
                max_ind_range=40000000,
                split="train"
            )

            # load all days into self.train_data. Then treat all days as a single huge day, so rest of code thinks just one day is being trained.
            # Do this because torhrec code uses a 2d [which day][which record in that day] format, whereas dlrm oss uses a 1d [which record in all days records]
            
            print("\nconverting dataset iterator to list START\n")
            self.train_data = list(self.train_loader)
            print("\nconverting dataset iterator to list DONE\n")
            self.dense_arrs = [[r[0] for r in self.train_data]] #   [self.train_data[:][0]]
            self.sparse_arrs = [[r[2] for r in self.train_data]] # [self.train_data[:][2]]
            self.labels_arrs = [[r[3] for r in self.train_data]]

            print("\nStacking data\n")

            self.dense_arrs = np.stack(self.dense_arrs[0][:-1])
            self.sparse_arrs = np.stack(self.sparse_arrs[0][:-1])
            self.labels_arrs = np.stack(self.labels_arrs[0][:-1])

            print("data prep done!\n")

        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################
        #################################################################################################


        self.dense_paths = dense_paths
        self.sparse_paths = sparse_paths
        self.labels_paths = labels_paths
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle_batches = shuffle_batches
        self.hashes = hashes
        self.path_manager_key = path_manager_key
        self.path_manager: PathManager = PathManagerFactory().get(path_manager_key)

        #print("\n_load_data_for_rank\n")
        self._load_data_for_rank()
        #print("\n_load_data_for_rank DONE\n")
        self.num_rows_per_file: List[int] = [a.shape[0] for a in self.dense_arrs]
        self.num_batches: int = sum(self.num_rows_per_file) // batch_size

        # These values are the same for the KeyedJaggedTensors in all batches, so they
        # are computed once here. This avoids extra work from the KeyedJaggedTensor sync
        # functions.
        self._num_ids_in_batch: int = CAT_FEATURE_COUNT * batch_size
        self.keys: List[str] = DEFAULT_CAT_NAMES
        self.lengths: torch.Tensor = torch.ones(
            (self._num_ids_in_batch,), dtype=torch.int32
        )
        self.offsets: torch.Tensor = torch.arange(
            0, self._num_ids_in_batch + 1, dtype=torch.int32
        )
        self.stride = batch_size
        self.length_per_key: List[int] = CAT_FEATURE_COUNT * [batch_size]
        self.offset_per_key: List[int] = [
            batch_size * i for i in range(CAT_FEATURE_COUNT + 1)
        ]
        self.index_per_key: Dict[str, int] = {
            key: i for (i, key) in enumerate(self.keys)
        }

    # From OSS
    # not in use because OSS's train_data and train_loader variables are redundant, and this is for train_data is no longer used.
    #    def __getitem__(self, index):
    #
    #        if isinstance(index, slice):
    #            return [
    #                self[idx] for idx in range(
    #                    index.start or 0, index.stop or len(self), index.step or 1
    #                )
    #            ]
    #
    #        if self.memory_map:
    #            if self.split == 'none' or self.split == 'train':
    #                # check if need to swicth to next day and load data
    #                if index == self.offset_per_file[self.day]:
    #                    # print("day_boundary switch", index)
    #                    self.day_boundary = self.offset_per_file[self.day]
    #                    fi = self.npzfile + "_{0}_reordered.npz".format(
    #                        self.day
    #                    )
    #                    # print('Loading file: ', fi)
    #                    with np.load(fi) as data:
    #                        self.X_int = data["X_int"]  # continuous  feature
    #                        self.X_cat = data["X_cat"]  # categorical feature
    #                        self.y = data["y"]          # target
    #                    self.day = (self.day + 1) % self.max_day_range
    #
    #                i = index - self.day_boundary
    #            elif self.split == 'test' or self.split == 'val':
    #                # only a single day is used for testing
    #                i = index + (0 if self.split == 'test' else self.test_size)
    #            else:
    #                sys.exit("ERROR: dataset split is neither none, nor train or test.")
    #        else:
    #            i = index
    #
    #        if self.max_ind_range > 0:
    #            return self.X_int[i], self.X_cat[i] % self.max_ind_range, self.y[i]
    #        else:
    #            return self.X_int[i], self.X_cat[i], self.y[i]    



    def _load_data_for_rank(self) -> None:
        file_idx_to_row_range = BinaryCriteoUtils.get_file_idx_to_row_range(
            lengths=[
                BinaryCriteoUtils.get_shape_from_npy(
                    path, path_manager_key=self.path_manager_key
                )[0]
                for path in self.dense_paths
            ],
            rank=self.rank,
            world_size=self.world_size,
        )

        self.dense_arrs, self.sparse_arrs, self.labels_arrs = [], [], []
        for arrs, paths in zip(
            [self.dense_arrs, self.sparse_arrs, self.labels_arrs],
            [self.dense_paths, self.sparse_paths, self.labels_paths],
        ):
            for idx, (range_left, range_right) in file_idx_to_row_range.items():
                arrs.append(
                    BinaryCriteoUtils.load_npy_range(
                        paths[idx],
                        range_left,
                        range_right - range_left + 1,
                        path_manager_key=self.path_manager_key,
                    )
                )

        #if self.hashes is not None:
        #    hashes_np = np.array(self.hashes).reshape((1, CAT_FEATURE_COUNT))
        #    for sparse_arr in self.sparse_arrs:
        #        sparse_arr %= hashes_np

    def _np_arrays_to_batch(
        self, dense: np.ndarray, sparse: np.ndarray, labels: np.ndarray
    ) -> Batch:
        if self.shuffle_batches:
            # Shuffle all 3 in unison
            shuffler = np.random.permutation(len(dense))
            dense = dense[shuffler]
            sparse = sparse[shuffler]
            labels = labels[shuffler]

        return Batch(
            dense_features=torch.from_numpy(dense),
            sparse_features=KeyedJaggedTensor(
                keys=self.keys,
                # transpose + reshape(-1) incurs an additional copy.
                values=torch.from_numpy(sparse.transpose(1, 0).reshape(-1)),
                lengths=self.lengths,
                offsets=self.offsets,
                stride=self.stride,
                length_per_key=self.length_per_key,
                offset_per_key=self.offset_per_key,
                index_per_key=self.index_per_key,
            ),
            labels=torch.from_numpy(labels.reshape(-1)),
        )

    def __iter__(self) -> Iterator[Batch]:

        #return iter(
        #    _batch_generator(
        #        self.data_filename, self.data_directory, self.days,
        #        self.batch_size, self.split, self.drop_last_batch, self.max_ind_range
        #    )
        #)








        # Invariant: buffer never contains more than batch_size rows.
        buffer: Optional[List[np.ndarray]] = None

        def append_to_buffer(
            dense: np.ndarray, sparse: np.ndarray, labels: np.ndarray
        ) -> None:
            nonlocal buffer
            if buffer is None:
                buffer = [dense, sparse, labels]
            else:
                for idx, arr in enumerate([dense, sparse, labels]):
                    buffer[idx] = np.concatenate((buffer[idx], arr))

        # Maintain a buffer that can contain up to batch_size rows. Fill buffer as
        # much as possible on each iteration. Only return a new batch when batch_size
        # rows are filled.
        file_idx = 0
        row_idx = 0
        batch_idx = 0
        while batch_idx < self.num_batches:
            buffer_row_count = 0 if buffer is None else none_throws(buffer)[0].shape[0]
            if buffer_row_count == self.batch_size:
                yield self._np_arrays_to_batch(*none_throws(buffer))
                batch_idx += 1
                buffer = None
            else:
                rows_to_get = min(
                    self.batch_size - buffer_row_count,
                    self.num_rows_per_file[file_idx] - row_idx,
                )
                slice_ = slice(row_idx, row_idx + rows_to_get) #

                #for simple nn test:
                #slice_ = slice(0, 2048)
                if SETTING == 1:
                    slice_ = slice(0, BATCH_SIZE)
                if SETTING == 2:
                    slice_ = slice(0, BATCH_SIZE)

                #print("slice: ", row_idx/2048)
                #slice_ = slice(0, 2048)
                #self.dense_arrs[file_idx][slice_, :] = 1.0
                #self.sparse_arrs[file_idx][slice_, :] = 0.0
                #self.labels_arrs[file_idx][slice_, :] = 0.0     
                append_to_buffer(
                    self.dense_arrs[file_idx][slice_, :],
                    self.sparse_arrs[file_idx][slice_, :],
                    self.labels_arrs[file_idx][slice_, :],
                )
                row_idx += rows_to_get

                if row_idx >= self.num_rows_per_file[file_idx]:
                    if SETTING == 4:
                        file_idx += 1
                    row_idx = 0

    def __len__(self) -> int:
        return self.num_batches
