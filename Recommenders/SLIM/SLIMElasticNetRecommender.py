#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Massimo Quadrana, Cesare Bernardis
"""


# import numpy as np
# import scipy.sparse as sps
# from Recommenders.Recommender_utils import check_matrix
# from sklearn.linear_model import ElasticNet
# from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
# from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
# import time, sys
# from tqdm import tqdm
# from sklearn.utils._testing import ignore_warnings
# from sklearn.exceptions import ConvergenceWarning

# # os.environ["PYTHONWARNINGS"] = ('ignore::exceptions.ConvergenceWarning:sklearn.linear_model')
# # os.environ["PYTHONWARNINGS"] = ('ignore:Objective did not converge:ConvergenceWarning:')

# class SLIMElasticNetRecommender(BaseItemSimilarityMatrixRecommender):
#     """
#     Train a Sparse Linear Methods (SLIM) item similarity model.
#     NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
#           make use of half the cores available

#     See:
#         Efficient Top-N Recommendation by Linear Regression,
#         M. Levy and K. Jack, LSRS workshop at RecSys 2013.

#         SLIM: Sparse linear methods for top-n recommender systems,
#         X. Ning and G. Karypis, ICDM 2011.
#         http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
#     """

#     RECOMMENDER_NAME = "SLIMElasticNetRecommender"

#     def __init__(self, URM_train, verbose = True):
#         super(SLIMElasticNetRecommender, self).__init__(URM_train, verbose = verbose)

#     @ignore_warnings(category=ConvergenceWarning)
#     def fit(self, l1_ratio=0.1, alpha = 1.0, positive_only=True, topK = 100):

#         assert l1_ratio>= 0 and l1_ratio<=1, "{}: l1_ratio must be between 0 and 1, provided value was {}".format(self.RECOMMENDER_NAME, l1_ratio)

#         self.l1_ratio = l1_ratio
#         self.positive_only = positive_only
#         self.topK = topK


#         # initialize the ElasticNet model
#         self.model = ElasticNet(alpha=alpha,
#                                 l1_ratio=self.l1_ratio,
#                                 positive=self.positive_only,
#                                 fit_intercept=False,
#                                 copy_X=False,
#                                 precompute=True,
#                                 selection='random',
#                                 max_iter=100,
#                                 tol=1e-4)

#         URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)

#         n_items = URM_train.shape[1]

#         # Use array as it reduces memory requirements compared to lists
#         dataBlock = 10000000

#         rows = np.zeros(dataBlock, dtype=np.int32)
#         cols = np.zeros(dataBlock, dtype=np.int32)
#         values = np.zeros(dataBlock, dtype=np.float32)

#         numCells = 0

#         start_time = time.time()
#         start_time_printBatch = start_time

#         # fit each item's factors sequentially (not in parallel)
#         for currentItem in range(n_items):

#             # get the target column
#             y = URM_train[:, currentItem].toarray()

#             # set the j-th column of X to zero
#             start_pos = URM_train.indptr[currentItem]
#             end_pos = URM_train.indptr[currentItem + 1]

#             current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
#             URM_train.data[start_pos: end_pos] = 0.0

#             # fit one ElasticNet model per column
#             self.model.fit(URM_train, y)

#             # self.model.coef_ contains the coefficient of the ElasticNet model
#             # let's keep only the non-zero values

#             # Select topK values
#             # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
#             # - Partition the data to extract the set of relevant items
#             # - Sort only the relevant items
#             # - Get the original item index

#             nonzero_model_coef_index = self.model.sparse_coef_.indices
#             nonzero_model_coef_value = self.model.sparse_coef_.data

#             local_topK = min(len(nonzero_model_coef_value)-1, self.topK)

#             relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
#             relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
#             ranking = relevant_items_partition[relevant_items_partition_sorting]

#             for index in range(len(ranking)):

#                 if numCells == len(rows):
#                     rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
#                     cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
#                     values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


#                 rows[numCells] = nonzero_model_coef_index[ranking[index]]
#                 cols[numCells] = currentItem
#                 values[numCells] = nonzero_model_coef_value[ranking[index]]

#                 numCells += 1

#             # finally, replace the original values of the j-th column
#             URM_train.data[start_pos:end_pos] = current_item_data_backup

#             elapsed_time = time.time() - start_time
#             new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)


#             if time.time() - start_time_printBatch > 300 or currentItem == n_items-1:
#                 self._print("Processed {} ({:4.1f}%) in {:.2f} {}. Items per second: {:.2f}".format(
#                     currentItem+1,
#                     100.0* float(currentItem+1)/n_items,
#                     new_time_value,
#                     new_time_unit,
#                     float(currentItem)/elapsed_time))

#                 sys.stdout.flush()
#                 sys.stderr.flush()

#                 start_time_printBatch = time.time()

#         # generate the sparse weight matrix
#         self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
#                                        shape=(n_items, n_items), dtype=np.float32)




# from multiprocessing import Pool, cpu_count, shared_memory
# from functools import partial


# def create_shared_memory(a):
#     shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
#     b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
#     b[:] = a[:]
#     return shm


# @ignore_warnings(category=ConvergenceWarning)
# def _partial_fit(items, topK, alpha, l1_ratio, urm_shape, positive_only=True, shm_names=None, shm_shapes=None, shm_dtypes=None):

#     model = ElasticNet(
#         alpha=alpha,
#         l1_ratio=l1_ratio,
#         positive=positive_only,
#         fit_intercept=False,
#         copy_X=False,
#         precompute=True,
#         selection='random',
#         max_iter=100,
#         tol=1e-4
#     )

#     indptr_shm = shared_memory.SharedMemory(name=shm_names[0], create=False)
#     indices_shm = shared_memory.SharedMemory(name=shm_names[1], create=False)
#     data_shm = shared_memory.SharedMemory(name=shm_names[2], create=False)

#     X_j = sps.csc_matrix((
#             np.ndarray(shm_shapes[2], dtype=shm_dtypes[2], buffer=data_shm.buf).copy(),
#             np.ndarray(shm_shapes[1], dtype=shm_dtypes[1], buffer=indices_shm.buf),
#             np.ndarray(shm_shapes[0], dtype=shm_dtypes[0], buffer=indptr_shm.buf),
#         ), shape=urm_shape)

#     values, rows, cols = [], [], []

#     for currentItem in items:

#         y = X_j[:, currentItem].toarray()

#         backup = X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]]
#         X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]] = 0.0

#         model.fit(X_j, y)

#         nonzero_model_coef_index = model.sparse_coef_.indices
#         nonzero_model_coef_value = model.sparse_coef_.data

#         local_topK = min(len(nonzero_model_coef_value) - 1, topK)

#         relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[:local_topK]
#         relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
#         ranking = relevant_items_partition[relevant_items_partition_sorting]

#         values.extend(nonzero_model_coef_value[ranking])
#         rows.extend(nonzero_model_coef_index[ranking])
#         cols.extend([currentItem] * len(ranking))

#         X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]] = backup

#     indptr_shm.close()
#     indices_shm.close()
#     data_shm.close()

#     return values, rows, cols




# # 
# import multiprocessing
# from multiprocessing import Pool
# from functools import partial


# class MultiThreadSLIM_ElasticNet(SLIMElasticNetRecommender, BaseItemSimilarityMatrixRecommender):

#     def __init__(self, URM_train, verbose=True):
#         super(MultiThreadSLIM_ElasticNet, self).__init__(URM_train, verbose=verbose)

#     def _partial_fit(self, currentItem, X, topK):
#         model = ElasticNet(alpha=self.alpha,
#                            l1_ratio=self.l1_ratio,
#                            positive=self.positive_only,
#                            fit_intercept=False,
#                            copy_X=False,
#                            precompute=True,
#                            selection='random',
#                            max_iter=self.max_iter,
#                            tol=self.tol)

#         # WARNING: make a copy of X to avoid race conditions on column j
#         # TODO: We can probably come up with something better here.
#         X_j = X.copy()
#         # get the target column
#         y = X_j[:, currentItem].toarray()
#         # set the j-th column of X to zero
#         X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]] = 0.0
#         # fit one ElasticNet model per column
#         model.fit(X_j, y)
#         # self.model.coef_ contains the coefficient of the ElasticNet model
#         # let's keep only the non-zero values
#         # nnz_idx = model.coef_ > 0.0

#         relevant_items_partition = (-model.coef_).argpartition(topK)[0:topK]
#         relevant_items_partition_sorting = np.argsort(-model.coef_[relevant_items_partition])
#         ranking = relevant_items_partition[relevant_items_partition_sorting]

#         notZerosMask = model.coef_[ranking] > 0.0
#         ranking = ranking[notZerosMask]

#         values = model.coef_[ranking]
#         rows = ranking
#         cols = [currentItem] * len(ranking)

#         return values, rows, cols

#     def fit(self, l1_ratio=0.1, alpha=1.0, tol =1e-4, positive_only=True, topK=100, max_iter=100,
#             workers=multiprocessing.cpu_count()):
#         assert l1_ratio >= 0 and l1_ratio <= 1, "SLIM_ElasticNet: l1_ratio must be between 0 and 1, provided value was {}".format(
#             l1_ratio)

#         self.alpha = alpha
#         self.tol = tol
#         self.l1_ratio = l1_ratio
#         self.positive_only = positive_only
#         self.topK = topK
#         self.max_iter = max_iter

#         self.workers = workers

#         self.URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)
#         n_items = self.URM_train.shape[1]
#         # fit item's factors in parallel

#         # oggetto riferito alla funzione nel quale predefinisco parte dell'input
#         _pfit = partial(self._partial_fit, X=self.URM_train, topK=self.topK)

#         # creo un pool con un certo numero di processi
#         pool = Pool(processes=self.workers)

#         # avvio il pool passando la funzione (con la parte fissa dell'input)
#         # e il rimanente parametro, variabile
#         res = pool.map(_pfit, np.arange(n_items))

#         # res contains a vector of (values, rows, cols) tuples
#         values, rows, cols = [], [], []
#         for values_, rows_, cols_ in res:
#             values.extend(values_)
#             rows.extend(rows_)
#             cols.extend(cols_)

#         # generate the sparse weight matrix
#         self.W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

import numpy as np
import scipy.sparse as sps
from Recommenders.Recommender_utils import check_matrix
from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
import time, sys, warnings


class SLIMElasticNetRecommender(BaseItemSimilarityMatrixRecommender):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available
    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.
        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    RECOMMENDER_NAME = "SLIMElasticNetRecommender"

    def __init__(self, URM_train, verbose=True):
        super(SLIMElasticNetRecommender, self).__init__(URM_train, verbose=verbose)

    def fit(self, l1_ratio=0.1, alpha=1.0, tol =1e-4, positive_only=True, topK=100):


        # Display ConvergenceWarning only once and not for every item it occurs
        warnings.simplefilter("once", category=ConvergenceWarning)

        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=alpha,
                                l1_ratio=l1_ratio,
                                positive=positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-2)

        URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        n_items = URM_train.shape[1]

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for currentItem in range(n_items):

            # get the target column
            y = URM_train[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = URM_train.indptr[currentItem]
            end_pos = URM_train.indptr[currentItem + 1]

            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.model.fit(URM_train, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            nonzero_model_coef_index = self.model.sparse_coef_.indices
            nonzero_model_coef_value = self.model.sparse_coef_.data

            local_topK = min(len(nonzero_model_coef_value) - 1, topK)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):

                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1

            # finally, replace the original values of the j-th column
            URM_train.data[start_pos:end_pos] = current_item_data_backup

            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            if time.time() - start_time_printBatch > 300 or currentItem == n_items - 1:
                self._print("Processed {} ( {:.2f}% ) in {:.2f} {}. Items per second: {:.2f}".format(
                    currentItem + 1,
                    100.0 * float(currentItem + 1) / n_items,
                    new_time_value,
                    new_time_unit,
                    float(currentItem) / elapsed_time))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(n_items, n_items), dtype=np.float32)


import multiprocessing
from multiprocessing import Pool
from functools import partial


class MultiThreadSLIM_ElasticNet(SLIMElasticNetRecommender, BaseItemSimilarityMatrixRecommender):

    def __init__(self, URM_train, verbose=True):
        super(MultiThreadSLIM_ElasticNet, self).__init__(URM_train, verbose=verbose)

    def _partial_fit(self, currentItem, X, topK):
        model = ElasticNet(alpha=self.alpha,
                           l1_ratio=self.l1_ratio,
                           positive=self.positive_only,
                           fit_intercept=False,
                           copy_X=False,
                           precompute=True,
                           selection='random',
                           max_iter=self.max_iter,
                           tol=self.tol)

        # WARNING: make a copy of X to avoid race conditions on column j
        # TODO: We can probably come up with something better here.
        X_j = X.copy()
        # get the target column
        y = X_j[:, currentItem].toarray()
        # set the j-th column of X to zero
        X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]] = 0.0
        # fit one ElasticNet model per column
        model.fit(X_j, y)
        # self.model.coef_ contains the coefficient of the ElasticNet model
        # let's keep only the non-zero values
        # nnz_idx = model.coef_ > 0.0

        relevant_items_partition = (-model.coef_).argpartition(topK)[0:topK]
        relevant_items_partition_sorting = np.argsort(-model.coef_[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        notZerosMask = model.coef_[ranking] > 0.0
        ranking = ranking[notZerosMask]

        values = model.coef_[ranking]
        rows = ranking
        cols = [currentItem] * len(ranking)

        return values, rows, cols

    def fit(self, l1_ratio=0.1, alpha=1.0, tol =1e-4, positive_only=True, topK=100, max_iter=100,
            workers=multiprocessing.cpu_count()):
        assert l1_ratio >= 0 and l1_ratio <= 1, "SLIM_ElasticNet: l1_ratio must be between 0 and 1, provided value was {}".format(
            l1_ratio)

        self.alpha = alpha
        self.tol = tol
        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.topK = topK
        self.max_iter = max_iter

        self.workers = workers

        self.URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)
        n_items = self.URM_train.shape[1]
        # fit item's factors in parallel

        # oggetto riferito alla funzione nel quale predefinisco parte dell'input
        _pfit = partial(self._partial_fit, X=self.URM_train, topK=self.topK)

        # creo un pool con un certo numero di processi
        pool = Pool(processes=self.workers)

        # avvio il pool passando la funzione (con la parte fissa dell'input)
        # e il rimanente parametro, variabile
        res = pool.map(_pfit, np.arange(n_items))

        # res contains a vector of (values, rows, cols) tuples
        values, rows, cols = [], [], []
        for values_, rows_, cols_ in res:
            values.extend(values_)
            rows.extend(rows_)
            cols.extend(cols_)

        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)