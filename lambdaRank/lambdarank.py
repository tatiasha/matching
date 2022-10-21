import math
import pickle
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm


class Solution:
    def __init__(self, n_estimators: int = 100, lr: float = 0.5, ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.9,
                 max_depth: int = 5, min_samples_leaf: int = 8):
        self._prepare_data()

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        
        self.best_ndcg = 0
        self.trees = []
        self.trees_features = []
        self.best_tree_idx = 0


    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()
        self.X_train = torch.FloatTensor(self._scale_features_in_query_groups(X_train, self.query_ids_train))
        self.X_test = torch.FloatTensor(self._scale_features_in_query_groups(X_test, self.query_ids_test))
        self.ys_train = torch.FloatTensor(y_train).reshape(-1, 1)
        self.ys_test = torch.FloatTensor(y_test).reshape(-1, 1)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        scale_feat_array = np.zeros_like(inp_feat_array)
        for q in set(inp_query_ids):
            mask = np.where(inp_query_ids==q)
            scaler = StandardScaler()
            scale_feat_array[mask] = scaler.fit_transform(inp_feat_array[mask])
        return scale_feat_array


    def _train_one_tree(self, cur_tree_idx: int,
                        train_preds: torch.FloatTensor
                        ) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        
        idx_f = list(range(self.X_train.shape[1]))
        idx_s = list(range(self.X_train.shape[0]))
        
        # n_features = int(self.X_train.shape[1] * self.colsample_bytree)
        # n_samples =  int(self.X_train.shape[0] * self.subsample)
        
        lambdas = torch.FloatTensor(torch.zeros(self.ys_train.shape))
        
        for qid in set(self.query_ids_train):
            mask = np.where(self.query_ids_train == qid)
            lambdas[mask] = self._compute_lambdas(self.ys_train[mask], train_preds[mask])
        
        one_tree = DecisionTreeRegressor(random_state=cur_tree_idx, 
                                         max_depth=self.max_depth,
                                         min_samples_leaf=self.min_samples_leaf)
        
        features = idx_f # np.random.choice(idx_f, n_features)
        samples = idx_s # np.random.choice(idx_s, n_samples)
        
        X = self.X_train[samples, :][:, features]
        y = lambdas.flatten().numpy()[samples]
        one_tree.fit(X, y)
        
        return one_tree, features


    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
        ndcgs = []
        for qid in set(queries_list):
            mask = np.where(queries_list==qid)
            batch_pred = preds[mask]
            batch_true = true_labels[mask]
            ndcg = self._ndcg_k(ys_true=batch_true.flatten(), ys_pred=batch_pred.flatten(), ndcg_top_k=self.ndcg_top_k)
            ndcgs.append(ndcg)

        return float(np.mean(ndcgs))

    def fit(self):
        np.random.seed(0)
        train_preds = torch.FloatTensor(np.zeros_like(self.ys_train))
        test_preds = torch.FloatTensor(np.zeros_like(self.ys_test))
        
        trees_ = []
        trees_features_ = []
        for k in tqdm(range(self.n_estimators)):
            tree, features = self._train_one_tree(k, train_preds)
            
            curr_train_preds = torch.FloatTensor(tree.predict(self.X_train[:, features])).reshape(-1, 1)
            train_preds -= self.lr * curr_train_preds
            
            curr_test_preds = torch.FloatTensor(tree.predict(self.X_test[:, features])).reshape(-1, 1)
            test_preds -= self.lr * curr_test_preds
            
            ndcg = self._calc_data_ndcg(self.query_ids_test, self.ys_test, test_preds)
            
            if ndcg >= self.best_ndcg:
                self.best_ndcg, self.best_tree_idx = ndcg, k + 1
                
            trees_.append(tree)
            trees_features_.append(features)
        
        self.trees = trees_[:self.best_tree_idx + 1]
        self.trees_features = trees_features_[:self.best_tree_idx + 1]

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        
        y_pred = torch.FloatTensor(np.zeros((data.shape[0], 1)))
        for i in range(self.best_tree_idx + 1):
            i_pred = self.trees[i].predict(data[:, self.trees_features[i]])
            y_pred -= self.lr * torch.FloatTensor(i_pred).reshape(-1, 1)

        return y_pred


    def _compute_labels_in_batch(self, y_true):
        rel_diff = y_true - y_true.t()

        pos_pairs = (rel_diff > 0).type(torch.float32)

        neg_pairs = (rel_diff < 0).type(torch.float32)
        Sij = pos_pairs - neg_pairs
        return Sij


    def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        ideal_dcg = self._dcg(y_true.flatten(), y_true.flatten(), -1)
        if ideal_dcg == 0:
            N = 0
        else:
            N = 1 / ideal_dcg

        _, rank_order = torch.sort(y_true, descending=True, axis=0)
        rank_order += 1

        with torch.no_grad():
            pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))

            Sij = self._compute_labels_in_batch(y_true)
            gain_diff = self._compute_gain_diff(y_true)

            decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            lambda_update =  (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)

            return lambda_update


    def _compute_gain_diff(self, y_true, gain_scheme: str = 'exp2'):
        if gain_scheme == "exp2":
            gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())
        elif gain_scheme == "diff":
            gain_diff = y_true - y_true.t()
        else:
            raise ValueError(f"{gain_scheme} method not supported")
        return gain_diff
    
    
    def _dcg(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, dcg_top_k: int) -> float:
        _, idxs = torch.sort(ys_pred, descending=True)
        if dcg_top_k == -1:
            return float(sum([float((2**(i) - 1)/math.log2(idx+2)) for idx,i in enumerate(ys_true[idxs])]))
        else:
            return float(sum([float((2**(i) - 1)/math.log2(idx+2)) for idx,i in enumerate(ys_true[idxs][:dcg_top_k])]))
    
    
    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        dcg_val = self._dcg(ys_true, ys_pred, ndcg_top_k)
        dcg_ideal = self._dcg(ys_true, ys_true, ndcg_top_k)
        if dcg_ideal == 0:
            return 0
        return dcg_val / dcg_ideal

    
    def save_model(self, path: str):
        state = self.__dict__.copy()
        with open(path, 'wb') as wfile:
            pickle.dump(state, wfile)

    def load_model(self, path: str):
        with open(path, 'rb') as rfile:
            self.__dict__ = pickle.load(rfile)
