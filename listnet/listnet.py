import math

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List


class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


class Solution:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 30,
                 lr: float = 0.001, ndcg_top_k: int = 10):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(
            self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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
        self.ys_train = torch.FloatTensor(y_train)
        self.ys_test = torch.FloatTensor(y_test)
        

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        scale_feat_array = np.zeros_like(inp_feat_array)
        for q in set(inp_query_ids):
            idx = np.where(inp_query_ids==q)
            scaler = StandardScaler()
            scale_feat_array[idx] = scaler.fit_transform(inp_feat_array[idx])
        return scale_feat_array

    def _create_model(self, listnet_num_input_features: int,
                      listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        net = ListNet(listnet_num_input_features, listnet_hidden_dim)
        return net

    def fit(self) -> List[float]:
        metric = []
        for e in range(self.n_epochs):
            self._train_one_epoch()
            metric.append(self._eval_test_set())
        return metric
            

    def _calc_loss(self, batch_ys: torch.FloatTensor,
                   batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        P_ys = torch.softmax(batch_ys, dim=0)
        P_pred = torch.softmax(batch_pred, dim=0)
        return -torch.sum(P_ys * torch.log(P_pred))

    def _train_one_epoch(self) -> None:
        self.model.train()
        for q in set(self.query_ids_train):
            idx = np.where(self.query_ids_train==q)
            batch_X = self.X_train[idx]
            batch_ys = self.ys_train[idx]

            self.optimizer.zero_grad()
            batch_pred = self.model(batch_X)
            batch_loss = self._calc_loss(batch_ys.flatten(), batch_pred.flatten())
            batch_loss.backward()
            self.optimizer.step()

    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            for q in set(self.query_ids_test):
                idx = np.where(self.query_ids_test==q)
                valid_pred = self.model(self.X_test[idx])
                ndcg_q = self._ndcg_k(self.ys_test[idx].flatten(), valid_pred.flatten(), self.ndcg_top_k)
                ndcgs.append(ndcg_q)
            return np.mean(ndcgs)
    
    def _dcg(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, dcg_top_k: int) -> float:
        _, idxs = torch.sort(ys_pred, descending=True)
        return float(sum([float((2**(i) - 1)/math.log2(idx+2)) for idx,i in enumerate(ys_true[idxs][:dcg_top_k])]))
    
    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        dcg_val = self._dcg(ys_true, ys_pred, ndcg_top_k)
        dcg_ideal = self._dcg(ys_true, ys_true, ndcg_top_k)
        return dcg_val / dcg_ideal