# -*- coding: utf-8 -*-
# @Time    : 2024/4/9 16:12
# @Author  : Karry Ren

""" The pre-train stage: Contrastive Mechanisms. """

import copy
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sacred import Ingredient


def model_config():
    # architecture
    input_shape = [6, 60]
    rnn_type = 'LSTM'  # LSTM/GRU
    rnn_layer = 2
    hid_size = 64
    dropout = 0
    # optimization
    optim_method = 'Adam'
    optim_args = {'lr': 1e-3}
    loss_fn = 'mse'
    eval_metric = 'corr'
    verbose = 500
    max_steps = 50
    early_stopping_rounds = 5
    negative_sample = 5


class CM_Model(nn.Module):
    def __init__(
            self, input_shape: List[List[int]], rnn_type: str = "LSTM", rnn_layer: int = 2,
            hidden_size: int = 64, dropout: float = 0.0,
            optim_method: str = "Adam", optim_params: dict = {"lr": 1e-3}, loss_fn: str = "mse",
            eval_metric: str = "corr", negative_sample_num: int = 5
    ):
        """ The init function of CM_Model.

        :param input_shape: the input shape list [[D, K1, T], [D, K2, T]]
        :param rnn_type: the type of rnn
        :param rnn_layer: the layer of rnn
        :param hidden_size: the hidden size of rnn
        :param dropout: the dropout ratio
        :param optim_method: the optimizing method
        :param optim_params: the optimizing param
        :param loss_fn: the loss function
        :param eval_metric: the eval metric
        :param negative_sample_num: the number of negative sample when using the 2 contrastive learning

        """

        super(CM_Model, self).__init__()

        # ---- Get the param ---- #
        self.hidden_size = hidden_size
        self.input_size = input_shape[0][0]  # D
        self.input_day = input_shape[0][2]  # T
        self.input_daily_K = input_shape[0][1]  # K1
        self.input_hf_K = input_shape[1][1]  # K2
        self.dropout = dropout
        self.rnn_layer = rnn_layer
        self.rnn_type = rnn_type
        self.negative_sample_num = negative_sample_num

        # ---- Using the param to buidl model ---- #
        self._build_model()

        # ---- The optimizer ---- #
        self.optimizer = getattr(optim, optim_method)(self.parameters(), **optim_params)

        # self.loss_fn = get_loss_fn(loss_fn)
        # self.metric_fn = get_metric_fn(eval_metric)
        #
        # if torch.cuda.is_available():
        #     self.cuda()

    def _build_model(self):
        # ---- Build the rnn ---- #
        try:
            rnn_module = getattr(nn, self.rnn_type.upper())
        except:
            raise ValueError(f"unknown rnn_type `{self.rnn_type}` !!!")

        # ---- Encoder 1. ---- #
        # for daily data
        self.net_daily_1 = nn.Sequential()
        self.net_daily_1.add_module("fc_in_daily_1", nn.Linear(in_features=self.input_size, out_features=self.hidden_size))
        self.net_daily_1.add_module("act_daily_1", nn.Tanh())
        # for hf data
        self.net_hf_1 = nn.Sequential()
        self.net_hf_1.add_module("fc_in_hf_1", nn.Linear(in_features=self.input_size, out_features=self.hidden_size))
        self.net_hf_1.add_module("act_hf_1", nn.Tanh())
        self.rnn_hf_1 = rnn_module(
            input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.rnn_layer, batch_first=True, dropout=self.dropout
        )

        # ---- Encoder 2. ---- #
        # for daily data
        self.net_daily_2 = nn.Sequential()
        self.net_daily_2.add_module("fc_in_daily_2", nn.Linear(in_features=self.input_size, out_features=self.hidden_size))
        self.net_daily_2.add_module("act_daily_2", nn.Tanh())
        # for hf data
        self.net_hf_2 = nn.Sequential()
        self.net_hf_2.add_module("fc_in_hf_2", nn.Linear(in_features=self.input_size, out_features=self.hidden_size))
        self.net_hf_2.add_module('act_hf_2', nn.Tanh())
        self.rnn_hf_2 = rnn_module(
            input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.rnn_layer, batch_first=True, dropout=self.dropout
        )

        self.rnn_daily = rnn_module(
            input_size=self.hidden_size * 2, hidden_size=self.hidden_size, num_layers=self.rnn_layer, batch_first=True, dropout=self.dropout
        )

        self.fc_pred = nn.Linear(in_features=self.hidden_size, out_features=1)  # output fc
        self.fc_point = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)  # point contrast weight
        self.fc_trend = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size * 2)  # trend contrast weight

    def forward(self, x_daily: torch.Tensor, x_hf: torch.Tensor):
        """ Forward function of CM_Model.

        :param x_daily: the daily feature, shape=(bs, D, 1, T)
        :param x_hf: the high frequency feature, shape=(bs, D, K, T)

        """

        # ---- Step 1. Encoding the high freq feature ---- #
        x_hf = x_hf.permute(0, 3, 1, 2)  # shape from (bs, D, K, T) to (bs, T, D, K)
        out_arr_hf_1, out_arr_hf_2 = [], []
        for i in range(self.input_day):  # day by day
            # prepare the input
            x_hf_1day = x_hf[:, i, :, :]  # shape=(bs, D, K)
            x_hf_1day = x_hf_1day.permute(0, 2, 1)  # shape from (bs, D, K) to (bs, K, D)
            # encoder 1
            out_hf_1, _ = self.rnn_hf_1(self.net_hf_1(x_hf_1day))  # fc the x_hf_1day to (bs, K, hidden_size)
            out_hf_1 = out_hf_1[:, -1, :].unsqueeze(1)  # get the last step hidden state, (bs, 1, hidden_size)
            out_arr_hf_1.append(out_hf_1)
            # encoder 2
            out_2, _ = self.rnn_hf_2(self.net_hf_2(x_hf_1day))
            out_2 = out_2[:, -1, :].unsqueeze(1)  # get the last step hidden state, (bs, 1, hidden_size)
            out_arr_hf_2.append(out_2)
        x_hf_day_reps_1 = torch.cat(out_arr_hf_1, dim=1)  # shape from input_day*(bs, 1, hidden_size) to (bs, input_day, input_size)
        x_hf_day_reps_2 = torch.cat(out_arr_hf_2, dim=1)  # shape from input_day*(bs, 1, hidden_size) to (bs, input_day, input_size)

        # ---- Step 2. Encoding the daily feature ---- #
        x_daily = x_daily.reshape(-1, self.input_size, self.input_day)  # shape from (bs, D, 1, T) to (bs, D, T)
        x_daily = x_daily.permute(0, 2, 1)  # shape from (bs, D, T) to (bs, T, D)
        x_daily_1 = self.net_daily_1(x_daily)  # shape from (bs, T, D) to (bs, T, hidden_size)
        x_daily_2 = self.net_daily_2(x_daily)  # shape from (bs, T, D) to (bs, T, hidden_size)

        # ---- Step 3. Get prediction ---- #
        pred_1, _ = self.rnn_daily(torch.cat((x_daily_2 + x_hf_day_reps_2, x_daily_1 + x_hf_day_reps_2), dim=2))
        out = self.fc_pred(pred_1[:, -1, :])  # (bs, 1)

        # ---- Step 4. Point contrast ---- #
        context = self.fc_point(x_daily_1)  # (bs, T, hidden_size)
        point_contrast_loss = 0
        for i in range(self.input_day):  # for-loop each day
            daily_input = context[:, i:i + 1, :]  # shape from (batch, T, hidden_size) to (batch, 1, hidden_size)
            hf_daily_input = x_hf_day_reps_1[:, i:i + 1, :]  # shape from (batch, T, hidden_size) to (batch, 1, hidden_size)
            pn_hf_daily_input = self._generate_hf_data(hf_daily_input)  # generate the negative daily input, (batch, 1+n_num, hidden_size)
            dot_product = torch.mean(pn_hf_daily_input * daily_input, -1)  # shape=(batch, 1+negative_sample_num)
            log_l1 = torch.nn.functional.log_softmax(dot_product, dim=1)[:, 0]  # only get the positive one ! (bs)
            point_contrast_loss += -torch.mean(log_l1)  # pair contrast loss

        # ---- Step 5. Trend contrast ---- #
        new_data_daily = torch.reshape(
            torch.cat((x_daily_2 + x_hf_day_reps_2, x_daily_1 + x_hf_day_reps_2), dim=2),
            (-1, self.input_day * 2 * self.hidden_size)
        )  # shape=(bs, T*2*hidden_size)
        new_data_daily = self._generate_data(new_data_daily, size=self.hidden_size * 2)
        new_data_daily = torch.reshape(new_data_daily, (-1, self.input_day + self.negative_sample_num, self.hidden_size * 2))
        next = new_data_daily[:, -1 - self.negative_sample_num:, :]  # last step, shape=(bs, 1+negative_sample_num, hidden_size)

        context_trend = self.fc_trend(pred_1[:, -2, :])  # last 2, shape=(bs, hidden_size)
        context_trend = torch.unsqueeze(context_trend, 1)  # shape from (bs, hidden_size) to shape=(bs, 1, hidden_size)
        dot_product_trend = torch.mean(next * context_trend, -1)  # can be changed to bmm no need mean
        log_lf2 = torch.nn.functional.log_softmax(dot_product_trend, dim=1)[:, 0]  # only the positive one
        trend_contrast_loss = -torch.mean(log_l2)

        return 0.05 * point_contrast_loss + trend_contrast_loss, out[..., 0]

    def fit(self,
            train_set,
            valid_set,
            run=None,
            max_steps=100,
            early_stopping_rounds=10,
            verbose=100):
        best_score = np.inf
        stop_steps = 0
        best_params = copy.deepcopy(self.state_dict())

        for step in range(max_steps):

            pprint('Step:', step)
            if stop_steps >= early_stopping_rounds:
                if verbose:
                    pprint('\tearly stop')
                break
            stop_steps += 1
            # training
            self.train()
            train_loss = AverageMeter()
            train_eval = AverageMeter()
            for i, (data_daily, data_highfreq, label) in enumerate(train_set):
                data_daily = torch.tensor(data_daily, dtype=torch.float)
                data_highfreq = torch.tensor(data_highfreq, dtype=torch.float)
                label = torch.tensor(label, dtype=torch.float)
                if torch.cuda.is_available():
                    data_daily, data_highfreq, label = data_daily.cuda(), data_highfreq.cuda(), label.cuda()
                loss_contrast, pred = self(data_daily, data_highfreq)
                loss = self.loss_fn(pred, label)
                loss = loss + loss_contrast
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_ = loss.item()
                eval_ = self.metric_fn(pred, label).item()
                train_loss.update(loss_, len(data_daily))
                train_eval.update(eval_)

                if verbose and i % verbose == 0:
                    pprint('iter %s: train_loss %.6f, train_eval %.6f' %
                           (i, train_loss.avg, train_eval.avg))
            # evaluation
            self.eval()
            valid_loss = AverageMeter()
            valid_eval = AverageMeter()
            for i, (data_daily, data_highfreq, label) in enumerate(valid_set):
                data_daily = torch.tensor(data_daily, dtype=torch.float)
                data_highfreq = torch.tensor(data_highfreq, dtype=torch.float)
                label = torch.tensor(label, dtype=torch.float)
                if torch.cuda.is_available():
                    data_daily, data_highfreq, label = data_daily.cuda(), data_highfreq.cuda(), label.cuda()
                with torch.no_grad():
                    loss_contrast, pred = self(data_daily, data_highfreq)
                loss = self.loss_fn(pred, label)
                loss = loss + loss_contrast
                valid_loss_ = loss.item()
                valid_eval_ = self.metric_fn(pred, label).item()
                valid_loss.update(valid_loss_, len(data_daily))
                valid_eval.update(valid_eval_)

            if run is not None:
                run.add_scalar('Train/Loss', train_loss.avg, step)
                run.add_scalar('Train/Eval', train_eval.avg, step)
                run.add_scalar('Valid/Loss', valid_loss.avg, step)
                run.add_scalar('Valid/Eval', valid_eval.avg, step)
            if verbose:
                pprint("current step: train_loss {:.6f}, valid_loss {:.6f}, "
                       "train_eval {:.6f}, valid_eval {:.6f}".format(
                    train_loss.avg, valid_loss.avg, train_eval.avg,
                    valid_eval.avg))
            if valid_eval.avg < best_score:
                if verbose:
                    pprint(
                        '\tvalid update from {:.6f} to {:.6f}, save checkpoint.'
                        .format(best_score, valid_eval.avg))
                best_score = valid_eval.avg
                stop_steps = 0
                best_params = copy.deepcopy(self.state_dict())
        # restore
        self.load_state_dict(best_params)

    def _random_select_ng_sample(self, data, size):
        """ Generate the negative data from the same mini-batch.

        :param data: raw high-frequency data (positive), shape=(bs, 1, hidden_size)

        return:
            - new_data, shape=(bs, 1+negative_sample_num, hidden_size), the first one is positive

        """

        new_data = data.clone()
        data_lastday = data.clone()[:, -size:]
        for i in range(self.negative_sample_num):
            random_list_1 = torch.randperm(data_lastday.size(0))
            random_list_2 = torch.randperm(data_lastday.size(0))
            random_list = torch.where(random_list_1 - torch.arange(data_lastday.size(0)) == 0, random_list_2, random_list_1)
            random_lastday = data_lastday[random_list]
            new_data = torch.cat((new_data, random_lastday), 1)
        return new_data

    def _generate_hf_data(self, raw_hf_data: torch.Tensor):
        """ Generate the negative high-frequency data from the same mini-batch.

        :param raw_hf_data: raw high-frequency data (positive), shape=(bs, 1, hidden_size)

        return:
            - new_data, shape=(bs, 1+negative_sample_num, hidden_size), the first one is positive

        """

        # ---- Step 1. Clone the data ---- #
        new_data = raw_hf_data.clone()

        # ---- Step 2. Generate the negative sample ---- #
        for i in range(self.negative_sample_num):
            random_list = torch.randperm(raw_hf_data.shape[0])  # [0, bs-1] random
            for j in range(raw_hf_data.shape[0]):
                if j != random_list[j]:
                    continue
                if j == random_list[j] and j == 0:
                    random_list[j] = torch.randint(1, raw_hf_data.shape[0], (1,))[0]
                elif j == random_list[j]:
                    random_list[j] = torch.randint(0, j, (1,))[0]
            random_data = raw_hf_data[random_list]  # gen 1 negative sample for each sample
            new_data = torch.cat((new_data, random_data), 1)
        return new_data

    def predict(self, test_set):
        self.eval()
        preds = []
        for i, (data_daily, data_highfreq, _) in enumerate(test_set):
            data_daily = torch.tensor(data_daily, dtype=torch.float)
            data_highfreq = torch.tensor(data_highfreq, dtype=torch.float)

            if torch.cuda.is_available():
                data_daily = data_daily.cuda()
                data_highfreq = data_highfreq.cuda()
            with torch.no_grad():
                preds.append(self(data_daily, data_highfreq)[1].cpu().numpy())
        return np.concatenate(preds)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path, strict=True):
        self.load_state_dict(torch.load(model_path), strict=strict)


if __name__ == '__main__':  # test the model
    bs, T, D = 2, 2, 3
    x_daily = torch.randn((bs, D, 1, T))
    x_1h = torch.randn((bs, D, 4, T))
    model = CM_Model(input_shape=[[3, 1, 2], [3, 4, 2]])
    y = model(x_daily, x_1h)
