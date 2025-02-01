import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.optimizers.ensure_model import ensure_model
from flcore.clients.clientbase import Client
from collections import defaultdict

class clientRG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.sample_per_class = torch.zeros(self.num_classes)
        trainloader = self.load_train_data()
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        self.classes_index = []
        self.index_classes = torch.zeros(self.num_classes, dtype=torch.int64)
        for idx, c in enumerate(self.sample_per_class):
            if c > 0:
                self.classes_index.append(idx)
                self.index_classes[idx] += len(self.classes_index) - 1
        self.classes_index = torch.tensor(self.classes_index, device=self.device)
        self.cnum_classes = torch.sum(self.sample_per_class > 0).item()
        print(f'Client {self.id} has {self.cnum_classes} classes. classes_index: {self.classes_index}')

        if args.model_structure == 'heterogeneity':
            self.model = ensure_model(args, id)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

        self.protos = None
        self.reps = None
        self.loss_mse = nn.MSELoss()

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        start_time = time.time()
        max_local_epochs = self.local_epochs

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if self.reps is not None:
                    rep_new = copy.deepcopy(rep.detach())
                    for j, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.reps[y_c]) != type([]):
                            rep_new[j, :] = self.reps[y_c].data
                    loss += self.loss_mse(rep_new, rep)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.collect_protos()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)

                for j, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[j, :].detach().data)

        self.protos = agg_func(protos)

    def set_rep(self, reps):
        self.reps = reps

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

        return test_acc, test_num, 0

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if self.reps is not None:
                    rep_new = copy.deepcopy(rep.detach())
                    for j, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.reps[y_c]) != type([]):
                            rep_new[j, :] = self.reps[y_c].data
                    loss += self.loss_mse(rep_new, rep)

            train_num += y.shape[0]
            losses += loss.item() * y.shape[0]

        return losses, train_num


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos
