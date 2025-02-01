import time
from flcore.clients.clientfedrg import clientRG
from flcore.servers.serverbase import Server
from flcore.optimizers.aggregator import rep_aggregate
from torch.utils.data import DataLoader
import copy
import torch
import torch.nn as nn
from utils.data_utils import normalize_adj
import torch.nn.functional as F

class FedRG(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.uploaded_protos_head = None
        self.client_class_weight = None
        self.uploaded_protos = None
        self.set_slow_clients()
        self.set_clients(clientRG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.CEloss = nn.CrossEntropyLoss()
        self.server_learning_rate = args.server_learning_rate

        self.head = self.clients[0].model.head
        self.opt_h = torch.optim.SGD(self.head.parameters(), lr=self.server_learning_rate)

        self.classes_id = [copy.deepcopy(c.classes_index) for c in self.clients]
        self.per_Rep = [None for _ in range(self.num_classes)]

        # self.global_model.head = copy.deepcopy(args.global_model.head)

    def train(self):

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_protos()
            self.train_head()
            self.set_rep()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

    def set_rep(self):

        assert (len(self.clients) > 0)
        # class_weight = calculate_model_performance(self.head, self.uploaded_ids, self.uploaded_protos)

        Rep_list = [[] for _ in range(self.num_classes)]
        Rep_id_list = [[] for _ in range(self.num_classes)]
        # data_weight = [[] for _ in range(self.num_classes)]
        re_weights = [[] for _ in range(self.num_classes)]

        if len(self.uploaded_ids) > 0:
            for client_id, client_protos in zip(self.uploaded_ids, self.uploaded_protos):
                for [label, proto_list] in client_protos.items():
                    # model_weight = calculate_data_performance(self.head, proto_list, label)
                    re_weight_class = calculate_performance(self.head, proto_list, label)
                    re_weights[label].append(re_weight_class)

                    Rep_list[label].append(proto_list)
                    Rep_id_list[label].append(client_id)
                    # perf_weight[label].append(model_weight)
                    # data_weight[label].append(class_weight[client_id].item())

            for cla in range(self.num_classes):
                if len(Rep_list[cla]) > 0:
                    re_weight = torch.tensor(re_weights[cla])
                    similar = cos_similar(Rep_list[cla])

                    #GNN
                    a_weight = re_weight.unsqueeze(1) * similar
                    self.per_Rep[cla] = rep_aggregate(self.args, Rep_list[cla], a_weight)

                    #RE
                    # self.per_Rep[cla] = proto_aggregation(re_weight, Rep_list[cla])
        #Re
        # for client in self.clients:
        #     client.set_rep(self.per_Rep)

        # GNN
        for client, cla_list in zip(self.clients, self.classes_id):
            tensor_dict = []

            rep = [None for _ in range(self.num_classes)]
            for key, value in self.uploaded_protos[self.uploaded_ids.index(client.id)].items():
                tensor_dict.append(torch.tensor(value))

            if len(self.uploaded_ids) > 0:
                for cid in range(len(cla_list)):
                    cla = cla_list[cid].item()
                    idx = Rep_id_list[cla].index(client.id)
                    rep[cla] = self.per_Rep[cla][idx]

            client.set_rep(rep)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        self.uploaded_protos_head = []
        self.uploaded_weights = []
        self.client_class_weight = []
        tot_samples = 0
        total_per_class = torch.zeros(self.num_classes)
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)
            self.uploaded_weights.append(client.train_samples)
            total_per_class += client.sample_per_class
            
            for cc in client.protos.keys():
                y = torch.tensor(cc, dtype=torch.int64, device=self.device)
                self.uploaded_protos_head.append((client.protos[cc], y))
                
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        for client in self.selected_clients:
            self.client_class_weight.append(client.sample_per_class / total_per_class)

    def train_head(self):
        proto_loader = DataLoader(self.uploaded_protos_head, self.batch_size, drop_last=False, shuffle=True)

        for p, y in proto_loader:
            out = self.head(p)
            loss = self.CEloss(out, y)
            self.opt_h.zero_grad()
            loss.backward()
            self.opt_h.step()


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(proto_weight,protos_list):
    # 计算 L2 范数
    # norm = torch.norm(proto_weight, p=2)
    # L2 归一化
    # proto_weight_normalized = proto_weight / norm
    proto_weight_normalized = proto_weight / proto_weight.sum()

    proto = 0 * protos_list[0].data
    for weight, i in zip(proto_weight_normalized, protos_list):
        proto += i.data * weight

    return proto


# reliability1
# def calculate_model_performance(model, uploaded_ids, uploaded_protos):
#     model.eval()
#     model_class = [0.0 for _ in range(len(uploaded_ids))]
#     with torch.no_grad():
#         for client_id, client_protos in zip(uploaded_ids, uploaded_protos):
#             for [label, proto_list] in client_protos.items():
#                 output = model(proto_list)
#                 performance = F.softmax(output, dim=0)
#                 performance_excluded = torch.tensor(
#                     [performance[i].item() for i in range(len(performance)) if i != label])
#                 model_class[client_id] += performance[label].item() * (1 - torch.max(performance_excluded))
#     return model_class

# reliability1
def calculate_performance(model, prototype, cla):
    model.eval()
    with torch.no_grad():
        output = model(prototype)
        performance = F.softmax(output, dim=0)
        performance_excluded = torch.tensor(
            [performance[i].item() for i in range(len(performance)) if i!= cla])
        data_class = 1 - pow((1 - performance[cla].item()), 0.5)
        model_class = performance[cla].item() * (1 - torch.max(performance_excluded))
    return data_class * model_class.item()


# def reliability(rep_list, data_weight, perf_weight):
#     size = len(rep_list)
#     Re = torch.zeros(size)
#     cosine_sim = torch.zeros(size, size)
#
#     for i in range(size):
#         for j in range(i, size):
#             cosine_sim[i][j] = cosine_sim[j][i] = F.cosine_similarity(rep_list[i].unsqueeze(0),
#                                                                       rep_list[j].unsqueeze(0), dim=-1).item()
#     for i in range(size):
#         # for j in range(size):
#         Re[i] = data_weight[i] * perf_weight[i]
#     return Re,cosine_sim

def cos_similar(rep_list):
    size = len(rep_list)
    cos_sim = torch.zeros(size, size)

    for i in range(size):
        for j in range(i, size):
            cos_sim[i][j] = cos_sim[j][i] = F.cosine_similarity(rep_list[i].unsqueeze(0), rep_list[j].unsqueeze(0), dim=-1).item()
    return cos_sim