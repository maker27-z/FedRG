# coding=utf-8
import argparse
import time
import warnings
import numpy as np
import logging

from flcore.servers.serverfedrg import FedRG
from flcore.trainmodel.transformer import *
from utils.result_utils import average_data

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

vocab_size = 98635
max_len=200
emb_dim=32


def run(args):
    time_list = []

    for i in range(args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # select algorithm
        if args.algorithm == "FedRG":
            server = FedRG(args, i)
        else:
            raise NotImplementedError

        server.train()
        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-mst', "--model_structure", type=str, default="heterogeneity",
                        help="The model_sturcture for this experiment")
    parser.add_argument('-hete', "--hete", type=str, default="Hete3",
                        help="The model heterogeneous pool")
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-id', "--id", type=int, default=5)
    parser.add_argument('-data', "--dataset", type=str, default="cifar100",)
    parser.add_argument('-nb', "--num_classes", type=int, default=102)
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=10)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedRG")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)


    #practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # Graph Learning
    parser.add_argument('-k', "--k", type=int, default=5,  # 5
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-agg', "--agg", type=str, default='graph_v2', help='averaging strategy')
    parser.add_argument('-sal', "--serveralpha", type=float, default=0.1, help='server prop alpha')
    parser.add_argument('-las', "--layers", type=int, default=2, help='number of layers')
    parser.add_argument('-node', "--node_dim", type=int, default=40, help='dim of nodes')

    parser.add_argument('-ss', "--subgraph_size", type=int, default=200, help='k')
    parser.add_argument('-adja', "--adjalpha", type=float, default=3, help='adj alpha')
    parser.add_argument('-gce', "--gc_epoch", type=int, default=10, help='')
    parser.add_argument('-gnnwd', "--gnn_weight_decay", type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('-gnnlr', "--gnn_learning_rate", type=float, default=0.001, help='learning rate')  # 0.01

    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-dim', "--dim", type=int, default=1600)
    parser.add_argument('-indim', "--input_dim", type=int, default=3072)


    #out
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=0.01)

    args = parser.parse_args()

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices: {num_devices}")
        for i in range(num_devices):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")

    torch.cuda.set_device(args.id)

    if torch.cuda.is_available():
        current_gpu = torch.cuda.current_device()
        print(f"Current GPU device: {current_gpu}")
    else:
        print("CUDA is not available.")

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("hete pool: {}".format(args.hete))
    print("feature_dim: {}".format(args.feature_dim))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("torch.cuda.current_device()",torch.cuda.current_device())
    print("=" * 50)
    run(args)

