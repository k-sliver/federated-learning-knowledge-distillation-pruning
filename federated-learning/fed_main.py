import matplotlib
from models.Update import LocalUpdate
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from utils.sampling import cifar_noniid, cifar_iid
from utils.options import args_parser
from models.Nets import stu_model
from models.Fed import FedAvg
from models.test import test_img
import models.teacher_model as tm

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    if args.iid:
        dict_users = cifar_iid(dataset_train, args.num_users)
    else:
        dict_users = cifar_noniid(dataset_train, args.num_users)
    #
    net_glob = stu_model(args=args).to(args.device)
    # print(net_glob)

    #end
    teach_model = tm.ConvNet(args=args)
    dict = teach_model.state_dict()

    filepath = "models/teacher_model_statedict/teach_model.pth"
    torch.save(dict, filepath)
    # copy weights
    w_glob = net_glob.state_dict()

    loss_train = []
    val_acc_list, net_list = [], []

    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            net = copy.deepcopy(net_glob).to(args.device)
            w, loss = local.train(net)
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # update global weights ??????????????????
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("Testing accuracy: {:.2f}".format(acc_test))
        # print loss
        loss_avg = sum(loss_locals) / len()
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

