import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import torch.utils.data
import numpy as np
import math


def to_var(x, requires_grad=False):
    """
    Automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return x.clone().detach().requires_grad_(requires_grad)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False

    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight * self.mask
            return F.conv2d(x, weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)


class ConvNet(nn.Module):
    def __init__(self):#,args
        super(ConvNet, self).__init__()

        self.conv1 = MaskedConv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = MaskedConv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv3 = MaskedConv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = MaskedConv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv5 = MaskedConv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = MaskedConv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = MaskedConv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv8 = MaskedConv2d(256, 512, kernel_size=3, padding=1, stride=1)
        self.relu8 = nn.ReLU(inplace=True)

        self.conv9 = MaskedConv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.relu9 = nn.ReLU(inplace=True)

        self.conv10 = MaskedConv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.relu10 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(2)

        self.conv11 = MaskedConv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.relu11 = nn.ReLU(inplace=True)

        self.conv12 = MaskedConv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.relu12 = nn.ReLU(inplace=True)

        self.conv13 = MaskedConv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.relu13 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(2)

        self.classifier = nn.Sequential(
            # 14, 512=>4096
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 15, 4096=>4096
            nn.Linear(4096, 10),
            # nn.ReLU(True),
            # nn.Dropout(),
            # # 16, 4096=>output_sizes
            # nn.Linear(256, 10),
        )

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.maxpool1(self.relu2(self.conv2(out)))
        out = self.relu3(self.conv3(out))
        out = self.maxpool2(self.relu4(self.conv4(out)))
        out = self.relu5(self.conv5(out))
        out = self.relu6(self.conv6(out))
        out = self.maxpool3(self.relu7(self.conv7(out)))
        out = self.relu8(self.conv8(out))
        out = self.relu9(self.conv9(out))
        out = self.maxpool4(self.relu10(self.conv10(out)))
        out = self.relu11(self.conv11(out))
        out = self.relu12(self.conv12(out))
        out = self.maxpool5(self.relu13(self.conv13(out)))
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.classifier(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.conv1.set_mask(torch.from_numpy(masks[0]))
        self.conv2.set_mask(torch.from_numpy(masks[1]))
        self.conv3.set_mask(torch.from_numpy(masks[2]))
        self.conv4.set_mask(torch.from_numpy(masks[3]))
        self.conv5.set_mask(torch.from_numpy(masks[4]))
        self.conv6.set_mask(torch.from_numpy(masks[5]))
        self.conv7.set_mask(torch.from_numpy(masks[6]))
        self.conv8.set_mask(torch.from_numpy(masks[7]))
        self.conv9.set_mask(torch.from_numpy(masks[8]))
        self.conv10.set_mask(torch.from_numpy(masks[9]))
        self.conv11.set_mask(torch.from_numpy(masks[10]))
        self.conv12.set_mask(torch.from_numpy(masks[11]))
        self.conv13.set_mask(torch.from_numpy(masks[12]))


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)
        print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
              (epoch, total, len(train_loader.dataset),
               '-' * progress + '>', progress * 2), end='')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)


"""Reference https://github.com/zepx/pytorch-weight-prune/"""


def prune_rate(model, verbose=False):
    """
    Print out prune rate for each layer and the whole network
    """
    total_nb_param = 0
    nb_zero_param = 0

    layer_id = 0

    for parameter in model.parameters():

        param_this_layer = 1
        for dim in parameter.data.size():
            param_this_layer *= dim
        total_nb_param += param_this_layer

        # only pruning linear and conv layers
        if len(parameter.data.size()) != 1:
            layer_id += 1
            zero_param_this_layer = \
                np.count_nonzero(parameter.cpu().data.numpy() == 0)
            nb_zero_param += zero_param_this_layer

            if verbose:
                print("Layer {} | {} layer | {:.2f}% parameters pruned" \
                    .format(
                    layer_id,
                    'Conv' if len(parameter.data.size()) == 4 \
                        else 'Linear',
                    100. * zero_param_this_layer / param_this_layer,
                ))
    pruning_perc = 100. * nb_zero_param / total_nb_param
    if verbose:
        print("Final pruning rate: {:.2f}%".format(pruning_perc))
    return pruning_perc


def arg_nonzero_min(a):
    """
    nonzero argmin of a non-negative array
    """

    if not a:
        return

    min_ix, min_v = None, None
    # find the starting value (should be nonzero)
    for i, e in enumerate(a):
        if e != 0:
            min_ix = i
            min_v = e
    if not min_ix:
        print('Warning: all zero')
        return np.inf, np.inf

    # search for the smallest nonzero
    for i, e in enumerate(a):
        if e < min_v and e != 0:
            min_v = e
            min_ix = i

    return min_v, min_ix


def prune_one_filter(model, masks):
    '''
    Pruning one least ``important'' feature map by the scaled l2norm of kernel weights
    arXiv:1611.06440
    '''
    NO_MASKS = False
    # construct masks if there is not yet
    if not masks:
        masks = []
        NO_MASKS = True

    values = []
    for p in model.parameters():

        if len(p.data.size()) == 4:  # nasty way of selecting conv layer
            p_np = p.data.cpu().numpy()

            # construct masks if there is not
            if NO_MASKS:
                masks.append(np.ones(p_np.shape).astype('float32'))

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1) \
                                   .sum(axis=1) / (p_np.shape[1] * p_np.shape[2] * p_np.shape[3])
            # normalization (important)
            value_this_layer = value_this_layer / \
                               np.sqrt(np.square(value_this_layer).sum())
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            values.append([min_value, min_ind])

    assert len(masks) == len(values), "something wrong here"

    values = np.array(values)

    # set mask corresponding to the filter to prune
    to_prune_layer_ind = np.argmin(values[:, 0])
    to_prune_filter_ind = int(values[to_prune_layer_ind, 1])
    masks[to_prune_layer_ind][to_prune_filter_ind] = 0.

    #     print('Prune filter #{} in layer #{}'.format(
    #         to_prune_filter_ind,
    #         to_prune_layer_ind))
    return masks


def filter_prune(model, pruning_perc):
    '''
    Prune filters one by one until reach pruning_perc(not iterative pruning)
    '''
    masks = []
    current_pruning_perc = 0.

    while current_pruning_perc < pruning_perc:
        masks = prune_one_filter(model, masks)
        model.set_masks(masks)
        current_pruning_perc = prune_rate(model, verbose=False)
    #         print('{:.2f} pruned'.format(current_pruning_perc))
    return masks


# def main():
#     epochs = 8
#     batch_size = 64
#     torch.manual_seed(0)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     trainset = torchvision.datasets.CIFAR10(root='./data/cifar', train=True, transform=transforms.ToTensor(), download=True)
#     testset = torchvision.datasets.CIFAR10(root='./data/cifar', train=False, transform=transforms.ToTensor(), download=True)
#     train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
#
#     model = ConvNet().to(device)
#     optimizer = torch.optim.Adadelta(model.parameters())
#
#     for epoch in range(1, epochs + 1):
#         train(model, device, train_loader, optimizer, epoch)
#         _, acc = test(model, device, test_loader)
#
#     print('\npruning 50%')
#     mask = filter_prune(model, 50)
#     model.set_masks(mask)
#     _, acc = test(model, device, test_loader)
#
#     # finetune
#     print('\nfinetune')
#     train(model, device, train_loader, optimizer, epoch)
#     _, acc = test(model, device, test_loader)
#
# main()