from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from loss_function_0712Alpha import loss_soft_add
from loss_function_0712Alpha import test_soft_add
from loss_function_0712Alpha import test_soft_batch

def pgd_attack(model,
               X,
               y,
               device,
               epsilon=0.5,
               num_steps=20,
               step_size=0.1,
               random = True,
               ):
    X_pgd = Variable(X, requires_grad=True)
    X, y = Variable(X, requires_grad=True), Variable(y)
    criterion = loss_soft_add().cuda(device)
    criterion_test = test_soft_add().cuda(device)
    out = model(X)
    loss_nat = criterion(out, y)
    # accuracy = criterion_test(out, y)
    acc = (test_soft_batch()(out, y) < 0.125).sum().item() / X.shape[0]
    # acc = (out.data.max(1)[1] == y.data).sum().item()
    if random:
        random_noise = torch.FloatTensor(*X.shape).uniform_(-epsilon, epsilon).to(X.device)
        X_pgd = Variable(torch.clamp(X.data + random_noise, 0.0, 12.0), requires_grad=True)

    for _ in range(num_steps):
        # opt = optim.SGD([X_pgd], lr=1e-3)
        # opt.zero_grad()

        with torch.enable_grad():
            loss = criterion_test(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        eta += X_pgd.data - X.data
        eta = torch.clamp(eta, -epsilon, epsilon)
        X_pgd = Variable(torch.clamp(X.data + eta, 0, 12), requires_grad=True)
    X_pgd = torch.round(X_pgd)
    out_pgd = model(X_pgd)
    loss_pgd = criterion(out_pgd, y)
    # accuracy_2 = criterion_test(out_pgd, y)
    acc_pgd = (test_soft_batch()(out_pgd, y) < 0.125).sum().item() / X.shape[0]
    # acc_pgd = (out_pgd.data.max(1)[1] ==y.data).sum().item()
    #print('err pgd (white-box): ', err_pgd)
    return loss_nat, loss_pgd, acc, acc_pgd

