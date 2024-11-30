import torch
import torch.nn as nn
import torch.nn.functional as func

global expand
expand = 1


class test_soft_add(nn.Module):

    def __init__(self):
        super().__init__()

    def trans(self, outtensor):
        device = torch.device("cuda")
        outtensor.cuda(device)
        return outtensor

    def forward(self, pre, tar):
        loss = 0
        device = torch.device("cuda")
        for j in range(0, tar.size(0)):
            prenew = pre[j]
            tarnew = tar[j]

            out = torch.zeros(25)
            # out = self.trans(out)
            out = out.cuda(device)

            # print(out)
            for i in range(0, 6):
                oneStatus = prenew[i * 4:(i + 1) * 4]
                out[i * 4:(i + 1) * 4] = 3 * expand * torch.softmax(oneStatus, 0)
            out[24]=torch.relu(prenew[24])
            # if prenew[24]>0.8:
            #     out[24]=1
            # elif prenew[24]<-0.8 :
            #     out[24]=-1
            # else :
            #     out[24]=prenew[24]

            #out[24] = torch.relu(prenew[24])
            #out[24] = prenew[24]
            loss += torch.mean(((out[0:24] - tarnew[0:24]).pow(2)))#-(out[24]*torch.log(out[24]) - (1-tarnew[24])*torch.log(1-tarnew[24]))
        # print("out it",out)
        # print("target is ", tarnew)
        return loss / tar.size(0)


class loss_soft_add(nn.Module):

    def __init__(self):
        super().__init__()

    def trans(self, outtensor):
        device = torch.device("cuda")
        outtensor.cuda(device)
        return outtensor

    def forward(self, pre, tar):
        loss = 0
        device = torch.device("cuda")
        for j in range(0, tar.size(0)):
            prenew = pre[j]
            tarnew = tar[j]

            out = torch.zeros(25).cuda(device)
            for i in range(0, 6):
                oneStatus = prenew[i * 4:(i + 1) * 4]
                out[i * 4:(i + 1) * 4] = 3 * expand * torch.softmax(oneStatus, 0)
            out[24] = torch.relu(prenew[24])

            # 修改为 L1 损失
            loss += torch.mean(torch.abs(out[0:24] - tarnew[0:24]))
        return loss


class test_recall(nn.Module):  # 注意这个召回率不是严格的召回率
    def __init__(self):
        super().__init__()

    def trans(self, outtensor):
        device = torch.device("cuda")
        outtensor.cuda(device)
        return outtensor

    def forward(self, pre, tar):
        loss = 0
        device = torch.device("cuda")
        batchsize = tar.size(0)
        for j in range(0, tar.size(0)):
            prenew = pre[j]
            tarnew = tar[j]
            out = torch.zeros(25)
            # out = self.trans(out)
            out = out.cuda(device)
            # print(out)
            for i in range(0, 6):
                oneStatus = prenew[i * 4:(i + 1) * 4]
                out[i * 4:(i + 1) * 4] = 3 * expand * torch.softmax(oneStatus, 0)

            #out[24] = torch.relu(prenew[24])
            #out[24] = prenew[24]
            #out[24] = pre[24]
            out[24] = torch.relu(prenew[24])
            if j == tar.size(0) - 1:
                print("out is ", out)
                print("target is ", tarnew)
            acc_recall = 0
            calc_num = 0
            for i in range(0, 6):
                oneStatus_tar = tarnew[i * 4:(i + 1) * 4]
                oneStatus_out = out[i * 4:(i + 1) * 4]
                jump_judge = judge_reall(3 * expand, 0.03, 0.01, oneStatus_out, oneStatus_tar)
                if jump_judge is False:
                    judge = torch.sum(oneStatus_tar == 0)
                    if judge < 2:

                        calc_num += 1
                        sort1 = torch.argsort(oneStatus_out)
                        sort2 = torch.argsort(oneStatus_tar)
                        if (sort1 == sort2).all():
                            acc_recall += 1
                    else:
                        sort1 = torch.argsort(oneStatus_out)
                        sort2 = torch.argsort(oneStatus_tar)
                        if judge == 2:
                            if sort1[2] == sort2[2] and sort1[3] == sort2[3]:
                                calc_num += 1
                                acc_recall += 1
                        if judge == 3:
                            if sort1[3] >= 3 * expand * 0.9:
                                calc_num += 1
                                acc_recall += 1
                else:
                    calc_num += 1
                    acc_recall += 1
            if calc_num != 0:
                loss += acc_recall / calc_num
            else:
                batchsize -= 1
        if batchsize == 0:
            print("error")
            return 0
        else:
            return loss / batchsize


def judge_reall(all, ratio1, ratio2, out, tar):  # all表示四个位置总和，ratio表示误差比例
    error = all * ratio1
    diff = out - tar
    diff = torch.abs(diff)
    judge = torch.sum(diff <= error)
    if judge != 4:  # 判断是否处于error区间内
        return False  # 直接进行排序判断
    sort_tar, tagtar = torch.sort(tar)
    sort_out, tagout = torch.sort(out)
    diff1 = torch.abs(sort_tar[0] - sort_tar[1])
    diff2 = torch.abs(sort_tar[2] - sort_tar[1])
    if diff1 < all * ratio2 and diff2 < all * ratio2:
        return True
    if diff1 < all * ratio2:
        indxout = torch.nonzero(tagout == 2).squeeze()
        indxtar = torch.nonzero(tagtar == 2).squeeze()
        if indxout == indxtar:
            return True  # 表示排序成立
    if diff2 < all * ratio2:
        indxout = torch.nonzero(tagout == 0).squeeze()
        indxtar = torch.nonzero(tagtar == 0).squeeze()
        if indxout == indxtar:
            return True  # 表示排序成立
    return False
