from ptflops import get_model_complexity_info
import torch
from try_resnet_0706 import ResNet
from try_resnet_0706 import BasicBlock


def main():
    device = torch.device("cuda")
    net = ResNet(BasicBlock, [1, 1, 1, 1])
    net = net.cuda(device)
    net.load_state_dict(torch.load('0830_1111_Alpha1.pt'))
    net.eval()

    macs, params = get_model_complexity_info(net, (10, 5, 5), as_strings=True, print_per_layer_stat=False)
    del net

    print('-{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('+{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == '__main__':
    main()
