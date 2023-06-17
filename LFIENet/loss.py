import torch
import torch.nn as nn

def ints_loss(raw, fusion):
    loss = torch.mean((raw - fusion) ** 2)
    return loss


def grad_loss(raw, fusion):
    laplacian = torch.tensor([[[[1, 1, 1], [1, -8, 1], [1, 1, 1]]]]).to('cuda:0').float()
    grad_raw = torch.nn.functional.conv2d(raw, laplacian, stride=1, padding=1)
    grad_fusion = torch.nn.functional.conv2d(fusion, laplacian, stride=1, padding=1)
    loss = torch.mean((grad_raw - grad_fusion) ** 2)
    return loss

class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        self.kernel = [[1,1,1],[1,-8,1],[1,1,1]]

    def forward(self, raw, fusion):
        batch = raw.size()[0]
        channels = raw.size()[1]
        out_channel = channels
        kernel = torch.FloatTensor(self.kernel).expand(out_channel, channels, 3, 3).cuda()
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        grad_raw = torch.nn.functional.conv2d(raw, self.weight, stride=1, padding=1)
        grad_fusion = torch.nn.functional.conv2d(fusion, self.weight, stride=1, padding=1)
        loss = torch.mean((grad_raw - grad_fusion) ** 2)

        return loss

if __name__ == '__main__':
    input = torch.randn(2, 1, 128, 128).cuda()
    output = torch.randn(2, 1, 128, 128).cuda()
    score = grad_loss(input, output)
    score1 = ints_loss(input, output)
    print(score1)