import argparse
import numpy as np
from torch.optim import Adam
import tqdm
from dataset import *
from utils import *
from model.net import FuseNet
from pytorch_msssim import ssim
from loss import *
from val import valdation


EPS = 1e-8
c = 3500
torch.autograd.set_detect_anomaly(True)


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    net = FuseNet().cuda()

    trainDataLoader = DataLoader(TrainData(args.dataset), batch_size=args.batch_size, shuffle=True)
    testDataLoader = DataLoader(TrainData(args.datasetv), batch_size=1, shuffle=False)

    optimizer = Adam(net.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss(reduction='mean')
    l1_loss = torch.nn.L1Loss()
    grad_loss = GradLoss()


    initial_epoch = 6
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        net.load_state_dict(torch.load(os.path.join(args.ckpt_path, 'epoch%d.pth' % initial_epoch)))
    for epoch in range(initial_epoch, args.epochs):

        net.train()
        agg_loss1 = 0.
        agg_loss2 = 0.
        agg_loss3 = 0.
        agg_loss4 = 0.
        agg_lossA = 0.
        for lf, dslr, lfh, dslrh in tqdm.tqdm(trainDataLoader):
            lf, dslr = lf.cuda(), dslr.cuda()
            lfh, dslrh = lfh.cuda(), dslr.cuda()

            optimizer.zero_grad()
            out, dl, dd = net(lf, dslr, lfh, dslrh)

            loss1_list = []
            loss2_list = []
            loss3_list = []
            loss4_list = []
            for i in range(5):
                for j in range(5):
                    loss_1 = l1_loss(dl[:, :, i * 5 + j, :, :], lf[:, :, i, j, :, :]) \
                             +  l1_loss(dd[:, :, i * 5 + j, :, :], dslr[:, :, i, j, :, :])
                    loss1_list.append(loss_1)

                    loss_2 = (1 - ssim(out[:, :, i, j, :, :], lf[:, :, i, j, :, :])) \
                             + (1 - ssim(out[:, :, i, j, :, :], dslr[:, :, i, j, :, :]))
                    loss2_list.append(loss_2)

                    loss_3 = mse_loss(out[:, :, i, j, :, :], lf[:, :, i, j, :, :]) \
                             +  mse_loss(out[:, :, i, j, :, :], dslr[:, :, i, j, :, :])
                    loss3_list.append(loss_3)

                    loss_4 = grad_loss(out[:, :, i, j, :, :], lf[:, :, i, j, :, :]) \
                             +  grad_loss(out[:, :, i, j, :, :], dslr[:, :, i, j, :, :])
                    loss4_list.append(loss_4)

            loss1 = (sum(loss1_list) / len(loss1_list))
            loss2 = (sum(loss2_list) / len(loss1_list)) * 20
            loss3 = (sum(loss3_list) / len(loss3_list)) * 1.5
            loss4 = (sum(loss4_list) / len(loss4_list)) * 16.0
            loss = loss1 + loss2 + loss3 + loss4

            agg_loss1 += loss1.item()
            agg_loss2 += loss2.item()
            agg_loss3 += loss3.item()
            agg_loss4 += loss4.item()
            agg_lossA += loss.item()
            loss.backward()
            optimizer.step()
        # print('epoch:'+str(epoch)+' | loss:'+str(agg_lossA)+' | loss1:'+str(agg_loss1)+' | loss2:'+str(agg_loss2)+' | loss3:'+ str(agg_loss3))                                                                                                                                                               ))
        print('epoch:' + str(epoch) + ' | loss:' + str(agg_lossA / len(trainDataLoader)) + ' | l3:' + str(
            agg_loss3 / len(trainDataLoader)) + ' | l4:' + str(agg_loss4 / len(trainDataLoader)))

        img_list = valdation(args, net, testDataLoader, args.val_dir, 128, 96, 512)
        torch.save(net.state_dict(), os.path.join(args.ckpt_path, 'epoch.pth'))

        if epoch == args.epochs - 1:
            net.eval()
            torch.save(net.state_dict(), os.path.join(args.ckpt_path, 'epoch100.pth'))
            print('model saved ! ')


def main():
    parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    parser.add_argument("--command", type=int, default=0, help="train/train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument('--workers', type=int, default=8, help='num workers of dataloader')

    parser.add_argument("--dataset", type=str, default='train')
    parser.add_argument("--datasetv", type=str, default='val')
    parser.add_argument("--ckpt_path", type=str, default="ckpt/")

    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--seed", type=int, default=24)

    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate, default is 1e-3")

    parser.add_argument('--val_dir', dest='val_dir', default='val/', help='directory for outputs')

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
