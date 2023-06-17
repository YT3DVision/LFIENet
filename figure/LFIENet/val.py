from torchvision.transforms import ToPILImage
from dataset import *
from model.net import FuseNet
from utils import *
import argparse

EPS = 1e-8
c = 3500


def test(args, data_path, save_path, patch_size, stride, img_size):
    print("test start.......................................................")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    testDataSet = TrainData(data_path)
    testDataLoader = DataLoader(testDataSet, batch_size=1, shuffle=False)

    net = FuseNet().cuda()
    initial_epoch = findLastCheckpoint(save_dir=args.ckpt_path)
    print('resuming by loading epoch %d' % initial_epoch)
    net.load_state_dict(torch.load(os.path.join(args.ckpt_path, 'epoch%d.pth' % initial_epoch)))
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    bound = math.ceil((img_size - patch_size) / stride + 1)
    num = bound * bound
    test = torch.zeros(num, 1, 9, patch_size, patch_size)
    print(test.device)
    index = 0
    with torch.no_grad():
        for iter, (lf, dslr, lfh, dslrh) in enumerate(testDataLoader):
            out_y, a1, a2, la, da = net(lf.cuda(), dslr.cuda(),lfh.cuda(), dslrh.cuda())

            print(iter)
            for i in range(3):
                for j in range(3):
                    fused_img = out_y[:, :, i, j, :,:]
                    fused_img = fused_img.squeeze(0).cpu()

                    test[index, :, i * 3 + j, :, :] = fused_img
            index = index + 1
            if index == num:
                index = 0
                out_save_path = os.path.join(save_path, str(iter // 16))
                print(out_save_path)
                if not os.path.exists(out_save_path):
                    os.makedirs(out_save_path)
                t = image_compose(test, out_save_path, 0, patch_size, stride, img_size)

    print("test finsh.......................................................")


def valdation(args, net, dataloader, save_path, patch_size, stride, img_size):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    testDataLoader = dataloader

    bound = math.ceil((img_size - patch_size) / stride + 1)
    num = bound * bound
    test = torch.zeros(num, 1, 25, patch_size, patch_size)
    index = 0
    img_list = []
    with torch.no_grad():
        for iter, (lf, dslr, lfh, dslrh) in enumerate(testDataLoader):
            out_y, la, da, dl, dd = net(lf.cuda(), dslr.cuda(),lfh.cuda(), dslrh.cuda())

            for i in range(5):
                for j in range(5):
                    fused_img = out_y[:, :, i,j, :,:]
                    fused_img = fused_img.squeeze(0).cpu()

                    test[index, :, i * 5 + j, :, :] = fused_img
            index = index + 1
            if index == num:
                index = 0
                out_save_path = os.path.join(save_path, str(iter // 16).zfill(3))
                if not os.path.exists(out_save_path):
                    os.makedirs(out_save_path)
                t = image_compose(test, out_save_path, 0, patch_size, stride, img_size)
                img_list.append(t)
    return img_list


def image_compose(mylist, save_path, epoch, size, stride, img_size):
    all_img = torch.zeros(1, 25, img_size, img_size)
    mylist = mylist.to(torch.device('cpu'))
    n = int(math.sqrt(len(mylist)))
    bound = int((size - stride) / 2 + stride)
    for ix in range(n):
        for iy in range(n):
            ind = iy + (n * ix)
            iby = 0 if iy == 0 else int(iy * stride + (size - stride) / 2)
            iby = img_size - bound if iy == n - 1 else iby
            iey = min(img_size, iby + bound)
            ibx = 0 if ix == 0 else int(ix * stride + (size - stride) / 2)
            ibx = img_size - bound if ix == n - 1 else ibx
            iex = min(img_size, ibx + bound)
            ipx = 0 if ix == 0 else int((size - stride) / 2)
            ipy = 0 if iy == 0 else int((size - stride) / 2)
            all_img[:, :, ibx:iex, iby:iey] = mylist[ind, :, :, ipx:ipx + (iex - ibx), ipy:ipy + (iey - iby)]

    for iv in range(25):
        fused_img = all_img[:, iv, :, :]

        save_image(save_path + '//%d.png' % (iv), fused_img)

    return all_img[:, 12, :, :]


def save_image(filename, data):
    train_img = data.cpu()
    train_img = ToPILImage()(train_img)
    train_img.save(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    parser.add_argument("--command", type=int, default=1, help="train/test")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument('--workers', type=int, default=8, help='num workers of dataloader')

    parser.add_argument("--dataset", type=str, default='data/')
    parser.add_argument("--ckpt_path", type=str, default="model/")
    parser.add_argument("--log_path", type=str, default="logs/", help='path to save log files')

    parser.add_argument("--gpu", type=str, default='1')
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate, default is 1e-3")

    parser.add_argument('--save_dir', dest='save_dir', default='result/', help='directory for outputs')
    parser.add_argument("--datasett", type=str, default='data')

    args = parser.parse_args()
    test(args, 'data/', 'result/', 128, 96, 512)
