from __future__ import print_function

import argparse
import os
import time
import platform
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DATASET_NAMES, BipedDataset, dataset_info, TestDataset
from loss2 import cats_loss, bdcn_loss2
from modelB4 import LDC
from utils.img_processing import visualize_result, count_parameters, save_image_batch_to_disk

IS_LINUX = True if platform.system() == "Linux" else False


def train_one_epoch(epoch, dataloader, model, criterions, optimizer, device,
                    log_interval_vis, tb_writer, args=None):
    imgs_res_folder = os.path.join(args.output_dir, 'current_res')
    os.makedirs(imgs_res_folder, exist_ok=True)

    if isinstance(criterions, list):
        criterion1, criterion2 = criterions
    else:
        criterion1 = criterions

    model.train()
    l_weight0 = [0.7, 0.7, 1.1, 0.7, 1.3]
    l_weight = [[0.05, 2.], [0.05, 2.], [0.05, 2.],
                [0.1, 1.], [0.1, 1.], [0.1, 1.],
                [0.01, 4.]]
    loss_avg = []

    for batch_id, sample_batched in enumerate(dataloader):
        images = sample_batched['images'].to(device)
        labels = sample_batched['labels'].to(device)
        preds_list = model(images)

        loss = sum([criterion1(preds, labels, l_w, device) for preds, l_w in zip(preds_list, l_weight)])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_avg.append(loss.item())

        if epoch == 0 and (batch_id == 100 and tb_writer is not None):
            tmp_loss = np.array(loss_avg).mean()
            tb_writer.add_scalar('loss', tmp_loss, epoch)

        if batch_id % 10 == 0:
            print(time.ctime(), 'Epoch: {0} Sample {1}/{2} Loss: {3}'
                  .format(epoch, batch_id, len(dataloader), format(loss.item(), '.4f')))
        if batch_id % log_interval_vis == 0:
            res_data = []

            img = images.cpu().numpy()
            res_data.append(img[2])

            ed_gt = labels.cpu().numpy()
            res_data.append(ed_gt[2])

            for i in range(len(preds_list)):
                tmp = preds_list[i]
                tmp = tmp[2]
                tmp = torch.sigmoid(tmp).unsqueeze(dim=0)
                tmp = tmp.cpu().detach().numpy()
                res_data.append(tmp)

            vis_imgs = visualize_result(res_data, arg=args)
            del tmp, res_data

            vis_imgs = cv2.resize(vis_imgs,
                                  (int(vis_imgs.shape[1] * 0.8), int(vis_imgs.shape[0] * 0.8)))
            img_test = 'Epoch: {0} Sample {1}/{2} Loss: {3}' \
                .format(epoch, batch_id, len(dataloader), loss.item())

            BLACK = (0, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 1.1
            font_color = BLACK
            font_thickness = 2
            x, y = 30, 30
            vis_imgs = cv2.putText(vis_imgs,
                                   img_test,
                                   (x, y),
                                   font, font_size, font_color, font_thickness, cv2.LINE_AA)
            cv2.imwrite(os.path.join(imgs_res_folder, 'results.png'), vis_imgs)
    loss_avg = np.array(loss_avg).mean()
    return loss_avg


def validate_one_epoch(epoch, dataloader, model, device, output_dir, arg=None):
    model.eval()
    with torch.no_grad():
        for _, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            preds = model(images)
            save_image_batch_to_disk(preds[-1],
                                     output_dir,
                                     file_names, img_shape=image_shape,
                                     arg=arg)


def parse_args():
    parser = argparse.ArgumentParser(description='trainer.')
    parser.add_argument('--choose_test_data',
                        type=int,
                        default=-1,
                        help='Choose a dataset for testing: 0 - 8')

    TEST_DATA = DATASET_NAMES[parser.parse_args().choose_test_data]
    test_inf = dataset_info(TEST_DATA, is_linux=IS_LINUX)
    test_dir = test_inf['data_dir']
    is_testing = False

    TRAIN_DATA = DATASET_NAMES[0]
    train_inf = dataset_info(TRAIN_DATA, is_linux=IS_LINUX)
    train_dir = train_inf['data_dir']

    parser.add_argument('--input_dir',
                        type=str,
                        default=train_dir,
                        help='the path to the directory with the input data.')
    parser.add_argument('--input_val_dir',
                        type=str,
                        default=test_inf['data_dir'],
                        help='the path to the directory with the input data for validation.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='checkpoints',
                        help='the path to output the results.')
    parser.add_argument('--train_data',
                        type=str,
                        choices=DATASET_NAMES,
                        default=TRAIN_DATA,
                        help='Name of the dataset.')
    parser.add_argument('--test_data',
                        type=str,
                        choices=DATASET_NAMES,
                        default=TEST_DATA,
                        help='Name of the dataset.')
    parser.add_argument('--test_list',
                        type=str,
                        default=test_inf['test_list'],
                        help='Dataset sample indices list.')
    parser.add_argument('--train_list',
                        type=str,
                        default=train_inf['train_list'],
                        help='Dataset sample indices list.')
    parser.add_argument('--is_testing', type=bool,
                        default=is_testing,
                        help='Script in testing mode.')
    parser.add_argument('--resume',
                        type=bool,
                        default=False,
                        help='use previous trained data')
    parser.add_argument('--checkpoint_data',
                        type=str,
                        default='16/16_model.pth',
                        help='Checkpoint path.')
    parser.add_argument('--epochs',
                        type=int,
                        default=25,
                        metavar='N',
                        help='Number of training epochs (default: 25).')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Initial learning rate. =5e-5')
    parser.add_argument('--wd', type=float, default=0., metavar='WD',
                        help='weight decay (Good 5e-6)')
    parser.add_argument('--adjust_lr', default=[6, 12, 18], type=int,
                        help='Learning rate step size.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        metavar='B',
                        help='the mini-batch size (default: 8)')
    parser.add_argument('--workers',
                        default=8,
                        type=int,
                        help='The number of workers for the dataloaders.')
    parser.add_argument('--tensorboard', type=bool,
                        default=False,
                        help='Use Tensorboard for logging.')
    parser.add_argument('--img_width',
                        type=int,
                        default=352,
                        help='Image width for training.')
    parser.add_argument('--img_height',
                        type=int,
                        default=352,
                        help='Image height for training.')
    parser.add_argument('--crop_img',
                        default=True,
                        type=bool,
                        help='If true crop training images, else resize images to match image width and height.')
    parser.add_argument('--mean_pixel_values',
                        default=[103.939, 116.779, 123.68, 137.86],
                        type=float)
    args = parser.parse_args()
    return args


def main(args):
    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")

    tb_writer = None
    training_dir = os.path.join(args.output_dir, args.train_data)
    os.makedirs(training_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, args.train_data, args.checkpoint_data)
    if args.tensorboard and not args.is_testing:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=training_dir)
        training_notes = ['Xavier Normal Init, LR= ' + str(args.lr) + ' WD= '
                          + str(args.wd) + ' image size = ' + str(args.img_width)
                          + ' adjust LR=' + str(args.adjust_lr) + ' LRs= '
                          + str(args.lr) + ' Loss Function= CAST-loss2.py '
                          + str(time.asctime()) + 'Version notes']
        info_txt = open(os.path.join(training_dir, 'training_settings.txt'), 'w')
        info_txt.write(str(training_notes))
        info_txt.close()

    device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')
    model = LDC().to(device)

    ini_epoch = 0
    if not args.is_testing:
        if args.resume:
            checkpoint_path2 = os.path.join(args.output_dir, 'BIPED-54-B4', args.checkpoint_data)
            ini_epoch = 8
            model.load_state_dict(torch.load(checkpoint_path2, map_location=device))
        dataset_train = BipedDataset(args.input_dir,
                                     img_width=args.img_width,
                                     img_height=args.img_height,
                                     mean_bgr=args.mean_pixel_values[0:3] if len(
                                         args.mean_pixel_values) == 4 else args.mean_pixel_values,
                                     train_mode='train',
                                     arg=args)
        dataloader_train = DataLoader(dataset_train,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)

    dataset_val = TestDataset(args.input_val_dir,
                              test_data=args.test_data,
                              img_width=args.img_width,
                              img_height=args.img_height,
                              mean_bgr=args.mean_pixel_values[0:3] if len(
                                  args.mean_pixel_values) == 4 else args.mean_pixel_values,
                              test_list=args.test_list, arg=args)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.workers)

    criterion1 = cats_loss
    criterion2 = bdcn_loss2
    criterion = [criterion1, criterion2]
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.wd)

    num_param = count_parameters(model)
    print('-------------------------------------------------------')
    print('parameters:')
    print(num_param)
    print('-------------------------------------------------------')

    seed = 1021
    adjust_lr = args.adjust_lr
    k = 0
    set_lr = [25e-4, 5e-6]
    for epoch in range(ini_epoch, args.epochs):
        if epoch % 7 == 0:
            seed = seed + 1000
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print("------ Random seed applied-------------")
        if adjust_lr is not None:
            if epoch in adjust_lr:
                lr2 = set_lr[k]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr2
                k += 1

        output_dir_epoch = os.path.join(args.output_dir, args.train_data, str(epoch))
        img_test_dir = os.path.join(output_dir_epoch, args.test_data + '_res')
        os.makedirs(output_dir_epoch, exist_ok=True)
        os.makedirs(img_test_dir, exist_ok=True)

        avg_loss = train_one_epoch(epoch, dataloader_train,
                                   model, criterion,
                                   optimizer,
                                   device,
                                   args.log_interval_vis,
                                   tb_writer=tb_writer,
                                   args=args)
        validate_one_epoch(epoch,
                           dataloader_val,
                           model,
                           device,
                           img_test_dir,
                           arg=args)

        torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                   os.path.join(output_dir_epoch, '{0}_model.pth'.format(epoch)))
        if tb_writer is not None:
            tb_writer.add_scalar('loss',
                                 avg_loss,
                                 epoch + 1)
        print('Last learning rate> ', optimizer.param_groups[0]['lr'])

    num_param = count_parameters(model)
    print('-------------------------------------------------------')
    print('parameters:')
    print(num_param)
    print('-------------------------------------------------------')


if __name__ == '__main__':
    args = parse_args()
    main(args)
