import torch
import numpy as np
import os
from pcc_model import PCCModel
# from coder import Coder
# from coder_ori import Coder
# from coder_ori2 import Coder  #VAE
from coder_ori3 import Coder   #ANF
import time
import matplotlib.pyplot as plt

# from data_utils import load_sparse_tensor, sort_spare_tensor, scale_sparse_tensor
from data_utils import write_ply_ascii_geo, read_ply_ascii_geo
from data_loader import dense_to_sparse, sparse_to_dense
from processing import RangeProjection, repro
from torch import nn

from pc_error import pc_error
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def test_rangeimage(filedir, ori, ckptdir_list, outdir, resultdir):
    # load data
    start_time = time.time()
    ori_img = torch.from_numpy(ori)
    x = torch.from_numpy(ori)
    x = torch.unsqueeze(x, dim=0)
    shape = x.shape
    x = dense_to_sparse(x)
    print('Loading Time:\t', round(time.time() - start_time, 4), 's')

    # output filename
    if not os.path.exists(outdir): os.makedirs(outdir)
    filename = os.path.join(outdir, os.path.split(filedir)[-1].split('.')[0])
    print('output filename:\t', filename)

    # load model
    model = PCCModel().to(device)

    for idx, ckptdir in enumerate(ckptdir_list):
        print('=' * 10, idx + 1, '=' * 10)
        # load checkpoints
        assert os.path.exists(ckptdir)
        ckpt = torch.load(ckptdir)
        model.load_state_dict(ckpt['model'])
        print('load checkpoint from \t', ckptdir)
        coder = Coder(model=model, filename=filename)

        # postfix: rate index
        postfix_idx = '_r' + str(idx + 1)
        _, main_coord = model.get_coordinate(x)
        # encode
        start_time = time.time()
        encoded_coords = coder.encode(x, postfix=postfix_idx)
        # y = model.encoder(x)
        print('Enc Time:\t', round(time.time() - start_time, 3), 's')
        time_enc = round(time.time() - start_time, 3)

        # decode
        start_time = time.time()
        # x_dec = coder.decode(postfix=postfix_idx, rho=rho)
        x_dec = coder.decode(rho=1, postfix=postfix_idx, y_C=main_coord)

        # x_dec = model.decoder(y[0])
        print('Dec Time:\t', round(time.time() - start_time, 3), 's')
        time_dec = round(time.time() - start_time, 3)
        x_dec = sparse_to_dense(x_dec, shape)
        print('--')
    # return all_results
    return x_dec, ori_img


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filedir", default='/home/student/Desktop/Salmon/PCGCv2-master_0414/testdataset/001.ply')
    # parser.add_argument("--filedir", default='/home/student/Desktop/Salmon/PCGCv2-master_0414/testdataset/Ford_01_vox1mm-0100.ply')
    # parser.add_argument("--filedir", default='/home/student/Desktop/Salmon/PCGCv2-master_0414/testdataset/campus/campus_0000000000.bin')
    parser.add_argument("--image_h", type=int, default=64)
    parser.add_argument("--image_w", type=int, default=2048)
    parser.add_argument("--normalize_factor", type=int, default=100) #kitti
    # parser.add_argument("--normalize_factor", type=int, default=116027) #ford
    parser.add_argument("--outdir", default='./recdata/0619_lambda100000_ANF_w_entropy/kitti_64_2048_1000epoch')
    parser.add_argument("--outdir_ply", default='/rec.ply')
    parser.add_argument("--outdir_ori_img", default='/ori.png')
    parser.add_argument("--outdir_rec_img", default='/rec.png')
    parser.add_argument("--resultdir", default='./results')
    # parser.add_argument("--scaling_factor", type=float, default=1.0, help='scaling_factor')
    # parser.add_argument("--res", type=int, default=1024, help='resolution')
    # parser.add_argument("--rho", type=float, default=1.0, help='the ratio of the number of output points to the number of input points')
    args = parser.parse_args()

    if not os.path.exists(args.outdir): os.makedirs(args.outdir)
    if not os.path.exists(args.resultdir): os.makedirs(args.resultdir)
    ckptdir_list = ['/home/student/Desktop/Salmon/PCGCv2-master_0414/ckpts/240604_alpha100000_kitti_stage2_ANF/epoch_249.pth']


    # -----------------project-------------
    h, w = args.image_h, args.image_w
    normalize_factor = args.normalize_factor
    projection = RangeProjection(proj_h=h, proj_w=w)
    data = read_ply_ascii_geo(args.filedir)
    # data = np.fromfile(args.filedir, dtype=np.float32).reshape(-1, 4)
    proj_pointcloud, proj_range, proj_idx, proj_mask, w, h, proj_x, proj_y, yaw, pitch = projection.doProjection(
        data)
    proj_range_n = proj_range / normalize_factor
    #-------------------------------------------------


    # all_results = test(args.filedir, ckptdir_list, args.outdir, args.resultdir, scaling_factor=args.scaling_factor, rho=args.rho, res=args.res)
    x_dec, ori_img = test_rangeimage(args.filedir, proj_range_n, ckptdir_list, args.outdir, args.resultdir)
    rec_img = x_dec.squeeze(0)
    ori_img = ori_img.cpu().numpy()
    rec_img = rec_img.cpu().numpy()

    # ----------------------------------------------------------------------Analysis
    # # Differences for different threshold
    # ori_img = ori_img.cpu().numpy()
    # rec_img = rec_img.cpu().numpy()
    # differences = rec_img - ori_img
    # abs_differences = np.abs(differences)
    # locations = np.where(abs_differences > 0.50)
    # mask = np.zeros_like(ori_img)
    # mask[locations] = 1
    # large_error_locations = ori_img * mask
    # large_error_value_locations = abs_differences * mask
    #
    #
    # # Plotting the original image with the mask overlay
    # # plt.imshow(ori_img, cmap='gray')
    # plt.imshow(differences, cmap='gray')
    # plt.imshow(mask, cmap='jet', alpha=0.5)  # alpha controls the transparency of the overlay
    # plt.title('Differences > 0.01')
    # plt.axis('off')
    # plt.savefig(args.outdir + args.outdir_rec_img, dpi=300)
    #
    #
    #
    # # MSE
    # ori_img = ori_img.cpu().unsqueeze(0)
    # rec_img = rec_img.cpu().unsqueeze(0)
    #
    # ri1 = dense_to_sparse(ori_img)
    # ri2 = dense_to_sparse(rec_img)
    # mse_loss = nn.MSELoss()
    # mse = mse_loss(ri1.F, ri2.F)
    # print('MSE: ', mse, mse * 100)
    # # print('MSE: ', mse, mse * 116.027)
    #
    # # Calculate element-wise differences and squared differences
    # differences = ri1.F - ri2.F
    # squared_differences = differences ** 2
    #
    # # Flatten the squared differences to analyze them as a single series
    # flat_squared_differences = squared_differences.flatten().cpu()
    #
    # # Calculating max, median, mean, and min values
    # max_value = flat_squared_differences.max()
    # median_value = flat_squared_differences.median()
    # mean_value = flat_squared_differences.mean()
    # min_value = flat_squared_differences.min()
    #
    # # Printing the calculated values
    # print(f"Max Value: {max_value.item()}")
    # print(f"Median Value: {median_value.item()}")
    # print(f"Mean Value: {mean_value.item()}")
    # print(f"Min Value: {min_value.item()}")
    #
    # # Plotting the histogram of squared differences
    # plt.figure(figsize=(10, 6))
    # plt.hist(flat_squared_differences, bins=100, alpha=0.75)
    # plt.title('Histogram of Squared Differences')
    # plt.xlabel('Squared Difference')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()
    #
    #
    # # Define a function to identify outliers based on IQR
    # def detect_outliers(data):
    #     Q1 = torch.quantile(data, 0.25)
    #     Q3 = torch.quantile(data, 0.75)
    #     IQR = Q3 - Q1
    #     threshold = 1.5 * IQR
    #     outliers_lower = data[(data < Q1 - threshold)]
    #     outliers_upper = data[(data > Q3 + threshold)]
    #     outliers_total = data[(data > Q3 + threshold) | (data < Q1 - threshold)]
    #     return outliers_lower, outliers_upper, outliers_total
    #
    #
    # outliers_lower, outliers_upper, outliers_total = detect_outliers(flat_squared_differences)
    #
    # print(f"Number of lower: {len(outliers_lower)}")
    # print(f"Number of upper: {len(outliers_upper)}")
    # print(f"Number of outliers: {len(outliers_total)}")
    # ----------------------------------------------------------------------Analysis

    # Save the original data
    plt.imshow(ori_img, cmap='viridis')  # Adjust the colormap as needed
    # plt.colorbar()  # Add a color bar if necessary
    plt.axis('off')
    plt.savefig( args.outdir + args.outdir_ori_img, dpi=300)

    # Save the reconstructed data
    plt.imshow(rec_img, cmap='viridis')  # Adjust the colormap as needed
    # plt.colorbar()  # Add a color bar if necessary
    plt.axis('off')
    plt.savefig( args.outdir + args.outdir_rec_img, dpi=300)

    # -----------------reproject-------------
    re_pc = repro(rec_img * normalize_factor, h, w)
    zero_rows_mask = np.all(re_pc == 0, axis=1)
    re_pc = re_pc[~zero_rows_mask]
    write_ply_ascii_geo( args.outdir + args.outdir_ply, re_pc)
    #-------------------------------------------------


