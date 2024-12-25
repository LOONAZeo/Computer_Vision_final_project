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

import argparse


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
    filename = os.path.join(outdir, os.path.basename(filedir).split('.')[0])
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
        print('Enc Time:\t', round(time.time() - start_time, 3), 's')
        time_enc = round(time.time() - start_time, 3)

        # decode
        start_time = time.time()
        x_dec = coder.decode(rho=1, postfix=postfix_idx, y_C=main_coord)
        print('Dec Time:\t', round(time.time() - start_time, 3), 's')
        time_dec = round(time.time() - start_time, 3)
        x_dec = sparse_to_dense(x_dec, shape)
        print('--')
    # return all_results
    return x_dec, ori_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--folder_dir", default='/media/student/e15ac0f9-6b08-41c6-b970-5f45828d563b/Hou/data_object_velodyne/testing/velodyne', help="Directory containing .ply files")
    parser.add_argument("--image_h", type=int, default=64)
    parser.add_argument("--image_w", type=int, default=2048)
    parser.add_argument("--normalize_factor", type=int, default=100) # Adjust according to your dataset specifics
    parser.add_argument("--outdir", default='/media/student/e15ac0f9-6b08-41c6-b970-5f45828d563b/Hou/kitti_rec_ANF_250Epochs_100000/testing/to_be_deleted')
    parser.add_argument("--outdir2", default='/media/student/e15ac0f9-6b08-41c6-b970-5f45828d563b/Hou/kitti_rec_ANF_250Epochs_100000/testing/velodyne')
    parser.add_argument("--resultdir", default='./results')
    args = parser.parse_args()

    if not os.path.exists(args.outdir): os.makedirs(args.outdir)
    if not os.path.exists(args.resultdir): os.makedirs(args.resultdir)
    ckptdir_list = ['/home/student/Desktop/Salmon/PCGCv2-master_0414/ckpts/240604_alpha100000_kitti_stage2_ANF/epoch_199.pth']

    h, w = args.image_h, args.image_w
    normalize_factor = args.normalize_factor
    projection = RangeProjection(proj_h=h, proj_w=w)

    # Get all .bin files
    files = [f for f in os.listdir(args.folder_dir) if f.endswith('.bin')]

    # Loop through all .bin files in the directory
    for filedir in files:
        full_path = os.path.join(args.folder_dir, filedir)

        # Read .bin file
        with open(full_path, 'rb') as file:
            binary_data = file.read()
        data = np.fromfile(full_path, dtype=np.float32).reshape(-1, 4)

        # read .ply file
        # data = read_ply_ascii_geo(full_path)

        proj_pointcloud, proj_range, proj_idx, proj_mask, w, h, proj_x, proj_y, yaw, pitch = projection.doProjection(
            data)
        proj_range_n = proj_range / normalize_factor

        x_dec, ori_img = test_rangeimage(full_path, proj_range_n, ckptdir_list, args.outdir, args.resultdir)
        rec_img = x_dec.squeeze(0)
        ori_img = ori_img.cpu().numpy()
        rec_img = rec_img.cpu().numpy()

        # Reproject and save
        re_pc = repro(rec_img * normalize_factor, h, w)
        re_pc = re_pc.astype(np.float32)  # Convert dtype to np.float32
        zero_rows_mask = np.all(re_pc == 0, axis=1)
        re_pc = re_pc[~zero_rows_mask]

        # Ensure the output directory exists
        os.makedirs(args.outdir2, exist_ok=True)
        out_bin = os.path.join(args.outdir2, os.path.splitext(filedir)[0] + '.bin')
        re_pc.tofile(out_bin)