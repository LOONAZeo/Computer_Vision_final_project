import torch
import numpy as np
import os
from pcc_model import PCCModel
# from coder import Coder
# from coder_ori import Coder
from coder_ori2 import Coder
import time
import matplotlib.pyplot as plt

# from data_utils import load_sparse_tensor, sort_spare_tensor, scale_sparse_tensor
# from data_utils import write_ply_ascii_geo, read_ply_ascii_geo
from data_loader import dense_to_sparse, sparse_to_dense

from pc_error import pc_error
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_rangeimage(filedir, ckptdir_list, outdir, resultdir):
    # load data
    start_time = time.time()
    ori = np.load(filedir)
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
        # _, encoded_coords = coder.encode(x, postfix=postfix_idx)
        y = coder.encode(x, postfix=postfix_idx)
        print('Enc Time:\t', round(time.time() - start_time, 3), 's')
        time_enc = round(time.time() - start_time, 3)

        # decode
        start_time = time.time()
        # x_dec = coder.decode(postfix=postfix_idx, rho=rho)
        # x_dec = coder.decode(encoded_coords, postfix=postfix_idx)
        x_dec = coder.decode(postfix=postfix_idx, z=y, y_key=main_coord.coordinate_map_key,
                             y_manager=main_coord.coordinate_manager)
        print('Dec Time:\t', round(time.time() - start_time, 3), 's')
        time_dec = round(time.time() - start_time, 3)
        x_dec = sparse_to_dense(x_dec, shape)

        # bitrate
        bits = np.array([os.path.getsize(filename + postfix_idx + postfix) * 8 \
                         for postfix in ['_F.bin', '_H.bin', '_num_points.bin']])
        bpps = (bits / len(x)).round(3)
        print('bits:\t', sum(bits), '\nbpps:\t', sum(bpps).round(3))

        # # distortion
        # start_time = time.time()
        # write_ply_ascii_geo(filename + postfix_idx + '_dec.ply', x_dec.C.detach().cpu().numpy()[:, 1:])
        # print('Write PC Time:\t', round(time.time() - start_time, 3), 's')
        #
        # start_time = time.time()
        # pc_error_metrics = pc_error(filedir, filename + postfix_idx + '_dec.ply',
        #                             res=res, normal=True, show=False)
        # print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
        # print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])
        #
        # # save results
        # results = pc_error_metrics
        # results["num_points(input)"] = len(x)
        # results["num_points(output)"] = len(x_dec)
        # results["resolution"] = res
        # results["bits"] = sum(bits).round(3)
        # results["bits"] = sum(bits).round(3)
        # results["bpp"] = sum(bpps).round(3)
        # results["bpp(coords)"] = bpps[0]
        # results["bpp(feats)"] = bpps[1]
        # results["time(enc)"] = time_enc
        # results["time(dec)"] = time_dec
        # if idx == 0:
        #     all_results = results.copy(deep=True)
        # else:
        #     all_results = all_results.append(results, ignore_index=True)
        # csv_name = os.path.join(resultdir, os.path.split(filedir)[-1].split('.')[0] + '.csv')
        # all_results.to_csv(csv_name, index=False)
        # print('Wrile results to: \t', csv_name)

    # return all_results
    return x_dec, ori_img

# def test(filedir, ckptdir_list, outdir, resultdir, scaling_factor=1.0, rho=1.0, res=1024):
#     # load data
#     start_time = time.time()
#     x = load_sparse_tensor(filedir, device)
#     print('Loading Time:\t', round(time.time() - start_time, 4), 's')
#     # x = sort_spare_tensor(input_data)
#
#     # output filename
#     if not os.path.exists(outdir): os.makedirs(outdir)
#     filename = os.path.join(outdir, os.path.split(filedir)[-1].split('.')[0])
#     print('output filename:\t', filename)
#
#     # load model
#     model = PCCModel().to(device)
#
#     for idx, ckptdir in enumerate(ckptdir_list):
#         print('='*10, idx+1, '='*10)
#         # load checkpoints
#         assert os.path.exists(ckptdir)
#         ckpt = torch.load(ckptdir)
#         model.load_state_dict(ckpt['model'])
#         print('load checkpoint from \t', ckptdir)
#         coder = Coder(model=model, filename=filename)
#
#         # postfix: rate index
#         postfix_idx = '_r'+str(idx+1)
#
#         # down-scale
#         if scaling_factor!=1:
#             x_in = scale_sparse_tensor(x, factor=scaling_factor)
#         else:
#             x_in = x
#
#         # encode
#         start_time = time.time()
#         _ = coder.encode(x_in, postfix=postfix_idx)
#         print('Enc Time:\t', round(time.time() - start_time, 3), 's')
#         time_enc = round(time.time() - start_time, 3)
#
#         # decode
#         start_time = time.time()
#         x_dec = coder.decode(postfix=postfix_idx, rho=rho)
#         print('Dec Time:\t', round(time.time() - start_time, 3), 's')
#         time_dec = round(time.time() - start_time, 3)
#
#         # up-scale
#         if scaling_factor!=1:
#             x_dec = scale_sparse_tensor(x_dec, factor=1.0/scaling_factor)
#
#         # bitrate
#         bits = np.array([os.path.getsize(filename + postfix_idx + postfix)*8 \
#                                 for postfix in ['_C.bin', '_F.bin', '_H.bin', '_num_points.bin']])
#         bpps = (bits/len(x)).round(3)
#         print('bits:\t', sum(bits), '\nbpps:\t',  sum(bpps).round(3))
#
#         # distortion
#         start_time = time.time()
#         write_ply_ascii_geo(filename+postfix_idx+'_dec.ply', x_dec.C.detach().cpu().numpy()[:,1:])
#         print('Write PC Time:\t', round(time.time() - start_time, 3), 's')
#
#         start_time = time.time()
#         pc_error_metrics = pc_error(filedir, filename+postfix_idx+'_dec.ply',
#                                     res=res, normal=True, show=False)
#         print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
#         print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])
#
#         # save results
#         results = pc_error_metrics
#         results["num_points(input)"] = len(x)
#         results["num_points(output)"] = len(x_dec)
#         results["resolution"] = res
#         results["bits"] = sum(bits).round(3)
#         results["bits"] = sum(bits).round(3)
#         results["bpp"] = sum(bpps).round(3)
#         results["bpp(coords)"] = bpps[0]
#         results["bpp(feats)"] = bpps[1]
#         results["time(enc)"] = time_enc
#         results["time(dec)"] = time_dec
#         if idx == 0:
#             all_results = results.copy(deep=True)
#         else:
#             all_results = all_results.append(results, ignore_index=True)
#         csv_name = os.path.join(resultdir, os.path.split(filedir)[-1].split('.')[0]+'.csv')
#         all_results.to_csv(csv_name, index=False)
#         print('Wrile results to: \t', csv_name)
#
#     return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filedir", default='/home/student/Desktop/Salmon/rangeimage/testdata/001.npy')
    parser.add_argument("--outdir", default='./rec')
    parser.add_argument("--resultdir", default='./results')
    # parser.add_argument("--scaling_factor", type=float, default=1.0, help='scaling_factor')
    # parser.add_argument("--res", type=int, default=1024, help='resolution')
    # parser.add_argument("--rho", type=float, default=1.0, help='the ratio of the number of output points to the number of input points')
    args = parser.parse_args()

    if not os.path.exists(args.outdir): os.makedirs(args.outdir)
    if not os.path.exists(args.resultdir): os.makedirs(args.resultdir)
    # ckptdir_list = ['./ckpts/r1_0.025bpp.pth', './ckpts/r2_0.05bpp.pth',
    #                 './ckpts/r3_0.10bpp.pth', './ckpts/r4_0.15bpp.pth',
    #                 './ckpts/r5_0.25bpp.pth', './ckpts/r6_0.3bpp.pth',
    #                 './ckpts/r7_0.4bpp.pth']
    ckptdir_list = ['./ckpts/240414_alpha1000_kitti_mae_stage1/epoch_21.pth']

    # all_results = test(args.filedir, ckptdir_list, args.outdir, args.resultdir, scaling_factor=args.scaling_factor, rho=args.rho, res=args.res)
    x_dec, ori_img = test_rangeimage(args.filedir, ckptdir_list, args.outdir, args.resultdir)
    rec_img = x_dec.squeeze(0)
    # rec_img = rec_img.squeeze(0)
    ori_img = ori_img.cpu().numpy()
    rec_img = rec_img.cpu().numpy()
    np.save('/home/student/Desktop/Salmon/PCGCv2-master_0414/rec/' + 'rec001_240414_alpha1000_1layer_mae_kitti_stage1',
                rec_img)

    # Plot the original data
    plt.imshow(ori_img, cmap='viridis')  # Adjust the colormap as needed
    # plt.colorbar()  # Add a color bar if necessary
    plt.axis('off')
    # plt.title('Original')  # Set the title of the plot
    # plt.xlabel('X Label')  # Set the label for the x-axis
    # plt.ylabel('Y Label')  # Set the label for the y-axis

    # Save the plot in a folder (replace 'path/to/your/folder' with the actual folder path)
    plt.savefig('/home/student/Desktop/Salmon/PCGCv2-master_0414/rec/ori001.png', dpi=300)

    # Plot the reconstructed data
    plt.imshow(rec_img, cmap='viridis')  # Adjust the colormap as needed
    # plt.colorbar()  # Add a color bar if necessary
    plt.axis('off')
    # plt.title('My Plot')  # Set the title of the plot
    # plt.xlabel('X Label')  # Set the label for the x-axis
    # plt.ylabel('Y Label')  # Set the label for the y-axis

    # Show the plots
    plt.savefig('/home/student/Desktop/Salmon/PCGCv2-master_0414/rec/rec001_240414_alpha1000_1layer_mae_kitti_stage1.png', dpi=300)


    # plot RD-curve
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(7, 4))
    # plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2point)"][:]),
    #         label="D1", marker='x', color='red')
    # plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2plane)"][:]),
    #         label="D2", marker='x', color='blue')
    # filename = os.path.split(args.filedir)[-1][:-4]
    # plt.title(filename)
    # plt.xlabel('bpp')
    # plt.ylabel('PSNR')
    # plt.grid(ls='-.')
    # plt.legend(loc='lower right')
    # fig.savefig(os.path.join(args.resultdir, filename+'.jpg'))

