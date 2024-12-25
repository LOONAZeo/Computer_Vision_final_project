import torch
import MinkowskiEngine as ME

from data_utils import isin, istopk
criterion = torch.nn.BCEWithLogitsLoss()

def get_bce(data, groud_truth):
    """ Input data and ground_truth are sparse tensor.
    """
    mask = isin(data.C, groud_truth.C)
    bce = criterion(data.F.squeeze(), mask.type(data.F.dtype))
    bce /= torch.log(torch.tensor(2.0)).to(bce.device)
    sum_bce = bce * data.shape[0]
    
    return sum_bce

def get_bits(likelihood):
    bits = -torch.sum(torch.log2(likelihood))

    return bits

def get_metrics(groud_truth, data):
    # mask_real = isin(data.C, groud_truth.C)
    # nums = [len(C) for C in groud_truth.decomposed_coordinates]
    # mask_pred = istopk(data, nums, rho=1.0)
    # metrics = get_cls_metrics(mask_pred, mask_real)
    metrics = psnr(groud_truth, data)

    return metrics

def psnr(ori, rec):
    mse = torch.mean((ori - rec) ** 2)
    max_pixel_value = torch.max(ori) * 100 # Assuming the maximum pixel value is the maximum value in ori
    psnr_value = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr_value.item()  # Convert the tensor to a Python scalar


def get_cls_metrics(pred, real):
    TP = (pred * real).cpu().nonzero(as_tuple=False).shape[0]
    FN = (~pred * real).cpu().nonzero(as_tuple=False).shape[0]
    FP = (pred * ~real).cpu().nonzero(as_tuple=False).shape[0]
    TN = (~pred * ~real).cpu().nonzero(as_tuple=False).shape[0]

    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    IoU = TP / (TP + FP + FN + 1e-7)

    return [round(precision, 4), round(recall, 4), round(IoU, 4)]

