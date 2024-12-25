import torch
import MinkowskiEngine as ME

from autoencoder import Encoder, Decoder, get_coordinate
from entropy_model import EntropyBottleneck


class PCCModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = Encoder(channels=[1,16,32,64,32,32])
        # self.decoder = Decoder(channels=[32,64,32,16])
        # self.encoder = Encoder(channels=[1, 64, 128])
        # self.decoder = Decoder(channels=[1, 64, 128])
        self.encoder1 = Encoder(channels=[1, 64, 128])
        self.encoder2 = Encoder(channels=[1, 64, 128])
        self.decoder1 = Decoder(channels=[1, 64, 128])
        self.decoder2 = Decoder(channels=[1, 64, 128])
        self.entropy_bottleneck = EntropyBottleneck(128)
        self.get_coordinate = get_coordinate()

    def get_likelihood(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)

        return data_Q, likelihood

    def forward(self, x, training=True):
        #---------------------------ANF
        # Encoder
        z1_list = self.encoder1(x)
        z1 = z1_list[0]
        xp = self.decoder1(z1)
        x1 = x - xp
        xpp_list = self.encoder2(x1)
        xpp = xpp_list[0]
        z2 = xpp + z1

        ground_truth_list = xpp_list[1:] + [x]
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
            for ground_truth in ground_truth_list]

        # Quantizer & Entropy Model
        z2_q, likelihood = self.get_likelihood(z2,
            quantize_mode="noise" if training else "symbols")

        # Decoder
        x1_q = self.decoder2(z2_q) #z2_q
        xs_list = self.encoder2(x1_q)
        xs = xs_list[0]
        z1_q = z2_q - xs
        out = x1_q + self.decoder1(z1_q)

        return {'out': out,
                # 'out_cls_list':out_cls_list,
                'prior': z2_q,
                'likelihood': likelihood,
                'ground_truth_list': ground_truth_list}
        # ---------------------------ANF

        # ---------------------------VAE
        # # Encoder
        # y_list = self.encoder(x)
        # y = y_list[0]
        # ground_truth_list = y_list[1:] + [x]
        # nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
        # for ground_truth in ground_truth_list]
        #
        # # Quantizer & Entropy Model
        # y_q, likelihood = self.get_likelihood(y,
        # quantize_mode="noise" if training else "symbols")
        #
        # # Decoder
        # # out = self.decoder(y)
        # out = self.decoder(y_q)
        #
        # return {'out':out,
        # # 'out_cls_list':out_cls_list,
        # 'prior':y_q,
        # 'likelihood':likelihood,
        # 'ground_truth_list':ground_truth_list}
        # ---------------------------VAE


if __name__ == '__main__':
    model = PCCModel()
    print(model)

