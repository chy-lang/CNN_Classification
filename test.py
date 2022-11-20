import torch
import torch.nn.functional as F


class my_batchnorm():
    def __init__(self, num):
        self.layer_num = num
        self.mean = torch.zeros(num)
        self.var = torch.ones(num)
        self.gama = torch.nn.Parameter(torch.ones(num))
        self.beta = torch.nn.Parameter(torch.zeros(num))
        self.train = 1

    def batchnorm(self, x):
            for i in range(self.layer_num):
                x_feature = x[:, i, :, :]
                feature_mean = x_feature.mean()
                # 总体标准差
                feature_std_t1 = torch.tensor(x_feature.detach().numpy().std())
                # 样本标准差
                feature_std_t2 = torch.tensor(x_feature.detach().numpy().std(ddof=1))

                x_feature = (x_feature - feature_mean) / torch.sqrt(feature_std_t1 ** 2 + 1e-5)

                if self.train == 1:
                    self.mean[i] = self.mean[i] * 0.9 + feature_mean * 0.1
                    self.var[i] = self.var[i] * 0.9 + (feature_std_t2 ** 2) * 0.1

                x_feature = F.linear(x_feature, self.gama[i], self.beta[i])
                x[:, i, :, :] = x_feature
            return x

    def on(self):
        self.train = 1
    def off(self):
        self.train = 0



