import torch

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.
      """

    def __init__(self,
                 means=[0.4914, 0.4822, 0.4465],
                 sds=[0.2023, 0.1994, 0.2010], device ='cuda'):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.mymeans = torch.tensor(means).to(device)
        self.register_buffer('means', self.mymeans)
        self.mysds = torch.tensor(sds).to(device)
        self.register_buffer('sds', self.mysds)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat(
            (batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat(
            (batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
