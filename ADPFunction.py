#G2
import torch
import torch.nn.functional as F

log_offset = 1e-10
det_offset = 1e-6


## Function ##
def Entropy(x):
    #print(x)
    entropy = x * torch.log(x + log_offset)  #bs * num_classes
    return -1. * entropy.sum(1)


def Ensemble_Entropy(y_pred):
    y_pred = F.softmax(y_pred, dim=2)  # num_modles * bs * num_classes
    y_all = y_pred.mean(0)  #bs * num_classes
    Ensemble = Entropy(y_all)
    return Ensemble


def log_det(y_true, y_pred):
    y_pred = F.softmax(y_pred, dim=2)
    num_model, batch_size, num_class = y_pred.shape
    mask = torch.tensor(True).repeat(y_pred.shape)
    mask[:, range(batch_size), y_true] = False
    M = y_pred[mask].view(num_model, batch_size, num_class -
                          1)  #num_models * batch_size * num_classes-1
    M = M / (M.norm(2, 2) + log_offset).unsqueeze(-1)  #normalize
    M = M.permute(1, 2, 0)  #batch_size * num_classes-1 * num_models
    matrix = torch.matmul(M.transpose(1, 2), M)
    return torch.logdet(
        matrix +
        det_offset * torch.eye(num_model).cuda().repeat(matrix.shape[0], 1, 1))
