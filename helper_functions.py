import random
import numpy as np

import torch


def set_seed(seed):
    #python
    random.seed(seed)
    #numpy
    np.random.seed(seed)
    #Pytorch
    torch.manual_seed(seed)
    if torch.cuda.is_available ():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    #pytorch deterministic behavior
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    