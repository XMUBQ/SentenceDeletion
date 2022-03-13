import torch
from paramsparser import args

cuda_available = torch.cuda.is_available()
print("If cuda available:", cuda_available)
dv = "cuda:" + str(args.cuda)
device = torch.device(dv)
torch.cuda.set_device(int(dv[-1]))