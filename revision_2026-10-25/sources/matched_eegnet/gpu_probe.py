"""Synthetic GPU software/resource check; generates no scientific results."""
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import time
import torch
from models import build_model

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
assert torch.cuda.is_available(), 'CUDA is unavailable'
print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
                      device=torch.cuda.get_device_name(0))), flush=True)
for seconds in [10,30]:
    torch.manual_seed(20261024)
    model = build_model('eegnet_combined', 9, seconds * 200).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=.001)
    x = torch.randn(64,30,seconds*200,device='cuda')
    side = torch.randn(64,5370,device='cuda')
    y = torch.arange(64,device='cuda') % 9
    start = time.monotonic()
    for _ in range(3):
        optim.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x, side), y)
        assert torch.isfinite(loss)
        loss.backward(); optim.step(); model.apply_constraints()
    torch.cuda.synchronize()
    print(json.dumps(dict(seconds=seconds, batch_size=64, steps=3,
                          elapsed_seconds=time.monotonic()-start,
                          max_memory_gb=torch.cuda.max_memory_allocated()/1e9,
                          purpose='synthetic software check only')), flush=True)
    del model,optim,x,side,y
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
