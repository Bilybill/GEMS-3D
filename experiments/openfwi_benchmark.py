"""Matched-budget OpenFWI velocity-to-seismic surrogate benchmark."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.models.common import ConvBlock2d, ResidualTransformerBlock, sinusoidal_position_encoding
from experiments.models.operator_fno import FNOBlock2d

class OpenFWIPairs(Dataset):
    def __init__(self, velocity_files, seismic_files, indices, time_stride=4):
        if len(velocity_files) != len(seismic_files): raise ValueError("Velocity/seismic file count mismatch")
        self.velocity=[np.load(p,mmap_mode="r") for p in velocity_files]; self.seismic=[np.load(p,mmap_mode="r") for p in seismic_files]
        for v,s in zip(self.velocity,self.seismic):
            if len(v)!=len(s) or v.shape[1:]!=(1,70,70) or s.shape[1:]!=(5,1000,70): raise ValueError(f"Unexpected OpenFWI shapes {v.shape}/{s.shape}")
        self.chunk_sizes=[len(x) for x in self.velocity]; self.offsets=np.cumsum([0]+self.chunk_sizes); self.indices=np.asarray(indices); self.time_stride=time_stride
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        j=int(self.indices[i]); chunk=int(np.searchsorted(self.offsets,j,side="right")-1); local=j-self.offsets[chunk]
        v=np.asarray(self.velocity[chunk][local],dtype=np.float32); s=np.asarray(self.seismic[chunk][local,:,::self.time_stride,:],dtype=np.float32)
        v=(v-v.mean())/max(v.std(),1e-8); s=s/max(np.sqrt(np.mean(s*s)),1e-8)
        return torch.from_numpy(v.copy()),torch.from_numpy(s.copy())


class InputEncoder2D(nn.Module):
    def __init__(self, channels=(32, 64, 128)):
        super().__init__()
        blocks = []
        current = 1
        for out_channels in channels:
            blocks.append(ConvBlock2d(current, out_channels))
            current = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AvgPool2d(2)

    def forward_features(self, x):
        features = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)
            if index < len(self.blocks) - 1:
                x = self.pool(x)
        return features

    def forward(self, x):
        features = self.forward_features(x)
        latent = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)
        return features, latent


class UpBlock2D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = ConvBlock2d(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class UNetSeismic2D(nn.Module):
    def __init__(self, base_channels=32, out_shape=(5, 1000, 70)):
        super().__init__()
        self.out_shape = out_shape
        self.encoder = InputEncoder2D((base_channels, base_channels * 2, base_channels * 4))
        self.bottleneck = ConvBlock2d(base_channels * 4, base_channels * 8)
        self.up2 = UpBlock2D(base_channels * 8, base_channels * 2, base_channels * 4)
        self.up1 = UpBlock2D(base_channels * 4, base_channels, base_channels * 2)
        self.head = nn.Conv2d(base_channels * 2, out_shape[0], 1)

    def forward(self, x):
        features = self.encoder.forward_features(x)
        z = self.bottleneck(features[-1])
        z = self.up2(z, features[-2])
        z = self.up1(z, features[-3])
        return F.interpolate(self.head(z), size=self.out_shape[1:], mode="bilinear", align_corners=False)


class FNOSeismic2D(nn.Module):
    def __init__(self,width=64,modes=12,out_shape=(5,250,70)):
        super().__init__(); self.out_shape=out_shape; self.lift=nn.Conv2d(1,width,1); self.blocks=nn.ModuleList([FNOBlock2d(width,modes,modes) for _ in range(4)]); self.head=nn.Sequential(nn.Conv2d(width,128,1),nn.GELU(),nn.Conv2d(128,out_shape[0],1))
    def forward(self,x):
        z=self.lift(x)
        for block in self.blocks: z=block(z)
        return torch.nn.functional.interpolate(self.head(z),size=self.out_shape[1:],mode="bilinear",align_corners=False)


def coordinate_grid(height, width, device, dtype):
    time = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    receiver = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_time, grid_receiver = torch.meshgrid(time, receiver, indexing="ij")
    return torch.stack((grid_time, grid_receiver), dim=-1).reshape(-1, 2)


class PINNSeismic2D(nn.Module):
    """Coordinate-conditioned supervised surrogate prepared for physics residuals."""

    def __init__(self, latent_dim=128, hidden_dim=256, out_shape=(5, 1000, 70)):
        super().__init__()
        self.out_shape = out_shape
        self.encoder = InputEncoder2D()
        self.latent_projection = nn.Linear(128, latent_dim)
        self.coord_frequencies = (1.0, 2.0, 4.0, 8.0, 16.0)
        coordinate_dim = 2 + 4 * len(self.coord_frequencies)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + coordinate_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_shape[0]),
        )

    def encode_coordinates(self, coords):
        features = [coords]
        for frequency in self.coord_frequencies:
            argument = math.pi * frequency * coords
            features.extend((torch.sin(argument), torch.cos(argument)))
        return torch.cat(features, dim=-1)

    def forward(self, x):
        _, latent = self.encoder(x)
        latent = self.latent_projection(latent)
        coords = coordinate_grid(self.out_shape[1], self.out_shape[2], x.device, x.dtype)
        coords = self.encode_coordinates(coords)
        coords = coords.unsqueeze(0).expand(x.shape[0], -1, -1)
        latent = latent.unsqueeze(1).expand(-1, coords.shape[1], -1)
        output = self.decoder(torch.cat((latent, coords), dim=-1))
        return output.transpose(1, 2).reshape(x.shape[0], *self.out_shape)


class DeepONetSeismic2D(nn.Module):
    def __init__(self, basis_dim=128, hidden_dim=256, out_shape=(5, 1000, 70)):
        super().__init__()
        self.out_shape = out_shape
        self.encoder = InputEncoder2D()
        self.branch = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_shape[0] * basis_dim),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, basis_dim),
        )
        self.bias = nn.Parameter(torch.zeros(out_shape[0]))

    def forward(self, x):
        _, latent = self.encoder(x)
        branch = self.branch(latent).reshape(x.shape[0], self.out_shape[0], -1)
        coords = coordinate_grid(self.out_shape[1], self.out_shape[2], x.device, x.dtype)
        trunk = self.trunk(coords)
        output = torch.einsum("bck,nk->bcn", branch, trunk)
        output = output.reshape(x.shape[0], *self.out_shape)
        return output + self.bias.reshape(1, -1, 1, 1)


class ImprovedDeepONetSeismic2D(nn.Module):
    """DeepONet with a spatial branch and a Fourier-feature trunk."""

    def __init__(self, basis_dim=256, hidden_dim=512, out_shape=(5, 1000, 70)):
        super().__init__()
        self.out_shape = out_shape
        self.basis_dim = basis_dim
        self.encoder = InputEncoder2D()
        self.branch_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_shape[0] * basis_dim),
        )
        self.coord_frequencies = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
        coordinate_dim = 2 + 4 * len(self.coord_frequencies)
        self.trunk = nn.Sequential(
            nn.Linear(coordinate_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, basis_dim),
        )
        self.bias = nn.Parameter(torch.zeros(out_shape[0]))

    def encode_coordinates(self, coords):
        features = [coords]
        for frequency in self.coord_frequencies:
            argument = math.pi * frequency * coords
            features.extend((torch.sin(argument), torch.cos(argument)))
        return torch.cat(features, dim=-1)

    def forward(self, x):
        features = self.encoder.forward_features(x)
        branch = self.branch(self.branch_pool(features[-1]))
        branch = branch.reshape(x.shape[0], self.out_shape[0], self.basis_dim)
        coords = coordinate_grid(self.out_shape[1], self.out_shape[2], x.device, x.dtype)
        trunk = self.trunk(self.encode_coordinates(coords))
        branch = F.normalize(branch, dim=-1) * math.sqrt(self.basis_dim)
        trunk = F.normalize(trunk, dim=-1)
        output = torch.einsum("bck,nk->bcn", branch, trunk)
        output = output.reshape(x.shape[0], *self.out_shape)
        return output + self.bias.reshape(1, -1, 1, 1)


class GNOTSeismic2D(nn.Module):
    def __init__(self, width=128, depth=4, num_heads=4, patch_size=5, out_shape=(5, 1000, 70)):
        super().__init__()
        self.out_shape = out_shape
        self.patch_embed = nn.Conv2d(1, width, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList(
            ResidualTransformerBlock(width, num_heads=num_heads) for _ in range(depth)
        )
        self.head = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv2d(width, out_shape[0], 1),
        )

    def forward(self, x):
        tokens = self.patch_embed(x)
        batch, channels, height, width = tokens.shape
        sequence = tokens.flatten(2).transpose(1, 2)
        position = sinusoidal_position_encoding(
            sequence.shape[1], sequence.shape[2], device=x.device, dtype=x.dtype
        )
        sequence = sequence + position.unsqueeze(0)
        for block in self.blocks:
            sequence = block(sequence)
        tokens = sequence.transpose(1, 2).reshape(batch, channels, height, width)
        return F.interpolate(self.head(tokens), size=self.out_shape[1:], mode="bilinear", align_corners=False)


MODEL_BUILDERS = {
    "unet": UNetSeismic2D,
    "pinn": PINNSeismic2D,
    "deeponet": DeepONetSeismic2D,
    "deeponet_v2": ImprovedDeepONetSeismic2D,
    "fno": FNOSeismic2D,
    "gnot": GNOTSeismic2D,
}


def evaluate(model,loader,device,optimizer=None):
    model.train(optimizer is not None); vals=[]
    for v,s in loader:
        v,s=v.to(device,non_blocking=True),s.to(device,non_blocking=True)
        with torch.set_grad_enabled(optimizer is not None):
            pred=model(v); loss=torch.nn.functional.mse_loss(pred,s)
            if optimizer: optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        p,t=pred.detach().flatten(1),s.flatten(1); pc,tc=p-p.mean(1,keepdim=True),t-t.mean(1,keepdim=True)
        r=(pc*tc).sum(1)/(pc.norm(dim=1).clamp_min(1e-8)*tc.norm(dim=1).clamp_min(1e-8)); rl2=(p-t).norm(dim=1)/t.norm(dim=1).clamp_min(1e-8)
        vals.extend(zip(r.cpu().tolist(),rl2.cpu().tolist()))
    a=np.asarray(vals); return {"pearson_r":{"mean":float(a[:,0].mean()),"std":float(a[:,0].std(ddof=1))},"relative_l2":{"mean":float(a[:,1].mean()),"std":float(a[:,1].std(ddof=1))}}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--train-velocity",type=Path,nargs="+",required=True); p.add_argument("--train-seismic",type=Path,nargs="+",required=True); p.add_argument("--test-velocity",type=Path,nargs="+",required=True); p.add_argument("--test-seismic",type=Path,nargs="+",required=True); p.add_argument("--train-size",type=int,default=680); p.add_argument("--val-size",type=int,default=120); p.add_argument("--test-size",type=int,default=120); p.add_argument("--epochs",type=int,default=30); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--time-stride",type=int,default=4); p.add_argument("--seed",type=int,default=42); p.add_argument("--device",default="cuda:0"); p.add_argument("--model",choices=tuple(MODEL_BUILDERS),default="fno"); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    ntrain=sum(len(np.load(x,mmap_mode="r")) for x in a.train_velocity); ntest=sum(len(np.load(x,mmap_mode="r")) for x in a.test_velocity); rng=np.random.default_rng(a.seed); order=rng.permutation(ntrain)
    if a.train_size+a.val_size>ntrain or a.test_size>ntest: raise ValueError("Requested split exceeds available samples")
    tr,va=order[:a.train_size],order[a.train_size:a.train_size+a.val_size]; te=np.arange(a.test_size)
    datasets=[OpenFWIPairs(a.train_velocity,a.train_seismic,tr,a.time_stride),OpenFWIPairs(a.train_velocity,a.train_seismic,va,a.time_stride),OpenFWIPairs(a.test_velocity,a.test_seismic,te,a.time_stride)]
    loaders=[DataLoader(d,a.batch_size,shuffle=i==0,num_workers=2,pin_memory=True) for i,d in enumerate(datasets)]
    device=torch.device(a.device); model=MODEL_BUILDERS[a.model](out_shape=(5,1000//a.time_stride,70)).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); best=float("inf"); state=None; history=[]
    for e in range(1,a.epochs+1):
        train=evaluate(model,loaders[0],device,opt)
        with torch.no_grad(): val=evaluate(model,loaders[1],device)
        score=val["relative_l2"]["mean"]; history.append({"epoch":e,"train":train,"validation":val}); print(f"epoch={e:02d} val_r={val['pearson_r']['mean']:.4f} val_rl2={score:.4f}",flush=True)
        if score<best: best=score; state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    with torch.no_grad(): test=evaluate(model,loaders[2],device)
    result={"dataset":"OpenFWI CurveFault-B","model":f"dimension-adapted {a.model.upper()}","model_key":a.model,"parameter_count":sum(parameter.numel() for parameter in model.parameters()),"setting":{"dimension_and_geology":"2-D curved-layer velocity models with one difficult fault structure (CurveFault-B)","input_tensor":"V_p: (1, 70, 70)","output_tensor":f"seismic: (5, {1000//a.time_stride}, 70)","acquisition_scale":"5 shot gathers, 1,000 time samples and 70 receivers per sample; full waveform unless --time-stride is changed"},"normalization":{"velocity":"sample standardization","seismic":"sample RMS"},"time_stride":a.time_stride,"split":{"train":a.train_size,"validation":a.val_size,"heldout_test":a.test_size,"heldout_source":"official validation chunks"},"training":{"epochs":a.epochs,"batch_size":a.batch_size,"optimizer":"AdamW","learning_rate":1e-3,"weight_decay":1e-4,"selection_metric":"validation relative L2"},"seed":a.seed,"heldout_test":test,"history":history}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(test,indent=2))
if __name__=="__main__": main()
