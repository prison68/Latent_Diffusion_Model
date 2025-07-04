import argparse, os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from imwatermark import WatermarkEncoder

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

dict={
    'prompt': 'A bicycle replica with a clock as the front wheel.',
    'outdir': "./outputs/outputs_imgs",
    'steps': 50,
    'plms': True,
    'dpm': False,
    'fixed_code': False,
    'ddim_eta': 0.0,
    'n_iter': 1,
    'H': 512,
    'W': 512,
    'C': 4,
    'f': 8,
    'n_samples': 4,
    'n_rows': 0,
    'scale': 9.0,
    'from-file': False,
    'config': "./configs/stable-diffusion/v2-inference.yaml",
    'ckpt':  './checkpoints/v2-1_512-ema-pruned.ckpt',
    'seed': 42,
    'precision': "autocast",
    'repeat': 4,
    'device': 'cuda',
    'torchscript': False,
    'ipex': False,
    'bf16': False
}

def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in sd.items() if k in model_dict and (v.shape == model_dict[k].shape)}
    model_dict.update(pretrained_dict)

    m, u = model.load_state_dict(model_dict, strict=True)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    # ldm.models.diffusion.ddpm.LatentDiffusion
    # 返回扩散模型的模型，在这里加载了预训练权重
    return model

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())
def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

# 采样生成图片
def sample_for_test(sampler, model, device, dict):
    os.makedirs(dict['outdir'], exist_ok=True)
    outpath = dict['outdir']

    # 设置水印
    wm = "SDV2"  # 水印字符
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = dict['n_samples']
    n_rows = dict['n_rows'] if dict['n_rows'] > 0 else batch_size

    # 从文件读文本prompt还是直接来自命令行
    if not dict['from-file']:
        print(f"reading prompts:{dict['prompt']}")
        prompt = dict['prompt']
        assert prompt is not None
        data = [batch_size * [prompt]]  # data是张数乘以prompt的个数

    else:
        print(f"reading prompts from {dict['from-file']}")
        with open(dict['from-file'], 'r') as f:
            data = f.read().splitlines()
            data = [p for p in data for i in range(dict['repeat'])]
            # islice的作用是从迭代器中截取size个元素，如果不够的话就停止。
            # 而返回的是这个lambda函数的迭代器，也就是每次yield这个元组，直到it被耗尽。
            # 将prompt分batch
            data = list(chunk(data, batch_size))


    sample_count = 0
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if dict['fixed_code']:
        start_code = torch.randn([dict['n_samples'], dict['C'], dict['H'] // dict['f'], dict['W'] // dict['f']],
                                 device=device)

    precision_scope = autocast if dict['precision'] == 'autocast' or dict['bf16'] else nullcontext
    with torch.no_grad(), \
            precision_scope(dict['device']), \
            model.ema_scope():
        all_samples = list()
        for n in trange(dict['n_iter'], desc='Sampling'):
            for prompts in tqdm(data, desc='data'):
                uc = None
                if dict['scale'] != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])  # 空字符生成
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                # 对文本prompts进行编码
                c = model.get_learned_conditioning(prompts)
                shape = [dict['C'], dict['H'] // dict['f'], dict['W'] // dict['f']]
                samples, _ = sampler.sample(
                    S=dict['steps'],
                    conditioning=c,
                    batch_size=dict['n_samples'],
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=dict['scale'],
                    unconditional_conditioning=uc,
                    eta=dict['ddim_eta'],
                    x_T=start_code)
                x_samples = model.decode_first_stage(samples)
                # 用于将输入张量（Tensor）的值限制在一个指定的范围内。
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                # 保存为RGB图像并加上水印，并行采样到一个batch_size的图片
                i = 0
                for x_sample in x_samples:
                    path = prompts[i]
                    i += 1
                    sample_path = str(os.path.join(outpath, path))
                    os.makedirs(sample_path, exist_ok=True)
                    base_count = len(os.listdir(sample_path))
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                    sample_count += 1

                all_samples.append(x_samples)

        # 由多张图像组成的网格图，并加上水印
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=n_rows)

        # 转换为rgb图片
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        grid = Image.fromarray(grid.astype(np.uint8))
        grid = put_watermark(grid, wm_encoder)
        grid.save(os.path.join(outpath, f'grid-{grid_count:05}.png'))
        grid_count += 1

if __name__ == "__main__":
    seed_everything(dict['seed'])
    config = OmegaConf.load(dict['config'])
    device = torch.device('cuda') if dict['device'] == 'cuda' else torch.device('cpu')

    model = load_model_from_config(config, dict['ckpt'], device)
    if dict['plms']:
        sampler = PLMSSampler(model, device=device)
    elif dict['dpm']:
        sampler = DPMSolverSampler(model, device=device)
    else:
        sampler = DDIMSampler(model, device=device)

    # 采样
    dict['prompt'] = 'A bicycle replica with a clock as the front wheel.'
    sample_for_test(sampler, model, device, dict)
