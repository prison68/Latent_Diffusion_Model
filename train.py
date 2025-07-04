from datetime import datetime

from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import os
from itertools import islice
from tqdm import tqdm, trange
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from ldm.models.diffusion.plms import PLMSSampler
from sampler import sample_for_test
from torch.utils.tensorboard import SummaryWriter

dict = {
    'prompt': 'A bicycle replica with a clock as the front wheel.', # the prompt to render
    'outdir': "./outputs/outputs_imgs/", # dir to write results to
    'steps': 50, # number of ddim sampling steps
    'plms': True,
    'dpm': False,
    'fixed_code': False,
    'ddim_eta': 0.0, # ddim eta (eta=0.0 corresponds to deterministic sampling
    'n_iter':1, # number of iterations during sampling
    'batch_size': 16,
    'epochs': 50, # number of iterations during sampling
    'H': 512,
    'W': 512,
    'C': 4,
    'f': 8,
    'n_samples': 3,  # how many samples to produce for each given prompt. A.k.a batch size
    'n_rows': 0,
    'scale': 9.0, # unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
    'from-file': False,
    'config': "./configs/stable-diffusion/v2-inference.yaml", # path to config which constructs model
    'ckpt': "./checkpoints/v2-1_512-ema-pruned.ckpt", # path to checkpoint of model
    'seed': 42,
    'precision': "autocast",  # ["full", "autocast"],
    'repeat': 1,
    'device': 'cuda',
    'torchscript': False,
    'ipex': False,
    'bf16': False
}

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 封装数据集类
class LaionDataset(Dataset):
    def __init__(self, image, prompt, data_dir, transform=None):
        self.transform = transform
        self.image = image
        self.prompt = prompt
        self.data_dir = data_dir

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx]
        prompt = self.prompt[idx]
        # 加载图像
        i = Image.open(os.path.join(self.data_dir, image)).convert('RGB')
        if self.transform:
            i = self.transform(i)

        return i, prompt

# 创建数据集
def get_loader(path, batch_size):
    files = os.listdir(path)
    image = [file for file in files if file.lower().endswith(".jpg")]
    txt_files = [file for file in files if file.lower().endswith(".txt")]
    prompts = []
    for file in txt_files:
        with open(os.path.join(path, file), 'r', encoding='utf-8', ) as f:
            prompts.append(f.read())
    dataset = LaionDataset(image=image, prompt=prompts, transform=transform, data_dir=path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

# 从配置文件中获取latent diffusion模型
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
        model.cuda().half()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    # 返回扩散模型的模型，在这里加载了预训练权重
    return model


if __name__ == "__main__":
    device = torch.device('cuda') if dict['device'] == 'cuda' else torch.device('cpu')

    path = './datasets'
    train_loader = get_loader(path, batch_size=dict['batch_size'])

    net = load_model_from_config(OmegaConf.load(dict['config']), dict['ckpt'], device)
    optimizer = net.configure_optimizers()

    writer = SummaryWriter(log_dir='./logs')

    # 训练
    for n in trange(dict['epochs'], desc='epochs'):
        l = []
        for i, (images, prompts) in enumerate(tqdm(train_loader, desc='training')):
            optimizer.zero_grad()

            images = images.to('cuda:0').half()
            net.on_train_batch_start(images, i, n)
            # encode_first_stage、get_first_stage_encoding将图像映射到隐空间
            p = net.encode_first_stage(images)
            z = net.get_first_stage_encoding(p).half() # torch.Size([4, 4, 64, 64])
            # 对文本prompts进行编码
            c = net.get_learned_conditioning(prompts).to('cuda:0').half()
            loss, loss_dict = net(z, c)

            loss.backward()
            l.append(loss.item())
            # torch.nn.utils.clip_grad_norm_ 的作用是将模型的所有梯度的范数限制在一个最大值（max_norm）内。
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0) # 如果梯度的范数超过了 max_norm，则会按比例缩放所有梯度，使其范数不超过 max_norm。
            optimizer.step()

        writer.add_scalar(tag='every epoch loss', scalar_value=sum(l) / len(l), global_step=n+1)

        if n % 10 == 0:
            # 每10个epoch采样一次
            sampler = PLMSSampler(net, device=device)
            sample_for_test(sampler, net, device, dict)
            # 保存模型参数
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'epoch_{1 + n}_{timestamp}.pth'
            torch.save(net.state_dict(), f'./checkpoints/{save_path}')

    print("训练完成")
