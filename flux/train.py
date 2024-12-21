import logging
import os
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from flux.sampling import get_schedule, prepare
from flux.util import get_flux_models

from flux.dataset import sft_dataset_loader
import wandb
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class TrainConfig:
    model_name: str = "flux-dev"

    # Data config. 
    train_batch_size: int = 1
    max_img_size: int = 512
    img_dir : str = '/home/azureuser/workspace/x-flux/accel/srishti_rautela_/'
    random_ratio: str = False
    
    # wandb config.
    wandb_project_name: str = "flux-lora"
    wandb_run_name: str = "lora-training"
    
    # IO config. 
    output_dir: str = "outputs"
    logging_dir: str = "logs"

    # Saving & Restart config. 
    resume_from_checkpoint: bool = False
    
    # Training config. 
    num_train_steps: int = 1_000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0  
    
    # optimizer config.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    
    # LR Config
    lr_scheduler: str = "constant"
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    
    # Sampling config.
    disable_sampling: bool = False
    sample_every: int = 500
    checkpointing_steps: int = 500
    inference_steps: int = 50 if model_name == "flux-dev" else 4

def get_models(name: str, device):
    t5, clip, dit, ae = get_flux_models(name=name, device=device)

    ## Load models & set grad false for non-lora weights. 
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    ae.requires_grad_(False)

    dit.train()
    allowed_params = [f'double_blocks.{i}.' for i in range(12,19)]

    for n, param in dit.named_parameters():
        if all(x not in n for x in allowed_params):
            param.requires_grad = False
        else:
            # print(n)
            pass
    print(sum([p.numel() for p in dit.parameters() if p.requires_grad]) / 1e6, 'M parameters')
    
    return t5, clip, dit, ae

def main():    
    # Set logging verbosity.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    config = TrainConfig()
    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)
    logger.info(f"  Total train batch size: {config.train_batch_size} * {config.gradient_accumulation_steps} = {config.train_batch_size * config.gradient_accumulation_steps}")


    # Init wandb. 
    # wandb.init(project=config.wandb_project_name, name=config.wandb_run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t5, clip, dit, ae = get_models(config.model_name, device)

    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        [p for p in dit.parameters() if p.requires_grad],
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=config.num_train_steps,
    )
    
    train_dataloader = sft_dataset_loader(
        num_train_steps=config.num_train_steps,
        train_batch_size=config.train_batch_size,
        img_dir = config.img_dir,
        max_size=config.max_img_size,
        random_ratio=config.random_ratio)

    if config.resume_from_checkpoint:
        # resume_from_checkpoint()
        pass
    else:
        global_step = 0

    progress_bar = tqdm(
        range(0, config.num_train_steps),
        initial=global_step,
        desc="Steps",
    )

    for _, batch in enumerate(train_dataloader):
        img, prompts = batch
        with torch.no_grad():
            # print(f"{img.shape=} | {img.dtype} | {img.device=}")
            img = img.to(device)
            x_1 = ae.encode(img).to(torch.bfloat16)
            inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)
            # x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        # print(f"{inp['img'].shape=}")
        timesteps = get_schedule(config.inference_steps, inp["img"].shape[1], shift=(config.model_name != "flux-schnell"))
        x_1 = inp['img'].squeeze(dim=1)
        t = torch.tensor([timesteps[random.randint(0, config.inference_steps)]]).to(device).to(x_1.dtype)
        x_0 = torch.randn_like(x_1).to(device)
        x_t = (1 - t) * x_1 + t * x_0
        guidance_vec = torch.full((x_t.shape[0],), 4, device=x_1.device, dtype=x_1.dtype)

        model_pred = dit(img=x_t,
                        img_ids=inp['img_ids'],
                        txt=inp['txt'],
                        txt_ids=inp['txt_ids'],
                        y=inp['vec'],
                        timesteps=t,
                        guidance=guidance_vec,)

        loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dit.parameters(), config.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()

        progress_bar.update(1)
        global_step += 1

        if not config.disable_sampling and global_step % config.sample_every == 0:
            # sample()
            pass

        if global_step % config.checkpointing_steps == 0:
            # save_checkpoint()
            pass

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

        if global_step >= config.num_train_steps:
            break

if __name__ == "__main__":
    main()
