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
from flux.train_utils import sample_images
from flux.modules.lora import replace_linear_with_lora

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
    
    # NOTE: When =1, it significantly slows down the training from 7 mins -> 12 mins for 1K steps. only use for debugging.
    detailed_metrics_log_steps : int = None
    
    # optimizer config.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    
    # LR Config
    lr_scheduler: str = "constant"
    learning_rate: float = 4e-4
    lr_warmup_steps: int = 10
    
    # Sampling config.
    disable_sampling: bool = False
    sample_every: int = 100
    checkpointing_steps: int = 500
    inference_steps: int = 50 if model_name == "flux-dev" else 4
    sample_prompts = ['me'] ## , 'me wearing red corset'
    sample_width = 480
    sample_height = 680
    sample_steps = 30
    sample_seed = 0

def get_models(name: str, device):
    t5, clip, dit, ae = get_flux_models(name=name, device=device)

    ## Load models & set grad false for non-lora weights. 
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    ae.requires_grad_(False)

    dit.train()
    replace_linear_with_lora(dit, max_rank=16, scale=1.0)

    for n, param in dit.named_parameters():
        if 'lora' not in n:
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
    wandb.init(project=config.wandb_project_name, name=config.wandb_run_name)

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

    wandb.config.update(config.__dict__)
    for _, batch in enumerate(train_dataloader):
        if not config.disable_sampling and (global_step % config.sample_every == 0 or global_step + 1 == config.num_train_steps):
            print(f"Sampling images for step {global_step}...")
            images = sample_images(config, clip, t5, ae, dit, device)
            wandb.log({'result' : [wandb.Image(img, caption=config.sample_prompts[i]) for i, img in enumerate(images)]}, step=global_step)
        
        img, prompts = batch
        with torch.no_grad():
            img = img.to(device) # (b, 3, h, w)
            x_1 = ae.encode(img).to(torch.bfloat16) # (b, 16, h/8, w/8) => only 4x reduction??
            inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)
            # x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        timesteps = get_schedule(config.inference_steps, inp["img"].shape[1], shift=(config.model_name != "flux-schnell"))
        x_1 = inp['img']
        t = torch.tensor([timesteps[random.randint(0, config.inference_steps - 1)]]).to(device).to(x_1.dtype)
        x_0 = torch.randn_like(x_1).to(device)
        x_t = (1 - t) * x_1 + t * x_0
        # NOTE: guidance_vec only works at guidance=1.0. at any other guidance training doesn't work, and produces very bad images. 
        # This is probably an artifact of de-distillation, but not  sure. 
        guidance_vec = torch.full((x_t.shape[0],), 1.0, device=x_1.device, dtype=x_1.dtype)

        model_pred = dit(img=x_t,
                        img_ids=inp['img_ids'],
                        txt=inp['txt'],
                        txt_ids=inp['txt_ids'],
                        y=inp['vec'],
                        timesteps=t,
                        guidance=guidance_vec,)

        loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")
        

        loss.backward()
        torch.nn.utils.clip_grad_norm_(dit.parameters(), config.max_grad_norm)
    
        # Log norm and std of all trainable parameters
        if config.detailed_metrics_log_steps and config.detailed_metrics_log_steps % global_step == 0:
            for name, param in dit.named_parameters():
                if param.requires_grad:
                    wandb.log({
                        f"{name}_norm": param.norm().item(),
                        f"{name}_std": param.std().item(),
                        f"{name}_grad": param.grad.norm().item() if param.grad is not None else 0
                    }, step=global_step)

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        wandb.log(logs, step=global_step)
        
        
        # Take next step. 
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.set_postfix(**logs)
        progress_bar.update(1)
        global_step += 1
            

        if global_step % config.checkpointing_steps == 0:
            # save_checkpoint()
            pass

if __name__ == "__main__":
    main()
