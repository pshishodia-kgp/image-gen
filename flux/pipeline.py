import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
from einops import rearrange
from fire import Fire
from transformers import pipeline
from PIL import Image

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5, save_image #, replace_single_stream_with_lora
from flux.modules.lora import replace_single_stream_with_lora
from safetensors.torch import load_file as load_sft

NSFW_THRESHOLD = 0.85

def get_output_filename(output_dir):
    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0
    return output_name.format(idx=idx)

class FluxSampler:
    def __init__(
        self,
        name: str = "flux-schnell",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = None,
    ):
        self.name = name
        self.torch_device = torch.device(device)
        self.output_dir = output_dir
        
        # init all components
        if name not in configs:
            available = ", ".join(configs.keys())
            raise ValueError(f"Got unknown model name: {name}, chose from {available}")
        
        t0 = time.perf_counter()
        self.t5 = load_t5(self.torch_device, max_length=256 if name == "flux-schnell" else 512)
        print(f"Loading T5 took {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        self.clip = load_clip(self.torch_device)
        print(f"Loading CLIP took {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        self.model = load_flow_model(name, self.torch_device, verbose=True)
        print(f"Loading flow model took {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        self.ae = load_ae(name, self.torch_device)
        print(f"Loading autoencoder took {time.perf_counter() - t0:.2f}s")
        
    def add_fal_lora(self, lora_path: str, scale: float = 1, device="cuda"):
        lora_sd = load_sft(lora_path, device=device)
        replace_single_stream_with_lora(self.model, lora_state_dict=lora_sd, scale=scale)

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str = "Indian girl sitting in an auto",
        width: int = 1360,
        height: int = 768,
        seed: int | None = None,
        num_steps: int | None = None,
        guidance: float = 3.5
    ):
        """
        Sample the flux model.

        Args:
            name: Name of the model to load
            height: height of the sample in pixels (should be a multiple of 16)
            width: width of the sample in pixels (should be a multiple of 16)
            seed: Set a seed for sampling
            output_name: where to save the output image, `{idx}` will be replaced
                by the index of the sample
            prompt: Prompt used for sampling
            device: Pytorch device
            num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
            guidance: guidance value used for guidance distillation
        """
        # nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

        if num_steps is None:
            num_steps = 4 if self.name == "flux-schnell" else 50

        # allow for packing and conversion to latent space
        height = 16 * (height // 16)
        width = 16 * (width // 16)

        rng = torch.Generator(device="cpu")

        if seed is None:
            seed = rng.seed()
        print(f"Generating image with {seed=}:\n{prompt=}")
        t0 = time.perf_counter()

        # prepare input
        x = get_noise(
            1,
            height,
            width,
            device=self.torch_device,
            dtype=torch.bfloat16,
            seed=seed,
        )

        inp = prepare(self.t5, self.clip, x, prompt=prompt)
        timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        # denoise initial noise
        x = denoise(self.model, **inp, timesteps=timesteps, guidance=guidance)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        print(f"Denoising took {t3 - t2:.1f}s.")
        
        # decode latents to pixel space
        x = unpack(x.float(), height, width)
        with torch.autocast(device_type=self.torch_device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
                
        print(f"Done in {t1 - t0:.1f}s.")

        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        if self.output_dir:
            fn = get_output_filename(self.output_dir)
            print(f"Saving image to {fn=}")
            img.save(fn)
        # nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]
        return img


def main(
    prompt: str = "Indian girl sitting in an auto",
    name: str = "flux-schnell",
    width: int = 1360,
    height: int = 768,
    seed: int | None = None,
    num_steps: int | None = None,
    guidance: float = 3.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "output"
):
    flux_sampler = FluxSampler(name=name, device=device, output_dir=output_dir)
    _ = flux_sampler(prompt, width, height, seed, num_steps, guidance)

def app():
    Fire(main)


if __name__ == "__main__":
    app()