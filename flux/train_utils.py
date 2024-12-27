from flux.pipeline import FluxSampler

def sample_images(config, clip, t5, ae, dit, device):
    flux_sampler = FluxSampler(name=config.model_name, clip=clip, t5=t5, ae=ae, model=dit, device=device)
    images = []
    for i, prompt in enumerate(config.sample_prompts):
        result = flux_sampler(prompt=prompt,
                            width=config.sample_width,
                            height=config.sample_height,
                            num_steps=config.sample_steps,
                            seed=config.sample_seed,
                            )
        images.append(result)
        print(f"Result for prompt #{i} is generated")
    return images
    