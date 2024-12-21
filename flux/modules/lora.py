import torch
from torch import nn
from typing import Dict
# from flux.model import Flux
from flux.modules.layers import SingleStreamLoraBlock

def replace_linear_with_lora(
    module: nn.Module,
    max_rank: int,
    scale: float = 1.0,
) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            new_lora = LinearLora(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias,
                rank=max_rank,
                scale=scale,
                dtype=child.weight.dtype,
                device=child.weight.device,
            )

            new_lora.weight = child.weight
            new_lora.bias = child.bias if child.bias is not None else None

            setattr(module, name, new_lora)
        else:
            replace_linear_with_lora(
                module=child,
                max_rank=max_rank,
                scale=scale,
            )

def replace_single_stream_with_lora(
    model: nn.Module,
    lora_state_dict: Dict[str, torch.Tensor],
    scale: float = 1.0,
    device: str = "cuda",
):
    single_block_base = "transformer.single_transformer_blocks"
    dtype = next(model.parameters()).dtype
    for i in range(len(model.single_blocks)):
        q_lora = lora_state_dict[f"{single_block_base}.{i}.attn.to_q.lora_B.weight"] @ lora_state_dict[f"{single_block_base}.{i}.attn.to_q.lora_A.weight"]
        k_lora = lora_state_dict[f"{single_block_base}.{i}.attn.to_k.lora_B.weight"] @ lora_state_dict[f"{single_block_base}.{i}.attn.to_k.lora_A.weight"]
        v_lora = lora_state_dict[f"{single_block_base}.{i}.attn.to_v.lora_B.weight"] @ lora_state_dict[f"{single_block_base}.{i}.attn.to_v.lora_A.weight"]
        
        q_lora = q_lora.to(device, dtype=dtype)
        k_lora = k_lora.to(device, dtype=dtype)
        v_lora = v_lora.to(device, dtype=dtype)
        
        new_single_block = SingleStreamLoraBlock(
            model.hidden_size, 
            model.num_heads, 
            mlp_ratio=model.mlp_ratio, 
            q_lora=q_lora, 
            k_lora=k_lora,
            v_lora=v_lora,
            lora_weight=scale
        ).to(device, dtype=dtype)
        
        model.single_blocks[i] = new_single_block
        
        
def remap_lora_keys(state_dict):
    """Remap LoRA keys to match Flux model structure"""
    new_state_dict = {}
    
    # Handle single transformer blocks
    for i in range(38):  # Based on depth_single_blocks in configs
        # Map Q,K,V weights to combined QKV
        base_key = f"transformer.single_transformer_blocks.{i}"
        qkv_key = f"single_blocks.{i}.qkv"
        
        # Combine Q,K,V LoRA weights
        q_a = state_dict[f"{base_key}.attn.to_q.lora_A.weight"]
        k_a = state_dict[f"{base_key}.attn.to_k.lora_A.weight"]
        v_a = state_dict[f"{base_key}.attn.to_v.lora_A.weight"]
        
        q_b = state_dict[f"{base_key}.attn.to_q.lora_B.weight"]
        k_b = state_dict[f"{base_key}.attn.to_k.lora_B.weight"]
        v_b = state_dict[f"{base_key}.attn.to_v.lora_B.weight"]
        
        # Stack them for combined QKV
        new_state_dict[f"{qkv_key}.lora_A.weight"] = torch.cat([q_a, k_a, v_a], dim=0)
        new_state_dict[f"{qkv_key}.lora_B.weight"] = torch.cat([q_b, k_b, v_b], dim=0)

    # Handle double transformer blocks similarly
    for i in range(19):  # Based on depth in configs
        base_key = f"transformer.transformer_blocks.{i}"
        
        # Map for img attention
        img_qkv_key = f"double_blocks.{i}.img_attn.qkv"
        new_state_dict[f"{img_qkv_key}.lora_A.weight"] = torch.cat([
            state_dict[f"{base_key}.attn.to_q.lora_A.weight"],
            state_dict[f"{base_key}.attn.to_k.lora_A.weight"],
            state_dict[f"{base_key}.attn.to_v.lora_A.weight"]
        ], dim=0)
        new_state_dict[f"{img_qkv_key}.lora_B.weight"] = torch.cat([
            state_dict[f"{base_key}.attn.to_q.lora_B.weight"],
            state_dict[f"{base_key}.attn.to_k.lora_B.weight"],
            state_dict[f"{base_key}.attn.to_v.lora_B.weight"]
        ], dim=0)
        
        # Map for txt attention
        txt_qkv_key = f"double_blocks.{i}.txt_attn.qkv"
        new_state_dict[f"{txt_qkv_key}.lora_A.weight"] = torch.cat([
            state_dict[f"{base_key}.attn.add_q_proj.lora_A.weight"],
            state_dict[f"{base_key}.attn.add_k_proj.lora_A.weight"],
            state_dict[f"{base_key}.attn.add_v_proj.lora_A.weight"]
        ], dim=0)
        new_state_dict[f"{txt_qkv_key}.lora_B.weight"] = torch.cat([
            state_dict[f"{base_key}.attn.add_q_proj.lora_B.weight"],
            state_dict[f"{base_key}.attn.add_k_proj.lora_B.weight"],
            state_dict[f"{base_key}.attn.add_v_proj.lora_B.weight"]
        ], dim=0)

    return new_state_dict

# def apply_fal_lora_to_flux(model, lora_path, device="cuda"):
#     """Apply LoRA weights to a Flux model"""
#     # Load LoRA state dict
#     lora_state_dict = torch.load(lora_path, map_location=device)
    
#     # Remap keys to match Flux structure
#     remapped_state_dict = remap_lora_keys(lora_state_dict)
    
#     # Convert model to LoRA if not already
#     if not isinstance(model, FluxLoraWrapper):
#         model = FluxLoraWrapper(
#             lora_rank=16,  # Use same rank as original LoRA
#             lora_scale=1.0,
#             params=model.params
#         )
        
#     # Load remapped weights
#     missing, unexpected = model.load_state_dict(remapped_state_dict, strict=False)
#     print(f"Missing keys: {missing}")
#     print(f"Unexpected keys: {unexpected}")
    
#     return model

class LinearLora(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
        lora_bias: bool = True,
        scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias is not None,
            device=device,
            dtype=dtype,
            *args,
            **kwargs,
        )

        assert isinstance(scale, float), "scale must be a float"

        self.scale = scale
        self.rank = rank
        self.lora_bias = lora_bias
        self.dtype = dtype
        self.device = device

        if rank > (new_rank := min(self.out_features, self.in_features)):
            self.rank = new_rank

        self.lora_A = nn.Linear(
            in_features=in_features,
            out_features=self.rank,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.lora_B = nn.Linear(
            in_features=self.rank,
            out_features=out_features,
            bias=self.lora_bias,
            dtype=dtype,
            device=device,
        )

    def set_scale(self, scale: float) -> None:
        assert isinstance(scale, float), "scalar value must be a float"
        self.scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base_out = super().forward(input)

        _lora_out_B = self.lora_B(self.lora_A(input))
        lora_update = _lora_out_B * self.scale

        return base_out + lora_update
