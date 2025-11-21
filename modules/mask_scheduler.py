import math
import torch
from tqdm import tqdm
import pytorch_lightning as pl
from typing import Optional, Union


def _truncated_normal(shape, *, device, dtype=torch.float32, mean=1.0, std=0.25, lower=0.0, upper=1.0):
    """Sample from a clipped normal distribution via inverse transform sampling."""
    sigma = std
    a = (lower - mean) / sigma
    b = (upper - mean) / sigma
    sqrt_two = math.sqrt(2.0)

    def _phi(x):
        return 0.5 * (1.0 + torch.erf(x / sqrt_two))

    cdf_a = _phi(torch.tensor(a, device=device, dtype=dtype))
    cdf_b = _phi(torch.tensor(b, device=device, dtype=dtype))
    u = torch.rand(shape, device=device, dtype=dtype) * (cdf_b - cdf_a) + cdf_a
    clipped = mean + sigma * sqrt_two * torch.erfinv(2 * u - 1)
    return clipped.clamp_(lower, upper)

class UnconditionalMaskGITScheduler(pl.LightningModule):
    def __init__(self, *, num_tokens, mask_value, codebook_size, num_predicted_frames =1, default_schedule_mode_train="arccos", default_schedule_mode="arccos", default_num_steps=12, disable_bar=False, max_block_size=1) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.num_predicted_frames = num_predicted_frames
        self.total_tokens = self.num_tokens * self.num_predicted_frames
        self.codebook_size = codebook_size
        self.mask_value = mask_value
        self.default_schedule_mode = default_schedule_mode
        self.default_schedule_mode_train = default_schedule_mode_train
        self.default_num_steps = default_num_steps
        self.disable_bar = disable_bar
        self.max_block_size = max_block_size

    def _schedule_transform(self, values: torch.Tensor, mode: str) -> torch.Tensor:
        sqrt_half_pi = math.pi * 0.5

        if mode == "linear":
            return values
        if mode == "square":
            return values.pow(2)
        if mode == "cosine":
            return torch.cos(values * sqrt_half_pi)
        if mode == "arccos":
            return torch.arccos(values).div(sqrt_half_pi)
        if mode == "cubic":
            return 1 - values.pow(3)
        if mode == "pow4":
            return 1 - values.pow(4)
        if mode == "pow6":
            return 1 - values.pow(6)
        if mode == "inv_root":
            return values.sqrt()
        if mode == "arccos2":
            return torch.arccos(values.pow(2)).div(sqrt_half_pi)
        if mode == "power_trunc":
            return _truncated_normal(values.shape, device=values.device, dtype=values.dtype)

        raise ValueError(f"Unknown schedule mode '{mode}'.")

    def _sample_mask_ratio(self, batch_size: int, mode: str, device: torch.device) -> torch.Tensor:
        random_values = torch.rand(batch_size, device=device)
        return self._schedule_transform(random_values, mode)

    def _apply_mask_value(self, mask_code: torch.Tensor, mask: torch.Tensor, value: Optional[Union[torch.Tensor, int]]) -> torch.Tensor:
        value = self.mask_value if value is None else value
        mask = mask.to(mask_code.device)
        expanded_mask = mask
        if mask_code.dim() == mask.dim() + 1:
            expanded_mask = mask.unsqueeze(-1).expand_as(mask_code)

        if isinstance(value, int):
            if value > 0:
                mask_code[expanded_mask] = value
            else:
                count = int(expanded_mask.sum().item())
                if count:
                    random_vals = torch.randint(self.codebook_size, (count,), device=mask_code.device, dtype=mask_code.dtype)
                    mask_code[expanded_mask] = random_vals
        
        # Use for Mar 
        # elif isinstance(value, torch.Tensor):
        #     replacement = value.to(mask_code.device, mask_code.dtype)
        #     if replacement.numel() == 1:
        #         mask_code[expanded_mask] = replacement.item()
        #     elif replacement.shape == mask_code.shape:
        #         mask_code[expanded_mask] = replacement[expanded_mask]
        #     else:
        #         raise ValueError("Replacement tensor must be scalar or match code shape.")
        else:
            raise TypeError("Value should be int.")

        return mask_code

    def get_mask_code(self, code, mode=None, value=None):
        """ Replace the code token by *value* according the the *mode* scheduler
           :param
            code  -> torch.LongTensor(): bsize * f * 16 * 16 or bsize * f * 16 * 16 * c, the unmasked code
            mode  -> str:                the rate of value to mask
            value -> int:                mask the code by the value
           :return
            masked_code -> torch.LongTensor(): bsize * f * 16 * 16, the masked version of the code
            mask        -> torch.LongTensor(): bsize * f * 16 * 16, the binary mask of the mask
        """
        if self.max_block_size > 1:
            return self.get_block_mask_code(code, mode, value, self.max_block_size)
        
        mode = mode or self.default_schedule_mode_train
        device = code.device
        ratios = self._sample_mask_ratio(code.size(0), mode, device)
        spatial_dims = code.shape[2:4] #if code.dim() >= 3 else code.shape[1:]
        t_dims = code.shape[1]
        mask_shape = (code.size(0),t_dims) + tuple(spatial_dims)
        random_tensor = torch.rand(mask_shape, device=device)
        ratio_view = ratios.view(-1, *([1] * (random_tensor.dim() - 1)))
        mask = random_tensor < ratio_view

        mask_code = self._apply_mask_value(code.clone(), mask, value)
        return mask_code, mask

    def get_block_mask_code(self, code, mode=None, value=None, block_size_max=4):
        """ Replace the code token by *value* according the the *mode* scheduler
           :param
            code  -> torch.LongTensor(): bsize * f * 16 * 16 or bsize * f * 16 * 16 * c, the unmasked code
            mode  -> str:                the rate of value to mask
            value -> int:                mask the code by the value
           :return
            masked_code -> torch.LongTensor(): bsize * f * 16 * 16, the masked version of the code
            mask        -> torch.LongTensor(): bsize * f * 16 * 16, the binary mask of the mask
        """
        mode = mode or self.default_schedule_mode_train
        device = code.device
        ratios = self._sample_mask_ratio(code.size(0), mode, device)

        batch, f, height, width = code.shape[:4]
        mask = torch.zeros((batch, f, height, width), dtype=torch.bool, device=device)

        for idx in range(batch):
            block_size = int(torch.randint(1, block_size_max + 1, (1,), device=device).item())
            aspect_ratio = float((torch.rand(1, device=device) * 0.75 + 0.75).item())
            block_height = max(1, int(block_size * aspect_ratio))
            block_width = max(1, int(block_size / aspect_ratio))

            grid_h = math.ceil(height / block_height)
            grid_w = math.ceil(width / block_width)

            downsampled = torch.rand((grid_h, grid_w), device=device) < ratios[idx]
            expanded = downsampled.repeat_interleave(block_height, dim=0).repeat_interleave(block_width, dim=1)
            mask[idx, :] = expanded[:height, :width]

        mask_code = self._apply_mask_value(code.clone(), mask, value)
        return mask_code, mask


    def adap_sche(self, num_steps=None, schedule_mode=None, leave=False):
        """ Create a sampling scheduler
           :param
            step  -> int:  number of prediction during inference
            mode  -> str:  the rate of value to unmask
            leave -> bool: tqdm arg on either to keep the bar or not
           :return
            scheduler -> torch.LongTensor(): the list of token to predict at each step
        """
        num_steps = num_steps or self.default_num_steps
        schedule_mode = schedule_mode or self.default_schedule_mode
        r = torch.linspace(1, 0, num_steps)
        transforms = {
            "root": lambda t: 1 - t.sqrt(),
            "linear": lambda t: 1 - t,
            "square": lambda t: 1 - t.pow(2),
            "cubic": lambda t: 1 - t.pow(3),
            "pow4": lambda t: 1 - t.pow(4),
            "pow6": lambda t: 1 - t.pow(6),
            "inv_root": lambda t: t.sqrt(),
            "cosine": lambda t: torch.cos(t * math.pi * 0.5),
            "arccos": lambda t: torch.arccos(t) / (math.pi * 0.5),
            "arccos2": lambda t: torch.arccos(t.pow(2)) / (math.pi * 0.5),
            "power_trunc": lambda t: _truncated_normal(t.shape, device=t.device, dtype=t.dtype),
        }

        if schedule_mode not in transforms:
            raise ValueError(f"Unknown schedule mode '{schedule_mode}'.")

        val_to_mask = transforms[schedule_mode](r)

        #print (val_to_mask)
        # fill the scheduler by the ratio of tokens to predict at each step
        sche = (val_to_mask / val_to_mask.sum()) * (self.total_tokens) 
        sche = sche.round()
        sche[sche == 0] = 1                                                  # add 1 to predict a least 1 token / step
        sche[-1] += (self.total_tokens) - sche.sum()         # need to sum up nb of code

        if num_steps == 1:
            sche = torch.ones_like(sche) * (self.total_tokens)


        return tqdm(sche.int(), leave=leave, disable=self.disable_bar)
