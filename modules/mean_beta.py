from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import prod
from typing import Generic, Literal, Sequence, TypeVar

import torch
from jaxtyping import Float
from torch import Tensor, device
from torch.distributions.beta import Beta
from torch.nn import functional as F


# Scalar time samplers ----------------------------------------------------


@dataclass
class ScalarTimeSamplerCfg:
    name: str


TScalarCfg = TypeVar("TScalarCfg", bound=ScalarTimeSamplerCfg)


class ScalarTimeSampler(Generic[TScalarCfg], ABC):
    def __init__(self, cfg: TScalarCfg) -> None:
        self.cfg = cfg

    @abstractmethod
    def __call__(
        self,
        shape: Sequence[int],
        device: device | str = "cpu",
    ) -> Float[Tensor, "*shape"]:
        raise NotImplementedError


@dataclass
class UniformCfg(ScalarTimeSamplerCfg):
    name: Literal["uniform"] = "uniform"


class Uniform(ScalarTimeSampler[UniformCfg]):
    def __call__(
        self,
        shape: Sequence[int],
        device: device | str = "cpu",
    ) -> Float[Tensor, "*shape"]:
        return torch.rand(shape, device=device)


SCALAR_TIME_SAMPLERS: dict[str, type[ScalarTimeSampler]] = {
    "uniform": Uniform,
}


def get_scalar_time_sampler(cfg: ScalarTimeSamplerCfg) -> ScalarTimeSampler:
    try:
        sampler_cls = SCALAR_TIME_SAMPLERS[cfg.name]
    except KeyError as exc:
        raise ValueError(f"Unknown scalar time sampler '{cfg.name}'") from exc
    return sampler_cls(cfg)


# Time samplers -----------------------------------------------------------


@dataclass
class HistogramPdfEstimatorCfg:
    num_bins: int = 1000
    blur_kernel_size: int = 5
    blur_kernel_sigma: float = 0.2


class HistogramPdfEstimator:
    """Estimate the density of samples and return the inverse density as weights."""

    def __init__(
        self,
        initial_samples: Float[Tensor, "sample"],
        cfg: HistogramPdfEstimatorCfg,
    ) -> None:
        self.cfg = cfg
        self.device = initial_samples.device
        self.histogram = self.get_smooth_density_histogram(initial_samples)
        assert self.histogram.min().item() > 0.001, (
            "Histogram too inaccurate, please use a different time_sampler"
        )

    def get_gaussian_1d_kernel(self) -> Float[Tensor, "kernel_size"]:
        if self.cfg.blur_kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        center = self.cfg.blur_kernel_size // 2
        x = torch.arange(-center, center + 1, dtype=torch.float32, device=self.device)

        kernel = torch.exp(-0.5 * (x / self.cfg.blur_kernel_sigma) ** 2)
        kernel /= kernel.sum()
        return kernel

    def get_smooth_density_histogram(
        self,
        vals: Float[Tensor, "sample"],
    ) -> Float[Tensor, "num_bins"]:
        assert vals.min() >= 0, "Timesteps must be nonnegative"
        assert vals.max() <= 1, "Timesteps must be less or equal than 1"
        histogram_torch = torch.histc(vals, self.cfg.num_bins, min=0, max=1).to(
            self.device
        )

        kernel = self.get_gaussian_1d_kernel()
        padded_hist = F.pad(
            histogram_torch.unsqueeze(0).unsqueeze(0),
            (self.cfg.blur_kernel_size // 2, self.cfg.blur_kernel_size // 2),
            mode="reflect",
        )
        histogram_torch_conv = F.conv1d(
            padded_hist, kernel.unsqueeze(0).unsqueeze(0)
        ).to(self.device)

        return histogram_torch_conv.squeeze() / histogram_torch_conv.mean()

    def __call__(self, t: Float[Tensor, "sample"]) -> Float[Tensor, "sample"]:
        bin_ids = (t * self.cfg.num_bins).long()
        bin_ids.clamp_(0, self.cfg.num_bins - 1)
        return self.histogram[bin_ids]


@dataclass
class TimeSamplerCfg:
    name: str
    histogram_pdf_estimator: HistogramPdfEstimatorCfg = field(
        default_factory=HistogramPdfEstimatorCfg
    )
    num_normalization_samples: int = 80000
    eps: float = 1e-6
    add_zeros: bool = False


TTimeSamplerCfg = TypeVar("TTimeSamplerCfg", bound=TimeSamplerCfg)


class TimeSampler(Generic[TTimeSamplerCfg], ABC):
    def __init__(self, cfg: TTimeSamplerCfg, resolution: tuple[int, int]) -> None:
        self.cfg = cfg
        self.resolution = resolution
        self.dim = prod(self.resolution)

    @abstractmethod
    def get_time(
        self,
        batch_size: int,
        num_samples: int = 1,
        device: device | str = "cpu",
    ) -> Float[Tensor, "batch sample height width"]:
        raise NotImplementedError

    def get_normalization_samples(self, device: device | str) -> Float[Tensor, "sample"]:
        return self.get_time(self.cfg.num_normalization_samples, device=device).flatten()

    def get_normalization_weights(
        self,
        t: Float[Tensor, "*batch"],
    ) -> Float[Tensor, "*#batch"]:
        if self.cfg.histogram_pdf_estimator is None:
            return torch.ones_like(t)

        if not hasattr(self, "histogram_pdf_estimator"):
            self.histogram_pdf_estimator = HistogramPdfEstimator(
                self.get_normalization_samples(t.device),
                self.cfg.histogram_pdf_estimator,
            )

        shape = t.shape
        probs = self.histogram_pdf_estimator(t.flatten())
        weights = (1 + self.cfg.eps) / (probs + self.cfg.eps)
        return weights.view(shape)

    def __call__(
        self,
        batch_size: int,
        num_samples: int = 1,
        device: device | str = "cpu",
    ) -> tuple[
        Float[Tensor, "batch sample height width"],
        Float[Tensor, "batch #sample height width"],
    ]:
        t = self.get_time(batch_size, num_samples, device)
        weights = self.get_normalization_weights(t)

        if self.cfg.add_zeros:
            t = t.flatten(-2).contiguous()
            weights = weights.flatten(-2).contiguous()
            zero_ratios = torch.rand((batch_size,), device=device)
            zero_mask = (
                torch.linspace(1 / self.dim, 1, self.dim, device=device)
                < zero_ratios[:, None, None]
            )
            idx = torch.rand_like(t).argsort(dim=-1)
            t[zero_mask] = 0
            weights[zero_mask] = 0
            t = t.gather(-1, idx).reshape(batch_size, -1, *self.resolution)
            weights = weights.gather(-1, idx).reshape(batch_size, -1, *self.resolution)

        return t, weights


@dataclass
class TwoStageTimeSamplerCfg(TimeSamplerCfg):
    scalar_time_sampler: ScalarTimeSamplerCfg = field(default_factory=UniformCfg)


TTwoStageCfg = TypeVar("TTwoStageCfg", bound=TwoStageTimeSamplerCfg)


class TwoStageTimeSampler(TimeSampler[TTwoStageCfg], ABC):
    scalar_time_sampler: ScalarTimeSampler

    def __init__(self, cfg: TTwoStageCfg, resolution: tuple[int, int]) -> None:
        super(TwoStageTimeSampler, self).__init__(cfg, resolution)
        self.scalar_time_sampler = get_scalar_time_sampler(cfg.scalar_time_sampler)


# MeanBeta sampler --------------------------------------------------------


@dataclass
class MeanBetaCfg(TwoStageTimeSamplerCfg):
    name: Literal["mean_beta"]
    beta_sharpness: float = 1.0


# modules/mean_beta.py
class MeanBeta(TwoStageTimeSampler[MeanBetaCfg]):
    def __init__(self, resolution: tuple[int, int], cfg: MeanBetaCfg | dict | None = None, **kwargs) -> None:
        if cfg is None:
            cfg = kwargs
        if isinstance(cfg, dict):
            # allow scalar_time_sampler to be given as a string or dict
            scalar_cfg = cfg.get("scalar_time_sampler", {"name": "uniform"})
            if isinstance(scalar_cfg, str):
                scalar_cfg = {"name": scalar_cfg}
            cfg = MeanBetaCfg(
                scalar_time_sampler=ScalarTimeSamplerCfg(**scalar_cfg),
                name=cfg.get("name", "mean_beta"),
                beta_sharpness=cfg.get("beta_sharpness", 1.0),
            )

        super().__init__(cfg, resolution)
        self.dim = prod(resolution)
        self.betas: dict[int, Beta] = {}
        self.init_betas(self.dim)


    def init_betas(self, dim: int) -> None:
        if dim > 1 and dim not in self.betas:
            a = b = (dim - 1 - (dim % 2)) ** 1.05 * self.cfg.beta_sharpness
            self.betas[dim] = Beta(a, b)
            half_dim = dim // 2
            self.init_betas(half_dim)
            self.init_betas(dim - half_dim)

    def _get_uniform_l1_conditioned_vector_list(
        self,
        l1_norms: Float[Tensor, "batch"],
        dim: int,
    ) -> list[Float[Tensor, "batch"]]:
        if dim == 1:
            return [l1_norms]

        device = l1_norms.device
        half_cells = dim // 2

        max_first_contribution = l1_norms.clamp(max=half_cells)
        max_second_contribution = l1_norms.clamp(max=dim - half_cells)
        min_first_contribution = (l1_norms - max_second_contribution).clamp_(min=0)

        random_matrix = self.betas[dim].sample((l1_norms.shape[0],)).to(device=device)
        ranges = max_first_contribution - min_first_contribution

        assert ranges.min() >= 0
        first_contribution = min_first_contribution + ranges * random_matrix
        second_contribution = l1_norms - first_contribution

        return self._get_uniform_l1_conditioned_vector_list(
            first_contribution, half_cells
        ) + self._get_uniform_l1_conditioned_vector_list(
            second_contribution, dim - half_cells
        )

    def _sample_time_matrix(
        self,
        l1_norms: Float[Tensor, "batch"],
        dim: int,
    ) -> Float[Tensor, "batch dim"]:
        vector_list = self._get_uniform_l1_conditioned_vector_list(l1_norms, dim)
        t = torch.stack(vector_list, dim=1)  # [batch_size, dim]
        idx = torch.rand_like(t).argsort()
        t = t.gather(1, idx)
        return t

    def get_time_with_mean(
        self,
        mean: Float[Tensor, "*batch"],
        resolution: tuple[int, int] | None = None,
    ) -> Float[Tensor, "*batch height width"]:
        shape = mean.shape
        resolution = self.resolution if resolution is None else resolution
        dim = self.dim if resolution is None else prod(resolution)
        l1_norms = mean.flatten() * dim
        t = self._sample_time_matrix(l1_norms, dim)
        return t.view(*shape, *resolution)

    def get_time(
        self,
        batch_size: int,
        num_samples: int = 1,
        resolution: tuple[int, int] | None = None,
        device: device | str = "cpu",
    ) -> Float[Tensor, "batch sample height width"]:
        mean = self.scalar_time_sampler((batch_size, num_samples), device)
        return self.get_time_with_mean(mean, resolution)
