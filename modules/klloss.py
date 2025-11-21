import torch
import torch.nn as nn

from modules.discriminator import NLayerDiscriminator, weights_init
from .lpips import LPIPS 
from modules.vqloss import adopt_weight, hinge_d_loss, vanilla_d_loss


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0, adaptive_disc_weight=True, distill_loss_weight=0.1, se_weight =0.25,
                 perceptual_weight=1.0, warmup_steps=1000, use_actnorm=False, disc_conditional=False, beta_1=0.5, beta_2=0.9,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.adaptive_disc_weight = adaptive_disc_weight
        self.warmup_steps = warmup_steps
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.se_weight = se_weight
        self.distill_loss_weight = distill_loss_weight

        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=(1,)) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, posteriors, distill_loss, x, xrec,  optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        
        # now the GAN part
        if optimizer_idx == 0:
            rec_loss = torch.tensor(0.0, device=distill_loss.device)
            p_loss = torch.tensor(0.0, device=distill_loss.device)
            
            if isinstance(x, list):
                for idx, (input, recon) in enumerate(zip(x, xrec)):
                    l_weight = 1 if idx == 0 else self.se_weight
                    rec_loss += torch.abs(input - recon).mean() * l_weight
                    if self.perceptual_weight > 0:
                        p_loss += self.perceptual_loss(input, recon).mean() * l_weight
            else:
                rec_loss = torch.abs(x - xrec).mean()
                if self.perceptual_weight > 0:
                    p_loss = self.perceptual_loss(x, xrec).mean()
            rec_loss_log = rec_loss
            rec_loss = rec_loss + self.perceptual_weight * p_loss


            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            # weighted_nll_loss = nll_loss
            # if weights is not None:
            #     weighted_nll_loss = weights*nll_loss
            # weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            # kl_loss = posteriors.kl()
            
            kl_loss = posteriors.kl()
            kl_loss = torch.mean(kl_loss)
            
            # generator update
            reconstruction = xrec[0] if isinstance(xrec, list) else xrec
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstruction)
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstruction, cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.adaptive_disc_weight:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0, device=distill_loss.device)
            else:
                d_weight = torch.tensor(self.disc_weight, device=distill_loss.device)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss + self.distill_loss_weight * distill_loss.mean()
            log = {
                    "{}/kl_loss".format(split): kl_loss.detach().mean(),
                    "{}/total_loss".format(split): loss.clone().detach().mean(),
                    "{}/logvar".format(split): self.logvar.detach(),
                    "{}/rec_loss".format(split): rec_loss_log.detach().mean(),
                    "{}/nll_loss".format(split): nll_loss.detach(),
                    "{}/d_weight".format(split): d_weight.detach(),
                    "{}/p_loss".format(split): p_loss.detach().mean(),
                    "{}/disc_factor".format(split): torch.tensor(disc_factor),
                    "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if isinstance(x, list):
                x = x[0]
                xrec = xrec[0]
            if cond is None:
                logits_real = self.discriminator(x.detach())
                logits_fake = self.discriminator(xrec.detach())
            else:
                logits_real = self.discriminator(torch.cat((x.detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((xrec.detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


