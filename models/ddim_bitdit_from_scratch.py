from random import random
import math
from collections import namedtuple
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel, AutoConfig
from .bert import BertModel
from .utils import decimal_to_bits, bits_to_decimal
from data.cws.cws_dataset import LabelSet
from .dit_discrete import DiT

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def identity(t, *args, **kwargs):
    return t


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class BitDit(nn.Module):
    def __init__(self,
                 device: torch.device,
                 num_classes: int,
                 backbone: str,
                 time_steps: int,
                 sampling_steps: int,
                 encoder_depth: int,
                 decoder_depth: int,
                 max_length: int,
                 ddim_sampling_eta: float,
                 self_condition: bool,
                 snr_scale: float,
                 dataset: str,
                 dim_model: int,
                 dim_time: int,
                 noise_schedule: str = 'linear',
                 objective: str = 'pred_x0',
                 loss_type: str = 'l2',
                 add_lstm: bool = False,
                 freeze_bert: bool = False):
        super().__init__()

        self.device = torch.device(device)

        # entity classes
        self.num_classes = num_classes
        self.dim_model = dim_model
        self.add_lstm = add_lstm

        self.config = AutoConfig.from_pretrained(backbone)
        self.config.max_position_embeddings = max_length
        self.config.num_hidden_layers = encoder_depth
        self.backbone = BertModel(self.config)

        # # backbone: pretrained BERT or RoBERTA, name or path
        # self.backbone = AutoModel.from_pretrained(backbone)

        # build diffusion
        # 100
        self.timesteps = time_steps
        self.time_difference = 0
        sampling_timesteps = sampling_steps
        self.objective = objective
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}
        if noise_schedule == "linear":
            betas = linear_beta_schedule(time_steps)
        elif noise_schedule == "cosine":
            betas = cosine_beta_schedule(time_steps)
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps, \
            'sampling steps must be smaller than time steps for ddim and equal to for ddpm'
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        # 1.
        self.ddim_sampling_eta = ddim_sampling_eta
        # whether or not use self condition, default to False
        self.self_condition = self_condition
        # 1.0 for image generation, 0.1 for semantic segmentation, 2.0 for object detection
        self.scale = snr_scale

        self.label_set = LabelSet(dataset)
        self.bits = torch.ceil(torch.log2(torch.tensor(len(self.label_set._labelset)))).long()
        self.loss_type = loss_type
        self.model = DiT(in_channels=self.bits.item(),
                         hidden_size=dim_model,
                         num_steps=self.timesteps,
                         time_dim=dim_time,
                         depth=decoder_depth,
                         num_heads=8,
                         mlp_ratio=4.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.to(self.device)
        if freeze_bert:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, bert_features, attention_mask, x_self_cond=None, clip_x_start=True):
        """
        make model predictions, get pred_x_start and pred_noise at the same time
        Args:
            x: [bsz, len, bits]
            t: [bsz,]
            bert_features: [bsz, len, hid]
            attention_mask: [bsz, len]
            x_self_cond: [bsz, len, bits] / None
            clip_x_start: True/False

        Returns:
            (x_start, nosie): [bsz, len, bits]
        """
        model_output = self.model(x, t, bert_features, attention_mask, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-self.scale, max=self.scale) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, bert_features, attention_mask, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, bert_features, attention_mask, x_self_cond=x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-self.scale, self.scale)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t, bert_features, attention_mask, x_self_cond=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batched_times,
                                                                          bert_features, attention_mask,
                                                                          x_self_cond=x_self_cond,
                                                                          clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_bits = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_bits, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, bert_features, attention_mask):
        batch, device = shape[0], self.betas.device

        bit_seq = torch.randn(shape, device=device)

        x_start = None

        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            bit_seq, x_start = self.p_sample(bit_seq, t, bert_features, attention_mask, self_cond)

        return bit_seq

    @torch.no_grad()
    def ddim_sample(self, shape, bert_features, attention_mask):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = \
            shape[
                0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        bit_seq = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(bit_seq, time_cond, bert_features,
                                                             attention_mask, self_cond, clip_x_start=True)

            if time_next < 0:
                bit_seq = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(bit_seq)

            bit_seq = x_start * alpha_next.sqrt() + \
                      c * pred_noise + \
                      sigma * noise

        return bit_seq

    @torch.no_grad()
    def sample(self, shape, bert_features, attention_mask):
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(shape, bert_features, attention_mask)

    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, seq_labels: torch.tensor,
                words2pieces: torch.tensor = None, ensemble: bool = False):
        """

        Args:
            input_ids: [bsz, len]
            attention_mask: [bsz, len]
            words2pieces: [bsz, word_len, piece_len]
            seq_labels: [bsz, len]

        Returns:

        """
        # feature extraction: [bsz, len_piece, d_model]
        label_mask = (seq_labels != -100).long()
        bert_output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        if self.add_lstm:
            min_value = torch.min(bert_output).item()
            bert_output = bert_output.unsqueeze(dim=1).expand(-1, words2pieces.shape[1], -1, -1)
            bert_output = torch.masked_fill(bert_output, words2pieces.eq(0).unsqueeze(dim=-1), min_value)
            # [bsz, len_word, d_model]
            features, _ = torch.max(bert_output, dim=2)
            lengths = words2pieces.sum(dim=-1).gt(0).sum(dim=-1).cpu()

            packed_features = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
            packed_outs, (hidden, _) = self.lstm(packed_features)
            # [bsz, word_len, d_model]
            features = pad_packed_sequence(packed_outs, batch_first=True, total_length=max(lengths))[0]
        else:
            # [bsz, pieces_len, d_model]
            features = bert_output

        if not self.training:
            shape = (*features.shape[:2], self.bits.item())
            results = self.sample(shape, features, label_mask)
            results = bits_to_decimal(results, self.bits.item())
            return results

        if self.training:
            # categorical data to bits [bsz, ]
            # gold and noised sequence labels
            bits_seq_labels = decimal_to_bits(seq_labels, self.bits)
            bits_seq_labels *= self.scale
            noise_bits_seq_labels, ts, noise = self.prepare_targets(bits_seq_labels)

            self_cond = None
            if random() < 0.5:
                with torch.no_grad():
                    self_cond = self.model_predictions(noise_bits_seq_labels, ts, features, label_mask).pred_x_start
                    self_cond.detach_()

            pred = self.model_predictions(noise_bits_seq_labels, ts, features, label_mask)

            targets = bits_seq_labels
            targets_mask = label_mask.unsqueeze(dim=-1).expand(-1, -1, self.bits.item())
            if self.objective == 'pred_noise':
                targets = noise
                pred = pred.pred_noise
            elif self.objective == 'pred_x0':
                pred = pred.pred_x_start
            if self.loss_type == 'l2':
                loss = F.mse_loss(pred[targets_mask.bool()], targets[targets_mask.bool()])
            elif self.loss_type == 'l1':
                loss = F.l1_loss(pred[targets_mask.bool()], targets[targets_mask.bool()])
            else:
                raise NotImplementedError
            return loss

    def prepare_targets(self, gold_seq_labels):
        """
        run forward process to get
        Args:
            gold_seq_labels: [bsz, len]

        Returns:
            diffused_labels: [bsz, len]
            ts: [bsz]
        """

        bsz = gold_seq_labels.shape[0]
        ts = torch.randint(0, self.num_timesteps, (bsz,), device=self.device).long()
        noise = torch.randn_like(gold_seq_labels, device=self.device)
        diffused_labels = self.q_sample(x_start=gold_seq_labels, t=ts, noise=noise)

        return diffused_labels, ts, noise

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
