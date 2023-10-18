import math
from collections import namedtuple
from functools import partial
from torch.special import expm1
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel
from .utils import decimal_to_bits, bits_to_decimal
from data import LabelSet1D
from .dit1D import DiT
from einops import repeat

__all__ = ["DiffusionEntityRecognizer"]

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


def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))


def alpha_cosine_log_snr(t, s: float = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1,
                eps=1e-5)  # not sure if this accounts for beta being clipped to 0.999 in discrete version


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


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


class DiffusionEntityRecognizer(nn.Module):
    def __init__(self,
                 device: torch.device,
                 num_classes: int,
                 backbone: str,
                 time_steps: int,
                 sampling_steps: int,
                 ddim_sampling_eta: float,
                 self_condition: bool,
                 snr_scale: float,
                 dataset: str,
                 dim_model: int,
                 dim_time: int,
                 noise_schedule: str = 'cosine',
                 objective: str = 'pred_x0',
                 loss_type: str = 'l2',
                 add_lstm: bool = False,
                 freeze_bert: bool = False):
        super().__init__()

        self.device = torch.device(device)

        # entity classes
        self.num_classes = num_classes
        self.add_lstm = add_lstm

        # backbone: pretrained BERT or RoBERTA, name or path
        self.backbone = AutoModel.from_pretrained(backbone)

        # build diffusion
        # 1000 or 2000
        self.timesteps = time_steps
        self.time_difference = 0
        sampling_timesteps = sampling_steps
        self.objective = objective
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')
        # betas = cosine_beta_schedule(timesteps)
        # alphas = 1. - betas
        # alphas_cumprod = torch.cumprod(alphas, dim=0)
        # alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        # timesteps, = betas.shape
        # self.num_timesteps = int(timesteps)
        #
        # self.sampling_timesteps = default(sampling_timesteps, timesteps)
        # assert self.sampling_timesteps <= timesteps, \
        #     'sampling steps must be smaller than time steps for ddim and equal to for ddpm'
        # self.is_ddim_sampling = self.sampling_timesteps < timesteps
        # 1.
        self.ddim_sampling_eta = ddim_sampling_eta
        # whether or not use self condition, default to False
        self.self_condition = self_condition
        # 1.0 for image generation, 0.1 for semantic segmentation, 2.0 for object detection
        self.scale = snr_scale

        self.label_set = LabelSet1D(dataset)
        self.bits = torch.ceil(torch.log2(torch.tensor(len(self.label_set._labelset)))).long()
        self.loss_type = loss_type
        # self.predictor = DiffusionPredictor(dim_model, 16, self.bits.item())
        self.predictor = DiT(in_channels=self.bits.item(),
                             hidden_size=dim_model,
                             depth=6,
                             num_heads=8,
                             mlp_ratio=4.0,
                             max_length=512)
        # self.register_buffer('betas', betas)
        # self.register_buffer('alphas_cumprod', alphas_cumprod)
        # self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        #
        # # calculations for diffusion q(x_t | x_{t-1}) and others
        #
        # self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        # self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        # self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        # self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        #
        # # calculations for posterior q(x_{t-1} | x_t, x_0)
        #
        # posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        #
        # # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        #
        # self.register_buffer('posterior_variance', posterior_variance)
        #
        # # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        #
        # self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        # self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        # self.register_buffer('posterior_mean_coef2',
        #                      (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

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
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-self.scale, max=self.scale) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    # @torch.no_grad()
    # def ddim_sample(self, backbone_feats):
    #     batch = backbone_feats.shape[0]
    #     shape = (*backbone_feats.shape[:2], self.bits.item())
    #     total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
    #
    #     # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    #     times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
    #     times = list(reversed(times.int().tolist()))
    #     time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
    #
    #     x = torch.randn(shape, device=self.device)
    #
    #     x_start = None
    #     for time, time_next in time_pairs:
    #         time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
    #         self_cond = x_start if self.self_condition else None
    #         x_start = self.predictor(x, time_cond, backbone_feats).clamp_(-self.scale, self.scale)
    #         pred_noise = self.predict_noise_from_start(x, time_cond, x_start)
    #
    #         if time_next < 0:
    #             x = x_start
    #             continue
    #         alpha = self.alphas_cumprod[time]
    #         alpha_next = self.alphas_cumprod[time_next]
    #
    #         sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
    #         c = (1 - alpha_next - sigma ** 2).sqrt()
    #
    #         noise = torch.randn_like(x, device=self.device)
    #         x = x_start * alpha_next.sqrt() + \
    #             c * pred_noise + \
    #             sigma * noise
    #     x_decimal = bits_to_decimal(x, self.bits)
    #     # [bsz, len]
    #     return x_decimal

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.timesteps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    @torch.no_grad()
    def ddim_sample(self, features, attention_mask, shape, time_difference=0):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device=device)

        ner_labels = torch.randn(shape, device=device)

        x_start = None

        for times, times_next in time_pairs:
            # get times and noise levels

            log_snr = self.log_snr(times)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr, padded_log_snr_next = map(partial(right_pad_dims_to, ner_labels), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            # add the time delay

            times_next = (times_next - time_difference).clamp(min=0.)

            # predict x0

            x_start = self.predictor(ner_labels, log_snr, features, attention_mask)

            # clip x0

            x_start.clamp_(-self.scale, self.scale)

            # get predicted noise

            pred_noise = (ner_labels - alpha * x_start) / sigma.clamp(min=1e-8)

            # calculate x next

            ner_labels = x_start * alpha_next + pred_noise * sigma_next

        return bits_to_decimal(ner_labels, self.bits.item())
        # return bits_to_decimal(x_start, self.bits.item())


    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start, device=self.device)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, seq_labels: torch.tensor,
                words2pieces: torch.tensor = None):
        """

        Args:
            input_ids: [bsz, len]
            attention_mask: [bsz, len]
            words2pieces: [bsz, word_len, piece_len]
            seq_labels: [bsz, len]

        Returns:

        """
        # feature extraction: [bsz, len_piece, d_model]
        bsz = input_ids.shape[0]
        bert_output = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

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
            results = self.ddim_sample(features, attention_mask, shape)
            return results

        if self.training:
            times = torch.zeros((bsz,), device=self.device).float().uniform_(0, 1.)
            # categorical data to bits [bsz, ]
            bits_seq_labels = decimal_to_bits(seq_labels, self.bits)
            bits_seq_labels *= self.scale
            noise = torch.randn_like(bits_seq_labels)

            noise_level = self.log_snr(times)
            padded_noise_level = right_pad_dims_to(bits_seq_labels, noise_level)
            alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)

            noised_seqs = alpha * bits_seq_labels + sigma * noise

            # self_cond = None
            # if random() < 0.5:
            #     with torch.no_grad();
            #         self_cond = self.predictor(noised_seqs, noise_level)

            pred = self.predictor(noised_seqs, noise_level, features, attention_mask)

            targets = bits_seq_labels
            targets_mask = (seq_labels != -100).unsqueeze(dim=-1).expand(-1, -1, self.bits)
            if self.objective == 'pred_noise':
                targets = noise
            if self.loss_type == 'l2':
                loss = F.mse_loss(pred[targets_mask], targets[targets_mask])
            elif self.loss_type == 'l1':
                loss = F.l1_loss(pred[targets_mask], targets[targets_mask])
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
        ts = torch.randint(0, self.num_timesteps, [bsz], device=self.device).long()
        noise = torch.randn_like(gold_seq_labels, device=self.device)
        diffused_labels = self.q_sample(x_start=gold_seq_labels, t=ts, noise=noise)

        return diffused_labels, ts, noise
