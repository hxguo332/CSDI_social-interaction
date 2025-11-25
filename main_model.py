import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_models import diff_CSDI
from scenario_map_embedding import ResnetMapEncoder
from collision_loss import compute_collision_loss


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.add_collision_loss = config["model"].get("add_collision_loss", False)

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask

        # feature embedding
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask


    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, sdf=None
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t, sdf=sdf
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1, sdf=None
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        #predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)
        try:
            predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)
        except Exception as e:
            print("⚠️ Exception during self.diffmodel call:", e)
            print("DEBUG: diffusion_step t =", t)
            print("  t dtype:", t.dtype, "min:", t.min().item(), "max:", t.max().item())
            print("  t NaN:", torch.isnan(t).any().item(), "Inf:", torch.isinf(t).any().item())

            print("DEBUG: total_input stats")
            print("  min:", total_input.min().item(), "max:", total_input.max().item(), "std:", total_input.std().item())
            print("  NaN:", torch.isnan(total_input).any().item(), "Inf:", torch.isinf(total_input).any().item())

            print("DEBUG: side_info stats")
            print("  min:", side_info.min().item(), "max:", side_info.max().item(), "std:", side_info.std().item())
            print("  NaN:", torch.isnan(side_info).any().item(), "Inf:", torch.isinf(side_info).any().item())
            raise  # re-raise the exception

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        # Insert runtime checks for NaN, Inf, and zero-sized evaluation mask:
        if torch.isnan(predicted).any() or torch.isinf(predicted).any():
            raise ValueError("predicted contains NaN or Inf")
        if torch.isnan(residual).any() or torch.isinf(residual).any():
            raise ValueError("residual contains NaN or Inf")
        if num_eval == 0:
            raise ValueError("No valid target points to evaluate (num_eval=0). Check target_mask logic.")
        noise_loss = (residual ** 2).sum() / num_eval

        if self.add_collision_loss:
            # 1) Reconstruct the denoised trajectory x0_hat
            x0_hat = self.estimate_x0_from_xt(noisy_data, predicted, current_alpha) * target_mask # (B,K,L)
            #print("target_mask shape:", target_mask.shape)
            #print("x0_hat shape:", x0_hat.shape)
            #print("x0_hat stats - min:", x0_hat.min().item(), "max:", x0_hat.max().item(), "std:", x0_hat.std().item())
            #print("predicted stats - min:", predicted.min().item(), "max:", predicted.max().item(), "std:", predicted.std().item())
            #print("current_alpha stats - min:", current_alpha.min().item(), "max:", current_alpha.max().item(), "std:", current_alpha.std().item())
            #print("noisy_data stats - min:", noisy_data.min().item(), "max:", noisy_data.max().item(), "std:", noisy_data.std().item())
            idx_x = 0  # assuming x coordinate is at index 0
            idx_y = 1  # assuming y coordinate is at index 1

            # 2) Extract the (x, y) coordinates only
            xy = torch.stack([x0_hat[:, idx_x, :], x0_hat[:, idx_y, :]], dim=-1)  # [B,L,2]

            # 3) Create a mask for missing (imputed) time steps
            ta_time_mask = (target_mask.max(dim=1).values > 0).float()  # [B,L]

            # 4) Compute the collision loss
            collision_loss = compute_collision_loss(
                xy, ta_time_mask, self.gen_raster_sdf_fn(sdf),
                w_obs=1, w_clear=0, margin=0, reduction="mean"
            )

            collision_loss_weight = 10.0  # You can adjust this weight as needed
            # Combine with your main noise-prediction loss
            loss = noise_loss + collision_loss["loss"] * collision_loss_weight
        else:
            loss = noise_loss
        return loss

    def calc_loss_debug(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1, scenario_map_raw=None, sdf=None
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        #predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)
        try:
            predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)
        except Exception as e:
            print("⚠️ Exception during self.diffmodel call:", e)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        # Insert runtime checks for NaN, Inf, and zero-sized evaluation mask:
        if torch.isnan(predicted).any() or torch.isinf(predicted).any():
            raise ValueError("predicted contains NaN or Inf")
        if torch.isnan(residual).any() or torch.isinf(residual).any():
            raise ValueError("residual contains NaN or Inf")
        if num_eval == 0:
            raise ValueError("No valid target points to evaluate (num_eval=0). Check target_mask logic.")
        noise_loss = (residual ** 2).sum() / num_eval

        if self.add_collision_loss:
            print(f"t is {t}")
            # 1) Reconstruct the denoised trajectory x0_hat
            x0_hat = self.estimate_x0_from_xt(noisy_data, predicted, current_alpha) * target_mask # (B,K,L)
            print("target_mask shape:", target_mask.shape)
            print("x0_hat shape:", x0_hat.shape)
            print("x0_hat stats - min:", x0_hat.min().item(), "max:", x0_hat.max().item(), "std:", x0_hat.std().item())
            print("predicted stats - min:", predicted.min().item(), "max:", predicted.max().item(), "std:", predicted.std().item())
            print("current_alpha stats - min:", current_alpha.min().item(), "max:", current_alpha.max().item(), "std:", current_alpha.std().item())
            print("noisy_data stats - min:", noisy_data.min().item(), "max:", noisy_data.max().item(), "std:", noisy_data.std().item())
            idx_x = 0  # assuming x coordinate is at index 0
            idx_y = 1  # assuming y coordinate is at index 1

            # 2) Extract the (x, y) coordinates only
            xy = torch.stack([x0_hat[:, idx_x, :], x0_hat[:, idx_y, :]], dim=-1)  # [B,L,2]

            # 3) Create a mask for missing (imputed) time steps
            ta_time_mask = (target_mask.max(dim=1).values > 0).float()  # [B,L]

            # 4) Compute the collision loss
            collision_loss = compute_collision_loss(
                xy, ta_time_mask, self.gen_raster_sdf_fn(sdf),
                w_obs=1, w_clear=0.0, margin=0, reduction="mean"
            )
            print("Collision loss components:", collision_loss)
            print(f"Noise loss: {noise_loss.item()}, Collision loss: {collision_loss['loss'].item()}")
            return xy, ta_time_mask, target_mask

            # Combine with your main noise-prediction loss
            loss = noise_loss + collision_loss["loss"]
        else:
            loss = noise_loss
        return loss

    def gen_raster_sdf_fn(
            self, 
            sdf, 
            align_corners: bool = True,
            padding_mode: str = "border",):
        """
        Parameters
            sdf: (B, W, H), signed distance function
        returns: sdf_fn(points) function that computes SDF values at given points"""
        B, H, W = sdf.shape
        sdf = sdf.unsqueeze(1)  # (B, 1, H, W)
        #print("SDF stats - min:", sdf.min().item(), "max:", sdf.max().item(), "std:", sdf.std().item())
        def sdf_fn(points):
            """
            Sample the precomputed SDF at given (normalized) points.

            Parameters
            ----------
            points : torch.Tensor, shape (B, L, 2)
                Normalized coordinates in [0, 1] x [0, 1], (x, y).
                If your points are already in pixel coordinates, convert them to normalized coords before calling.

            Returns
            -------
            torch.Tensor, shape (B, L)
                Signed distances (positive free, negative obstacle), bilinearly sampled from the raster SDF.
            """
            assert points.dim() == 3 and points.shape[-1] == 2, "points should be (B, L, 2)"
            Bp, Lp, _ = points.shape
            assert Bp == B, f"Batch size mismatch: points B={Bp}, map B={B}"

            # Do NOT modify the input tensor in-place (avoid breaking autograd).
            # Convert [0,1] -> [-1,1] for grid_sample (x → u, y → v).
            #print("Points stats before normalization - min:", points.min().item(), "max:", points.max().item(), "std:", points.std().item())
            u = 2.0 * points[..., 0] - 1.0  # (B, L)
            v = 2.0 * points[..., 1] - 1.0  # (B, L)
            #print("u stats - min:", u.min().item(), "max:", u.max().item(), "std:", u.std().item())
            #print("v stats - min:", v.min().item(), "max:", v.max().item(), "std:", v.std().item())

            # Build a sampling grid of shape (B, L, 1, 2)
            grid = torch.stack([u, v], dim=-1).view(B, Lp, 1, 2)
            grid = grid.to(device=sdf.device, dtype=sdf.dtype)

            # Bilinear sample. We expand batch-wise SDFs to match B if needed (already B here).
            # align_corners must match the normalization above
            d = F.grid_sample(
                sdf, grid,
                mode="bilinear",
                padding_mode=padding_mode,
                align_corners=align_corners,
            )  # -> (B, 1, L, 1)
            d = d.view(B, Lp)  # (B, L)
            return d

        return sdf_fn

    def sdf_fn(self, points):
        """
        points: (B, L, 2) tensor representing (x, y) coordinates
        returns: (B, L) tensor representing SDF values at the given points
        """
        # Placeholder implementation; replace with actual SDF computation
        B, L, _ = points.shape
        sdf_values = torch.ones(B, L).to(points.device)  # Dummy values
        return sdf_values

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp

    def process_data(self, batch):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def estimate_x0_from_xt(self, x_t, eps_pred, alpha_t_scalar_or_tensor):
        """
        x_t, eps_pred: (B, K, L)
        alpha_t_scalar_or_tensor: 标量 or (B,1,1), that represent alpha_t
        return: x0_hat (B, K, L)
        """
        if not torch.is_tensor(alpha_t_scalar_or_tensor):
            alpha_t = torch.tensor(alpha_t_scalar_or_tensor, device=x_t.device, dtype=x_t.dtype)
            alpha_t = alpha_t.view(1,1,1)
        else:
            alpha_t = alpha_t_scalar_or_tensor
        return (x_t - (1.0 - alpha_t).sqrt() * eps_pred) / alpha_t.sqrt()


class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )

class CSDI_Simulation(CSDI_base):
    def __init__(self, config, device, target_dim=2):
        super(CSDI_Simulation, self).__init__(target_dim, config, device)

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy == "two_ends":
            cond_mask = gt_mask
        elif self.target_strategy == "mix_two_ends":
            if np.random.rand() > 0.8:
                cond_mask = gt_mask
            else:
                cond_mask = self.get_randmask(observed_mask)
        elif self.target_strategy == "hist" or self.target_strategy == "mix":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        elif self.target_strategy == "random":
            cond_mask = self.get_randmask(observed_mask)
        else:
            raise NotImplementedError(f"Target strategy {self.target_strategy} not implemented.")

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )

class CSDI_Forecasting(CSDI_base):
    def __init__(self, config, device, target_dim):
        super(CSDI_Forecasting, self).__init__(target_dim, config, device)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        feature_id=torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0],-1).to(self.device)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            feature_id, 
        )        

    def sample_features(self,observed_data, observed_mask,feature_id,gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []
        
        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k,ind[:size]])
            extracted_mask.append(observed_mask[k,ind[:size]])
            extracted_feature_id.append(feature_id[k,ind[:size]])
            extracted_gt_mask.append(gt_mask[k,ind[:size]])
        extracted_data = torch.stack(extracted_data,0)
        extracted_mask = torch.stack(extracted_mask,0)
        extracted_feature_id = torch.stack(extracted_feature_id,0)
        extracted_gt_mask = torch.stack(extracted_gt_mask,0)
        return extracted_data, extracted_mask,extracted_feature_id, extracted_gt_mask


    def get_side_info(self, observed_tp, cond_mask,feature_id=None):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)

        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim).to(self.device)
            )  # (K,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1,L,-1,-1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch)
        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask,feature_id,gt_mask = \
                    self.sample_features(observed_data, observed_mask,feature_id,gt_mask)
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        if is_train == 0:
            cond_mask = gt_mask
        else: #test pattern
            cond_mask = self.get_test_pattern_mask(
                observed_mask, gt_mask
            )

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)



    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask * (1-gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp

class CSDI_SimulationScenmap(CSDI_base):
    def __init__(self, config, device, target_dim=2, scale_embed_dim=4):
        super(CSDI_base, self).__init__()
        self.device = device
        self.target_dim = target_dim
        self.scale_embed_dim = scale_embed_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        # init the scenario map embedding layer
        self.emb_scenmap_dim = config["model"]["scenmapemb"]
        self.add_collision_loss = config["model"].get("add_collision_loss", False)

        # use MLP to embed the scale
        self.scale_dim = 2 # scale of the scenario map, x direction and y direction
        self.scale_mlp = nn.Sequential(
            nn.Linear(self.scale_dim, 16),
            nn.ReLU(),
            nn.Linear(16, self.scale_embed_dim),
        )

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + self.emb_scenmap_dim + self.scale_embed_dim # we can add the shape later
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask

        # use CNN to embed the scenario map
        me_cfg = (config.get("model", {}).get("map_encoder", {}) if isinstance(config.get("model", {}), dict) else {})
        grid_size = me_cfg.get("grid_size", 7)
        finetune_from = me_cfg.get("finetune_from", "layer4")
        self.emb_scenmap = ResnetMapEncoder(
            output_dim=self.emb_scenmap_dim,
            grid_size=grid_size,
            finetune_from=finetune_from,
        )

        # feature embedding
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def get_side_info(self, observed_tp, cond_mask, scenmap, scenmap_scales):
        # scenmap_scales shape: (B,2)
        # K: number of features, L: number of time points
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        # get the scenario map embedding
        scenmap_embed = self.emb_scenmap(scenmap) # (B, emb_scenmap_dim)
        # convert the shape from (B, emb_scenmap_dim) to (B, emb_scenmap_dim, K, L)
        scenmap_embed = scenmap_embed.unsqueeze(2).unsqueeze(2).expand(-1, -1, K, L)

        side_info = torch.cat([side_info, scenmap_embed], dim=1)

        # scenmap_scales shape: (B, 2), embed with MLP and expand to (B, scale_embed_dim, K, L)
        scale_embed = self.scale_mlp(scenmap_scales) # (B, scale_embed_dim)
        scale_embed = scale_embed.unsqueeze(-1).unsqueeze(-1).expand(-1, self.scale_embed_dim, K, L)
        # add the scenario map scales embedding to the side info
        side_info = torch.cat([side_info, scale_embed], dim=1)

        return side_info

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
            scenmap,
            scenmap_scales,
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy == "two_ends":
            cond_mask = gt_mask
        elif self.target_strategy == "mix_two_ends":
            if np.random.rand() > 0.8:
                cond_mask = gt_mask
            else:
                cond_mask = self.get_randmask(observed_mask)
        elif self.target_strategy == "hist" or self.target_strategy == "mix":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        elif self.target_strategy == "random":
            cond_mask = self.get_randmask(observed_mask)
        else:
            raise NotImplementedError(f"Target strategy {self.target_strategy} not implemented.")


        side_info = self.get_side_info(observed_tp, cond_mask, scenmap, scenmap_scales)

        if self.add_collision_loss:
            sdf = batch["sdf"].to(self.device).float()  # (B, H, W)
        else:
            sdf = None

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train, sdf=sdf)

    def debug_loss(self, batch):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
            scenmap,
            scenmap_scales,
        ) = self.process_data(batch)

        scenmap_raw = batch["scen_map_raw"]
        #sdf = batch["sdf"] # (B, H, W)
        sdf = batch["sdf"].to(self.device).float()  # (B, H, W)

        #cond_mask = self.get_hist_mask(
        #    observed_mask, for_pattern_mask=for_pattern_mask
        #)
        cond_mask = gt_mask

        side_info = self.get_side_info(observed_tp, cond_mask, scenmap, scenmap_scales)

        loss = self.calc_loss_debug(
            observed_data, cond_mask, observed_mask, side_info, is_train=1, sdf=sdf,
        )
        return loss, observed_data, observed_mask, cond_mask, scenmap_raw

    def evaluate(self, batch, n_samples, mode="normalized"):
        (
            observed_data, # groundtruth data
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            scenmap,
            scenmap_scales,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask, scenmap, scenmap_scales)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0

        if mode == "unnormalized":
        # scenmap: (B, C, H, W), samples: (B, nsample, 2, L), observed_data: (B, 2, L)
            B, _, H, W = scenmap.shape
            scale_wh = torch.tensor([W, H], device=samples.device, dtype=samples.dtype)

            # Rescale samples
            samples = samples * scale_wh.view(1, 1, 2, 1)       # (B, nsample, 2, L)

            # Rescale observed/target too if needed
            observed_data = observed_data * scale_wh.view(1, 2, 1)  # (B, 2, L)

            # Un-normalize generated samples and observed data
            # scenmap_scales: (B, 2)
            samples = samples * scenmap_scales.unsqueeze(1).unsqueeze(3)  # (B, nsample, K, L)
            observed_data = observed_data * scenmap_scales.unsqueeze(2)  # (B, K, L)

        return samples, observed_data, target_mask, observed_mask, observed_tp

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        scenmap = batch["scen_map"].to(self.device).float() # N, H, W, C
        scenmap_scales = batch["scen_map_scale"].to(self.device).float() # N

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        scenmap = scenmap.permute(0, 3, 1, 2) # N, C, H, W

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            scenmap,
            scenmap_scales,
        )
