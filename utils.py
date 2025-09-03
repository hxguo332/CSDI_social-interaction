import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
    batch_sampler_train=None,
    batch_sampler_valid=None,
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"
    else:
        output_path = None

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        if batch_sampler_train is not None:
            batch_sampler_train.set_epoch(epoch_no)
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model(train_batch)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ Skipping batch {batch_no} due to NaN/Inf loss.")
                    continue

                loss.backward()
                avg_loss += loss.item()
                optimizer.step()

                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 1000000
            loss_sum = 0
            if batch_sampler_valid is not None:
                batch_sampler_valid.set_epoch(epoch_no)
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        loss_sum += loss.item()
                        avg_loss_valid = loss_sum / batch_no
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid,
                    "at",
                    epoch_no,
                )

    if output_path is not None:
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    ).item()


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS / len(quantiles)

def process_batch_data(output, return_scen_map):
    """
    Process and permute batch data for evaluation.

    Args:
        output: Output from the model's evaluation method.
        return_scen_map: Whether scenario mapping is included in the output.

    Returns:
        Processed tensors for samples, target, eval_points, observed_points, and observed_time.
    """
    scen_map = None
    scen_map_scale = None
    if return_scen_map:
        samples, c_target, eval_points, observed_points, observed_time, scen_map, scen_map_scale = output
    else:
        samples, c_target, eval_points, observed_points, observed_time = output

    # Permute dimensions for further processing
    samples = samples.permute(0, 1, 3, 2)  # (B, nsample, L, K)
    c_target = c_target.permute(0, 2, 1)  # (B, L, K)
    eval_points = eval_points.permute(0, 2, 1)
    observed_points = observed_points.permute(0, 2, 1)

    return samples, c_target, eval_points, observed_points, observed_time, scen_map, scen_map_scale


def compute_batch_metrics(samples_median, c_target, eval_points, scaler):
    """
    Compute MSE and MAE for a batch.

    Args:
        samples_median: Median of generated samples.
        c_target: Ground truth target values.
        eval_points: Evaluation points mask.
        scaler: Scaling factor for the target values.

    Returns:
        mse_current: Mean Squared Error for the batch.
        mae_current: Mean Absolute Error for the batch.
    """
    mse_current = (
        ((samples_median - c_target) * eval_points) ** 2
    ) * (scaler ** 2)
    mae_current = (
        torch.abs((samples_median - c_target) * eval_points)
    ) * scaler

    return mse_current.sum().item(), mae_current.sum().item()


def save_generated_outputs(foldername, nsample, all_target, all_evalpoint, all_observed_point, all_observed_time, all_generated_samples, scaler, mean_scaler, mode=None):
    """
    Save generated outputs to a pickle file.

    Args:
        foldername: Directory to save the file.
        nsample: Number of samples generated.
        all_target, all_evalpoint, all_observed_point, all_observed_time, all_generated_samples: Evaluation data.
        scaler: Scaling factor for the target values.
        mean_scaler: Mean scaler for the target values.
    """
    if mode is not None:
        file_path = foldername + f"/generated_outputs_nsample{nsample}_{mode}.pk"
    else:
        file_path = foldername + f"/generated_outputs_nsample{nsample}.pk"
    with open(file_path, "wb") as f:
        pickle.dump(
            [
                all_generated_samples,
                all_target,
                all_evalpoint,
                all_observed_point,
                all_observed_time,
                scaler,
                mean_scaler,
            ],
            f,
        )


def compute_crps_metrics(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler):
    """
    Compute CRPS and CRPS_sum metrics.

    Args:
        all_target: Ground truth target values.
        all_generated_samples: Generated samples from the model.
        all_evalpoint: Evaluation points mask.
        mean_scaler: Mean scaler for the target values.
        scaler: Scaling factor for the target values.

    Returns:
        CRPS: Continuous Ranked Probability Score.
        CRPS_sum: Summed CRPS metric.
    """
    CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)
    CRPS_sum = calc_quantile_CRPS_sum(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)
    return CRPS, CRPS_sum

class CollisionEvaluator:
    """
    A class to compute and store collision metrics (collision rate and invalid rate)
    across multiple batches, and calculate the average metrics at the end.

    Attributes:
        total_collisions (int): Total number of collisions across all batches.
        total_paths (int): Total number of paths across all batches.
        invalid_rate_all (list): List of invalid rates for each batch.
    """

    def __init__(self, scenmap_scale=10):
        self.total_collisions = 0
        self.total_paths = 0

        self.invalid_rate_all = []

        self.scenmap_scale = scenmap_scale

    def update(self, samples_batch, scen_maps_batch, eval_points, scenmap_scale):
        """
        Update the collision metrics with a new batch of data.

        Args:
            samples_batch (torch.Tensor): Predicted paths. Shape: (batch_size, sample_length, 2).
            scen_maps_batch (torch.Tensor): Scenario map with obstacles and forbidden areas.
                - Channel 1 (Red): Obstacles (rectangles, circles) & optionally standing persons.
                - Channel 2 (Green): Walls & forbidden areas.
                - Channel 3 (Blue): Entrances.
            eval_points (torch.Tensor): Evaluation points mask. Shape: (batch_size, sample_length, 2).
            scenmap_scale (int or torch.Tensor): Scale factor for the scenario map. Shape: (batch_size, 1) if it is torch.Tensor.
        """
        # Generate a mask for collisions
        collision_mask = (scen_maps_batch[:, 0] > 0) | (scen_maps_batch[:, 1] > 0)

        batch_size = samples_batch.shape[0]
        sample_length = samples_batch.shape[1]

        # Conver scenmap_scale to a long tensor
        if not isinstance(scenmap_scale, torch.Tensor):
            scenmap_scale = torch.full((batch_size, 1), scenmap_scale, device=samples_batch.device, dtype=samples_batch.dtype)
        else:
            scenmap_scale = scenmap_scale.view(batch_size, 1)

        # Get x and y coordinates, multiply eval_points to avoid out-of-bounds
        x = ((samples_batch[..., 0] * eval_points[..., 0]) * scenmap_scale).long()  # Shape: (batch_size, sample_length)
        # Do we need to reverse the y-axis?
        # y = (samples_batch[..., 1] * eval_points[..., 1]).long() * scenmap_scale
        y = ((scen_maps_batch.shape[2] - samples_batch[..., 1] * scenmap_scale) * eval_points[..., 1]).long()  # Shape: (batch_size, sample_length)

        # Deal with out-of-bounds indices
        # If the coordinates are out of bounds, they are collisons
        collision_position = torch.logical_or(torch.logical_or(x<0, x>=scen_maps_batch.shape[3]), torch.logical_or(y<0, y>=scen_maps_batch.shape[2]))

        # Ensure x and y are within the bounds of the scenario map
        x = torch.clamp(x, 0, scen_maps_batch.shape[3] - 1)
        y = torch.clamp(y, 0, scen_maps_batch.shape[2] - 1)

        # Generate batch indices
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, sample_length)  

        # Check how many samples collide with obstacles under eval_points
        collision = collision_mask[batch_indices, y, x] * eval_points[..., 0]  # Shape: (batch_size, sample_length)
        collision = collision.bool() | collision_position  # Combine with collision_position

        # Calculate collision rate for the batch, calculated by each sample
        batch_collisions = torch.any(collision, dim=1).sum().item()  # Number of paths with at least one collision

        # Calculate invalid rate for the batch
        batch_invalid_steps = collision.sum(dim=-1)  # Total number of invalid steps
        batch_total_steps = eval_points[...,0].sum(dim=-1)  # Total number of steps under eval_points

        batch_invalid_rate = batch_invalid_steps / batch_total_steps  # Invalid rate for each path
        # Handle NaN values
        batch_invalid_rate[batch_total_steps == 0] = 0.0  # Set invalid rate to 0 if total steps is 0
        batch_invalid_rate = batch_invalid_rate.tolist()  # Convert to list for easier handling

        # Update the totals
        self.invalid_rate_all.extend(batch_invalid_rate)
        self.total_collisions += batch_collisions
        self.total_paths += batch_size

    def compute_metrics(self):
        """
        Compute the average collision rate and invalid rate across all batches.

        Returns:
            collision_rate (float): Average collision rate across all batches.
            invalid_rate (float): Average invalid rate across all batches.
        """
        # Calculate the average collision rate across all batches
        collision_rate = self.total_collisions / self.total_paths if self.total_paths > 0 else 0.0

        # Calculate the average invalid rate across all batches
        invalid_rate = np.mean(self.invalid_rate_all) if len(self.invalid_rate_all) > 0 else 0.0
        return collision_rate, invalid_rate


def save_evaluation_metrics(foldername, nsample, metrics_dict, mode=None):
    """
    Save evaluation metrics to a CSV file.

    Args:
        foldername: Directory to save the file.
        nsample: Number of samples generated.
        metrics_dict: Dictionary containing evaluation metrics (RMSE, MAE, CRPS, etc.).
        mode: Sampling mode, either "normalized" or "unnormalized" (optional).
    """
    if mode is not None:
        result_path = foldername + f"/result_nsample{nsample}_{mode}.csv"
    else:
        result_path = foldername + f"/result_nsample{nsample}.csv"
    with open(result_path, "w") as f:
        f.write("Metric,Value\n")
        for key, value in metrics_dict.items():
            f.write(f"{key},{value}\n")


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", return_scen_map=False, mode="unnormalized"):
    """
    Evaluate the model on the test dataset and compute various metrics.

    Args:
        model: The trained model to evaluate.
        test_loader: DataLoader for the test dataset.
        nsample: Number of samples to generate for evaluation.
        scaler: Scaling factor for the target values.
        mean_scaler: Mean scaler for the target values.
        foldername: Directory to save evaluation results and generated outputs.
        return_scen_map: Whether scenario map is returned in each batch in dataloader (optional).
        mode: Sampling mode, either "normalized" or "unnormalized".

    Returns:
        None. Saves evaluation metrics and generated outputs to files.
    """
    with torch.no_grad():
        model.eval()
        mse_total, mae_total, evalpoints_total = 0, 0, 0

        collision_evaluator = None
        if return_scen_map:
            collision_evaluator = CollisionEvaluator()

        all_target, all_observed_point, all_observed_time = [], [], []
        all_evalpoint, all_generated_samples = [], []

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):

                if return_scen_map:
                    # Evaluate the model on the current batch
                    if mode is not None:
                        output = model.evaluate(test_batch, nsample, return_scenmap=True, mode=mode)
                    else:
                        output = model.evaluate(test_batch, nsample, return_scenmap=True)
                    # Process batch data
                    samples, c_target, eval_points, observed_points, observed_time, scen_map, scen_maps_scale = process_batch_data(output, return_scen_map)
                else:
                    # Evaluate the model on the current batch
                    if mode is not None:
                        output = model.evaluate(test_batch, nsample, mode=mode)
                    else:
                        output = model.evaluate(test_batch, nsample)
                    # Process batch data
                    samples, c_target, eval_points, observed_points, observed_time, scen_map, scen_maps_scale = process_batch_data(output, return_scen_map)
                # Compute the median of the generated samples
                samples_median = samples.median(dim=1).values
                
                if collision_evaluator is not None:
                    collision_evaluator.update(samples_median, scen_map, eval_points, scen_maps_scale)

                # Append results to the respective lists
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                # Compute MSE and MAE for the current batch
                mse_current, mae_current = compute_batch_metrics(samples_median, c_target, eval_points, scaler)

                # Accumulate MSE, MAE, and evaluation points
                mse_total += mse_current
                mae_total += mae_current
                evalpoints_total += eval_points.sum().item()

                # Update progress bar
                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

        # Save generated outputs
        save_generated_outputs(foldername, nsample, torch.cat(all_target, dim=0), torch.cat(all_evalpoint, dim=0),
                               torch.cat(all_observed_point, dim=0), torch.cat(all_observed_time, dim=0),
                               torch.cat(all_generated_samples, dim=0), scaler, mean_scaler, mode)

        # Compute CRPS metrics
        CRPS, CRPS_sum = compute_crps_metrics(torch.cat(all_target, dim=0), torch.cat(all_generated_samples, dim=0),
                                              torch.cat(all_evalpoint, dim=0), mean_scaler, scaler)
        # Compute collision metrics if applicable


        # Save evaluation metrics
        RMSE = np.sqrt(mse_total / evalpoints_total)
        MAE = mae_total / evalpoints_total
        if collision_evaluator is not None:
            collision_rate, invalid_rate = collision_evaluator.compute_metrics()
            print("Collision Rate:", collision_rate)
            print("Invalid Rate:", invalid_rate)
            metrics_dict = {
                "RMSE": RMSE,
                "MAE": MAE,
                "CRPS": CRPS,
                "CRPS_sum": CRPS_sum,
                "Collision Rate": collision_rate,
                "Invalid Rate": invalid_rate,
            }
        else:
            metrics_dict = {
                "RMSE": RMSE,
                "MAE": MAE,
                "CRPS": CRPS,
                "CRPS_sum": CRPS_sum,
            }
        save_evaluation_metrics(foldername, nsample, metrics_dict, mode)

        # Print evaluation metrics
        print("RMSE:", RMSE)
        print("MAE:", MAE)
        print("CRPS:", CRPS)
        print("CRPS_sum:", CRPS_sum)
