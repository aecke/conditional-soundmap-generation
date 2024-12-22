from torchvision import utils
import torch
import time
import sys
import math
import data_handler
from data_handler import transient
import helper
from .loss import *
from trainer.training_logger import TrainingLogger
import models


def init_train_configs(args):
    train_configs = {"reg_factor": args.reg_factor}  # lambda
    print(f"In [init_train_configs]: \ntrain_configs: {train_configs}\n")
    return train_configs


def adjust_lr(current_lr, initial_lr, step, epoch_steps):
    curr_epoch = math.ceil(
        step / epoch_steps
    )  # epoch_steps is the number of steps to complete an epoch
    threshold = 50  # linearly decay after threshold
    if curr_epoch > threshold:
        extra_epochs = curr_epoch - threshold
        decay = initial_lr * (extra_epochs / threshold)
        current_lr = initial_lr - decay

    print(f"In [adjust_lr]: step: {step}, curr_epoch: {curr_epoch}, lr: {current_lr}")
    return current_lr


def train(
    args,
    params,
    train_configs,
    model,
    optimizer,
    current_lr,
    comet_tracker=None,
    resume=False,
    last_optim_step=0,
    reverse_cond=None,
):
    # getting data loaders
    train_loader, val_loader = data_handler.init_data_loaders(args, params)

    # Initialize training logger
    training_logger = TrainingLogger(args, params)
    
    # Log numerical conditions configuration
    if any([args.use_temperature, args.use_humidity, args.use_db]):
        numerical_conditions = []
        if args.use_temperature:
            numerical_conditions.append('temperature')
        if args.use_humidity:
            numerical_conditions.append('humidity') 
        if args.use_db:
            numerical_conditions.append('db')
        print(f"Training with numerical conditions: {', '.join(numerical_conditions)}")

    # adjusting optim step
    optim_step = last_optim_step + 1 if resume else 1
    max_optim_steps = params["iter"]
    paths = helper.compute_paths(args, params)

    if resume:
        print(
            f"In [train]: resuming training from optim_step={optim_step} - max_step: {max_optim_steps}"
        )

    # optimization loop
    while optim_step < max_optim_steps:
        # after each epoch, adjust learning rate accordingly
        current_lr = adjust_lr(
            current_lr,
            initial_lr=params["lr"],
            step=optim_step,
            epoch_steps=len(train_loader),
        )  # now only supports with batch size 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print(f"In [train]: optimizer learning rate adjusted to: {current_lr}\n")

        for i_batch, batch in enumerate(train_loader):
            if optim_step > max_optim_steps:
                print(f"In [train]: reaching max_step or lr is zero. Terminating...")
                training_logger.write_summary()
                return  # ============ terminate training if max steps reached

            begin_time = time.time()
            # forward pass
            left_batch, right_batch, numerical_conditions = data_handler.extract_batches(
                batch, args
            )
            forward_output = forward_and_loss(
                args, params, model, left_batch, right_batch, numerical_conditions
            )

            # regularize left loss
            if train_configs["reg_factor"] is not None:
                loss = (
                    train_configs["reg_factor"] * forward_output["loss_left"]
                    + forward_output["loss_right"]
                )  # regularized
            else:
                loss = forward_output["loss"]

            metrics = {"loss": loss.item(), "learning_rate": current_lr}
            # also add left and right loss if available
            if "loss_left" in forward_output.keys():
                metrics.update(
                    {
                        "loss_right": forward_output["loss_right"].item(),
                        "loss_left": forward_output["loss_left"].item(),
                    }
                )

            # backward pass and optimizer step
            model.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"In [train]: Step: {optim_step} => loss: {loss.item():.3f}")

            # validation loss
            if (
                params["monitor_val"] and optim_step % params["val_freq"] == 0
            ) or current_lr == 0:
                val_loss_mean, _ = calc_val_loss(args, params, model, val_loader)
                metrics["val_loss"] = val_loss_mean
                print(f"====== In [train]: val_loss mean: {round(val_loss_mean, 3)}")
                
                # Log welche numerischen Bedingungen verwendet wurden
                if any([args.use_temperature, args.use_humidity, args.use_db]):
                    print("Validation includes numerical conditions:", 
                          "temperature" if args.use_temperature else "",
                          "humidity" if args.use_humidity else "",
                          "db" if args.use_db else "")

            # Log training metrics
            end_time = time.time()
            iteration_time = end_time - begin_time
            training_logger.log_iteration(optim_step, metrics, iteration_time)

            # tracking metrics in comet if enabled
            if args.use_comet:
                for key, value in metrics.items():  # track all metric values
                    comet_tracker.track_metric(key, round(value, 3), optim_step)

            # saving samples
            if (optim_step % params["sample_freq"] == 0) or current_lr == 0:
                samples_path = paths["samples_path"]
                helper.make_dir_if_not_exists(samples_path)
                sampled_images = models.take_samples(args, params, model, reverse_cond)
                
                # Dateiname mit numerischen Bedingungen
                filename = f"{str(optim_step).zfill(6)}"
                if any([args.use_temperature, args.use_humidity, args.use_db]):
                    conds = []
                    if args.use_temperature:
                        conds.append("temp")
                    if args.use_humidity:
                        conds.append("hum")
                    if args.use_db:
                        conds.append("db")
                    filename += f"_{'_'.join(conds)}"
                filename += ".png"
                
                utils.save_image(
                    sampled_images,
                    f"{samples_path}/{filename}",
                    nrow=10,
                )
                print(
                    f'\nIn [train]: Sample saved at iteration {optim_step} to: \n"{samples_path}"\n'
                )

            # saving checkpoint
            if (
                optim_step > 0 and optim_step % params["checkpoint_freq"] == 0
            ) or current_lr == 0:
                checkpoints_path = paths["checkpoints_path"]
                helper.make_dir_if_not_exists(checkpoints_path)
                
                # Speichere auch die numerischen Bedingungen
                additional_info = {
                    'numerical_conditions': {
                        'temperature': args.use_temperature,
                        'humidity': args.use_humidity,
                        'db': args.use_db
                    }
                }
                
                helper.save_checkpoint(
                    checkpoints_path, optim_step, model, optimizer, loss, current_lr,
                    additional_info=additional_info
                )
                print("In [train]: Checkpoint saved at iteration", optim_step, "\n")

            optim_step += 1
            print(f"Iteration took: {round(iteration_time, 2)}")
            helper.show_memory_usage()
            print("\n")

            if current_lr == 0:
                print("In [train]: current_lr = 0, terminating the training...")
                training_logger.write_summary()
                sys.exit(0)

    # Write final summary at end of training
    training_logger.write_summary()