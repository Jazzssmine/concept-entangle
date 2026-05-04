import argparse
import os
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import (
    setup_forget_horse_data,
    setup_forget_nsfw_data,
    setup_model,
)
from tqdm import tqdm

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f"{word}_loss")
    plt.legend(loc="upper left")
    plt.title("Average loss in trainings", fontsize=20)
    plt.xlabel("Data point", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)
    plt.savefig(path)


def nsfw_removal(
    train_method,
    alpha,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    mask_path,
    diffusers_config_path,
    device,
    image_size=512,
    ddim_steps=50,
    save_diffusers=False,
    concept_mode="nsfw",
    horse_data_path="data/horse",
    not_horse_data_path="data/not-horse",
    concept_name="concept",
    concept_prompt=None,
    remain_prompt=None,
    concept_data_path=None,
    not_concept_data_path=None,
):
    # MODEL TRAINING SETUP
    model = setup_model(config_path, ckpt_path, device)
    criteria = torch.nn.MSELoss()
    if concept_mode == "horse":
        forget_dl, remain_dl = setup_forget_horse_data(
            batch_size=batch_size,
            image_size=image_size,
            horse_data_path=horse_data_path,
            not_horse_data_path=not_horse_data_path,
        )
        forget_prompt = "a photo of a horse"
        remain_prompt = "a photo without a horse"
        run_tag = "horse"
    elif concept_mode == "concept":
        target_data_path = concept_data_path or horse_data_path
        non_target_data_path = not_concept_data_path or not_horse_data_path
        forget_dl, remain_dl = setup_forget_horse_data(
            batch_size=batch_size,
            image_size=image_size,
            horse_data_path=target_data_path,
            not_horse_data_path=non_target_data_path,
        )
        concept_name = concept_name.strip() if concept_name else "concept"
        forget_prompt = concept_prompt or f"a photo of a {concept_name}"
        remain_prompt = remain_prompt or f"a photo without a {concept_name}"
        run_tag = concept_name.replace(" ", "_")
    else:
        forget_dl, remain_dl = setup_forget_nsfw_data(batch_size, image_size)
        forget_prompt = "a photo of a nude person"
        remain_prompt = remain_prompt or "a photo of a person wearing clothes"
        run_tag = "nsfw"

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in name:
                parameters.append(param)
        # train all layers
        if train_method == "full":
            parameters.append(param)
    # set model to train
    model.train()

    losses = []
    optimizer = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()

    if mask_path:
        mask = torch.load(mask_path)
        name = f"compvis-{run_tag}-mask-method_{train_method}-lr_{lr}"
    else:
        name = f"compvis-{run_tag}-method_{train_method}-lr_{lr}"

    # TRAINING CODE
    for epoch in range(epochs):
        remain_iter = iter(remain_dl)
        with tqdm(total=len(forget_dl)) as time:
            # with tqdm(total=10) as time:
            for i, forget_images in enumerate(forget_dl):
                # for i in range(1):
                optimizer.zero_grad()

                try:
                    remain_images = next(remain_iter)
                except StopIteration:
                    remain_iter = iter(remain_dl)
                    remain_images = next(remain_iter)

                forget_bs = int(forget_images.shape[0])
                remain_bs = int(remain_images.shape[0])
                forget_prompts = [forget_prompt] * forget_bs

                # player -> truck
                pseudo_prompts = [remain_prompt] * forget_bs
                remain_prompts = [remain_prompt] * remain_bs

                # remain stage
                remain_batch = {
                    "jpg": remain_images.permute(0, 2, 3, 1),
                    "txt": remain_prompts,
                }
                remain_loss = model.shared_step(remain_batch)[0]

                # forget stage
                forget_batch = {
                    "jpg": forget_images.permute(0, 2, 3, 1),
                    "txt": forget_prompts,
                }

                pseudo_batch = {
                    "jpg": forget_images.permute(0, 2, 3, 1),
                    "txt": pseudo_prompts,
                }

                forget_input, forget_emb = model.get_input(
                    forget_batch, model.first_stage_key
                )
                pseudo_input, pseudo_emb = model.get_input(
                    pseudo_batch, model.first_stage_key
                )

                t = torch.randint(
                    0,
                    model.num_timesteps,
                    (forget_input.shape[0],),
                    device=model.device,
                ).long()
                noise = torch.randn_like(forget_input, device=model.device)

                forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
                pseudo_noisy = model.q_sample(x_start=pseudo_input, t=t, noise=noise)

                forget_out = model.apply_model(forget_noisy, t, forget_emb)
                pseudo_out = model.apply_model(pseudo_noisy, t, pseudo_emb).detach()

                forget_loss = criteria(forget_out, pseudo_out)

                # total loss
                loss = forget_loss + alpha * remain_loss
                loss.backward()
                losses.append(loss.item() / batch_size)

                if mask_path:
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            p.grad *= mask[n.split("model.diffusion_model.")[-1]].to(
                                device
                            )

                optimizer.step()
                time.set_description("Epoch %i" % epoch)
                time.set_postfix(loss=loss.item() / batch_size)
                sleep(0.1)
                time.update(1)

    model.eval()
    saved_model_path = save_model(
        model,
        name,
        None,
        save_compvis=True,
        save_diffusers=save_diffusers,
        compvis_config_file=config_path,
        diffusers_config_file=diffusers_config_path,
    )
    print(f"Saved model checkpoint: {saved_model_path}")


def save_model(
    model,
    name,
    num,
    compvis_config_file=None,
    diffusers_config_file=None,
    device="cpu",
    save_compvis=True,
    save_diffusers=True,
):
    # SAVE MODEL
    folder_path = f"/work/hdd/bcxt/anon3/unlearn_diff/salun/SD/models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f"{folder_path}/{name}-epoch_{num}.pt"
    else:
        path = f"{folder_path}/{name}.pt"
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        try:
            from convertModels import savemodelDiffusers
        except Exception as exc:
            print(
                "Skipping diffusers export because convertModels/diffusers import failed:"
                f" {exc}"
            )
            return

        print("Saving Model in Diffusers Format")
        savemodelDiffusers(
            name, compvis_config_file, diffusers_config_file, device=device
        )
    return path


def save_history(losses, name, word_print):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses, f"{folder_path}/loss.png", word_print, n=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="TrainESD",
        description="Finetuning stable diffusion model to erase concepts using ESD method",
    )

    parser.add_argument(
        "--train_method", help="method of training", type=str, required=True
    )
    parser.add_argument(
        "--alpha",
        help="guidance of start image used to train",
        type=float,
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=2,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=1
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=float,
        required=False,
        default=1e-5,
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/stable-diffusion/v1-inference.yaml",
    )
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default="/work/hdd/bcxt/anon3/unlearn_diff/stable-diffusion/sd-v1-4-full-ema.ckpt",
    )
    parser.add_argument(
        "--mask_path",
        help="mask path for stable diffusion v1-4",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--diffusers_config_path",
        help="diffusers unet config json path",
        type=str,
        required=False,
        default="diffusers_unet_config.json",
    )
    parser.add_argument(
        "--device",
        help="cuda devices to train on",
        type=str,
        required=False,
        default="0,0",
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=50,
    )
    parser.add_argument(
        "--save_diffusers",
        action="store_true",
        help="also export a diffusers-format checkpoint (requires diffusers dependencies)",
    )
    parser.add_argument(
        "--concept_mode",
        type=str,
        default="nsfw",
        choices=["nsfw", "horse", "concept"],
        help="training mode: nsfw (default), horse, or generic concept",
    )
    parser.add_argument(
        "--horse_data_path",
        type=str,
        default="/work/hdd/bcxt/anon3/unlearn_diff/salun/SD/data/horse",
        help="horse image folder for concept_mode=horse",
    )
    parser.add_argument(
        "--not_horse_data_path",
        type=str,
        default="/work/hdd/bcxt/anon3/unlearn_diff/salun/SD/data/not-horse",
        help="non-horse image folder for concept_mode=horse",
    )
    parser.add_argument(
        "--concept_name",
        type=str,
        default="concept",
        help="target concept name for concept_mode=concept (e.g., castle)",
    )
    parser.add_argument(
        "--concept_prompt",
        type=str,
        default=None,
        help="forget prompt for concept_mode=concept (default: 'a photo of a {concept_name}')",
    )
    parser.add_argument(
        "--remain_prompt",
        type=str,
        default=None,
        help="retain prompt text; optional override",
    )
    parser.add_argument(
        "--concept_data_path",
        type=str,
        default=None,
        help="target concept image folder for concept_mode=concept (falls back to --horse_data_path)",
    )
    parser.add_argument(
        "--not_concept_data_path",
        type=str,
        default=None,
        help="non-target image folder for concept_mode=concept (falls back to --not_horse_data_path)",
    )
    args = parser.parse_args()

    train_method = args.train_method
    alpha = args.alpha
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    mask_path = args.mask_path
    diffusers_config_path = args.diffusers_config_path
    device = f"cuda:{int(args.device)}"
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    save_diffusers = args.save_diffusers
    concept_mode = args.concept_mode
    horse_data_path = args.horse_data_path
    not_horse_data_path = args.not_horse_data_path
    concept_name = args.concept_name
    concept_prompt = args.concept_prompt
    remain_prompt = args.remain_prompt
    concept_data_path = args.concept_data_path
    not_concept_data_path = args.not_concept_data_path

    nsfw_removal(
        train_method=train_method,
        alpha=alpha,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        config_path=config_path,
        ckpt_path=ckpt_path,
        mask_path=mask_path,
        diffusers_config_path=diffusers_config_path,
        device=device,
        image_size=image_size,
        ddim_steps=ddim_steps,
        save_diffusers=save_diffusers,
        concept_mode=concept_mode,
        horse_data_path=horse_data_path,
        not_horse_data_path=not_horse_data_path,
        concept_name=concept_name,
        concept_prompt=concept_prompt,
        remain_prompt=remain_prompt,
        concept_data_path=concept_data_path,
        not_concept_data_path=not_concept_data_path,
    )
