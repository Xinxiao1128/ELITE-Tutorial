import argparse
import itertools
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, LMSDiscreteScheduler
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIP_TEXT_INPUTS_DOCSTRING, _expand_mask

from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

from typing import Optional, Tuple, Union
from datasets import OpenImagesDataset

# ... Keep Mapper class and other functions unchanged ...
class Mapper(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
    ):
        super(Mapper, self).__init__()

        for i in range(5):
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                         nn.LayerNorm(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, 1024),
                                         nn.LayerNorm(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, output_dim)))

            setattr(self, f'mapping_patch_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, output_dim)))

    def forward(self, embs):
        hidden_states = ()
        for i, emb in enumerate(embs):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(emb[:, 1:]).mean(dim=1, keepdim=True)
            hidden_states += (hidden_state, )
        hidden_states = torch.cat(hidden_states, dim=1)
        return hidden_states

# ... Keep other functions unchanged, including inj_forward_text and inj_forward_crossattention ...

logger = get_logger(__name__)

def save_progress(mapper, accelerator, args, step=None):
    logger.info("Saving embeddings")

    state_dict = accelerator.unwrap_model(mapper).state_dict()

    if step is not None:
        torch.save(state_dict, os.path.join(args.output_dir, f"mapper_{str(step).zfill(6)}.pt"))
    else:
        torch.save(state_dict, os.path.join(args.output_dir, "mapper.pt"))

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--global_mapper_path", type=str, default=None, help="If not none, the training will start from the given checkpoints."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # 🔥 New parameter: Select which token embedding to use
    parser.add_argument(
        "--token_index",
        type=str,
        default="full",
        help="Which token embedding to use during validation: 0-4 for specific layer, 'full' for all combined, 'compare' for generating all layers"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args

def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def unfreeze_params(params):
    for param in params:
        param.requires_grad = True

def th2image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)

# 🔥 Key modification: Validation function supporting different token embedding selection
@torch.no_grad()
def validation(example, tokenizer, image_encoder, text_encoder, unet, mapper, vae, device, guidance_scale, token_index='full', seed=None):
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    uncond_input = tokenizer(
        [''] * example["pixel_values"].shape[0],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder({'input_ids':uncond_input.input_ids.to(device)})[0]

    if seed is None:
        latents = torch.randn(
            (example["pixel_values"].shape[0], unet.in_channels, 64, 64)
        )
    else:
        generator = torch.manual_seed(seed)
        latents = torch.randn(
            (example["pixel_values"].shape[0], unet.in_channels, 64, 64), generator=generator,
        )

    latents = latents.to(example["pixel_values_clip"])
    scheduler.set_timesteps(100)
    latents = latents * scheduler.init_noise_sigma

    placeholder_idx = example["index"]
    image = F.interpolate(example["pixel_values_clip"], (224, 224), mode='bilinear')

    image_features = image_encoder(image, output_hidden_states=True)
    image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12],
                        image_features[2][16]]
    image_embeddings = [emb.detach() for emb in image_embeddings]
    inj_embedding = mapper(image_embeddings)

    # 🔥 Key modification: Select different token embeddings based on token_index
    if token_index == 'full':
        # Original method: Use all 5 token embeddings
        pass  # inj_embedding remains as is, shape: [batch, 5, 768]
    elif token_index.isdigit():
        # Select specific layer token embedding
        token_index = int(token_index)
        if 0 <= token_index <= 4:
            print(f"🔍 Using token embedding from layer {token_index}")
            # Only use specified layer token embedding, repeat 5 times to maintain shape consistency
            selected_embedding = inj_embedding[:, token_index:token_index + 1, :]  # [batch, 1, 768]
            inj_embedding = selected_embedding.repeat(1, 5, 1)  # [batch, 5, 768]
        else:
            print(f"⚠️ Invalid token_index {token_index}, using full embeddings")
    else:
        print(f"⚠️ Unknown token_index '{token_index}', using full embeddings")

    encoder_hidden_states = text_encoder({'input_ids': example["input_ids"],
                                          "inj_embedding": inj_embedding,
                                          "inj_index": placeholder_idx})[0]

    for t in tqdm(scheduler.timesteps):
        latent_model_input = scheduler.scale_model_input(latents, t)
        noise_pred_text = unet(
            latent_model_input,
            t,
            encoder_hidden_states={
                "CONTEXT_TENSOR": encoder_hidden_states,
            }
        ).sample

        latent_model_input = scheduler.scale_model_input(latents, t)

        noise_pred_uncond = unet(
            latent_model_input,
            t,
            encoder_hidden_states={
                "CONTEXT_TENSOR": uncond_embeddings,
            }
        ).sample

        noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    _latents = 1 / 0.18215 * latents.clone()
    images = vae.decode(_latents).sample
    ret_pil_images = [th2image(image) for image in images]

    return ret_pil_images

# 🔥 New function: Generate comparison images for all layers
@torch.no_grad()
def generate_layer_comparison(example, tokenizer, image_encoder, text_encoder, unet, mapper, vae, device, output_dir, step):
    """Generate comparison images using different token embedding layers"""

    print("🔍 Generating layer comparison images...")
    layer_names = ["Layer_0_deepest", "Layer_1", "Layer_2", "Layer_3", "Layer_4_shallowest", "Full_combined"]
    layer_images = []

    # Generate images for each layer
    for i in range(5):
        print(f"   Generating image for layer {i}...")
        images = validation(example, tokenizer, image_encoder, text_encoder, unet, mapper, vae, device, 5, token_index=str(i), seed=42)
        layer_images.append(images[0])

    # Generate image using all layers combined
    print("   Generating image for full combined layers...")
    images = validation(example, tokenizer, image_encoder, text_encoder, unet, mapper, vae, device, 5, token_index='full', seed=42)
    layer_images.append(images[0])

    # Create comparison grid
    target_size = (256, 256)
    resized_images = [img.resize(target_size) for img in layer_images]

    # Create 2x3 grid
    grid_width = 3 * target_size[0]
    grid_height = 2 * target_size[1]
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

    # Place images
    for i, (img, name) in enumerate(zip(resized_images, layer_names)):
        row = i // 3
        col = i % 3
        x = col * target_size[0]
        y = row * target_size[1]
        grid_image.paste(img, (x, y))

    # Save comparison grid
    grid_path = os.path.join(output_dir, f"layer_comparison_step_{str(step).zfill(5)}.jpg")
    grid_image.save(grid_path)
    print(f"📸 Layer comparison saved to: {grid_path}")

    # Save individual layer images
    for i, (img, name) in enumerate(zip(resized_images, layer_names)):
        img_path = os.path.join(output_dir, f"step_{str(step).zfill(5)}_{name}.jpg")
        img.save(img_path)

    return layer_images

def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer and add the placeholder token as a additional special token
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # replace the forward method of the text encoder to inject the word embedding
    for _module in text_encoder.modules():
        if _module.__class__.__name__ == "CLIPTextTransformer":
            _module.__class__.__call__ = inj_forward_text

    image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")

    mapper = Mapper(input_dim=1024, output_dim=768)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # replace the forward method of the crossattention to finetune the to_k and to_v layers
    for _name, _module in unet.named_modules():
        if _module.__class__.__name__ == "CrossAttention":
            if 'attn1' in _name: continue
            _module.__class__.__call__ = inj_forward_crossattention

            shape = _module.to_k.weight.shape
            to_k_global = nn.Linear(shape[1], shape[0], bias=False)
            to_k_global.weight.data = _module.to_k.weight.data.clone()
            mapper.add_module(f'{_name.replace(".", "_")}_to_k', to_k_global)

            shape = _module.to_v.weight.shape
            to_v_global = nn.Linear(shape[1], shape[0], bias=False)
            to_v_global.weight.data = _module.to_v.weight.data.clone()
            mapper.add_module(f'{_name.replace(".", "_")}_to_v', to_v_global)

            if args.global_mapper_path is None:
                _module.add_module('to_k_global', to_k_global)
                _module.add_module('to_v_global', to_v_global)

    if args.global_mapper_path is not None:
        mapper.load_state_dict(torch.load(args.global_mapper_path, map_location='cpu'))
        for _name, _module in unet.named_modules():
            if _module.__class__.__name__ == "CrossAttention":
                if 'attn1' in _name: continue
                _module.add_module('to_k_global', getattr(mapper, f'{_name.replace(".", "_")}_to_k'))
                _module.add_module('to_v_global', getattr(mapper, f'{_name.replace(".", "_")}_to_v'))

    # Freeze vae and unet, encoder
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    freeze_params(text_encoder.parameters())
    freeze_params(image_encoder.parameters())

    # Unfreeze the mapper
    unfreeze_params(mapper.parameters())

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        itertools.chain(mapper.parameters()),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    train_dataset = OpenImagesDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        set="test",
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    mapper, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        mapper, optimizer, train_dataloader, lr_scheduler
    )

    # Move vae, unet, and encoders to device
    vae.to(accelerator.device)
    unet.to(accelerator.device)
    image_encoder.to(accelerator.device)
    text_encoder.to(accelerator.device)
    # Keep vae, unet and image_encoder in eval model as we don't train these
    vae.eval()
    unet.eval()
    image_encoder.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("elite", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(args.num_train_epochs):
        mapper.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(mapper):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                placeholder_idx = batch["index"]
                image = F.interpolate(batch["pixel_values_clip"], (224, 224), mode='bilinear')

                image_features = image_encoder(image, output_hidden_states=True)
                image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12], image_features[2][16]]
                image_embeddings = [emb.detach() for emb in image_embeddings]
                inj_embedding = mapper(image_embeddings)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder({'input_ids': batch["input_ids"],
                                                      "inj_embedding": inj_embedding,
                                                      "inj_index": placeholder_idx.detach()})[0]

                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states={
                    "CONTEXT_TENSOR": encoder_hidden_states,
                }).sample

                loss_mle = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                loss_reg = torch.mean(torch.abs(inj_embedding)) * 0.01

                loss = loss_mle + loss_reg

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(mapper.parameters(), 1)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_progress(mapper, accelerator, args, global_step)

                    # 🔥 Modified validation section: Generate comparison images for different layers
                    if args.token_index == 'compare':
                        # Generate comparison images for all layers
                        generate_layer_comparison(batch, tokenizer, image_encoder, text_encoder, unet, mapper, vae, batch["pixel_values_clip"].device, args.output_dir, global_step)
                    else:
                        # Generate images for specified layer
                        syn_images = validation(batch, tokenizer, image_encoder, text_encoder, unet, mapper, vae, batch["pixel_values_clip"].device, 5, token_index=args.token_index)
                        gt_images = [th2image(img) for img in batch["pixel_values"]]
                        img_list = []
                        for syn, gt in zip(syn_images, gt_images):
                            img_list.append(np.concatenate((np.array(syn), np.array(gt)), axis=1))
                        img_list = np.concatenate(img_list, axis=0)

                        # Add layer information to filename
                        if args.token_index != 'full':
                            filename = f"{str(global_step).zfill(5)}_layer_{args.token_index}.jpg"
                        else:
                            filename = f"{str(global_step).zfill(5)}_full.jpg"
                        Image.fromarray(img_list).save(os.path.join(args.output_dir, filename))

            logs = {"loss_mle": loss_mle.detach().item(), "loss_reg": loss_reg.detach().item(),  "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        save_progress(mapper, accelerator, args)

    accelerator.end_training()


if __name__ == "__main__":
    main()
