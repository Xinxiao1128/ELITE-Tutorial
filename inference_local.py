#!/usr/bin/env python3
"""
修改后的inference_local.py
支持选择不同层的词嵌入进行推理
"""

import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

# 假设这些模块存在
from train_global import inj_forward_text, th2image, Mapper
from train_local import MapperLocal, inj_forward_crossattention
from datasets import OpenImagesDatasetWithMask


@torch.no_grad()
def validation_with_layer_selection(example, tokenizer, image_encoder, text_encoder, unet, mapper, mapper_local, vae, device, guidance_scale, seed=None, llambda=1, num_steps=100, layer_index=0, use_all_layers=False):
    """
    支持层级选择的validation函数
    """
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
    scheduler.set_timesteps(num_steps)
    latents = latents * scheduler.init_noise_sigma

    placeholder_idx = example["index"]

    image = F.interpolate(example["pixel_values_clip"], (224, 224), mode='bilinear')
    image_features = image_encoder(image, output_hidden_states=True)
    image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12], image_features[2][16]]
    image_embeddings = [emb.detach() for emb in image_embeddings]
    inj_embedding = mapper(image_embeddings)

    # 🔥 关键修改：根据参数选择不同的词嵌入层
    if use_all_layers:
        print(f"🔥 使用所有层的组合进行推理")
        # inj_embedding保持原样，包含所有层
    else:
        print(f"🔥 使用第{layer_index}层的词嵌入 (w_{layer_index}) 进行推理")
        inj_embedding = inj_embedding[:, layer_index:layer_index+1, :]

    encoder_hidden_states = text_encoder({'input_ids': example["input_ids"],
                                          "inj_embedding": inj_embedding,
                                          "inj_index": placeholder_idx})[0]

    image_obj = F.interpolate(example["pixel_values_obj"], (224, 224), mode='bilinear')
    image_features_obj = image_encoder(image_obj, output_hidden_states=True)
    image_embeddings_obj = [image_features_obj[0], image_features_obj[2][4], image_features_obj[2][8],
                            image_features_obj[2][12], image_features_obj[2][16]]
    image_embeddings_obj = [emb.detach() for emb in image_embeddings_obj]

    inj_embedding_local = mapper_local(image_embeddings_obj)
    mask = F.interpolate(example["pixel_values_seg"], (16, 16), mode='nearest')
    mask = mask[:, 0].reshape(mask.shape[0], -1, 1)
    inj_embedding_local = inj_embedding_local * mask

    for t in tqdm(scheduler.timesteps):
        latent_model_input = scheduler.scale_model_input(latents, t)
        noise_pred_text = unet(
            latent_model_input,
            t,
            encoder_hidden_states={
                "CONTEXT_TENSOR": encoder_hidden_states,
                "LOCAL": inj_embedding_local,
                "LOCAL_INDEX": placeholder_idx.detach(),
                "LAMBDA": llambda
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


def parse_args():
    parser = argparse.ArgumentParser(description="ELITE Local Inference with Layer Selection")
    
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        required=True,
        help="A folder containing the test data."
    )
    parser.add_argument(
        "--template",
        type=str,
        default="a photo of S",
        help="Template for text prompt"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/layer_analysis",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="test",
        help="Suffix for output files"
    )
    parser.add_argument(
        "--llambda",
        type=float,
        default=0.8,
        help="Lambda value for local attention"
    )
    parser.add_argument(
        "--global_mapper_path",
        type=str,
        required=True,
        help="Path to global mapper checkpoint"
    )
    parser.add_argument(
        "--local_mapper_path",
        type=str,
        required=True,
        help="Path to local mapper checkpoint"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    # 🔥 新增参数：选择哪一层的词嵌入
    parser.add_argument(
        "--layer_index",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="Which layer embedding to use (0-4, where 0 is the deepest layer w0, 4 is the shallowest w4)",
    )
    
    # 🔥 新增参数：是否使用所有层的组合
    parser.add_argument(
        "--use_all_layers",
        action="store_true",
        help="Use all layers combined instead of a single layer",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    print("加载模型...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    
    # 设置forward方法
    for _module in text_encoder.modules():
        if _module.__class__.__name__ == "CLIPTextTransformer":
            _module.__class__.__call__ = inj_forward_text
    
    # 加载mapper
    mapper = Mapper(input_dim=1024, output_dim=768)
    mapper_local = MapperLocal(input_dim=1024, output_dim=768)
    
    # 设置UNet的交叉注意力
    for _name, _module in unet.named_modules():
        if _module.__class__.__name__ == "CrossAttention":
            if 'attn1' in _name: continue
            _module.__class__.__call__ = inj_forward_crossattention
    
    # 加载预训练权重
    print("加载预训练权重...")
    mapper.load_state_dict(torch.load(args.global_mapper_path, map_location='cpu'))
    mapper_local.load_state_dict(torch.load(args.local_mapper_path, map_location='cpu'))
    
    # 移动到设备
    text_encoder.to(device)
    image_encoder.to(device)
    vae.to(device)
    unet.to(device)
    mapper.to(device)
    mapper_local.to(device)
    
    # 设置为评估模式
    text_encoder.eval()
    image_encoder.eval()
    vae.eval()
    unet.eval()
    mapper.eval()
    mapper_local.eval()
    
    # 加载测试数据
    print("加载测试数据...")
    test_dataset = OpenImagesDatasetWithMask(
        data_root=args.test_data_dir,
        tokenizer=tokenizer,
        size=512,
        placeholder_token="S",
        set="test"
    )
    
    # 运行推理
    print(f"开始推理...")
    if args.use_all_layers:
        print(f"使用模式：所有层组合")
        layer_info = "all_layers"
    else:
        print(f"使用模式：单层 w_{args.layer_index}")
        layer_info = f"w{args.layer_index}"
    
    for i, example in enumerate(test_dataset):
        print(f"处理第 {i+1}/{len(test_dataset)} 个样本...")
        
        # 添加batch维度
        for key in example:
            if isinstance(example[key], torch.Tensor):
                example[key] = example[key].unsqueeze(0).to(device)
        
        # 生成图像
        generated_images = validation_with_layer_selection(
            example, tokenizer, image_encoder, text_encoder, unet, 
            mapper, mapper_local, vae, device, 
            guidance_scale=7.5, seed=args.seed, llambda=args.llambda,
            layer_index=args.layer_index, use_all_layers=args.use_all_layers
        )
        
        # 保存结果
        for j, img in enumerate(generated_images):
            filename = f"{args.suffix}_{layer_info}_sample{i}_img{j}.png"
            output_path = os.path.join(args.output_dir, filename)
            img.save(output_path)
            print(f"保存图像: {output_path}")
    
    print("推理完成！")


if __name__ == "__main__":
    main()
