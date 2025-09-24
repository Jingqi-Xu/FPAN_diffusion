# FPAN:Mitigating Replication in Diffusion Models through the Fine-Grained Probabilistic Addition of Noise to Token Embeddings
This repo contains code for the paper: FPAN: Mitigating Replication in Diffusion Models through the Fine-Grained Probabilistic Addition of Noise to Token Embeddings.-[paper link](https://arxiv.org/abs/2505.21848)
## Overview
![Overview of FPAN](workflow_FPAN.png)
## Requirement
```txt
diffusers == 0.32.1
huggingface-hub == 0.27.1
img2dataset == 1.41.0
openai == 0.27.8
timm == 0.9.2
spacy == 3.7.6
torch == 2.5.1
torchvision == 0.20.1
transformers == 4.47.1
webdataset == 0.2.100
pytorch-fid == 0.3.0
```
## Pretrained Models
Before training or inference, please download the pretrained UNet weights for **Stable Diffusion v2.1**:

- [Stable Diffusion v2.1 UNet weights (diffusion_pytorch_model.bin)](https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/unet/diffusion_pytorch_model.bin)

After downloading, please place the file at: ./download_pretrain/stable_diffusion_2_1/diffusion_pytorch_model.bin
## Train the model

This is an example of **W=1.7** and **P=0.6**:

```bash
accelerate launch --main_process_port 27000 diff_train_baseline_randompick_tokenlevel_1.7_0.6_0_0.4noise.py --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1 --instance_data_dir "./laion_10k_data_2" --resolution=256 --gradient_accumulation_steps=1 --center_crop --random_flip --learning_rate=5e-6 --lr_scheduler constant_with_warmup --lr_warmup_steps=5000 --max_train_steps=100000 --train_batch_size=16 --save_steps=10000 --modelsavesteps 20000 --duplication nodup --output_dir="./out_laion_10k_baseline_randompick_tokenlevel_1.7_0.6_0_0.4noise_orig_capiton" --class_prompt laion_orig --num_train_epoch 200 --modify_unet --modify_unet_config "./unet_config/unet_config.json" --modify_unet_pretrain_path "./download_pretrain/stable_diffusion_2_1/diffusion_pytorch_model.bin"
```
## Inference
```bash
python diff_inference.py --modelpath "./out_laion_10k_baseline_randompick_tokenlevel_1.7_0.6_0_0.4noise_orig_capiton_laion_orig_nodup/" -nb 8201 --GPT_caption "no" --modify_unet_type "custom"
```
## Retrieval
```bash
python diff_retrieval.py --arch resnet50_disc --similarity_metric dotproduct --pt_style sscd --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 --query_dir "./inferences/laion_10k_frozentext/out_laion_10k_baseline_randompick_tokenlevel_1.7_0.6_0_0.4noise_orig_capiton_laion_orig_nodup/laion_orig/" --val_dir "./laion_10k_data_2/"

python -m pytorch_fid ./laion_10k_data_2/raw_images ./inferences/laion_10k_frozentext/out_laion_10k_baseline_randompick_tokenlevel_1.7_0.6_0_0.4noise_orig_capiton_laion_orig_nodup/laion_orig/generations --device cuda
```
## Cite Us
```bash
@misc{xu2025fpanmitigatingreplicationdiffusion,
      title={FPAN: Mitigating Replication in Diffusion Models through the Fine-Grained Probabilistic Addition of Noise to Token Embeddings}, 
      author={Jingqi Xu and Chenghao Li and Yuke Zhang and Peter A. Beerel},
      year={2025},
      eprint={2505.21848},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.21848}, 
}
```
