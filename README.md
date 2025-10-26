# Milmer

This is an implementation of the Milmer model, described in the following paper:

**Milmer: a Framework for Multiple Instance Learning based Multimodal Emotion Recognition**

![Preview](./Graphical%20Abstract.png)

## Abstract

Emotions play a crucial role in human behavior and decision-making, making emotion recognition a key area of interest in human-computer interaction (HCI). This study addresses the challenges of emotion recognition by integrating facial expression analysis with electroencephalogram (EEG) signals, introducing a novel multimodal framework-Milmer. The proposed framework employs a transformer-based fusion approach to effectively integrate visual and physiological modalities. It consists of an EEG preprocessing module, a facial feature extraction and balancing module, and a cross-modal fusion module. To enhance visual feature extraction, we fine-tune a pre-trained Swin Transformer on emotion-related datasets. Additionally, a cross-attention mechanism is introduced to balance token representation across modalities, ensuring effective feature integration. A key innovation of this work is the adoption of a multiple instance learning (MIL) approach, which extracts meaningful information from multiple facial expression images over time, capturing critical temporal dynamics often overlooked in previous studies. Extensive experiments conducted on the DEAP dataset demonstrate the superiority of the proposed framework, achieving a classification accuracy of 96.72\% in the four-class emotion recognition task. Ablation studies further validate the contributions of each module, highlighting the significance of advanced feature extraction and fusion strategies in enhancing emotion recognition performance. Our code are available at https://github.com/liangyubuaa/Milmer.

## Requirements

- Python 3.8
- For other dependencies, see [requirements.txt](./requirements.txt)

## Parameters

For detailed parameter configuration, please refer to the [config](./config) folder.

- **Pretrained model**: Use swin-tiny-patch4-window7-224-finetuned-face-emotion-v12 by default.
- **LR schedule**: Cosine decay reduces lr from its initial value to lrf×lr with lrf=0.1 by default.
- **Batch size**: Train with batch size 14 and use the same for test.
- **Epochs**: Train for 100 epochs; change it in config/multi_instance.json if needed.
- **Optimizer**: AdamW with lr 1e-4 and other PyTorch defaults.
- **MIL**: Use 10 instances per sample and select top 3 by attention-weighted top-k.
- **Fusion**: Fusion type is cross_attention; available options are none, cross_attention, and mlp.
- **Transformer encoder**: d_model 768, nhead 12, dim_feedforward 2048, num_layers 2, dropout 0.2, CLS dropout 0.1.
- **Seed**: Random seed 0.

## Reference

@article{wang2025milmer,
title={Milmer: a Framework for Multiple Instance Learning based Multimodal Emotion Recognition},
author={Wang, Zaitian and He, Jian and Liang, Yu and Hu, Xiyuan and Peng, Tianhao and Wang, Kaixin and Wang, Jiakai and Zhang, Chenlong and Zhang, Weili and Niu, Shuang and others},
journal={arXiv preprint arXiv:2502.00547},
year={2025}
}
