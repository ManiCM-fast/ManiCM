# [ManiCM: Real-time 3D Diffusion Policy via Consistency Model for Robotic Manipulation](https://manicm-fast.github.io/)

[Project Page](https://manicm-fast.github.io/) | [arXiv](http://arxiv.org/abs/2406.01586) | [Paper](https://arxiv.org/pdf/2406.01586)

[Guanxing Lu](https://guanxinglu.github.io/)\*, Zifeng Gao\*, [Tianxing Chen](https://tianxingchen.github.io), [Wenxun Dai](https://github.com/Dai-Wenxun), [Ziwei Wang](https://ziweiwangthu.github.io/), [Yansong Tang](https://andytang15.github.io/)<sup>†</sup>

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FManiCM-fast%2FManiCM&count_bg=%2349CCFF&title_bg=%23B038EF&icon=&icon_color=%23E7E7E7&title=ManiCM+Code+Viewers&edge_flat=false)](https://hits.seeyoufarm.com)

![](./files/2024-ManiCM.png)
<b>ManiCM Overview</b>: Given a raw action sequence a<sub>0</sub>, we first perform a forward diffusion to introduce noise over n + k steps. The resulting noisy sequence a<sub>n+k</sub> is then fed into both the online network and the teacher network to predict the clean action sequence. The target network uses the teacher network’s k-step estimation results to predict the action sequence. To enforce self-consistency, a loss function is applied to ensure that the outputs of the online network and the target network are consistent.

# 💻 Installation

See [INSTALL.md](INSTALL.md) for installation instructions. 



# 📚 Config

**Algorithms**. We provide the implementation of the following algorithms: 

- DP3: `dp3.yaml`
- ManiCM: `dp3_cm.yaml`

You can modify the configuration of the teacher model and ManiCM by editing these two files. Here are the meanings of some important configurations:

`num_inference_timesteps`: The inference steps of ManiCM.

`num_train_timesteps`: Total time step for adding noise.

`prediction_type`:  `epsilon` represents prediction noise, while `sample` represents predicted action.

For more detailed arguments, please refer to the scripts and the code.

# 🛠️ Usage

Scripts for generating demonstrations, training, and evaluation are all provided in the `scripts/` folder. 

The results are logged by `wandb`, so you need to `wandb login` first to see the results and videos.

We provide a simple instruction for using the codebase here.

1. Generate demonstrations by `gen_demonstration_adroit.sh` and `gen_demonstration_dexart.sh`. See the scripts for details. For example:
    ```bash
    bash scripts/gen_demonstration_adroit.sh hammer
    ```
    This will generate demonstrations for the `hammer` task in Adroit environment. The data will be saved in `3D-Diffusion-Policy/data/` folder automatically.


2. Train and evaluate a teacher policy with behavior cloning. For example:
    ```bash
    # bash scripts/train_policy.sh config_name task_name addition_info seed gpu_id 
    bash scripts/train_policy.sh dp3 adroit_hammer 0603 0 0
    ```
    This will train a DP3 policy on the `hammer` task in Adroit environment using point cloud modality. By default we **save** the ckpt (optional in the script). During training, teacher's model takes ~10G gpu memory and ~4 hours on an Nvidia 4090 GPU.
    
3. Move teacher's ckpt. For example:
    ```bash
    # bash scopy.sh alg_name task_name teacher_addition_info addition_info seed gpu_id
    bash scopy.sh dp3_cm adroit_hammer 0603 0603_cm 0 0
    ```
    
4. Train and evaluate ManiCM. For example:
    ```bash
    # bash scripts/train_policy_cm.sh config_name task_name addition_info seed gpu_id
    bash scripts/train_policy_cm.sh dp3_cm adroit_hammer 0603_cm 0 0
    ```
    This will train ManiCM use a DP3 policy teacher model on the `hammer` task in Adroit environment using point cloud modality. During training, ManiCM model takes ~10G gpu memory and ~4 hours on an Nvidia 4090 GPU.

# 🏞️ Checkpoints

We have updated the [pre-trained checkpoints](https://drive.google.com/drive/folders/1WhYQij_D3IisDpdLjKCQF3iaZB1bZ63f?usp=sharing) of `hammer` task in Adroit environment for your convenience. You can download them and place the folder into `data/outputs/`.

# 🏷️ License
This repository is released under the MIT license.

# 🙏 Acknowledgement

Our code is built upon [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy), [MotionLCM](https://github.com/Dai-Wenxun/MotionLCM), [Latent Consistency Model](https://github.com/luosiallen/latent-consistency-model), [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [VRL3](https://github.com/microsoft/VRL3), [Metaworld](https://github.com/Farama-Foundation/Metaworld), and [ManiGaussian](https://github.com/GuanxingLu/ManiGaussian). We would like to thank the authors for their excellent works.

# 🥰 Citation
If you find this repository helpful, please consider citing:

```
@article{lu2024manicm,
      title={ManiCM: Real-time 3D Diffusion Policy via Consistency Model for Robotic Manipulation}, 
      author={Guanxing Lu and Zifeng Gao and Tianxing Chen and Wenxun Dai and Ziwei Wang and Yansong Tang},
      journal={arXiv preprint arXiv:2406.01586},
      year={2024}
}
```

# ✅ 31 Tasks (Adroit and MetaWorld)

<div class="video-container">
  <!-- 示例中只添加了部分视频，你需要根据实际情况添加所有32个视频 -->
  
  <video class="video-item" controls>
    <source src="videos/1.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/2.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/3.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/4.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/5.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/6.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/7.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/8.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/9.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/10.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/11.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/12.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/13.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/14.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/15.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/16.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/17.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/18.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/19.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/20.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/21.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/22.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/23.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/24.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/25.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/26.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/27.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/28.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/29.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/30.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/31.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>

  <video class="video-item" controls>
    <source src="videos/32.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <!-- 其他视频标签 -->
</div>


<style>
    .video-container {
        display: grid;
        grid-template-columns: repeat(8, 1fr); /* 8列，每列占网格容器宽度的1/8 */
        grid-template-rows: repeat(4, 1fr); /* 4行，每行占网格容器高度的1/4 */
        gap: 5px; /* 网格项之间的间隙 */
        width: 100%; /* 容器宽度 */
        border: 2px dashed #333; /* 虚线边框，可以根据需要调整颜色 */
        /* 如果您希望网格容器的高度也是自适应的，可以添加以下属性 */
        height: auto; /* 容器高度自适应 */
    }
    .video-item {
        width: 100%; /* 视频宽度占满网格项宽度 */
        height: 100%; /* 视频高度占满网格项高度 */
        /* 如果视频原始比例不是1:1，可能需要以下属性来保持视频的宽高比 */
        object-fit: cover; /* 视频填充网格项，可能会被裁剪 */
    }
</style>