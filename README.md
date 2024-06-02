# [ManiCM: Real-time 3D Diffusion Policy via Consistency Model for Robotic Manipulation](https://manicm-fast.github.io/)

[Project Page](https://manicm-fast.github.io/) | [arXiv]() | [Paper]() 

[Guanxing Lu](https://guanxinglu.github.io/)\*, Zifeng Gao\*, [Tianxing Chen](https://tianxingchen.github.io), [Wenxun Dai](https://github.com/Dai-Wenxun), [Ziwei Wang](https://ziweiwangthu.github.io/), [Yansong Tang](https://andytang15.github.io/)

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FManiCM-fast%2FManiCM&count_bg=%2349CCFF&title_bg=%23B038EF&icon=&icon_color=%23E7E7E7&title=ManiCM+Code+Viewers&edge_flat=false)](https://hits.seeyoufarm.com)

![](./files/2024-ManiCM.png)
<b>ManiCM Overview</b>: Given a raw action sequence a<sub>0</sub>, we first perform a forward diffusion to introduce noise over n + k steps. The resulting noisy sequence a<sub>n+k</sub> is then fed into both the online network and the teacher network to predict the clean action sequence. The target network uses the teacher network’s k-step estimation results to predict the action sequence. To enforce self-consistency, a loss function is applied to ensure that the outputs of the online network and the target network are consistent.


# Code
Coming Soon
