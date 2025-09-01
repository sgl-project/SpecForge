<div align="center" id="sglangtop">
<img src="./assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![documentation](https://img.shields.io/badge/📖-Documentation-red.svg?style=flat)](https://docs.sglang.ai/SpecForge/)
[![github badge](https://img.shields.io/badge/📃%20LMSYS-Blog-black.svg?style=flat)](https://lmsys.org/blog/2025-07-25-spec-forge/)
[![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://sgl-fru7574.slack.com/archives/C09784E3EN6)
[![SGLang Eagle3](https://img.shields.io/badge/🤗%20Hugging%20Face-SGLang%20Eagle3-yellow.svg?style=flat)](https://huggingface.co/collections/lmsys/eagle-3-6886b2329f3998a8bc23f8ed)
[![license](https://img.shields.io/badge/License-MIT%202.0-blue)](./LICENSE)

</div>

## 📍 Overview

SpecForge is an ecosystem project developed by the SGLang team. It is a framework for training speculative decoding models so that you can smoothly port them over to the SGLang serving framework to speed up your inference.

We have seen many open-source projects for speculative decoding, but most of them are not well-maintained or not directly compatible with SGLang. We prepared this project because we wish that the open-source community can enjoy a speculative decoding framework that is
- regularly maintained by the SpecForge team: the code is runnable out-of-the-box
- directly compatible with SGLang: there is no additional efforts for porting to SGLang
- provide performant training capabilities: we provided online/offline/tensor-parallel/FSDP to suit your needs


Check out [**our documentation**](https://docs.sglang.ai/SpecForge/) to get started.

## 🎉 News

- [2025-08] 🔔 SpecForge is listed as a [flagship project](https://lmsys.org/about/) in LMSYS. Congratulations to the SpecForge team!
- [2025-08] 🔥 SpecForge powered the Eagle3 draft model for GPT-OSS. Check out the blog at [LMSYS.org](https://lmsys.org/blog/2025-08-27-gpt-oss/)
- [2025-07] 🔥 SpecForge is released together with Llama4-Eagle3 checkpoints. Check out our blog at [LMSYS.org](https://lmsys.org/blog/2025-07-25-spec-forge/)

## ✨ Acknowledgements

<img src="./assets/acknowledgements.png" alt="acknowledgements"></img>

We would like to express our sincere gratitude to the official EAGLE team, especially Hongyang Zhang and Yuhui Li, for their invaluable contributions and support. Our thanks also go to the NVIDIA team—particularly Avery H and Izzy Putterman—and to the Google team, especially Ying Wang, for their insightful discussions and generous assistance throughout the project.

We are especially grateful to Meituan for their strong backing and meaningful contributions, which played a vital role in driving this project forward.

This project has also been inspired by many outstanding open-source projects from the LLM community, including [EAGLE](https://github.com/SafeAILab/EAGLE), [BaldEagle](https://github.com/NickL77/BaldEagle), and [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) and others. Their contributions and shared knowledge have greatly benefited our work.

## 💡 Special Thanks to Voltage Park

We would like to extend our sincere thanks to [Voltage Park](https://www.voltagepark.com/), our official infrastructure partner. As part of a formal collaboration with the SGLang team, Voltage Park provided critical GPU resources that empowered us to train and evaluate large-scale speculative decoding models efficiently and reliably. This partnership was instrumental in making SpecForge possible. We deeply appreciate Voltage Park’s mission to make cutting-edge AI infrastructure more accessible, and we look forward to continued collaboration as we push the boundaries of open-source LLM serving and optimization.

## 📃 Citation

```bibtex
@misc{specforge2025,
  title={SpecForge: Train speculative decoding models effortlessly},
  author={Shenggui Li, Yikai Zhu, Chao Wang, Fan Yin, Shuai Shi, Yubo Wang, Yi Zhang, Yingyi Huang, Haoshuai Zheng, Yineng Zhang},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/sgl-project/specforge}},
}
