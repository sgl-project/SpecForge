# ⚡️ About SpecForge

## 💡 Motivation

Speculative decoding is an important and powerful technique for speeding up inference without losing performance. Industries have used it extensively in production to better serve their users with lower latency and higher throughput. We have seen some open-source projects for training speculative decoding models, but most of them are not well-maintained or not directly compatible with SGLang. We prepared this project because we wish that the open-source community can enjoy a speculative decoding framework that is

- regularly maintained by the SGLang team: the code is runnable out-of-the-box
- directly compatible with SGLang: there is no additional efforts for porting to SGLang
- provide performant training capabilities: we provided online/offline/tensor-parallel/FSDP to suit your needs

## ✅ SGLang-ready

As SpecForge is built by the SGLang, we ensure that the draft models trained with SpecForge are directly compatible with [SGLang](https://github.com/sgl-project/sglang). This means that no postprocessing or weights conversion is required, providing users with a seamless experience from training to serving. We export our data in the Hugging Face format, so you can load it to other serving frameworks as well if the model is supported by them.
