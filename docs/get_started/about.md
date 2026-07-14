# ⚡️ About SpecForge

## 💡 Motivation

Speculative decoding is an important and powerful technique for speeding up inference without losing performance. Industries have used it extensively in production to better serve their users with lower latency and higher throughput. We have seen some open-source projects for training speculative decoding models, but most of them are not well-maintained or not directly compatible with SGLang. We prepared this project because we wish that the open-source community can enjoy a speculative decoding framework that is

- regularly maintained by the SGLang team: the code is runnable out-of-the-box
- directly compatible with SGLang: there is no additional efforts for porting to SGLang
- provide one-GPU colocated/offline training and online-disaggregated data-parallel training through one runtime

## ✅ SGLang-ready

As SpecForge is built by the SGLang team, draft models trained with SpecForge
can be exported for [SGLang](https://github.com/sgl-project/sglang) serving.
Runtime checkpoints retain training state, so materialize a serving directory
with the shared `specforge export` command. SGLang and Hugging Face export
targets use the same checkpoint surface; there are no method-specific
conversion scripts.
