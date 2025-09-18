## 🤔 Why SpecForge?

We have seen many open-source projects for speculative decoding, but most of them are not well-maintained or not directly compatible with SGLang. We prepared this project because we wish that the open-source community can enjoy a speculative decoding framework that is
- regularly maintained by the SGLang team: the code is runnable out-of-the-box
- directly compatible with SGLang: there is no additional efforts for porting to SGLang
- provide performant training capabilities: we provided online/offline/tensor-parallel/FSDP to suit your needs

## 🚀 Which training mode should I use?

We provide two orthogonal paths so everyone can start training in minutes, regardless of hardware budget — both are actively maintained, battle-tested daily in our CI, and guaranteed runnable out-of-the-box. Below is a comparison of the two methods.

| Method | Target Model | Disk Space Requirement | GPU Requirement | One-liner rationale |
| --- | --- | --- | --- | --- |
| Online | Used during training | Small | More GPUs are needed if your target model is large | Generating auxiliary hidden states on the fly |
| Offline | Only used during data preparation | Huge (e.g. ultrachat+sharegpt will need 12TB storage ) | as low as 1 GPU, as only need to accommodate the draft model  | Preparing auxiliary hidden states beforehand and only once |

> **Why does disk matter?**
> During Eagle3 training, the frozen target model will first generate the hidden states for each token given the data sample. The hidden states are fed to the draft model for training.
> Offline mode stores these hidden states to the local disk, so a small disk can be filled up fast.
> Online mode only generates these hidden states on the fly without storing them to the disk, but needs to keep the target model resident in memory during training, trading GPU RAM for almost-zero disk footprint.

## ⚡️ SGLang-ready

Whichever mode you pick, the checkpoint format is **byte-for-byte compatible** with [SGLang](https://github.com/sgl-project/sglang). There is no post-processing or weights manipulation required.

Happy training!
