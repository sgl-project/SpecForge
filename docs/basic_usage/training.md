## 🚀 Training

### 🏎️ Online Training

We have provided a simple startup script to train the Eagle3 model for the Llama 3 and 4, Qwen3 models. You can run the following command to start the training.

```bash
# make sure you have sharegpt data prepared
# train llama3-8B-instruct
bash ./examples/run_llama3_eagle3_online.sh

# train llama4-scout
bash ./examples/run_llama4_eagle3_online.sh

# train Qwen3-30B-A3B
# Qwen3-235B-A22B online training is also supported;
bash ./examples/run_qwen3_moe_eagle3_online.sh

# train Qwen3-8B
bash ./examples/run_qwen3_dense_eagle3_online.sh

# train Qwq-32B
bash ./examples/run_qwq_eagle3_online.sh
```

### 💨 Offline Training

We have provided a simple startup script to train the Eagle3 model for Llama-3.1-8B-Instruct model in an offline manner. You can run the following command to start the training. Almost Everything is the same as the Online Training Step, except that you don't need to configure anything about target model. Instead, you need to pass `--train-hidden-states-path` to the file.

```bash
# make sure you have sharegpt data prepared
bash ./examples/run_llama3_eagle3_offline.sh
```

### 📈 Experiment Tracking

This project supports logging training progress to Wandb, TensorBoard, and SwanLab. You can enable tracking by adding the --report-to argument to the command line in your shell script.
