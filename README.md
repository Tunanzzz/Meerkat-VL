# Meerkat-VL

## Quick Start  

### 1. Download pretrained model from huggingface
```bash
# Download Qwen2-VL-7B-Instruct
hfi download Qwen/Qwen2-VL-7B-Instruct --repo-type model

# Download Qwen2.5-VL-7B-Instruct
hfi download Qwen/Qwen2.5-VL-7B-Instruct --repo-type model
```
### 2. Cold Start
```bash
bash zmy_script\cold_start.sh
```

### 2. Run RL Training
```bash
bash zmy_script\train.sh
```

---

## Eval
### 1. Model Deployment
```bash
bash eval\vllm_server.sh
```

### 2. Run Model Inference
Make sure the `port` matches the deployed service:
```bash
bash eval/sample_response.sh
```

### 3. Evaluation with GPT
Load your API key:
```bash
export OPENAI_API_KEY="your key"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```
Run the evaluation script:

```bash
bash SaFeR-VLM/eval/cal_result_by_gpt.sh
```

### ⚠️ Note  

If Hugging Face is blocked in your region, set the mirror before running:  

```bash
export HF_ENDPOINT=https://hf-mirror.com
```