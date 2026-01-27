import json
import os
import base64
import argparse
from io import BytesIO
from PIL import Image
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

BENCHMARK_BASE = "xxxxxxx"
IMAGE_ROOT = os.path.join(BENCHMARK_BASE, "images")

DATASET_CONFIG = {
    "siuo": {
        "input_file": os.path.join(BENCHMARK_BASE, "siuo.json"),
        "image_root": IMAGE_ROOT,
        "format": "json",
        "img_key": "image"
    },
    "mssbench": {
        "input_file": os.path.join(BENCHMARK_BASE, "mssbench.json"),
        "image_root": IMAGE_ROOT,
        "format": "json",
        "img_key": "unsafe_image_path"
    },
    "mm-safetybench": {
        "input_file": os.path.join(BENCHMARK_BASE, "mm-safetybench.jsonl"),
        "image_root": IMAGE_ROOT,
        "format": "jsonl",
        "img_key": "image_path"
    },
    "beavertails": {
        "input_file": os.path.join(BENCHMARK_BASE, "beavertails.jsonl"),
        "image_root": IMAGE_ROOT,
        "format": "jsonl",
        "img_key": "image"
    },
    "spa-vl": {
        "input_file": os.path.join(BENCHMARK_BASE, "spa-vl.jsonl"),
        "image_root": IMAGE_ROOT,
        "format": "jsonl",
        "img_key": "image"
    }
}

# ================= 2. 图像处理工具 =================
def get_task_id(item_data):
    """
    辅助函数：为每一条数据生成一个唯一标识 (Question + Image Filename)。
    用于对比 Input 任务和 Output 结果是否一致。
    """
    q = item_data.get('question', '')
    
    img_path = item_data.get('image_path') or item_data.get('image') or item_data.get('unsafe_image_path')
    
    if not img_path and 'meta' in item_data:
        meta = item_data['meta']
        img_path = meta.get('image') or meta.get('unsafe_image_path') or meta.get('image_path')

    img_name = os.path.basename(str(img_path)) if img_path else "no_image"
    
    return f"{q}_{img_name}"

def resize_and_encode_image(image_path, max_size=672):
    """读取图片 -> 缩放 -> 转 Base64。失败返回 None。"""
    if not image_path or not os.path.exists(image_path):
        return None

    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail((max_size, max_size))
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        return None


class DatasetHandler:
    @staticmethod
    def load_data(dataset_name):
        """
        根据 dataset_name 读取对应的 Config，加载数据并标准化
        """
        if dataset_name not in DATASET_CONFIG:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = DATASET_CONFIG[dataset_name]
        input_file = config["input_file"]
        base_img_root = config["image_root"]
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        data_list = []
        raw_items = []
        config = DATASET_CONFIG[dataset_name]
        input_file = config["input_file"]
        base_img_root = config["image_root"] 
        img_key = config["img_key"]
        
        if input_file.endswith('.jsonl'):
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        raw_items.append(json.loads(line))
        else:
            with open(input_file, 'r', encoding='utf-8') as f:
                raw_items = json.load(f)

        data_list = []
        for item in raw_items:
            rel_path = item.get(img_key, "")
            standard_item = {
                "meta": item,
                "image_path": os.path.join(base_img_root, rel_path) if rel_path else None,
                "question": item.get('question') or (item.get('queries')[0] if item.get('queries') else "")
            }
            data_list.append(standard_item)
            
        return data_list

    @staticmethod
    def format_output(dataset_name, result_item, response_text):
        """
        根据不同数据集的要求，筛选保存字段
        """
        meta = result_item['meta']
        
        if dataset_name == "mssbench":
            return {
                "question": result_item['question'],
                "image": meta.get('unsafe_image_path'),
                "intent": meta.get('intent'),
                "Type": meta.get('Type'),
                "response": response_text
            }
            
        elif dataset_name == "beavertails":
            return {
                "question": meta.get('question'),
                "image": meta.get('image'),
                "category": meta.get('category'),
                "image_severity": meta.get('image_severity'),
                "response": response_text
            }
            
        else:
            output = meta.copy()
            output['response'] = response_text 
            return output


def process_single_task(std_item, client, model_name, dataset_name, system_prompt=None):
    """
    接收标准化后的 item (含 path, question), 执行推理
    增加了 system_prompt 参数
    """
    image_path = std_item['image_path']
    question = std_item['question']
    
    image_url = resize_and_encode_image(image_path)
    
    if image_url is None:
        return DatasetHandler.format_output(dataset_name, std_item, "ERROR: Image processing failed or file missing.")

    messages = []

    if system_prompt:
        messages.append({
            "role": "system", 
            "content": system_prompt
        })

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    })

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=2048
        )
        ans = response.choices[0].message.content
    except Exception as e:
        ans = f"ERROR: API request failed: {str(e)}"
    
    return DatasetHandler.format_output(dataset_name, std_item, ans)

# ================= 5. 主函数 =================
def main():
    parser = argparse.ArgumentParser(description="Batch Multi-Dataset Inference")
    parser.add_argument("--datasets", nargs='+', required=True, 
                        choices=DATASET_CONFIG.keys(),
                        help="List of datasets to run (e.g. siuo mssbench)")
    
    parser.add_argument("--model_name", type=str, required=True, help="Model name (used for output directory)")
    parser.add_argument("--api_base", type=str, default="http://localhost:9600/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--system_prompt", type=str, default=None, help="Optional system prompt content")

    args = parser.parse_args()
    
    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    output_dir = os.path.join("/home/zhoupc/safe_alignment/sample_and_verify/benchmark_response", "responses", args.model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for ds_name in args.datasets:
        print(f"\n{'='*20} Processing: {ds_name} {'='*20}")
        
        try:
            tasks = DatasetHandler.load_data(ds_name)
            total_tasks = len(tasks)
            print(f"Loaded {total_tasks} items from {DATASET_CONFIG[ds_name]['input_file']}")
        except Exception as e:
            print(f"Skipping {ds_name} due to error: {e}")
            continue

        output_file = os.path.join(output_dir, f"{ds_name}_response.jsonl")
        
        processed_ids = set()
        if os.path.exists(output_file):
            print(f"Checking existing progress in: {output_file}")
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        res = json.loads(line)
                        processed_ids.add(get_task_id(res))
                    except json.JSONDecodeError:
                        continue
        
        tasks_to_run = [t for t in tasks if get_task_id(t) not in processed_ids]
        
        num_skipped = total_tasks - len(tasks_to_run)
        if num_skipped > 0:
            print(f"--> Skipped {num_skipped} already processed items.")
        
        if not tasks_to_run:
            print(f"--> All items for {ds_name} are already completed. Moving to next dataset.")
            continue

        print(f"--> Remaining tasks to run: {len(tasks_to_run)}")

        if args.system_prompt:
            print(f"Using System Prompt: {args.system_prompt}")
        
        with open(output_file, 'a', encoding='utf-8') as f_out:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                future_map = {
                    executor.submit(process_single_task, item, client, args.model_name, ds_name, args.system_prompt): item 
                    for item in tasks_to_run
                }
                
                for future in tqdm(as_completed(future_map), total=len(tasks_to_run), desc=f"Running {ds_name}"):
                    result = future.result()
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
        
        print(f"Finished {ds_name}.")

    print(f"\nAll requested datasets completed. Check folder: ./{output_dir}/")
    
if __name__ == "__main__":
    main()