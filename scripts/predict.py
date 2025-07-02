import argparse
from pathlib import Path
from filelock import FileLock
import json

def update_json_field(file_path, field, new_value):
    lock_path = str(file_path) + '.lock'
    lock = FileLock(lock_path)

    with lock:  # 自动加锁，使用完自动释放
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            data[field] = new_value
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    item[field] = new_value
        else:
            print("不支持的 JSON 数据格式")
            return

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def count_labels(jsonl_path):
    safe_count = 0
    unsafe_count = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                predict = data.get('predict', '')
                if predict.startswith('safe'):
                    safe_count += 1
                elif predict.startswith('unsafe'):
                    unsafe_count += 1
            except json.JSONDecodeError:
                print("Warning: Skipping invalid JSON line")

    print(f"Safe count: {safe_count}")
    print(f"Unsafe count: {unsafe_count}")

if __name__ == "__main__":
    from scripts.vllm_infer import vllm_infer

    parser = argparse.ArgumentParser(description="Run inference with vLLM.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model or model name.")
    parser.add_argument("--adapter_name_or_path", type=str, default=None, help="Path to the adapter model.")
    parser.add_argument("--template", type=str, default="default", help="Template to use for generation.")
    parser.add_argument("--dataset", type=str, default="directharm4", help="Name of the dataset to use.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty for generation.")

    args = parser.parse_args()
    save_name = f"data/{args.dataset}_ans/{args.model_name_or_path.split('/')[-1]}"
    if args.adapter_name_or_path:
        save_name += f"_{args.adapter_name_or_path.split('/')[-1]}"
    save_name += f".jsonl"

    vllm_infer(
        model_name_or_path=args.model_name_or_path,
        adapter_name_or_path=args.adapter_name_or_path,
        template=args.template,
        dataset=args.dataset,
        repetition_penalty=args.repetition_penalty,
        save_name=save_name
    )

    input_file = Path(save_name)
    output_file = input_file.parent / (input_file.name.replace(".jsonl", "_ans.jsonl"))
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line.strip())
            new_data = {
                'instruction': data['predict'],
                'input': "",
                'output': "",
            }
            outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
    print(f"Processed data saved to {output_file}")

    dataset = f"{args.dataset}-{args.model_name_or_path.split('/')[-1]}"
    update_json_field("data/dataset_info.json", dataset, {"file_name": str(output_file).replace("data/", "")})

    if args.adapter_name_or_path:
        save_name = f"saves/{args.model_name_or_path.split('/')[-1]}-{args.dataset}/{args.adapter_name_or_path.split('/')[-1]}.jsonl"
    else:
        save_name = f"saves/{args.model_name_or_path.split('/')[-1]}-{args.dataset}.jsonl"
    vllm_infer(
        model_name_or_path="/home/lizijian/Models/Llama-Guard-3-8B",
        template="llama3_safetycheck",
        dataset=dataset,
        save_name=save_name,
    )
    count_labels(save_name)