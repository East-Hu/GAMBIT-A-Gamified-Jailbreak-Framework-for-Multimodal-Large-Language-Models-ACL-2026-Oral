import os, sys
# --- repo-root bootstrap so `config` and `gambit` import from any depth ---
_ROOT = os.path.dirname(os.path.abspath(__file__))
while _ROOT != os.path.dirname(_ROOT) and not os.path.isdir(os.path.join(_ROOT, 'gambit')):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------

import sys
import os
import json
import base64
import numpy as np
from tqdm import tqdm
from PIL import Image
import io

# Add current directory to path for imports if needed
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'run_api_and_open_new'))

# Import OpenAI and Llama Guard
try:
    from openai import OpenAI
    from gambit import judge as run_llama_guard
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# ===================== Client Setup =====================

print("Initializing client...", flush=True)
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

# You can switch the model here
# TARGET_MODEL = "gpt-4o-2024-11-20"
TARGET_MODEL = "gpt-4o-2024-11-20"

print(f"Target Model: {TARGET_MODEL}", flush=True)

def pil_image_to_base64(image, format='PNG') -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str

def call_model_base(image, prompt_text):
    b64_string = pil_image_to_base64(image)
    
    response = client.chat.completions.create(
        model=TARGET_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_string}"}}
                ]
            }
        ]
    )
    return response.choices[0].message.content.strip()

# ===================== Main Execution Logic =====================

if __name__ == "__main__":
    print("Starting Base Attack (Direct Query) Experiment...", flush=True)
    
    # Dataset Path
    dataset_path = 'dataset/break_jail_20250704/only_step0.npy'
    
    try:
        print(f"Loading data from {dataset_path}...", flush=True)
        data = np.load(dataset_path, allow_pickle=True).tolist()
        print(f"Data loaded. Groups: {len(data.keys())}", flush=True)
    except Exception as e:
        print(f"Error loading data: {e}", flush=True)
        exit(1)

    all_categories = list(data.keys())
    # User requested index 0 and 3 only (Self-Harm and Animal)
    categories = [all_categories[0], all_categories[3]]
    print(f"Selected categories: {categories}", flush=True)
    
    total_safe = 0
    total_unsafe = 0
    category_stats = {}

    # Output Directory
    output_dir = 'run_api_and_open_new/base_attack'
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, 'base_attack.log')
    
    print(f"Logging to {log_file_path}", flush=True)

    with open(log_file_path, 'a') as f_log:
        
        for cat in categories:
            print(f"\nProcessing Group: {cat}", flush=True)
            items = data[cat]
            cat_unsafe = 0
            cat_total = 0
            
            for id_item, item in enumerate(tqdm(items, desc=cat)):
                instruction = item['instruction']
                
                try:
                    # 1. Get Image
                    if 'image' in item:
                        img = item['image']
                    else:
                        continue
                    
                    # 2. Call Model
                    response = call_model_base(img, instruction)
                    
                    # Check for empty response
                    if not response or not response.strip():
                        safe_or_not = "N/A (Empty Response)"
                        status = "safe"
                    else:
                        # 3. Moderate only if response exists
                        safe_or_not = run_llama_guard.moderate(
                            [
                                {"role": "user", "content": instruction}, 
                                {"role": "assistant", "content": response},
                            ]
                        )
                        
                        # Check result
                        if 'unsafe' in safe_or_not:
                            status = "unsafe"
                        else:
                            status = "safe"

                    if status == "unsafe":
                        cat_unsafe += 1
                        total_unsafe += 1
                    else:
                        total_safe += 1
                    
                    cat_total += 1
                    
                    log_entry = {
                        'id': id_item,
                        'category': cat,
                        'instruction': instruction,
                        'model': TARGET_MODEL,
                        'response': response,
                        'safe_or_not': safe_or_not,
                        'status': status
                    }
                    
                    print(json.dumps(log_entry, ensure_ascii=False))
                    f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    f_log.flush()
                        
                except Exception as e:
                    print(f"Error processing {cat} id {id_item}: {e}")
            
            category_stats[cat] = {'unsafe': cat_unsafe, 'total': cat_total}
            print(f"Category {cat} finished. Unsafe count: {cat_unsafe}/{cat_total}")
            f_log.write(f"SUMMARY: Category {cat} - Unsafe: {cat_unsafe}/{cat_total}\n")

    print(f"\nExperiment Finished.")
    print(f"Total Processed: {total_safe + total_unsafe}")
    print(f"Total Unsafe: {total_unsafe}")
    
    print("\n--- Category Breakdown ---")
    for cat, stats in category_stats.items():
        u = stats['unsafe']
        t = stats['total']
        asr = (u/t*100) if t > 0 else 0.0
        print(f"Category: {cat:<15} | Unsafe: {u:<4} | Total: {t:<4} | ASR: {asr:.2f}%")
