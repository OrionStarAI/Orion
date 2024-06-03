import os
import json
import codecs
import torch

from transformers import AutoTokenizer, AutoModel

from awq.models.auto import AWQ_CAUSAL_LM_MODEL_MAP, AutoAWQForCausalLM
from awq.models.base import TRANSFORMERS_AUTO_MAPPING_DICT

from orion import OrionAWQForCausalLM

import pdb

TRANSFORMERS_AUTO_MAPPING_DICT["orion"] = "AutoModelForCausalLM"
AWQ_CAUSAL_LM_MODEL_MAP['orion'] = OrionAWQForCausalLM

def load(path:str=None, key:str="text"):
    if path is None:
        path = "data/val.jsonl"
    texts = []
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line)[key]
            except Exception as e:
                print('exception ', e)
                continue
            texts += [text]
            if len(texts) >= 128:
                break
    return texts

def load_wikitext():
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    return [text for text in data["text"] if text.strip() != '' and len(text.split(' ')) > 20]

# def quant(group_size:int, version:str):
def quant(args):

    group_size = args.group_size
    version = args.version
    model_path = args.model_path
    save_path = args.save_path
    bits = 4

    # model_path = '/data/xp/chat/quant_orion/models/final/OrionStar-JpKr-650k-0112-ck19800'
    # model_path = './models'
    # save_path = f'./outputs/final/gs{group_size}_version{version}'

    os.makedirs(save_path, exist_ok=True)

    assert version in ["gemm", "gemv"]
    assert group_size in [32, 64, 128]

    quant_config = { "zero_point": True, "q_group_size": group_size, "w_bit": bits, "version": version.upper() }
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = OrionAWQForCausalLM.from_pretrained(model_path,
                                            model_type="orion",
                                            device_map="auto",
                                            trust_remote_code=True,
                                            torch_dtype=torch.float16)
    calib_data = load()

    # Quantize
    # model.quantize(tokenizer, quant_config=quant_config, calib_data=load_wikitext())
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

    # Save quantized model
    model.save_quantized(save_path)
    tokenizer.save_pretrained(save_path)

    print(f'Model is quantized and saved at "{save_path}"')

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Quantized model")
    parser.add_argument("--model_path", type=str, help="The original model path")
    parser.add_argument("--save_path", type=str, help="The quantized name")
    parser.add_argument("--group_size", type=int, default=128, help="The quantized name")
    parser.add_argument("--version", type=str, default="gemm", help="The quantized name")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    quant(args)
