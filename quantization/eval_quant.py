import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from awq import AutoAWQForCausalLM

def run(args):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
        legacy=True,
        use_fast=False,
    )
    # fuse_layers=True,
    # batch_size=args.batch,
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map='auto',
            torch_dtype=torch.float16,
            trust_remote_code=args.trust_remote_code,
        )
        .eval()
    )

    prompt = "count to 1000: 0 1 2 3"
    # prompt = "Hello! Do you hava a dream that "
    prompts = [prompt] * args.batch
    inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=False).to(
        "cuda:0"
    )
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=4096,
    )
    generate_tokens = tokenizer.batch_decode(output_ids)
    print(generate_tokens)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run AWQ quantized model")
    parser.add_argument("--model", type=str, help="The quantized name")
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="Trust remote code"
    )
    parser.add_argument("--batch", type=int, default=1)

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args()
    run(args)
