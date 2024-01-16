import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

def parse_inputs():
    parser = argparse.ArgumentParser(description="Orion-14B-Chat text generation demo")
    parser.add_argument(
        "--model",
        type=str,
        default="OrionStarAI/Orion-14B-Chat",
        help="pretrained model path locally or name on huggingface",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="OrionStarAI/Orion-14B-Chat",
        help="tokenizer path locally or name on huggingface",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="你好!",
        help="The prompt to start with",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="whether to enable streaming text generation",
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default="</s>",
        help="End of sentence token",
    )
    args = parser.parse_args()
    return args


def main(args):
    print(args)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True, use_safetensors=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    messages = [{"role": "user", "content": args.prompt}]
    if args.streaming:
        position = 0
        try:
            for response in model.chat(tokenizer, messages, streaming=True):
                print(response[position:], end='', flush=True)
                position = len(response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        except KeyboardInterrupt:
            pass
    else:
        response = model.chat(tokenizer, messages)
        print(response)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    args = parse_inputs()
    main(args)
