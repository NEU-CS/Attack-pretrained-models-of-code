"""
This example shows how to use the multi-LoRA functionality for offline inference.

Requires HuggingFace credentials for access to Llama2.
"""

from typing import Optional, List, Tuple

from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest
from datasets import load_dataset

def create_test_prompts(
        lora_path: str, codes
) -> List[Tuple[str, SamplingParams, Optional[LoRARequest]]]:
    """Create a list of test prompts with their sampling parameters.
    
    2 requests for base model, 4 requests for the LoRA. We define 2
    different LoRA adapters (using the same model for demo purposes).
    Since we also set `max_loras=1`, the expectation is that the requests
    with the second LoRA adapter will be ran after all requests with the
    first adapter have finished.
    """
    return [
        (prompt,
         SamplingParams(temperature=0.1,max_tokens= 512,stop_token_ids=[32022,32014],skip_special_tokens=False),
         LoRARequest("DeepSeek7bForCodeTrans",1,lora_path)) for prompt in codes
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams,
                                              Optional[LoRARequest]]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)


def initialize_engine() -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    engine_args = EngineArgs(model="deepseek-ai/deepseek-coder-6.7b-base",
                             enable_lora=True,
                             max_loras=1,
                             max_lora_rank=8,
                             max_cpu_loras=2,
                             max_num_seqs=256,
                             max_model_len= 512)
    return LLMEngine.from_engine_args(engine_args)


def main():
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine()
    lora_path = "DeepSeek7bForCodeTrans/checkpoint-6435"
    eval_dataset = load_dataset("json", data_files = "../../data/test_java2cs.jsonl")
    eval_dataset = eval_dataset['train']
    codes = []
    additional_special_tokens = {'additional_special_tokens':['<|begin_of_java_code|>','<|end_of_java_code|>'\
                                                           ,'<|begin_of_c-sharp_code|>','<|end_of_c-sharp_code|>',\
                                                            '<|translate|>']}
    for trans in eval_dataset['translation']:
        code = trans['java']
        code = additional_special_tokens['additional_special_tokens'][-1] + code + additional_special_tokens['additional_special_tokens'][3] + additional_special_tokens['additional_special_tokens'][2]
        codes.append(code)

    test_prompts = create_test_prompts(lora_path,codes)
    process_requests(engine, test_prompts)


if __name__ == '__main__':
    main()