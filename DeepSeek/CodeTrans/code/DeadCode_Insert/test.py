from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import torch 
from vllm import LLM, SamplingParams
from torch.utils.cpp_extension import CUDA_HOME
print(CUDA_HOME)


'''
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)

sampling_params = SamplingParams(
                    max_tokens=2048
                )
model = LLM("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, max_model_len=45120,quantization=)

model = LLM()
lora_weights = "../finetune/DeepSeek7bForCodeTrans"
if lora_weights != "":
    print("loading lora")
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )
    
    '''