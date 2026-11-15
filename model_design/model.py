import dspy
from llama_cpp import Llama

# Direct local GGUF model initialization
llm_native = Llama(
    model_path="./models/Qwen3-4B-Q4_K_M.gguf",
    n_ctx=4096,
    verbose=False
)

class LocalLM(dspy.LM):
    def __init__(self, model_instance):
        super().__init__(model="local/qwen3")
        self.llm = model_instance

    def __call__(self, prompt=None, messages=None, **kwargs):
        if messages:
            input_text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
        else:
            input_text = prompt or ""
        response = self.llm(
            input_text, 
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.1)
        )
        return [response["choices"][0]["text"]]

def init_dspy_model():
    local_lm = LocalLM(llm_native)
    dspy.configure(lm=local_lm)
    return local_lm