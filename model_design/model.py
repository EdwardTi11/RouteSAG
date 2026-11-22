import dspy
from dspy.lm15 import Message, Request, Response, Usage
from llama_cpp import Llama

llm = Llama(
    model_path="./models/Qwen3-4B-Q4_K_M.gguf",
    n_ctx=4096,
    verbose=False
)

class LlamaCppEngine:
    def __init__(self, llm):
        self.llm = llm

    def complete(self, request: Request) -> Response:
        messages = ([{"role": "system", "content": request.system}] if request.system else [])
        messages += [{"role": m.role, "content": m.text} for m in request.messages]

        result = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=request.config.max_tokens or 512,
            temperature=0.1 if request.config.temperature is None else request.config.temperature
        )

        usage = result.get("usage") or {}

        return Response(
            id=result["id"],
            model=request.model,
            message=Message.assistant(result["choices"][0]["message"]["content"]),
            finish_reason=result["choices"][0]["finish_reason"],
            usage=Usage(
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens")
            )
        )

def init_dspy_model():
    lm = dspy.LM(
        model="local/qwen3",
        engine=LlamaCppEngine(llm)
    )
    dspy.configure(lm=lm, adapter=dspy.XMLAdapter())
    return lm