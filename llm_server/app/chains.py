import os

from app.prompts import PromptDto
from langchain.callbacks import MlflowCallbackHandler
from langchain.chains.llm import LLMChain
from langchain_community.llms import Ollama

llm = Ollama(
    model="Meta-Llama-3-8B-Instruct.Q8_0.gguf",
    temperature=0,
    base_url=os.getenv("OLLAMA_URI"),
)


def get_chain(exp_id, room_uuid: str, run_id, p: PromptDto):
    callback_handler = MlflowCallbackHandler(
        experiment=exp_id,
        tags={"prompt_type": p.tag, "room_uuid": room_uuid},
        run_id=run_id,
    )

    return LLMChain(
        prompt=p.prompt,
        llm=llm,
        callbacks=[
            callback_handler,
        ],
    )
