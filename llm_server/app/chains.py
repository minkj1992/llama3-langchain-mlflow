import os

from app.custom_parser import MY_PARSER
from app.prompts import PromptDto
from langchain.callbacks import MlflowCallbackHandler
from langchain.chains.llm import LLMChain
from langchain_community.llms import Ollama

llm = Ollama(
    model=os.getenv("OLLAMA_MODEL"),
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
        output_parser=MY_PARSER,
    )
