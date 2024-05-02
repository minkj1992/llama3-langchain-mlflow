from typing import Any, Dict

from app.llm import CustomLLMResult
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import Generation

MY_PARSER = JsonOutputParser(pydantic_object=CustomLLMResult)

_output_key = "text"  # langchain.chains.llm.LLMChain.output_key


# mlflow cannot pickle json parser
# def parse(llm_result: Dict[str, Any]) -> Dict[str, Any]:
#     return MY_PARSER.parse_result([Generation(text=llm_result[_output_key])])


__all__ = [
    "MY_PARSER",
]
