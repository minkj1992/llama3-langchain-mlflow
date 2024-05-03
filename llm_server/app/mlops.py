from dataclasses import dataclass

import pandas as pd
from app.chains import get_chain
from app.const import EVAL_ENDPOINT_URI, EXP_ID, MLFLOW_DEPLOYMENTS_TARGET, get_logger
from app.prompts import PromptDto
from mlflow.deployments import set_deployments_target

import mlflow

logger = get_logger(__name__)


set_deployments_target(MLFLOW_DEPLOYMENTS_TARGET)


answer_relevance_metric = mlflow.metrics.genai.answer_relevance(model=EVAL_ENDPOINT_URI)
# faithfulness_metric = mlflow.metrics.genai.faithfulness(model=EVAL_ENDPOINT_URI)
answer_correctness_metric = mlflow.metrics.genai.answer_correctness(
    model=EVAL_ENDPOINT_URI
)


@dataclass
class TrackDto:
    run_id: str
    key: str
    question: str
    prediction: dict  # i.g. {'a_score': 85, 'a_reason': "Team A's arguments are generally coherent, but there are some minor inconsistencies. For example, they initially emphasize the importance of animal experimentation for medical research, but later acknowledge ethical concerns without fully addressing them.", 'b_score': 90, 'b_reason': 'Team B presents a clear and consistent argument against animal experimentation, highlighting both the ethical implications and the potential for alternative research methods. However, their points could be more effectively connected to strengthen their overall case.'}

    def to_response(self):
        return {
            self.key: {
                "question": self.question,
                "prediction": self.prediction,
                "run_id": self.run_id,
            }
        }

    def to_eval(self):
        return {
            "question": self.question,
            "prediction": self.prediction,
            "run_id": self.run_id,
        }


def track_llm(prompt_dto: PromptDto, room_uuid, topic, debate):
    with mlflow.start_run(run_name=str(prompt_dto.tag), nested=True) as child:
        # invoke
        chain = get_chain(EXP_ID, room_uuid, child.info.run_id, prompt_dto)

        logger.info("###############CHAIN INVOKED###############")
        output = chain.invoke({"topic": topic, "debate": debate})[chain.output_key]
        logger.info(output)

        result = TrackDto(
            run_id=child.info.run_id,
            key=prompt_dto.tag,
            question=prompt_dto.prompt.format(topic=topic, debate=debate),
            prediction=output,
        )
        return result


# https://learn.microsoft.com/ko-kr/azure/databricks/mlflow/llm-evaluate
def evaluate(run_id: str, question: str, prediction: dict):
    with mlflow.start_run(run_id=run_id):
        logger.info(
            f"############################## Background evaluate start ##########################"
        )
        logger.info(f"run_id: {run_id}")
        logger.info(f"question: {question}")
        logger.info(f"prediction: {prediction}")

        result = mlflow.evaluate(
            model_type="question-answering",
            data=pd.DataFrame(
                {
                    "inputs": [question],
                    "prediction": [str(prediction)],
                }
            ),
            feature_names=[
                "inputs",
            ],
            targets="prediction",
            predictions="prediction",
            extra_metrics=[
                answer_relevance_metric,
                answer_correctness_metric,
            ],
        )
        logger.info(f"result: {result}")
        logger.info("############### Background evaluate end ###############")
