import json

import pandas as pd
from app.chains import get_chain
from app.custom_parser import parse
from app.prompts import debate_prompts
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import mlflow

app = FastAPI()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class TopicDebate(BaseModel):
    room_uuid: str
    topic: str
    debate: str


EXP_ID = "debates-hf"


@app.post("/predict/")
async def predict_debate(item: TopicDebate):
    outputs = {}
    predictions = []
    questions = []
    with mlflow.start_run(run_name=item.room_uuid) as parent:
        for p in debate_prompts:
            prompt_category = str(p.tag)
            with mlflow.start_run(run_name=prompt_category, nested=True) as child:
                # invoke
                chain = get_chain(EXP_ID, item.room_uuid, child.info.run_id, p)
                outputs[p.tag] = parse(
                    chain.invoke({"topic": item.topic, "debate": item.debate})
                )

            questions.append(p.prompt.format(topic=item.topic, debate=item.debate))
            predictions.append(outputs[p.tag])

        mlflow.evaluate(
            model=f'{os.getenv("OLLAMA_URI")}/v1'
            model_type="question-answering",
            data=pd.DataFrame(
                {
                    "questions": questions,
                    "predictions": predictions,
                    # "answer": TODO chatgpt
                }
            ),
            feature_names=[
                "questions",
            ],
            predictions="predictions",
        )

    return Response(content=json.dumps(outputs), media_type="application/json")


if __name__ == "__main__":
    import os

    import uvicorn

    # MLFLOW_DEPLOYMENTS_TARGET=http://mlflow-deployments:5000
    # MLFLOW_TRACKING_URI=http://mlflow-server:5000
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    uvicorn.run(app, host="0.0.0.0", port=8000)
