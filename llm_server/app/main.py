import json

from app.const import EXP_ID, MLFLOW_TRACKING_URI, get_logger
from app.mlops import evaluate, track_llm
from app.prompts import debate_prompts
from fastapi import BackgroundTasks, FastAPI, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import mlflow

# from mlflow.metrics.genai

logger = get_logger(__name__)

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


class EvalItem(BaseModel):
    run_id: str
    question: str
    prediction: str


# for debug purpose
@app.post("/evalute/", status_code=201)
async def eval(item: EvalItem):
    await run_in_threadpool(
        evaluate,
        {
            "run_id": item.run_id,
            "question": item.question,
            "prediction": item.prediction,
        },
    )

    return {"msg": "evaluate registered"}


class TopicDebate(BaseModel):
    room_uuid: str
    topic: str
    debate: str


@app.post("/predict/")
async def predict_debate(item: TopicDebate, background_tasks: BackgroundTasks):
    outputs = {}
    tracks = None
    logger.info("###############PREDICT IS CALLED###############")
    with mlflow.start_run(run_name=item.room_uuid) as parent:
        logger.info("###############PARENT IS CALLED###############")

        # debate_prompts order
        tracks = [
            track_llm(
                p,
                item.room_uuid,
                item.topic,
                item.debate,
            )
            for p in debate_prompts
        ]
        coherence, rebut, persuasiveness, info = tracks

        outputs = {
            **coherence.to_response(),
            **rebut.to_response(),
            **persuasiveness.to_response(),
            **info.to_response(),
        }

    logger.info("###############REGISTER EVALUATE BACKGROUND###############")
    for track in tracks:
        background_tasks.add_task(evaluate, **track.to_eval())

    return Response(content=json.dumps(outputs), media_type="application/json")


if __name__ == "__main__":

    import uvicorn

    # MLFLOW_TRACKING_URI=http://mlflow-server:5000
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXP_ID)
    uvicorn.run(app, host="0.0.0.0", port=8000)
