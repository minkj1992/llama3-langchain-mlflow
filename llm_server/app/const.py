import logging
import os

EXP_ID = "debates-llama3"
EVAL_ENDPOINT_URI = "endpoints:/ollama"
# EVAL_ENDPOINT_URI = "http://mlflow-deployments:5000/gateway/ollama/invocations"

# # prod
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_DEPLOYMENTS_TARGET = os.getenv("MLFLOW_DEPLOYMENTS_TARGET")
OLLAMA_URI = os.getenv("OLLAMA_URI")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


# # local
# MLFLOW_TRACKING_URI = "http://0.0.0.0:5000/"
# OLLAMA_MODEL = "Meta-Llama-3-8B-Instruct.Q8_0.gguf"
# OLLAMA_URI = "http://localhost:11434"


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Default is stderr
        },
    },
    "loggers": {
        "": {  # root logger
            "level": "INFO",
            "handlers": ["default"],
            "propagate": False,
        },
        "uvicorn.error": {
            "level": "DEBUG",
            "handlers": ["default"],
        },
        "uvicorn.access": {
            "level": "DEBUG",
            "handlers": ["default"],
        },
    },
}


logging.config.dictConfig(LOGGING_CONFIG)


def get_logger(name):
    return logging.getLogger(name)
