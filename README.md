# Run

#### GPU Support (Optional)

If you have a GPU and want to leverage its power within a Docker container, follow these steps to install the NVIDIA Container Toolkit:

```sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure NVIDIA Container Toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU integration
docker run --gpus all nvidia/cuda:11.5.2-base-ubuntu20.04 nvidia-smi
```


> FYI, mac cannot use gpu so you have to use docker compose -f docker-compose-mac.yaml [Further information](https://chariotsolutions.com/blog/post/apple-silicon-gpus-docker-and-ollama-pick-two/)


```sh
# if gpu supported
docker compose up -d

# local mac
ollama run llama3
docker compose -f docker-compose-mac.yaml up
```



# LLama3 download 
## 1. Simple way

#### 1-1. Download ollama

- https://ollama.com/

#### 1-2. run llama3

```
ollama run llama3
```

#### (opt) docker

```
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama run llama3
```

## 2. Custom model
> Install manual model

#### 2-1. Download ollama

- https://ollama.com/

#### 2-2. Download model
> https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q8_0.gguf

```sh
$ pip install huggingface-hub
$ mkdir Meta-Llama-3-8B-Instruct-GGUF && cd Meta-Llama-3-8B-Instruct-GGUF
$ export CUR_DIR=$(pwd)
$ huggingface-cli download \
  QuantFactory/Meta-Llama-3-8B-Instruct-GGUF \
  Meta-Llama-3-8B-Instruct.Q8_0.gguf \
  --local-dir $CUR_DIR \
  --local-dir-use-symlinks False
```


#### 2-3. Modelfile

```
FROM Meta-Llama-3-8B-Instruct.Q8_0.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
PARAMETER num_keep 24
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
```


#### 2-4. Create Ollama model

```sh
> ollama create Meta-Llama-3-8B-Instruct.Q8_0.gguf -f Modelfile

# check
> ollama list | grep -i Meta
Meta-Llama-3-8B-Instruct.Q8_0.gguf:latest  2670b8c8ba2c    8.5 GB  About a minute ago

# run
> ollama run Meta-Llama-3-8B-Instruct.Q8_0.gguf:latest
```





## Why Should I need deployment

```
llama3-langchain-mlflow-app-1            | Run `python -m spacy download en_core_web_sm` to download en_core_web_sm model for text visualization.
llama3-langchain-mlflow-app-1            | /usr/local/lib/python3.11/site-packages/pydantic/_internal/_config.py:334: UserWarning: Valid config keys have changed in V2:
llama3-langchain-mlflow-app-1            | * 'schema_extra' has been renamed to 'json_schema_extra'
llama3-langchain-mlflow-app-1            |   warnings.warn(message, UserWarning)
llama3-langchain-mlflow-app-1            | INFO:     192.168.208.1:59398 - "POST /predict/ HTTP/1.1" 500 Internal Server Error
llama3-langchain-mlflow-app-1            | ERROR:    Exception in ASGI application
llama3-langchain-mlflow-app-1            | Traceback (most recent call last):
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/uvicorn/protocols/http/h11_impl.py", line 407, in run_asgi
llama3-langchain-mlflow-app-1            |     result = await app(  # type: ignore[func-returns-value]
llama3-langchain-mlflow-app-1            |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/uvicorn/middleware/proxy_headers.py", line 69, in __call__
llama3-langchain-mlflow-app-1            |     return await self.app(scope, receive, send)
llama3-langchain-mlflow-app-1            |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/fastapi/applications.py", line 1054, in __call__
llama3-langchain-mlflow-app-1            |     await super().__call__(scope, receive, send)
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/starlette/applications.py", line 123, in __call__
llama3-langchain-mlflow-app-1            |     await self.middleware_stack(scope, receive, send)
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/starlette/middleware/errors.py", line 186, in __call__
llama3-langchain-mlflow-app-1            |     raise exc
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/starlette/middleware/errors.py", line 164, in __call__
llama3-langchain-mlflow-app-1            |     await self.app(scope, receive, _send)
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/starlette/middleware/cors.py", line 93, in __call__
llama3-langchain-mlflow-app-1            |     await self.simple_response(scope, receive, send, request_headers=headers)
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/starlette/middleware/cors.py", line 148, in simple_response
llama3-langchain-mlflow-app-1            |     await self.app(scope, receive, send)
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/starlette/middleware/exceptions.py", line 65, in __call__
llama3-langchain-mlflow-app-1            |     await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/starlette/_exception_handler.py", line 64, in wrapped_app
llama3-langchain-mlflow-app-1            |     raise exc
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
llama3-langchain-mlflow-app-1            |     await app(scope, receive, sender)
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/starlette/routing.py", line 756, in __call__
llama3-langchain-mlflow-app-1            |     await self.middleware_stack(scope, receive, send)
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/starlette/routing.py", line 776, in app
llama3-langchain-mlflow-app-1            |     await route.handle(scope, receive, send)
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/starlette/routing.py", line 297, in handle
llama3-langchain-mlflow-app-1            |     await self.app(scope, receive, send)
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/starlette/routing.py", line 77, in app
llama3-langchain-mlflow-app-1            |     await wrap_app_handling_exceptions(app, request)(scope, receive, send)
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/starlette/_exception_handler.py", line 64, in wrapped_app
llama3-langchain-mlflow-app-1            |     raise exc
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
llama3-langchain-mlflow-app-1            |     await app(scope, receive, sender)
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/starlette/routing.py", line 72, in app
llama3-langchain-mlflow-app-1            |     response = await func(request)
llama3-langchain-mlflow-app-1            |                ^^^^^^^^^^^^^^^^^^^
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/fastapi/routing.py", line 278, in app
llama3-langchain-mlflow-app-1            |     raw_response = await run_endpoint_function(
llama3-langchain-mlflow-app-1            |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/fastapi/routing.py", line 191, in run_endpoint_function
llama3-langchain-mlflow-app-1            |     return await dependant.call(**values)
llama3-langchain-mlflow-app-1            |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
llama3-langchain-mlflow-app-1            |   File "/app/app/main.py", line 55, in predict_debate
llama3-langchain-mlflow-app-1            |     mlflow.evaluate(
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/mlflow/models/evaluation/base.py", line 1979, in evaluate
llama3-langchain-mlflow-app-1            |     model = _load_model_or_server(model, env_manager, model_config)
llama3-langchain-mlflow-app-1            |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/mlflow/pyfunc/__init__.py", line 979, in _load_model_or_server
llama3-langchain-mlflow-app-1            |     return load_model(model_uri, model_config=model_config)
llama3-langchain-mlflow-app-1            |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/mlflow/pyfunc/__init__.py", line 864, in load_model
llama3-langchain-mlflow-app-1            |     local_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
llama3-langchain-mlflow-app-1            |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/mlflow/tracking/artifact_utils.py", line 106, in _download_artifact_from_uri
llama3-langchain-mlflow-app-1            |     return get_artifact_repository(artifact_uri=root_uri).download_artifacts(
llama3-langchain-mlflow-app-1            |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/mlflow/store/artifact/artifact_repo.py", line 198, in download_artifacts
llama3-langchain-mlflow-app-1            |     if self._is_directory(artifact_path):
llama3-langchain-mlflow-app-1            |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/mlflow/store/artifact/artifact_repo.py", line 97, in _is_directory
llama3-langchain-mlflow-app-1            |     listing = self.list_artifacts(artifact_path)
llama3-langchain-mlflow-app-1            |               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
llama3-langchain-mlflow-app-1            |   File "/usr/local/lib/python3.11/site-packages/mlflow/store/artifact/http_artifact_repo.py", line 84, in list_artifacts
llama3-langchain-mlflow-app-1            |     url, tail = self.artifact_uri.split(endpoint, maxsplit=1)
llama3-langchain-mlflow-app-1            |     ^^^^^^^^^
llama3-langchain-mlflow-app-1            | ValueError: not enough values to unpack (expected 2, got 1)
```