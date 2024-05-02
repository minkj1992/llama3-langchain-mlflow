# Run

1. #Set Custom model

```sh
docker compose up -d

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
