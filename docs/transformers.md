# Hugging Face Transformers

> State-of-the-art pretrained models for text, vision, audio, and multimodal tasks, with one consistent API.

Hugging Face Transformers is the de facto library for downloading, running, fine-tuning, and sharing pretrained transformer models. It provides model definitions for hundreds of architectures (BERT, GPT-2, Llama, Mistral, Qwen, Gemma, T5, Whisper, ViT, CLIP, LLaVA, and many more), matching tokenizers and processors, the high-level `pipeline` API for inference, the `generate` API for text generation, and the `Trainer` for training, all integrated with the Hugging Face Hub.

Covers Transformers 4.4x through 4.5x (PyTorch backend), with notes on Transformers v5. Notable changes called out on this page: `eval_strategy` replaced `evaluation_strategy` in `TrainingArguments` (4.41; the old name was removed later), `processing_class` replaced the `tokenizer` argument of `Trainer` (4.46), `dtype` replaced `torch_dtype` in `from_pretrained` (4.56; the old name still works with a deprecation warning), quantization flags moved into `quantization_config` objects, `attn_implementation="sdpa"` became the default attention, chat templates (`apply_chat_template`) are the standard way to prompt chat models, and TensorFlow/Flax model classes are deprecated and removed in v5.

## Overview

### What it is

Transformers bundles three things for each supported architecture:

- **Configuration** (`PretrainedConfig` subclasses): hyperparameters such as hidden size, number of layers, vocabulary size.
- **Model** (`PreTrainedModel` subclasses): PyTorch `nn.Module`s, with task-specific heads (`...ForCausalLM`, `...ForSequenceClassification`, `...ForTokenClassification`, `...ForQuestionAnswering`, ...).
- **Preprocessor**: a tokenizer (text), image processor (vision), feature extractor (audio), or processor (multimodal, combining several).

On top of this sit:

- **`pipeline()`**: one-line inference for dozens of tasks.
- **Auto classes** (`AutoModel...`, `AutoTokenizer`, `AutoProcessor`): pick the right class from a checkpoint's `config.json`.
- **`generate()`** and `GenerationConfig`: decoding strategies (greedy, sampling, beam search, assisted decoding), streaming, caching.
- **`Trainer`** and `TrainingArguments`: a full training loop with mixed precision, gradient accumulation, distributed training (via Accelerate), checkpointing, evaluation, logging, and Hub upload.
- **Integrations**: quantization (bitsandbytes, GPTQ, AWQ, torchao, HQQ, and more), PEFT adapters (LoRA), FlashAttention, DeepSpeed, FSDP.

### History and maintainers

The library started in 2018 as `pytorch-pretrained-bert`, a PyTorch port of Google's BERT by Hugging Face, became `pytorch-transformers` in 2019, and then `transformers` with TensorFlow support later that year. The EMNLP 2020 paper "Transformers: State-of-the-Art Natural Language Processing" describes its design. It is maintained by Hugging Face with a very large open-source contributor base, and model authors (Meta, Google, Mistral, Alibaba Qwen, Microsoft, and others) often contribute their architectures on release day. Transformers v5 refocuses the library on PyTorch as the single backend and on being the reference "model definition" layer that other engines (vLLM, SGLang, TGI, llama.cpp, MLX) build on.

### When to use it

- You want to use or fine-tune a pretrained model from the Hugging Face Hub with minimal code.
- Classification, NER, question answering, summarization, translation, embeddings, speech recognition, image classification, vision-language tasks.
- LLM fine-tuning (full, LoRA, QLoRA) combined with PEFT and TRL.
- Research prototyping where you need the reference implementation of an architecture.

### When not to use it

- **High-throughput LLM serving**: use a dedicated engine (vLLM, SGLang, TGI, TensorRT-LLM) that implements continuous batching and paged attention. Transformers' `generate` is great for experiments and moderate loads.
- **Classical ML** on tabular data: use scikit-learn or gradient-boosted trees.
- **Tiny, custom architectures** you design from scratch: plain PyTorch is simpler than subclassing `PreTrainedModel`.
- **On-device inference**: export to ONNX/ExecuTorch, or use llama.cpp/MLX/Core ML conversions.

### Where it fits in the ML stack

```text
Applications: chatbots, RAG, classifiers, agents
        |
Training / alignment:  TRL (SFT, DPO, GRPO), PEFT (LoRA), Accelerate, DeepSpeed
        |
Transformers:  pipeline, Auto classes, tokenizers/processors, generate, Trainer
        |
Data / Hub:  datasets, tokenizers (Rust), safetensors, huggingface_hub
        |
Framework:  PyTorch (primary; TF and Flax legacy)
        |
Hardware:  NVIDIA / AMD GPUs, Apple MPS, Intel, CPU, TPU (via PyTorch/XLA)
```

## Installation

### pip

```bash
# Install PyTorch first, matching your hardware (see pytorch.org), then:
pip install -U transformers

# Common extras
pip install -U "transformers[torch]"          # pulls in torch and accelerate
pip install -U datasets evaluate accelerate    # data, metrics, distributed/mixed precision
pip install -U peft trl                         # adapters and LLM fine-tuning
pip install -U bitsandbytes                     # 4/8-bit quantization (NVIDIA GPUs; other backends experimental)
pip install -U sentencepiece protobuf           # needed by some older tokenizers (T5, some Llama variants)

# Latest development version
pip install -U git+https://github.com/huggingface/transformers
```

### conda

```bash
conda create -n hf python=3.11 -y
conda activate hf
pip install torch --index-url https://download.pytorch.org/whl/cu124
conda install -c conda-forge transformers datasets accelerate
```

### GPU, ROCm, Apple Silicon, TPU

Transformers runs on whatever devices PyTorch supports. Install the CUDA or ROCm build of PyTorch, and models move to the GPU with `.to("cuda")` or `device_map="auto"`. On Apple Silicon use `device="mps"`. For TPUs, use PyTorch/XLA (`torch_xla`) with Accelerate. `flash-attn` (`pip install flash-attn --no-build-isolation`) enables `attn_implementation="flash_attention_2"` on supported NVIDIA GPUs.

### Authentication for gated/private models and uploads

```bash
hf auth login                  # newer huggingface_hub CLI
huggingface-cli login          # older CLI name, still common
# or set an environment variable
export HF_TOKEN=hf_xxx
```

### Verifying the install

```python
import torch
import transformers

print(transformers.__version__)
print(torch.__version__, torch.cuda.is_available())

from transformers import pipeline
clf = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
print(clf("Transformers is installed correctly!"))
# [{'label': 'POSITIVE', 'score': 0.99...}]
```

```bash
transformers env               # prints versions of transformers, torch, accelerate, platform (older: transformers-cli env)
```

### Cache and environment variables

| Variable | Effect |
|---|---|
| `HF_HOME` | Root of the Hugging Face cache (default `~/.cache/huggingface`). |
| `HF_HUB_CACHE` | Where model/dataset repos are cached. |
| `HF_TOKEN` | Access token for gated/private repos and uploads. |
| `HF_HUB_OFFLINE=1` | Never hit the network; use cached files only. |
| `HF_HUB_ENABLE_HF_TRANSFER=1` | Faster downloads with the `hf_transfer` package. |
| `TRANSFORMERS_VERBOSITY=error` | Reduce log noise. |
| `CUDA_VISIBLE_DEVICES=0,1` | Restrict visible GPUs. |

## Core Concepts

### 1. Checkpoints, configs, and the Hub

A checkpoint is a Hub repository (or local folder) containing `config.json`, weights (`model.safetensors`, possibly sharded with an index file), tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`), and optionally `generation_config.json` and a chat template. `from_pretrained("org/name")` downloads and caches these files, then builds the object.

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")
print(config.model_type, config.hidden_size, config.num_hidden_layers, config.vocab_size)
# bert 768 12 30522
```

### 2. Auto classes and task heads

The same backbone can have different heads. The Auto class you choose decides the head:

```python
from transformers import AutoModel, AutoModelForSequenceClassification

base = AutoModel.from_pretrained("google-bert/bert-base-uncased")              # hidden states only
clf = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-uncased", num_labels=3)                             # + randomly initialized classifier
print(type(base).__name__, type(clf).__name__)   # BertModel BertForSequenceClassification
```

When a head is new, Transformers prints a warning listing newly initialized weights; that is expected before fine-tuning.

### 3. Tokenization

Models consume integer token IDs. The tokenizer splits text into subword tokens, maps them to IDs, adds special tokens (`[CLS]`, `<s>`, ...), and builds an `attention_mask` that marks real tokens (1) versus padding (0).

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
enc = tok(["Hello world!", "Tokenizers split words into subwords."],
          padding=True, truncation=True, max_length=16, return_tensors="pt")
print(enc.keys())                    # input_ids, token_type_ids, attention_mask
print(enc["input_ids"].shape)        # (2, length of the longest sequence)
print(tok.convert_ids_to_tokens(enc["input_ids"][1]))
print(tok.decode(enc["input_ids"][0], skip_special_tokens=True))
```

"Fast" tokenizers are backed by the Rust `tokenizers` library and support batching speedups and offset mappings; `AutoTokenizer` returns a fast tokenizer whenever one is available.

### 4. Forward pass and model outputs

Models return `ModelOutput` dataclasses that behave like both dicts and tuples. If you pass `labels`, the model computes the task loss itself.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name)

inputs = tok(["I loved it", "I hated it"], return_tensors="pt", padding=True)
with torch.no_grad():
    out = model(**inputs, labels=torch.tensor([1, 0]))
print(out.loss, out.logits.shape)                        # scalar, (2, 2)
probs = out.logits.softmax(-1)
print([model.config.id2label[i] for i in probs.argmax(-1).tolist()])   # ['POSITIVE', 'NEGATIVE']
```

### 5. Causal language models and generation

Decoder-only LLMs (`AutoModelForCausalLM`) predict the next token. `generate()` repeatedly runs the model, reusing a key/value cache, and applies a decoding strategy. Chat models expect input formatted with their **chat template**.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="auto")

messages = [{"role": "user", "content": "Explain gradient descent in one sentence."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                 return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

On Transformers versions before 4.56, write `torch_dtype=torch.bfloat16` instead of `dtype=`.

### 6. Training: Trainer or your own loop

Because every model is an `nn.Module` that returns a loss when given labels, you can train with a plain PyTorch loop, with Accelerate, or with `Trainer`, which handles batching, mixed precision, gradient accumulation, distributed training, evaluation, checkpointing, and logging from a `TrainingArguments` object.

### 7. Saving and sharing

`save_pretrained(dir)` writes config, weights (safetensors), and, for tokenizers/processors, their files. `push_to_hub(repo_id)` uploads them. `from_pretrained(dir_or_repo)` reloads them anywhere.

```python
model.save_pretrained("my-model")
tok.save_pretrained("my-model")
# model.push_to_hub("your-username/my-model"); tok.push_to_hub("your-username/my-model")
```

## API Reference

Signatures list the commonly used arguments; all of these classes accept more keyword arguments documented upstream.

### Pipelines

#### pipeline

```python
transformers.pipeline(task=None, model=None, config=None, tokenizer=None, feature_extractor=None,
                      image_processor=None, processor=None, revision=None, use_fast=True, token=None,
                      device=None, device_map=None, dtype=None, trust_remote_code=None,
                      model_kwargs=None, **kwargs) -> Pipeline
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | `str` | `None` | Task name (see table below). Inferred from the model if omitted. |
| `model` | `str` or `PreTrainedModel` | `None` | Hub ID, local path, or a loaded model. If omitted, a default model for the task is used (pin one explicitly in real code). |
| `tokenizer` / `processor` | `str` or object | `None` | Override the preprocessor. |
| `device` | `int`, `str`, `torch.device` | `None` | `0`, `"cuda:0"`, `"mps"`, `"cpu"`, `-1` (CPU). |
| `device_map` | `str` or `dict` | `None` | `"auto"` to spread a large model across available devices (needs `accelerate`). Do not combine with `device`. |
| `dtype` | `torch.dtype` or `"auto"` | `None` | Weight dtype (`torch_dtype` on versions before 4.56). |
| `revision` | `str` | `"main"` | Branch, tag, or commit to pin. |
| `trust_remote_code` | `bool` | `None` | Allow custom modeling code from the repo (review it first). |
| `model_kwargs` | `dict` | `None` | Extra arguments for `from_pretrained` (e.g. `quantization_config`). |

Returns a `Pipeline` object. Calling it accepts a single input or a list, plus task-specific keyword arguments; `batch_size=` enables batched inference.

#### Common tasks

| Task string | Input | Output | Example model |
|---|---|---|---|
| `"text-classification"` (alias `"sentiment-analysis"`) | text | `[{'label', 'score'}]` | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` |
| `"token-classification"` (alias `"ner"`) | text | entities with `entity_group`, `start`, `end` | `dslim/bert-base-NER` |
| `"question-answering"` | `question=`, `context=` | `{'answer', 'score', 'start', 'end'}` | `distilbert/distilbert-base-cased-distilled-squad` |
| `"fill-mask"` | text with mask token | top-k completions | `google-bert/bert-base-uncased` |
| `"zero-shot-classification"` | text, `candidate_labels=` | labels ranked by score | `facebook/bart-large-mnli` |
| `"summarization"` | text | `[{'summary_text'}]` | `facebook/bart-large-cnn` |
| `"translation_xx_to_yy"` / `"translation"` | text | `[{'translation_text'}]` | `Helsinki-NLP/opus-mt-en-fr` |
| `"text-generation"` | text or chat messages | `[{'generated_text'}]` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `"feature-extraction"` | text | hidden states | `sentence-transformers/all-MiniLM-L6-v2` |
| `"image-classification"` | image path/URL/PIL | labels with scores | `google/vit-base-patch16-224` |
| `"object-detection"` | image | boxes, labels, scores | `facebook/detr-resnet-50` |
| `"image-segmentation"` | image | masks | `facebook/detr-resnet-50-panoptic` |
| `"zero-shot-image-classification"` | image, `candidate_labels=` | labels with scores | `openai/clip-vit-base-patch32` |
| `"automatic-speech-recognition"` | audio file/array | `{'text'}` | `openai/whisper-small` |
| `"audio-classification"` | audio | labels | `superb/wav2vec2-base-superb-ks` |
| `"image-text-to-text"` | image + text/chat | generated text | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| `"depth-estimation"` | image | depth map | `depth-anything/Depth-Anything-V2-Small-hf` |

Some legacy task names (for example `"text2text-generation"` and the seq2seq-specific `"summarization"`/`"translation"` pipelines) are deprecated in recent releases; check the pipeline docs for your version.

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Hugging Face is based in New York City."))

qa = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")
print(qa(question="Where is Hugging Face based?", context="Hugging Face is based in New York City."))

zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print(zs("The new GPU doubles training throughput.", candidate_labels=["hardware", "sports", "politics"]))

gen = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device_map="auto")
chat = [{"role": "user", "content": "Give me three names for a cat."}]
print(gen(chat, max_new_tokens=40)[0]["generated_text"][-1]["content"])

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
# print(asr("speech.wav", return_timestamps=True))

clf = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
print(clf(["great", "awful", "fine I guess"], batch_size=8))
```

### Auto classes

All Auto classes are factories: call `from_pretrained` (load weights) or `from_config` (random init).

| Class | Loads |
|---|---|
| `AutoConfig` | Configuration only. |
| `AutoTokenizer` | Tokenizer (fast if available). |
| `AutoImageProcessor` | Image preprocessing (resize, normalize). |
| `AutoFeatureExtractor` | Audio feature extraction. |
| `AutoProcessor` | Combined processor for multimodal models (e.g. Whisper, CLIP, LLaVA). |
| `AutoModel` | Base model (hidden states, no head). |
| `AutoModelForCausalLM` | Decoder LMs (GPT, Llama, Qwen, Mistral, Gemma). |
| `AutoModelForMaskedLM` | Masked LMs (BERT, RoBERTa). |
| `AutoModelForSeq2SeqLM` | Encoder-decoder (T5, BART, Marian). |
| `AutoModelForSequenceClassification` | Text classification / regression. |
| `AutoModelForTokenClassification` | NER, POS tagging. |
| `AutoModelForQuestionAnswering` | Extractive QA (start/end logits). |
| `AutoModelForImageClassification` | ViT, ConvNeXt, ... |
| `AutoModelForObjectDetection` | DETR, ... |
| `AutoModelForSpeechSeq2Seq` | Whisper and similar. |
| `AutoModelForImageTextToText` | Vision-language chat models (LLaVA, Qwen2-VL, Gemma 3, SmolVLM). |

#### from_pretrained

```python
AutoModelForX.from_pretrained(pretrained_model_name_or_path, *model_args, config=None,
    cache_dir=None, force_download=False, local_files_only=False, token=None, revision="main",
    trust_remote_code=False, dtype=None, device_map=None, attn_implementation=None,
    quantization_config=None, low_cpu_mem_usage=None, subfolder="", **kwargs) -> PreTrainedModel
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pretrained_model_name_or_path` | `str` | required | Hub repo ID (`"org/name"`) or local directory. |
| `revision` | `str` | `"main"` | Branch, tag, or commit SHA (pin for reproducibility). |
| `dtype` (`torch_dtype` before 4.56) | `torch.dtype` or `"auto"` | `None` (float32 in v4; v5 defaults to `"auto"`) | Load weights in this dtype; `"auto"` uses the checkpoint's dtype. |
| `device_map` | `str`, `dict`, `int` | `None` | `"auto"` places layers across GPUs/CPU/disk (needs `accelerate`); `"cuda"` or `0` puts all on one GPU. |
| `attn_implementation` | `str` | `None` (`"sdpa"` when supported) | `"eager"`, `"sdpa"`, `"flash_attention_2"`, (`"flash_attention_3"`, `"flex_attention"` on newer versions). |
| `quantization_config` | config object | `None` | `BitsAndBytesConfig`, `GPTQConfig`, `AwqConfig`, `TorchAoConfig`, ... |
| `trust_remote_code` | `bool` | `False` | Execute custom code from the repo. |
| `token` | `str` or `bool` | `None` | Hub token (falls back to `HF_TOKEN` / saved login). |
| `local_files_only` | `bool` | `False` | Do not download. |
| `low_cpu_mem_usage` | `bool` | `None` | Load without building a randomly initialized copy first (implied by `device_map`). |
| `num_labels`, `id2label`, `label2id`, ... | | | Any config attribute can be overridden as a keyword. |

Returns a model in evaluation mode (`model.eval()` is called for you). Tokenizers' `from_pretrained` accepts `revision`, `token`, `use_fast`, `padding_side`, `trust_remote_code`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForTokenClassification

llm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
print(llm.device, llm.dtype, f"{llm.num_parameters() / 1e6:.0f}M params")

labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]
ner = AutoModelForTokenClassification.from_pretrained(
    "google-bert/bert-base-cased",
    num_labels=len(labels),
    id2label=dict(enumerate(labels)),
    label2id={l: i for i, l in enumerate(labels)},
)
```

#### Building from a config (random weights)

```python
from transformers import AutoConfig, AutoModelForCausalLM

cfg = AutoConfig.from_pretrained("openai-community/gpt2", n_layer=4, n_embd=256, n_head=4)
tiny_gpt = AutoModelForCausalLM.from_config(cfg)     # untrained model with GPT-2 architecture
print(sum(p.numel() for p in tiny_gpt.parameters()) / 1e6, "M")
```

### Tokenizers

#### Calling a tokenizer

```python
tokenizer(text, text_pair=None, add_special_tokens=True, padding=False, truncation=None,
          max_length=None, stride=0, is_split_into_words=False, return_tensors=None,
          return_token_type_ids=None, return_attention_mask=None,
          return_overflowing_tokens=False, return_offsets_mapping=False) -> BatchEncoding
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | `str`, `list[str]`, `list[list[str]]` | required | One text, a batch, or pre-split words (with `is_split_into_words=True`). |
| `text_pair` | same | `None` | Second segment (sentence pairs, QA context). |
| `padding` | `bool` or `str` | `False` | `True`/`"longest"`, `"max_length"`, or `False`. |
| `truncation` | `bool` or `str` | `None` | `True`/`"longest_first"`, `"only_first"`, `"only_second"`. |
| `max_length` | `int` | `None` | Max tokens (defaults to `model_max_length` when truncating). |
| `stride` | `int` | `0` | Overlap between chunks with `return_overflowing_tokens`. |
| `return_tensors` | `str` | `None` | `"pt"` (PyTorch), `"np"` (NumPy); lists if `None`. |
| `return_offsets_mapping` | `bool` | `False` | Character spans per token (fast tokenizers only). |
| `is_split_into_words` | `bool` | `False` | Input is already split into words (NER). |

Returns a `BatchEncoding` (dict-like) with `input_ids`, `attention_mask`, and optionally `token_type_ids`, `offset_mapping`, `overflow_to_sample_mapping`. Fast tokenizers also provide `.word_ids(batch_index)`.

#### Other tokenizer methods and attributes

| Method / attribute | Description |
|---|---|
| `decode(ids, skip_special_tokens=False)` | IDs to string. |
| `batch_decode(sequences, skip_special_tokens=False)` | Decode a batch. |
| `tokenize(text)` | String to token strings (no IDs). |
| `convert_tokens_to_ids(tokens)` / `convert_ids_to_tokens(ids)` | Lookups. |
| `encode(text)` | String to list of IDs. |
| `add_tokens(new_tokens)` / `add_special_tokens({"pad_token": "[PAD]"})` | Extend vocabulary (then call `model.resize_token_embeddings(len(tokenizer))`). |
| `pad_token`, `eos_token`, `bos_token`, `unk_token`, `mask_token` (+ `_id`) | Special tokens. |
| `padding_side` / `truncation_side` | `"right"` or `"left"`. Use left padding for batched generation with decoder-only models. |
| `model_max_length` | Maximum supported sequence length. |
| `chat_template` | Jinja template string for chat formatting. |
| `save_pretrained(dir)` / `push_to_hub(repo)` | Persist. |

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
words = ["Angela", "Merkel", "visited", "Paris"]
enc = tok(words, is_split_into_words=True)
print(enc.tokens())
print(enc.word_ids())     # word index for each token, None for special tokens

long_text = "word " * 1000
chunks = tok(long_text, truncation=True, max_length=128, stride=32,
             return_overflowing_tokens=True)
print(len(chunks["input_ids"]))   # number of overlapping 128-token windows
```

#### apply_chat_template

```python
tokenizer.apply_chat_template(conversation, tools=None, documents=None, chat_template=None,
    add_generation_prompt=False, continue_final_message=False, tokenize=True, padding=False,
    truncation=False, max_length=None, return_tensors=None, return_dict=False, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `conversation` | `list[dict]` (or batch of lists) | required | Messages with `role` (`system`, `user`, `assistant`, `tool`) and `content`. |
| `add_generation_prompt` | `bool` | `False` | Append the assistant-turn header so the model starts answering. Use `True` for inference. |
| `continue_final_message` | `bool` | `False` | Let the model continue a partially written final assistant message. |
| `tokenize` | `bool` | `True` | Return token IDs; `False` returns the formatted string. |
| `return_tensors` | `str` | `None` | `"pt"` for tensors. |
| `return_dict` | `bool` | `False` | Return a dict with `input_ids` and `attention_mask` (pass with `**` to `generate`). |
| `tools` | `list` | `None` | JSON schemas or Python functions for tool calling (if the template supports it). |

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
messages = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "What is 2 + 2?"},
]
print(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
```

### Processors for vision, audio, and multimodal

```python
from transformers import AutoImageProcessor, AutoProcessor
from PIL import Image

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
pixel_values = image_processor(images=Image.new("RGB", (640, 480)), return_tensors="pt")["pixel_values"]
print(pixel_values.shape)          # torch.Size([1, 3, 224, 224])

processor = AutoProcessor.from_pretrained("openai/whisper-small")   # feature extractor + tokenizer
```

Multimodal chat models accept messages whose `content` is a list of parts, such as `{"type": "image", "url": ...}` and `{"type": "text", "text": ...}`, formatted with `processor.apply_chat_template(...)`.

### Text generation

#### generate

```python
model.generate(inputs=None, generation_config=None, logits_processor=None, stopping_criteria=None,
               streamer=None, assistant_model=None, **kwargs) -> torch.LongTensor | GenerateOutput
```

Any `GenerationConfig` field can be passed as a keyword to override the model's default generation config. For decoder-only models, the output contains the prompt followed by new tokens; slice off `input_ids.shape[1]` tokens to get only the completion.

#### GenerationConfig key parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_new_tokens` | `int` | `None` | Number of tokens to generate (prefer over `max_length`). |
| `max_length` | `int` | `20` | Total length including the prompt (legacy; easy to misuse). |
| `min_new_tokens` | `int` | `None` | Minimum generated tokens. |
| `do_sample` | `bool` | `False` | Sample instead of greedy/beam. |
| `temperature` | `float` | `1.0` | Softmax temperature (with sampling). |
| `top_k` | `int` | `50` | Keep top-k tokens (with sampling). |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold. |
| `min_p` | `float` | `None` | Min-p sampling. |
| `num_beams` | `int` | `1` | Beam search width. |
| `num_return_sequences` | `int` | `1` | Sequences per input. |
| `repetition_penalty` | `float` | `1.0` | Penalize repeated tokens (values above 1). |
| `no_repeat_ngram_size` | `int` | `0` | Forbid repeating n-grams. |
| `eos_token_id` | `int` or list | from config | Stop token(s). |
| `pad_token_id` | `int` | from config | Padding token for batches. |
| `stop_strings` | `str` or list | `None` | Stop on strings (requires passing `tokenizer=` to `generate`). |
| `use_cache` | `bool` | `True` | Reuse key/value cache (much faster). |
| `return_dict_in_generate` | `bool` | `False` | Return a `GenerateOutput` with `sequences`, optional `scores`. |
| `output_scores` / `output_logits` | `bool` | `False` | Include per-step scores / raw logits. |

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(name, dtype="auto", device_map="auto")

prompts = [[{"role": "user", "content": "Name a prime number."}],
           [{"role": "user", "content": "Write a haiku about GPUs."}]]
batch = tok.apply_chat_template(prompts, add_generation_prompt=True, padding=True,
                                return_tensors="pt", return_dict=True).to(model.device)

gen_cfg = GenerationConfig(max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
out = model.generate(**batch, generation_config=gen_cfg)
completions = tok.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
print(completions)

# Stream tokens to stdout as they are generated
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
single = tok.apply_chat_template(prompts[1], add_generation_prompt=True,
                                 return_tensors="pt", return_dict=True).to(model.device)
model.generate(**single, max_new_tokens=64, streamer=streamer)
```

`TextIteratorStreamer` yields text chunks from another thread (for web UIs). `model.generation_config` holds the checkpoint's defaults; `model.generation_config.save_pretrained(dir)` writes them.

### Training: Trainer and TrainingArguments

#### TrainingArguments

```python
transformers.TrainingArguments(output_dir=None, eval_strategy="no", save_strategy="steps",
    learning_rate=5e-05, per_device_train_batch_size=8, per_device_eval_batch_size=8,
    gradient_accumulation_steps=1, num_train_epochs=3.0, max_steps=-1, weight_decay=0.0,
    warmup_ratio=0.0, warmup_steps=0, lr_scheduler_type="linear", logging_steps=500,
    eval_steps=None, save_steps=500, save_total_limit=None, load_best_model_at_end=False,
    metric_for_best_model=None, greater_is_better=None, bf16=False, fp16=False,
    gradient_checkpointing=False, optim="adamw_torch", report_to=..., seed=42,
    push_to_hub=False, hub_model_id=None, torch_compile=False, dataloader_num_workers=0,
    remove_unused_columns=True, ...)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output_dir` | `str` | required in older versions; `"trainer_output"` in newer | Where checkpoints and logs go. |
| `eval_strategy` | `str` | `"no"` | `"no"`, `"steps"`, `"epoch"`. Renamed from `evaluation_strategy`. |
| `save_strategy` | `str` | `"steps"` | `"no"`, `"steps"`, `"epoch"`, `"best"`. Must match `eval_strategy` when `load_best_model_at_end=True`. |
| `learning_rate` | `float` | `5e-5` | Peak LR for AdamW. |
| `per_device_train_batch_size` | `int` | `8` | Batch size per GPU/CPU. |
| `gradient_accumulation_steps` | `int` | `1` | Effective batch = per-device batch x accumulation x number of devices. |
| `num_train_epochs` / `max_steps` | `float` / `int` | `3.0` / `-1` | Training length; `max_steps > 0` overrides epochs. |
| `weight_decay` | `float` | `0.0` | AdamW decoupled decay. |
| `warmup_ratio` / `warmup_steps` | `float` / `int` | `0.0` / `0` | Linear warmup. |
| `lr_scheduler_type` | `str` | `"linear"` | `"cosine"`, `"constant"`, `"constant_with_warmup"`, `"polynomial"`, ... |
| `logging_steps` | `int` or `float` | `500` | Log every N steps (float = fraction of total). |
| `eval_steps` / `save_steps` | `int` | `None` / `500` | Frequency for `"steps"` strategies. |
| `save_total_limit` | `int` | `None` | Keep only the most recent N checkpoints (plus best). |
| `load_best_model_at_end` | `bool` | `False` | Reload best checkpoint per `metric_for_best_model`. |
| `metric_for_best_model` | `str` | `None` | e.g. `"eval_loss"` or `"accuracy"` (key from `compute_metrics`). |
| `bf16` / `fp16` | `bool` | `False` | Mixed precision (bf16 on Ampere+/TPU, fp16 on older GPUs). |
| `gradient_checkpointing` | `bool` | `False` | Trade compute for memory. |
| `optim` | `str` | `"adamw_torch"` | Also `"adamw_torch_fused"`, `"adafactor"`, `"paged_adamw_8bit"`, `"adamw_bnb_8bit"`, ... |
| `report_to` | `str` or list | all installed integrations (v4) | `"none"`, `"tensorboard"`, `"wandb"`, `"mlflow"`, `"trackio"`. Set explicitly. |
| `push_to_hub` / `hub_model_id` | `bool` / `str` | `False` / `None` | Upload checkpoints and the final model. |
| `torch_compile` | `bool` | `False` | Wrap the model with `torch.compile`. |
| `remove_unused_columns` | `bool` | `True` | Drop dataset columns not accepted by `model.forward` (disable for custom collators). |
| `dataloader_num_workers` | `int` | `0` | DataLoader worker processes. |
| `seed` | `int` | `42` | Seed for reproducibility. |

#### Trainer

```python
transformers.Trainer(model=None, args=None, data_collator=None, train_dataset=None,
    eval_dataset=None, processing_class=None, model_init=None, compute_loss_func=None,
    compute_metrics=None, callbacks=None, optimizers=(None, None),
    preprocess_logits_for_metrics=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `PreTrainedModel` or `nn.Module` | `None` | Model to train; must return a loss when given labels (or supply `compute_loss_func`). |
| `args` | `TrainingArguments` | `None` | Hyperparameters. |
| `data_collator` | callable | `None` | Batches examples; defaults to `DataCollatorWithPadding` if a tokenizer is given, else `default_data_collator`. |
| `train_dataset` / `eval_dataset` | `Dataset` or dict of datasets | `None` | Usually `datasets.Dataset` objects. |
| `processing_class` | tokenizer / processor | `None` | Saved with checkpoints and used for padding. Replaces the deprecated `tokenizer=` argument (4.46+). |
| `compute_metrics` | callable | `None` | `fn(EvalPrediction) -> dict` of metrics. |
| `callbacks` | list | `None` | e.g. `EarlyStoppingCallback(early_stopping_patience=3)`. |
| `optimizers` | tuple | `(None, None)` | Custom `(optimizer, lr_scheduler)`. |
| `compute_loss_func` | callable | `None` | Custom loss `fn(outputs, labels, num_items_in_batch)`. |
| `preprocess_logits_for_metrics` | callable | `None` | Reduce logits (e.g. argmax) before accumulation to save memory. |

Key methods:

| Method | Description |
|---|---|
| `train(resume_from_checkpoint=None)` | Run training; pass `True` or a path to resume. Returns `TrainOutput`. |
| `evaluate(eval_dataset=None)` | Returns a dict of metrics prefixed with `eval_`. |
| `predict(test_dataset)` | Returns `PredictionOutput(predictions, label_ids, metrics)`. |
| `save_model(output_dir=None)` | Save model (and processing class). |
| `push_to_hub(commit_message=...)` | Upload the model and a generated model card. |
| `add_callback(callback)` | Attach a callback. |

```python
import numpy as np
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}

args = TrainingArguments(
    output_dir="out",
    eval_strategy="epoch",          # NOT evaluation_strategy (renamed)
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    bf16=True,                      # set False on CPU / pre-Ampere GPUs (use fp16=True there)
    report_to="none",
)
# trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
#                   processing_class=tokenizer,          # NOT tokenizer= (deprecated)
#                   compute_metrics=compute_metrics,
#                   callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
# trainer.train()
```

`Seq2SeqTrainer` and `Seq2SeqTrainingArguments` add `predict_with_generate=True` and `generation_max_length` for encoder-decoder tasks. For LLM supervised fine-tuning and preference optimization, TRL's `SFTTrainer`, `DPOTrainer`, and `GRPOTrainer` build on `Trainer`.

#### Data collators

| Collator | Use |
|---|---|
| `DataCollatorWithPadding(tokenizer)` | Dynamic padding for classification. |
| `DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)` | MLM masking; `mlm=False` for causal LM (labels = inputs, padding set to -100). |
| `DataCollatorForSeq2Seq(tokenizer, model=None, label_pad_token_id=-100)` | Pads inputs and labels for encoder-decoder. |
| `DataCollatorForTokenClassification(tokenizer)` | Pads token-level labels with -100. |
| `default_data_collator` | Stacks already-padded features. |

Label value `-100` is ignored by the loss in all Transformers models.

### Quantization

Quantized loading reduces memory so large models fit on smaller GPUs. Pass a config object to `from_pretrained(..., quantization_config=...)`. (The old `load_in_8bit=True` / `load_in_4bit=True` keyword shortcuts are deprecated.)

#### BitsAndBytesConfig

```python
transformers.BitsAndBytesConfig(load_in_8bit=False, load_in_4bit=False, llm_int8_threshold=6.0,
    llm_int8_skip_modules=None, bnb_4bit_compute_dtype=None, bnb_4bit_quant_type="fp4",
    bnb_4bit_use_double_quant=False, bnb_4bit_quant_storage=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `load_in_8bit` | `bool` | `False` | LLM.int8() quantization. |
| `load_in_4bit` | `bool` | `False` | 4-bit quantization (QLoRA). |
| `bnb_4bit_quant_type` | `str` | `"fp4"` | `"nf4"` is recommended for normally distributed weights. |
| `bnb_4bit_compute_dtype` | dtype | `None` (float32) | Dtype for matmuls; use `torch.bfloat16`. |
| `bnb_4bit_use_double_quant` | `bool` | `False` | Quantize the quantization constants too (saves ~0.4 bits/param). |
| `llm_int8_skip_modules` | list | `None` | Modules kept in full precision (e.g. `["lm_head"]`). |

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                             quantization_config=bnb, device_map="auto")
print(f"{model.get_memory_footprint() / 2**30:.2f} GiB")
```

#### Other quantization methods

| Config | Method | Notes |
|---|---|---|
| `GPTQConfig(bits=4, dataset="c4", tokenizer=tok)` | GPTQ | Load pre-quantized GPTQ checkpoints directly; quantizing needs a calibration dataset (gptqmodel backend). |
| `AwqConfig(bits=4)` | AWQ | Mostly used to load pre-quantized AWQ checkpoints. |
| `TorchAoConfig(quant_type)` | torchao | int4/int8 weight-only, float8; works with `torch.compile`. |
| `HqqConfig(nbits=4, group_size=64)` | HQQ | Fast, calibration-free. |
| `FbgemmFp8Config()`, `FineGrainedFP8Config()` | FP8 | FP8 inference on Hopper-class GPUs. |
| `from_pretrained(repo, gguf_file="model.gguf")` | GGUF | Load (dequantize) llama.cpp GGUF files. |

Pre-quantized checkpoints (GPTQ, AWQ, bitsandbytes, FP8) store their `quantization_config` in `config.json`, so a plain `from_pretrained` loads them.

### PEFT and LoRA integration

PEFT (`pip install peft`) trains small adapter matrices instead of all weights. Two equivalent entry points exist.

#### Using the peft library directly

```python
peft.LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, target_modules=None, bias="none",
                task_type=None, modules_to_save=None, use_dora=False, init_lora_weights=True)
peft.get_peft_model(model, peft_config) -> PeftModel
peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) -> model
peft.PeftModel.from_pretrained(base_model, adapter_path_or_repo) -> PeftModel
```

| `LoraConfig` parameter | Type | Default | Description |
|---|---|---|---|
| `r` | `int` | `8` | Rank of the update matrices. |
| `lora_alpha` | `int` | `8` | Scaling (`alpha / r`); commonly 2 x r. |
| `lora_dropout` | `float` | `0.0` | Dropout on the LoRA path. |
| `target_modules` | list or `str` | `None` (per-architecture default) | e.g. `["q_proj", "v_proj"]` or `"all-linear"`. |
| `task_type` | `str` / `TaskType` | `None` | `"CAUSAL_LM"`, `"SEQ_CLS"`, `"SEQ_2_SEQ_LM"`, `"TOKEN_CLS"`. |
| `modules_to_save` | list | `None` | Extra modules trained fully and saved (e.g. a classifier head). |
| `use_dora` | `bool` | `False` | Weight-decomposed LoRA. |

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
                  task_type="CAUSAL_LM")
model = get_peft_model(base, lora)
model.print_trainable_parameters()     # e.g. trainable params: 8.8M || all params: 503M || trainable%: 1.75
# ... train ...
model.save_pretrained("qwen-lora")     # saves only the adapter (a few MB)
merged = model.merge_and_unload()      # fold adapters into base weights for deployment
```

#### Using the Transformers integration

Transformers models expose adapter methods directly when `peft` is installed:

| Method | Description |
|---|---|
| `model.add_adapter(peft_config, adapter_name=None)` | Attach a new adapter for training. |
| `model.load_adapter(path_or_repo, adapter_name=None)` | Load a trained adapter. |
| `model.set_adapter(name)` | Activate one adapter. |
| `model.disable_adapters()` / `model.enable_adapters()` | Toggle adapters. |
| `AutoModelForCausalLM.from_pretrained(adapter_repo)` | If the repo contains only an adapter (`adapter_config.json`), the base model is loaded and the adapter attached automatically. |

```python
from peft import LoraConfig
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model.add_adapter(LoraConfig(r=8, target_modules=["q_proj", "v_proj"]), adapter_name="my_lora")
# The model can now be passed to Trainer; only adapter weights are trainable.
```

### Hub: save, load, push, pull

| API | Description |
|---|---|
| `model.save_pretrained(save_directory, safe_serialization=True, max_shard_size=...)` | Save config + safetensors weights (sharded if large). |
| `model.push_to_hub(repo_id, commit_message=None, private=None, token=None, revision=None, create_pr=False)` | Upload to the Hub (creates the repo if needed). Same on tokenizers, processors, configs. |
| `from_pretrained(repo_id, revision="v1.0")` | Pull a specific branch/tag/commit. |
| `huggingface_hub.login(token=None)` | Authenticate in Python. |
| `huggingface_hub.snapshot_download(repo_id, revision=None, allow_patterns=None, local_dir=None)` | Download a whole repo (or a filtered subset). |
| `huggingface_hub.hf_hub_download(repo_id, filename, revision=None)` | Download one file and return its local path. |
| `huggingface_hub.HfApi().upload_folder(folder_path, repo_id, repo_type="model")` | Upload arbitrary files. |
| `huggingface_hub.create_repo(repo_id, private=False, exist_ok=True)` | Create a repo. |

```python
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

path = snapshot_download("distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                         allow_patterns=["*.json", "*.safetensors", "*.txt"])
model = AutoModelForSequenceClassification.from_pretrained(path)   # load from the local folder
cfg_file = hf_hub_download("google-bert/bert-base-uncased", "config.json")

tok = AutoTokenizer.from_pretrained(path)
model.save_pretrained("./sst2-local")
tok.save_pretrained("./sst2-local")
# model.push_to_hub("your-username/sst2-copy", private=True)
# tok.push_to_hub("your-username/sst2-copy", private=True)
```

### Utilities

| API | Description |
|---|---|
| `transformers.set_seed(seed)` | Seed Python, NumPy, and PyTorch. |
| `transformers.logging.set_verbosity_error()` | Silence warnings (`set_verbosity_info()` for more detail). |
| `model.num_parameters(only_trainable=False)` | Parameter count. |
| `model.get_memory_footprint()` | Bytes used by parameters and buffers. |
| `model.gradient_checkpointing_enable()` | Activation checkpointing. |
| `model.resize_token_embeddings(new_num_tokens)` | After adding tokens. |
| `model.get_input_embeddings()` / `get_output_embeddings()` | Embedding modules. |
| `model.config.use_cache = False` | Disable KV cache (needed with gradient checkpointing during training). |
| `model.to(device)` / `model.eval()` / `model.train()` | Standard `nn.Module` methods. |
| `TextStreamer`, `TextIteratorStreamer` | Streaming generation output. |
| `get_scheduler(name, optimizer, num_warmup_steps, num_training_steps)` | LR schedules for custom loops. |

## Tutorials

### Tutorial 1: Fine-tune DistilBERT for sentiment classification with Trainer

End-to-end text classification on IMDB movie reviews: load a dataset, tokenize, fine-tune, evaluate, save, and use the result in a pipeline.

```bash
pip install -U transformers datasets evaluate accelerate
```

```python
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed,
)

set_seed(42)
model_name = "distilbert/distilbert-base-uncased"

# 1. Data: subsample for a fast run (use the full splits for best accuracy).
train_raw = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=42).select(range(5000))
test_raw = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=42).select(range(2000))

# 2. Tokenize without padding; the collator pads each batch dynamically.
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)

train_ds = train_raw.map(preprocess, batched=True, remove_columns=["text"])
test_ds = test_raw.map(preprocess, batched=True, remove_columns=["text"])

# 3. Model with a fresh 2-way classification head and readable labels.
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id={v: k for k, v in id2label.items()})

# 4. Metrics.
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {**accuracy.compute(predictions=preds, references=labels),
            **f1.compute(predictions=preds, references=labels)}

# 5. Training configuration.
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
args = TrainingArguments(
    output_dir="imdb-distilbert",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    bf16=use_bf16,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# 6. Train, evaluate, save.
trainer.train()
print(trainer.evaluate())             # {'eval_loss': ..., 'eval_accuracy': ~0.88, 'eval_f1': ...}
trainer.save_model("imdb-distilbert/final")   # also saves the tokenizer (processing_class)

# 7. Inference with the fine-tuned model.
clf = pipeline("text-classification", model="imdb-distilbert/final")
print(clf(["An absolute masterpiece.", "Two hours I will never get back."]))
```

Step by step:

1. `load_dataset` downloads Parquet files from the Hub and returns Arrow-backed `Dataset`s; `select` keeps the run short.
2. Tokenizing with `batched=True` uses the fast tokenizer's batch path. Padding is deferred to `DataCollatorWithPadding`, which pads to the longest sequence in each batch (much less wasted compute than padding everything to 256).
3. `num_labels=2` creates a new classifier head; the "newly initialized weights" warning is expected. `id2label` makes predictions human-readable and is saved in `config.json`.
4. `compute_metrics` receives NumPy logits and labels for the whole eval set.
5. `eval_strategy` and `save_strategy` must match for `load_best_model_at_end`. The dataset's `label` column is automatically renamed to `labels`, the argument name the model expects.
6. `save_model` writes the best checkpoint plus tokenizer files, so the folder is self-contained.
7. `pipeline` reloads everything from the folder. Add `trainer.push_to_hub()` (after `hf auth login`) to share it.

### Tutorial 2: Semantic search with sentence embeddings

Compute normalized sentence embeddings with `AutoModel` and mean pooling, then rank documents by cosine similarity. No training required.

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

name = "sentence-transformers/all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name).to(device).eval()

def embed(texts, batch_size=32):
    chunks = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                          max_length=256, return_tensors="pt").to(device)
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state          # (B, T, 384)
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)   # mean over real tokens
        chunks.append(F.normalize(pooled, dim=-1))
    return torch.cat(chunks)

docs = [
    "Use mixed precision and larger batch sizes to speed up GPU training.",
    "The Eiffel Tower is located in Paris.",
    "Gradient checkpointing reduces memory usage at the cost of extra compute.",
    "LoRA fine-tunes large language models by training low-rank adapters.",
    "Bananas are rich in potassium.",
]
doc_emb = embed(docs)

def search(query, k=3):
    q = embed([query])
    scores = (q @ doc_emb.T).squeeze(0)                         # cosine similarity
    top = scores.topk(k)
    return [(docs[i], round(s, 3)) for s, i in zip(top.values.tolist(), top.indices.tolist())]

for hit in search("How can I make training faster and use less memory?"):
    print(hit)
```

Explanation: the base model outputs one vector per token; mean pooling with the attention mask averages only real (non-padding) tokens, which is how this checkpoint was trained. After L2 normalization, a dot product equals cosine similarity. For large corpora, store `doc_emb` in a vector index (FAISS, a vector database). The `sentence-transformers` library wraps exactly this pattern (`SentenceTransformer(name).encode(...)`).

### Tutorial 3: QLoRA instruction fine-tuning of a small LLM

Fine-tune `Qwen/Qwen2.5-0.5B-Instruct` on an instruction dataset with 4-bit quantization (bitsandbytes) and LoRA adapters (PEFT), masking the prompt so the loss only covers the response. Requires an NVIDIA GPU (about 6 GB of memory is enough for this size).

```bash
pip install -U transformers datasets accelerate peft bitsandbytes
```

```python
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 1. Data: format each example with the model's chat template and mask the prompt.
raw = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=0).select(range(2000))
MAX_LEN = 512

def build_example(ex):
    user = ex["instruction"] + (f"\n\n{ex['input']}" if ex["input"] else "")
    prompt_msgs = [{"role": "user", "content": user}]
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(
        prompt_msgs + [{"role": "assistant", "content": ex["output"]}], tokenize=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:MAX_LEN]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

train_ds = raw.map(build_example, remove_columns=raw.column_names)

# 2. Load the base model in 4-bit NF4.
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb,
                                             device_map={"": 0}, dtype=torch.bfloat16)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# 3. Attach LoRA adapters to every linear layer.
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
                  task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# 4. Train. DataCollatorForSeq2Seq pads input_ids with pad_token_id and labels with -100.
args = TrainingArguments(
    output_dir="qwen-alpaca-qlora",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=False,
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
)
trainer.train()

# 5. Save only the adapter (a few tens of MB).
model.save_pretrained("qwen-alpaca-lora")
tokenizer.save_pretrained("qwen-alpaca-lora")

# 6. Inference: load the base model in bf16, attach the adapter, optionally merge.
base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
tuned = PeftModel.from_pretrained(base, "qwen-alpaca-lora").merge_and_unload()
msgs = [{"role": "user", "content": "Give three tips for writing clean Python code."}]
inputs = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to(tuned.device)
out = tuned.generate(**inputs, max_new_tokens=150, do_sample=False)
print(tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))

# 7. Share (after `hf auth login`): the adapter alone, or the merged full model.
# model.push_to_hub("your-username/qwen2.5-0.5b-alpaca-lora")
# tuned.push_to_hub("your-username/qwen2.5-0.5b-alpaca-merged"); tokenizer.push_to_hub(...)
```

Why each piece matters:

- **Chat template formatting** makes training data look exactly like what the model sees at inference time.
- **Label masking** (`-100` on prompt tokens) trains the model to produce answers rather than to reproduce instructions.
- **4-bit NF4 + double quantization** shrinks the frozen base weights about 4x; computation happens in bfloat16.
- **`prepare_model_for_kbit_training`** casts norms to float32, enables input gradients, and turns on gradient checkpointing, which quantized training needs.
- **LoRA on `all-linear`** typically trains about 1-2 percent of parameters while matching much of full fine-tuning quality.
- **`paged_adamw_8bit`** keeps optimizer memory small and avoids spikes.
- **Merging** removes adapter overhead at inference; merge into a bf16 (not 4-bit) copy of the base model.

For larger projects, TRL's `SFTTrainer` automates chat formatting, completion-only loss, and packing on top of the same components.

### Tutorial 4: Abstractive summarization with T5 and Seq2SeqTrainer

```bash
pip install -U transformers datasets evaluate rouge_score accelerate sentencepiece
```

```python
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# 1. Data: news articles with human-written highlight summaries.
raw = load_dataset("abisee/cnn_dailymail", "3.0.0")
train_raw = raw["train"].shuffle(seed=42).select(range(4000))
val_raw = raw["validation"].shuffle(seed=42).select(range(200))

prefix = "summarize: "            # T5 was pretrained with task prefixes

def preprocess(batch):
    model_inputs = tokenizer([prefix + a for a in batch["article"]], max_length=512, truncation=True)
    labels = tokenizer(text_target=batch["highlights"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

cols = train_raw.column_names
train_ds = train_raw.map(preprocess, batched=True, remove_columns=cols)
val_ds = val_raw.map(preprocess, batched=True, remove_columns=cols)

# 2. Metrics: decode generated IDs and compute ROUGE.
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in scores.items()}

# 3. Train with generation-based evaluation.
args = Seq2SeqTrainingArguments(
    output_dir="t5-small-cnn",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    predict_with_generate=True,       # run generate() during evaluation
    generation_max_length=128,
    generation_num_beams=4,
    save_total_limit=1,
    logging_steps=50,
    report_to="none",
)
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
)
trainer.train()
print(trainer.evaluate())             # eval_rouge1, eval_rouge2, eval_rougeL, ...

# 4. Summarize new text.
article = val_raw[0]["article"]
inputs = tokenizer(prefix + article, return_tensors="pt", truncation=True, max_length=512).to(model.device)
summary_ids = model.generate(**inputs, max_new_tokens=80, num_beams=4, no_repeat_ngram_size=3)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
```

Notes: `text_target=` tokenizes labels with the target-side settings. `DataCollatorForSeq2Seq(model=model)` also prepares `decoder_input_ids` by shifting labels. With `predict_with_generate=True`, predictions are generated token IDs (padded with -100 across batches, hence the `np.where`). Avoid `fp16=True` with T5 (it overflows to NaN); bf16 or float32 are safe.

## Performance & Best Practices

### Inference

| Technique | How |
|---|---|
| Lower precision | `from_pretrained(..., dtype=torch.bfloat16)` or `dtype="auto"` (float16 on older GPUs). |
| Fast attention | Default `attn_implementation="sdpa"`; `"flash_attention_2"` with `flash-attn` installed for long sequences. |
| Fit large models | `device_map="auto"` spreads layers across GPUs/CPU; quantize with bitsandbytes (4-bit), torchao, AWQ, GPTQ. |
| Batch requests | Pass lists to pipelines with `batch_size=`; for decoder-only generation use left padding. |
| Avoid gradients | `model.eval()` and `torch.inference_mode()` in custom code (pipelines do this). |
| Compile | `model.forward = torch.compile(model.forward)` or a static KV cache: `model.generate(..., cache_implementation="static")` combined with `torch.compile` for low-latency decoding. |
| Speculative decoding | `model.generate(..., assistant_model=small_model)` with a small draft model of the same tokenizer family. |
| Serving | For production throughput use vLLM, SGLang, or TGI, which load the same Hub checkpoints. |

### Training

- **Dynamic padding** with `DataCollatorWithPadding` instead of padding everything to `max_length`; consider `group_by_length=True` to batch similar lengths.
- **Mixed precision**: `bf16=True` on Ampere+ GPUs and TPUs; `fp16=True` on older GPUs (not for T5).
- **Memory**: `gradient_checkpointing=True`, smaller `per_device_train_batch_size` with larger `gradient_accumulation_steps`, 8-bit optimizers (`optim="adamw_bnb_8bit"` or `"paged_adamw_8bit"`), LoRA/QLoRA instead of full fine-tuning.
- **Throughput**: `optim="adamw_torch_fused"`, `dataloader_num_workers=4`, `torch_compile=True`, `tf32=True` on Ampere+.
- **Multi-GPU**: launch the same script with `accelerate launch train.py` or `torchrun --nproc_per_node=N train.py`; `Trainer` uses DDP automatically, and FSDP or DeepSpeed via `fsdp=` / `deepspeed=` arguments or an Accelerate config.
- **Learning rates**: about 2e-5 to 5e-5 for full fine-tuning of BERT-class encoders; 1e-4 to 3e-4 for LoRA.
- **Evaluation cost**: use `preprocess_logits_for_metrics` to argmax logits early for LM tasks so evaluation does not store full vocabulary logits.

### Reproducibility and hygiene

- Pin `revision=` (a commit SHA) for production models, and pin library versions.
- Prefer safetensors checkpoints (the default) and avoid `trust_remote_code=True` unless you have reviewed the repo code.
- Save the tokenizer/processor with the model (`processing_class=` in Trainer does it automatically).
- Call `set_seed()` before creating the model so newly initialized heads are reproducible.
- Set `report_to` explicitly to avoid unexpected logging to installed integrations.

## Common Errors & Troubleshooting

### Missing padding token

```text
ValueError: Asking to pad but the tokenizer does not have a padding token. Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.
```

Cause: many decoder-only tokenizers (GPT-2, Llama) have no pad token. Fix: `tokenizer.pad_token = tokenizer.eos_token` (and pass `pad_token_id=tokenizer.pad_token_id` to `generate`), or add a new pad token and call `model.resize_token_embeddings(len(tokenizer))`.

### Right padding with decoder-only generation

```text
A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
```

Fix: `AutoTokenizer.from_pretrained(name, padding_side="left")` for batched generation. Keep right padding for training.

### Missing attention mask / pad token at generation

```text
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
```

Fix: pass the full tokenizer output with `model.generate(**inputs, ...)` (not just `input_ids`), and set `pad_token_id`.

### Output truncated after a few tokens

```text
UserWarning: Input length of input_ids is 25, but `max_length` is set to 20. This can lead to unexpected behavior. You should consider increasing `max_length` or, better yet, setting `max_new_tokens`.
```

Fix: use `max_new_tokens=` instead of relying on the default `max_length=20`.

### Renamed TrainingArguments and Trainer parameters

```text
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'
```

Fix: use `eval_strategy=` (renamed in 4.41; the old name was later removed). Similarly, `Trainer(tokenizer=...)` is deprecated since 4.46 and removed in v5:

```text
TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'
```

Fix: `Trainer(..., processing_class=tokenizer)`. In code that targets versions older than 4.46, `processing_class` does not exist yet; upgrade rather than branching.

### Strategy mismatch with load_best_model_at_end

```text
ValueError: --load_best_model_at_end requires the save and eval strategy to match, but found
- Evaluation strategy: IntervalStrategy.EPOCH
- Save strategy: SaveStrategy.STEPS
```

Fix: set `save_strategy` equal to `eval_strategy` (and matching `save_steps`/`eval_steps` for `"steps"`).

### No loss returned

```text
ValueError: The model did not return a loss from the inputs, only the following keys: logits. For reference, the inputs it received are input_ids,attention_mask.
```

Cause: the dataset has no `labels` (or `label`) column, or it was removed by `remove_unused_columns`. Fix: ensure a `labels` column exists (for causal LM, use `DataCollatorForLanguageModeling(tokenizer, mlm=False)` which creates labels), or set `remove_unused_columns=False` with custom collators.

### Sequence too long

```text
Token indices sequence length is longer than the specified maximum sequence length for this model (1043 > 512). Running this sequence through the model will result in indexing errors
```

Fix: tokenize with `truncation=True, max_length=...`, or chunk long documents with `stride` and `return_overflowing_tokens=True`. Running an over-long sequence through a model with absolute position embeddings causes `IndexError: index out of range in self` or a CUDA device-side assert.

### Model or repo not found, gated models

```text
OSError: org/model-name is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `hf auth login` or by passing `token=<your_token>`
```

```text
OSError: You are trying to access a gated repo. Make sure to have access to it at https://huggingface.co/meta-llama/Llama-3.2-1B.
401 Client Error. (Request ID: ...) Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/config.json.
```

Fix: check spelling (`org/name`), accept the license on the model page, and authenticate (`hf auth login` or `HF_TOKEN`). Offline machines: pre-download with `snapshot_download` and set `HF_HUB_OFFLINE=1`.

### Unknown architecture

```text
ValueError: The checkpoint you are trying to load has model type `qwen3` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.
```

Fix: `pip install -U transformers` (new architectures need new releases), or `trust_remote_code=True` if the repo ships custom code.

### Newly initialized weights warning

```text
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert/distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

Expected when adding a new head; train before use. If it appears when loading your own fine-tuned checkpoint, you used the wrong Auto class or the save was incomplete.

### Size mismatch when loading

```text
RuntimeError: Error(s) in loading state_dict for BertForSequenceClassification:
	size mismatch for classifier.weight: copying a param with shape torch.Size([2, 768]) from checkpoint, the shape in current model is torch.Size([5, 768]).
```

Fix: when changing `num_labels` on an already fine-tuned checkpoint, pass `ignore_mismatched_sizes=True`.

### Device mismatch

```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

Fix: move inputs to the model's device: `inputs = tokenizer(..., return_tensors="pt").to(model.device)`. With `device_map="auto"`, send inputs to `model.device` (the first layer's device) and do not call `model.to(...)`.

### Out of memory

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.95 GiB. GPU 0 has a total capacity of 15.77 GiB of which 1.20 GiB is free.
```

Fixes: load in bf16/fp16 (default float32 doubles memory), use 4-bit quantization, reduce `per_device_train_batch_size` and raise `gradient_accumulation_steps`, enable gradient checkpointing, shorten `max_length`, use LoRA, or use `device_map="auto"` to offload layers.

### bitsandbytes problems

```text
ImportError: Using `bitsandbytes` 4-bit quantization requires the latest version of bitsandbytes: `pip install -U bitsandbytes`
```

Fix: install/upgrade `bitsandbytes` (NVIDIA GPU builds are the most mature), and make sure a GPU is visible. Quantized models cannot be moved with `.to(...)` after loading; place them with `device_map`.

### Fine-tuning produces NaN or zero loss

Causes: fp16 overflow (T5, some LLMs; switch to bf16), learning rate too high, all labels masked to -100 (check your masking: if the prompt is longer than `max_length`, nothing is left to learn), or `use_cache=True` with gradient checkpointing (Trainer disables it; set `model.config.use_cache = False` in custom loops).

## Interoperability

### PyTorch

Every Transformers model is a `torch.nn.Module`, so you can use custom training loops, `torch.compile`, `torch.amp`, DDP/FSDP, `torch.export`, and any PyTorch tooling. Pass `return_tensors="pt"` to tokenizers and call `model(**batch)`.

```python
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler

tok = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=2)
opt = AdamW(model.parameters(), lr=2e-5)
sched = get_scheduler("linear", opt, num_warmup_steps=0, num_training_steps=10)

batch = tok(["good", "bad"], padding=True, return_tensors="pt")
for _ in range(10):
    loss = model(**batch, labels=torch.tensor([1, 0])).loss
    loss.backward()
    opt.step(); sched.step(); opt.zero_grad()
```

### TensorFlow and JAX/Flax

`TFAutoModel...` and `FlaxAutoModel...` classes exist in v4 but are deprecated, and Transformers v5 removes them to focus on PyTorch. For Keras users, KerasHub offers many of the same architectures; for JAX users, see the JAX ecosystem (MaxText, Flax-based implementations). Weights on the Hub are framework-neutral safetensors, so they can be converted.

### Hugging Face ecosystem

| Library | Role with Transformers |
|---|---|
| `datasets` | Load, map, and stream datasets; feeds `Trainer` directly. |
| `tokenizers` | Rust backend of fast tokenizers; train new tokenizers. |
| `accelerate` | Device placement, `device_map="auto"`, multi-GPU/TPU launching under `Trainer` or custom loops. |
| `peft` | LoRA and other adapters (`get_peft_model`, `model.add_adapter`). |
| `trl` | SFT, DPO, GRPO, reward modeling built on `Trainer`. |
| `evaluate` | Metrics (accuracy, F1, ROUGE, BLEU). |
| `optimum` | ONNX export, ONNX Runtime and Intel/AMD/NVIDIA backends. |
| `sentence-transformers` | Embedding models built on Transformers backbones. |
| `diffusers` | Diffusion pipelines that reuse Transformers text encoders (CLIP, T5). |
| `huggingface_hub` | Downloading, uploading, repos, inference endpoints. |
| `safetensors` | Safe, zero-copy weight format used by default. |

### Inference engines and runtimes

- **vLLM, SGLang, TGI**: serve Hub checkpoints with continuous batching; recent versions can also use the Transformers modeling code as a fallback backend.
- **llama.cpp / GGUF**: convert with llama.cpp's `convert_hf_to_gguf.py`; Transformers can load GGUF with `gguf_file=`.
- **ONNX / ONNX Runtime**: `optimum-cli export onnx --model distilbert/distilbert-base-uncased onnx_dir/`.
- **MLX (Apple)**: `mlx-lm` loads and converts Hub checkpoints.

### scikit-learn and pandas

Use embeddings from `AutoModel` (Tutorial 2) as features for scikit-learn classifiers; `datasets.Dataset.from_pandas(df)` and `dataset.to_pandas()` move data between pandas and the Trainer.

## Cheat Sheet

### Inference

| Task | Code |
|---|---|
| Quick inference | `pipeline("sentiment-analysis", model=name)(text)` |
| Pipeline on GPU | `pipeline(task, model=name, device=0)` |
| Big model across GPUs | `pipeline(task, model=name, device_map="auto", dtype="auto")` |
| Load tokenizer | `tok = AutoTokenizer.from_pretrained(name)` |
| Load LLM in bf16 | `AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="auto")` |
| Load in 4-bit | `from_pretrained(name, quantization_config=BitsAndBytesConfig(load_in_4bit=True))` |
| Tokenize batch | `tok(texts, padding=True, truncation=True, return_tensors="pt")` |
| Forward pass | `model(**inputs).logits` |
| Chat prompt | `tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)` |
| Generate | `model.generate(**inputs, max_new_tokens=128)` |
| Sample | `model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.9)` |
| Decode new tokens | `tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)` |
| Stream | `model.generate(**inputs, streamer=TextStreamer(tok, skip_prompt=True))` |
| Batched generation | `AutoTokenizer.from_pretrained(name, padding_side="left")` |
| Set pad token | `tok.pad_token = tok.eos_token` |
| Embeddings | `AutoModel.from_pretrained(name)(**inputs).last_hidden_state` + mean pooling |

### Training

| Task | Code |
|---|---|
| Classification model | `AutoModelForSequenceClassification.from_pretrained(name, num_labels=n)` |
| Tokenize dataset | `ds.map(lambda b: tok(b["text"], truncation=True), batched=True)` |
| Dynamic padding | `DataCollatorWithPadding(tok)` |
| Causal LM labels | `DataCollatorForLanguageModeling(tok, mlm=False)` |
| Seq2seq labels | `tok(text_target=summaries)` + `DataCollatorForSeq2Seq(tok, model=model)` |
| Arguments | `TrainingArguments(output_dir="out", eval_strategy="epoch", bf16=True, report_to="none")` |
| Trainer | `Trainer(model=m, args=args, train_dataset=tr, eval_dataset=ev, processing_class=tok)` |
| Train / resume | `trainer.train()` / `trainer.train(resume_from_checkpoint=True)` |
| Evaluate / predict | `trainer.evaluate()` / `trainer.predict(test_ds)` |
| Early stopping | `callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]` |
| Gradient checkpointing | `TrainingArguments(..., gradient_checkpointing=True)` |
| LoRA | `get_peft_model(model, LoraConfig(r=16, target_modules="all-linear", task_type="CAUSAL_LM"))` |
| QLoRA prep | `prepare_model_for_kbit_training(model)` |
| Merge adapter | `model.merge_and_unload()` |
| Multi-GPU | `accelerate launch train.py` |

### Saving and Hub

| Task | Code |
|---|---|
| Login | `hf auth login` (CLI) or `huggingface_hub.login()` |
| Save locally | `model.save_pretrained(d); tok.save_pretrained(d)` |
| Push | `model.push_to_hub("user/repo"); tok.push_to_hub("user/repo")` |
| Push from Trainer | `TrainingArguments(push_to_hub=True, hub_model_id="user/repo")`, `trainer.push_to_hub()` |
| Pin a version | `from_pretrained(name, revision="<commit-sha>")` |
| Download repo | `snapshot_download("org/name")` |
| Download file | `hf_hub_download("org/name", "config.json")` |
| Offline mode | `HF_HUB_OFFLINE=1` |
| Load adapter | `PeftModel.from_pretrained(base, "user/adapter")` or `model.load_adapter("user/adapter")` |

## Further Resources

- Documentation: https://huggingface.co/docs/transformers/index
- GitHub repository: https://github.com/huggingface/transformers
- Model Hub: https://huggingface.co/models
- Pipelines guide: https://huggingface.co/docs/transformers/main_classes/pipelines
- Trainer docs: https://huggingface.co/docs/transformers/main_classes/trainer
- Text generation strategies: https://huggingface.co/docs/transformers/generation_strategies
- Chat templates: https://huggingface.co/docs/transformers/chat_templating
- Quantization overview: https://huggingface.co/docs/transformers/quantization/overview
- PEFT documentation: https://huggingface.co/docs/peft/index
- TRL documentation: https://huggingface.co/docs/trl/index
- Datasets documentation: https://huggingface.co/docs/datasets/index
- Accelerate documentation: https://huggingface.co/docs/accelerate/index
- huggingface_hub documentation: https://huggingface.co/docs/huggingface_hub/index
- Hugging Face Learn (LLM course and others): https://huggingface.co/learn
- Forums: https://discuss.huggingface.co/
- Paper: Wolf et al., "Transformers: State-of-the-Art Natural Language Processing", EMNLP 2020 Demos: https://aclanthology.org/2020.emnlp-demos.6/
- Paper: Vaswani et al., "Attention Is All You Need", 2017: https://arxiv.org/abs/1706.03762
- Paper: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021: https://arxiv.org/abs/2106.09685
- Paper: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", 2023: https://arxiv.org/abs/2305.14314
- Book: Tunstall, von Werra, Wolf, "Natural Language Processing with Transformers" (O'Reilly): https://www.oreilly.com/library/view/natural-language-processing/9781098136789/
