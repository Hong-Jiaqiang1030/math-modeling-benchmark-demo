import os
from typing import Dict, List

from demo.benchmark.utils.env import load_env

# Make sure keys from `.env` are available.
load_env()

# User-specified defaults
DOUBAO_MODEL_ID = "ep-20260115090253-hbjtj"
DEFAULT_LLM_MODEL = DOUBAO_MODEL_ID
DEFAULT_EXPERT_MODEL = "qwen-max"

_doubao_client = None
_qwen_client = None


def _openai_available() -> bool:
    try:
        import openai  # noqa: F401

        return True
    except Exception:
        return False


def _provider_for_model(model: str) -> str:
    # Follow the user's routing rule exactly.
    return "doubao" if model == DOUBAO_MODEL_ID else "qwen"


def has_credentials(model: str) -> bool:
    # If the OpenAI SDK isn't installed, we cannot call any provider.
    if not _openai_available():
        return False
    provider = _provider_for_model(model)
    if provider == "doubao":
        return bool(os.getenv("API_KEY_DOUBAO"))
    return bool(os.getenv("API_KEY_QWEN"))


def _get_client(provider: str):
    """
    Lazily build OpenAI-compatible clients for each provider to avoid repeating setup
    across evaluators.
    """
    global _doubao_client, _qwen_client

    # Lazy import so the project can be imported even if openai isn't installed yet.
    from openai import OpenAI

    if provider == "doubao":
        if _doubao_client is None:
            _doubao_client = OpenAI(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=os.getenv("API_KEY_DOUBAO"),
            )
        return _doubao_client

    if _qwen_client is None:
        _qwen_client = OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("API_KEY_QWEN"),
        )
    return _qwen_client


def call_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    if not _openai_available():
        raise RuntimeError("Missing dependency: openai (pip install -r demo/requirements.txt)")
    provider = _provider_for_model(model)
    if not has_credentials(model):
        missing = "API_KEY_DOUBAO" if provider == "doubao" else "API_KEY_QWEN"
        raise RuntimeError(f"Missing env var {missing} for model={model}")

    client = _get_client(provider)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


# Backward-compatible alias for the exact name used in the user's snippet.
def _call_openai_chat(model: str, messages: List[Dict[str, str]]) -> str:
    return call_chat(model=model, messages=messages, temperature=0.0)
