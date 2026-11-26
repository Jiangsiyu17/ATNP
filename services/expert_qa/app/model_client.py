from app.performance import model_span
import os
import requests
from app.request_queue import check_cancelled


# --------------------------------------------------------
# 模型服务配置
#
# 在 jhc 上暂时可以保持默认值；
# 同步到 SLURM 后，可以通过环境变量覆盖。
# --------------------------------------------------------

VLLM_BASE_URL = os.getenv(
    "VLLM_BASE_URL",
    "http://127.0.0.1:8000"
)

VLLM_API_KEY = os.getenv(
    "VLLM_API_KEY",
    "EMPTY"
)

VLLM_MODEL = os.getenv(
    "VLLM_MODEL",
    "qwen32b-atnp"
)


def call_llm(
    prompt,
    model=None,
    temperature=0.0,
    max_tokens=1500,
    timeout=180
):
    """
    调用本地 OpenAI-compatible API；使用调用方提供的 HTTP 超时。

    Parameters
    ----------
    prompt : str
        完整 Strict RAG prompt

    model : str | None
        模型名称。
        如果为 None，则使用环境变量 VLLM_MODEL。

    temperature : float
        Strict RAG 推荐 0 或很低的值。

    max_tokens : int
        最大输出 token 数。

    timeout : int
        HTTP 请求超时时间，单位秒。

    Returns
    -------
    str
        模型生成的回答文本。
    """

    check_cancelled()
    if not prompt or not prompt.strip():
        raise ValueError("prompt cannot be empty")

    model_name = model or VLLM_MODEL

    url = f"{VLLM_BASE_URL.rstrip('/')}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {VLLM_API_KEY}"
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    with model_span(model=model_name, max_output_tokens=max_tokens, http_timeout_seconds=timeout,
                    prompt_tokens=None, completion_tokens=None, total_tokens=None,
                    finish_reason=None, http_status=None) as metrics:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        check_cancelled()

        # 如果 vLLM 返回 401 / 404 / 500 等，
        # 这里直接抛出异常，方便排查。
        http_status = getattr(response, 'status_code', None)
        metrics['http_status'] = http_status if type(http_status) is int else None
        response.raise_for_status()

        data = response.json()
        usage = (data.get('usage') or {}) if isinstance(data, dict) else {}
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            value = usage.get(key) if isinstance(usage, dict) else None
            metrics[key] = value if type(value) is int and value >= 0 else None

        try:
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected vLLM response format: {data}"
            ) from exc

        reason = data['choices'][0].get('finish_reason')
        metrics['finish_reason'] = reason if isinstance(reason, str) else None
        return answer


def check_model_server(timeout=10):
    """
    检查 vLLM 服务是否可访问。

    Returns
    -------
    dict
        /v1/models 返回的 JSON。
    """

    url = f"{VLLM_BASE_URL.rstrip('/')}/v1/models"

    headers = {
        "Authorization": f"Bearer {VLLM_API_KEY}"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=timeout
    )

    response.raise_for_status()

    return response.json()


if __name__ == "__main__":

    print("Base URL:", VLLM_BASE_URL)
    print("Model:", VLLM_MODEL)

    try:
        models = check_model_server()

        print("Model server is reachable.")
        print(models)

    except Exception as e:

        print("Model server check failed:")
        print(type(e).__name__, str(e))
