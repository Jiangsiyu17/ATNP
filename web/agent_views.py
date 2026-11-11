import json
import os

import requests
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


def _agent_url(path):
    base = os.environ.get("ATNP_QA_URL", "http://127.0.0.1:8080").rstrip("/")
    return f"{base}/expert-qa/api/{path.lstrip('/')}"


def expert_qa(request):
    return render(request, "web/expert_qa.html")


def _upstream_error(exc):
    return JsonResponse({"detail": f"问答服务暂时不可用：{exc}"}, status=503)


def agent_health(request):
    try:
        response = requests.get(_agent_url("health"), timeout=8)
        return JsonResponse(response.json(), status=response.status_code)
    except (requests.RequestException, ValueError) as exc:
        return _upstream_error(exc)


def agent_models(request):
    try:
        response = requests.get(_agent_url("models"), timeout=8)
        return JsonResponse(response.json(), status=response.status_code)
    except (requests.RequestException, ValueError) as exc:
        return _upstream_error(exc)


@csrf_exempt
def agent_chat_stream(request):
    if request.method != "POST":
        return JsonResponse({"detail": "Only POST is supported."}, status=405)
    try:
        payload = json.loads(request.body.decode("utf-8"))
        upstream = requests.post(
            _agent_url("chat/stream"),
            json=payload,
            stream=True,
            timeout=(10, 1800),
            headers={"Accept": "text/event-stream"},
        )
        if upstream.status_code >= 400:
            try:
                data = upstream.json()
            except ValueError:
                data = {"detail": upstream.text or "问答服务请求失败"}
            upstream.close()
            return JsonResponse(data, status=upstream.status_code)

        def events():
            try:
                for chunk in upstream.iter_content(chunk_size=1024):
                    if chunk:
                        yield chunk
            finally:
                upstream.close()

        response = StreamingHttpResponse(events(), content_type="text/event-stream")
        response["Cache-Control"] = "no-cache, no-transform"
        response["X-Accel-Buffering"] = "no"
        return response
    except (requests.RequestException, ValueError, UnicodeDecodeError) as exc:
        return _upstream_error(exc)


@csrf_exempt
def agent_chat_cancel(request, request_id):
    if request.method != "POST":
        return JsonResponse({"detail": "Only POST is supported."}, status=405)
    try:
        token = request.headers.get("X-Cancel-Token", "")
        response = requests.post(
            _agent_url(f"chat/{request_id}/cancel"),
            headers={"X-Cancel-Token": token},
            timeout=8,
        )
        try:
            data = response.json()
        except ValueError:
            data = {"detail": response.text or "取消请求失败"}
        return JsonResponse(data, status=response.status_code)
    except requests.RequestException as exc:
        return _upstream_error(exc)
