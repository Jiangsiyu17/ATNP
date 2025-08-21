from django.apps import AppConfig
import threading
import time
import sys

class WebConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'web'

    def ready(self):
        # 使用线程在后台预加载模型，避免阻塞服务器启动
        from .utils import identify

        def preload_models():
            try:
                print("[PRELOAD] Starting to load spec2vec models and HNSW indexes...", flush=True)
                start_time = time.time()
                # 预加载正负离子模式
                identify.load_model_index_refs("positive")
                identify.load_model_index_refs("negative")
                end_time = time.time()
                print(f"[PRELOAD] Finished loading models in {end_time - start_time:.2f}s", flush=True)
            except Exception as e:
                print(f"[PRELOAD] Error while loading models: {e}", file=sys.stderr, flush=True)

        threading.Thread(target=preload_models, daemon=True).start()