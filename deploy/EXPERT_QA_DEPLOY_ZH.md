# ATNP Expert Q&A 一体化部署

主站保留一次性问答页面。检索、Agent、生成和声明核验仍由独立
FastAPI 进程完成；模型由独立 vLLM 进程完成。这样用户看到的是一个
ATNP 网站，但 Django 不加载大模型，也不会因模型进程异常而整体退出。

## 目录与数据

- `services/expert_qa/`：问答 API 的运行时代码和小型实体配置。
- `crawler/knowledge_base_full.duckdb`：Knowledge Search 与 Expert Q&A
  共用的只读知识图谱。
- `crawler/passages.sqlite`：Expert Q&A 使用的全文 FTS5 索引。该大文件
  被 Git 忽略，迁移服务器时必须单独复制并校验哈希。
- 模型权重不放入代码仓库，通过 `ATNP_MODEL_PATH` 指向数据盘目录。

## 启动顺序

1. 配置 Django 的 MySQL、`DJANGO_SECRET_KEY` 和允许域名。
2. 启动模型：`bash deploy/start_expert_qa_model.sh`。
3. 可选启动神经重排：`bash deploy/start_expert_qa_reranker.sh`；未启动时
   hybrid 模式自动回退到规则重排。
4. 启动问答 API：`bash deploy/start_expert_qa_api.sh`。
5. 启动 Django/Gunicorn。
6. 配置 Nginx；将 `/expert-qa/api/` 直接转发到 `127.0.0.1:8080`，
   其他路径转发到 Django。

## 上线前检查

```bash
test -r crawler/knowledge_base_full.duckdb
test -r crawler/passages.sqlite
curl http://127.0.0.1:18000/v1/models
curl http://127.0.0.1:8080/expert-qa/api/health
curl http://127.0.0.1:8080/expert-qa/api/models
```

页面通信协议没有改变，仍使用 POST SSE。主站当前
`web/static/js/atnp-sse-client.js`、`web/agent_views.py` 和一次性页面可以继续使用。

## 重要说明

- 不要在两个目录保存两份知识图谱；两个功能通过 `ATNP_KG_DB` 指向同一文件。
- SQLite 和 DuckDB 均以只读方式检索。
- `crawler/` 已在 `.gitignore` 中，部署代码时不能依赖 Git 自动带上数据库。
- 当前页面不保存历史会话；每次提交仍会生成独立 `conversation_id`，用于排队、取消和日志隔离。
