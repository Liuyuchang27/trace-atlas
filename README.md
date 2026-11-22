# Trace Atlas

Trace Atlas 是一个深度研究工作台。输入一个问题，研究代理会规划调查方向，并行搜索和阅读原文，再把各分支的发现写成带来源引用的 Markdown 报告。在线索图中可以看到调查如何展开、哪些分支还在查证，以及它们在哪里汇合。

## 从问题到报告

- **规划和查证**。代理会根据已经读到的材料调整计划、补查线索或结束分支。网页检索和正文读取优先使用 DeepSeek Search & Fetch。
- **研究线索图**。点击规划、查证或汇总节点，右侧面板会显示对应的调查问题、发现和原文来源。工作台支持简体中文、繁體中文和 English。
- **事件时间线**。勾选「整理事件时间线」后，可按时间段查看进展和证据。时间线区分事件发生、公告、计划和来源发表，便于核对每个日期究竟代表什么。
- **报告导出**。各分支的发现会汇总到同一份报告。报告支持 Markdown 下载、引用链接、表格、图片和数据图。

## 本地运行

准备 Python 3.12、uv 和 Node.js 20。克隆仓库后，在第一个终端启动 API。

```bash
cd app
uv sync --frozen --extra documents
cp .env.example .env
# 在 .env 填写 DEEPSEEK_API_KEY 后启动
uv run --no-sync trace-atlas-server --profile atlas --env-file .env --port 8802
```

在第二个终端从仓库根目录启动工作台。

```bash
cd app/frontend/React
npm ci
npm run start -- --host 127.0.0.1 --port 8881
```

打开 [本地工作台](http://127.0.0.1:8881/)。只配置 DeepSeek 密钥即可进行文字调查。工作台会先搜索 DeepSeek。填写可选的 `TAVILY_API_KEY` 后，DeepSeek 遇到配额、限流或暂时性服务故障时可转用 Tavily，图片检索也可使用 Tavily。若需固定网页检索服务，可在 `.env` 设置 `ATLAS_SEARCH_ENGINE=deepseek` 或 `tavily`。

## 项目结构

| 目录 | 内容 |
| --- | --- |
| [app/trace_atlas](app/trace_atlas/) | 研究代理、搜索、报告生成、事件流和本地历史记录 |
| [app/frontend/React](app/frontend/React/) | 研究工作台、线索图、时间线和报告阅读界面 |
| [app/.env.example](app/.env.example) | 队友在本机填写的配置模板 |

调查历史和生成的报告保存在运行工作台的机器上。每位成员使用自己的 DeepSeek 密钥。
