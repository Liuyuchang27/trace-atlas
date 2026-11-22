# Trace Atlas

Trace Atlas 是一个面向深度研究和评测的工作台。输入问题后，研究内核按已有发现规划多条调查方向，并行检索、阅读原文，再决定补查或写作。工作台会实时画出分支和汇合点。点击节点可以查看该分支的结论、引文和来源。界面与报告支持简体中文、繁体中文和英文。

## 本地运行

需要 Python 3.11 或更高版本、uv 和 Node.js 22.12 或更高版本。首次运行，在项目根目录执行：

```sh
uv sync --locked
npm --prefix web ci
npm --prefix web run build
cp -n .env.example .env
```

在本机的 `.env` 填入 `DEEPSEEK_API_KEY`。也可以直接用 `--env-file` 指向已有的本地凭据文件。

```sh
uv run --locked atlas-serve --env-file .env --frontend-dist web/dist
```

打开 [本地工作台](http://127.0.0.1:8812/)。默认仅监听本机。SQLite 数据库位于 `.atlas/workbench.sqlite`。通过 `--port` 和 `--database` 可以启动独立实例。若要接已有工作台，可以将 `--frontend-dist` 指向其构建目录，或将原工作台开发代理的 `TRACE_ATLAS_API_URL` 指向这个服务。

## 调查过程

- 调度器根据当前证据选择下一批搜索。不同分支可以共用读过的原文。连续两轮没有新增可验证的原文片段时收束检索。常规流程没有应用层 token 总额或固定轮数。上游模型仍有自身的上下文和输出边界。长网页会分段阅读。
- 交互调查优先使用 DeepSeek Search 与 Fetch。提供 `TAVILY_API_KEY` 后，遇到相应的临时服务错误可回退。联网评测固定 DeepSeek，以便记录和比较搜索条件。
- 读者模型为每条发现返回原文片段。内核在实际读到的页面里匹配这段文字，并为来源分配引用编号。报告逐节生成并保留草稿，可下载 Markdown。表格、带数据的图表和有来源的图片可在工作台中展示。
- 可按需打开时间线。界面把相近进展归入时段，区分事件发生、公告、计划和来源发表。缺少事件日期时，有依据的来源日期会标为发表日期或“截至此日已有记录”。缺乏日期依据的材料仍保存在证据中。
- 调查支持停止、刷新后重连、利用已保存证据继续研究或直接写报告。新工作台会把上一轮结果带入追问。历史由本地服务保存。
- 要迁移上一版的 v2 会话，可以在启动时加 `--import-history /path/to/conversations.db`，输入应为已关闭并完成 SQLite 检查点的数据库备份。导入器只读打开备份、保留时间戳、跳过已删除项，并按会话 ID 去重。旧运行的中间检查点不能跨内核恢复，保存好的轮次和报告可以回读。

## 评测入口

`atlas-bench` 与工作台调用同一研究循环。输入使用含 `items` 的 JSON 或每行一题的 JSONL。每条记录提供 `id` 及 `question` 或 `prompt`，可附语言。Open-TLS 可提供 `end_date`。每次运行使用新的输出目录，保存配置、源码和输入指纹、token 统计、轨迹及预测。题目、评分参考、密钥和运行结果由使用者在本地准备。

| 套件 | 检索与输出 | 评分 |
| --- | --- | --- |
| Open-TLS | 联网检索，生成截止日内、有日级发生依据的事件。`--temporal-guidance` 控制规划时是否强调时间 | 对接 CHRONOS/TILSE，计算日期 F1 和对齐 ROUGE |
| BrowseComp-Plus | 使用官方 BM25 索引，锁定每查询 12 条候选，记录返回和打开的文档。生成前回读候选原文，核对题目要的属性 | 输出逐题官方提交格式，可用独立 qrels 算召回。答案裁判需另行运行 |
| DeepResearch Bench | 生成可阅读的报告和带显式来源链接的官方 JSONL | 支持生成 RACE/FACT 调用清单，或在裁判凭据齐备后执行 |

```sh
uv run atlas-bench open-tls --queries /path/to/tls.json --id TOPIC --temporal-guidance --env-file .env --out results/tls
/path/to/chronos-env/bin/python atlas_kernel/scoring.py open-tls results/tls --repo /path/to/CHRONOS --out results/tls-scores

uv run atlas-bench browsecomp-plus --queries /path/to/queries.json --id 772 --index /path/to/index --env-file .env --out results/browse
uv run atlas-score browsecomp-plus results/browse --qrels /path/to/qrel_evidence.txt --out results/browse-submission

uv run atlas-bench deepresearch-bench --queries /path/to/query.jsonl --id 31 --env-file .env --out results/reports
uv run atlas-score deepresearch-bench results/reports --repo /path/to/DeepResearchBench --judge both --plan --out results/report-submission
```

Open-TLS 可用 `--replay-evidence /path/to/previous/case/evidence.json` 在同一题、同一批已读材料上复测时间线整理。BrowseComp-Plus 需要 Java 21、`anserini-1.2.0-fatjar.jar` 及同目录的官方 `bm25/`。要实际执行 RACE/FACT，在最后一条命令中移除 `--plan`，通过 `--judge-python` 提供已安装官方依赖的解释器，通过 `--judge-env-file` 提供本地 OpenAI 或 OpenRouter 裁判凭据，FACT 另需 Jina 凭据。

## 项目结构

`atlas_kernel/research.py` 负责任务编排、取证和写作，`providers.py` 连接模型及检索服务，`evidence.py` 处理证据与时间线。`server.py` 和 `storage.py` 提供工作台协议及持久化。`bench.py` 与 `scoring.py` 分别负责生成和评分。`web/` 包含工作台、会话适配层及前端构建入口。
