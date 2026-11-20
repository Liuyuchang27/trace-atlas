# Trace Atlas

Trace Atlas is a local news investigation workbench. A research agent searches the web, chooses which pages to read, and connects source excerpts to an event timeline. The workbench shows actual searches, source reads, evidence checks, follow-up activity, and unknowns. Read the report alongside its citations; recorded findings still require judgment.

## Try it locally

Install [uv](https://docs.astral.sh/uv/) and clone this repository. The project uses Python 3.12.

```bash
uv sync --frozen
uv run --frozen news-agent serve --port 8768
```

Open [http://127.0.0.1:8768](http://127.0.0.1:8768). Choose **新建调查 → 先试离线合成样例** for a scripted demonstration that uses no API keys or paid requests. The sample uses a fictional acquisition; it does not answer a real question.

## Run a live investigation

```bash
cp .env.example .env
```

On PowerShell, use `Copy-Item .env.example .env`. Enter your own DeepSeek and Tavily keys in `.env` and restart the local workbench. Use **新建调查 → 开始调查** to run a question of your choice. Live use can incur API charges. Start with the default budget.

| Setting | What to enter |
| --- | --- |
| `NEWS_AGENT_MODEL` | `deepseek:deepseek-flash` |
| `DEEPSEEK_API_KEY` | Your DeepSeek API key |
| `TAVILY_API_KEY` | Your Tavily API key |

The workbench exposes the **调查过程** view for observing and replaying actions, **事件时间线** for dated milestones, and **研究报告** and **证据库** for checking citations. The local server binds to `127.0.0.1`. Each teammate should run it on their own machine and keep `.env`, question history, and generated `runs/` private. Avoid putting sensitive information in live research questions, which are sent to model and search providers as needed.

The staged profile uses DeepSeek Flash and DeepSeek V4 Pro and enables high reasoning for the isolated date and report reviews. Transient DeepSeek request failures can retry up to three times, with delays of 12, 16, and 20 seconds. Request counts and token limits apply to the shared run budget. You can inspect the retry in the workbench.

## Recent improvements

- For recent-fact questions, the agent checks the publisher and its disclosures. For sales questions it can look back to the previous reporting period and read the new candidates. The report leads with the available data, labels third-party preorder estimates, and states the investigation cutoff and unresolved questions.
- For monthly trends, the agent tracks each requested month separately and can switch to a historical table when a summary page leaves gaps. The report links the original passages behind the trend chart.
- The investigation can replan an evidence gap, try a different source or date range, and continue with other questions when one gap stalls. The report export preserves citations to saved passages.

## Command line and tests

Run the synthetic sample directly:

```bash
uv run --frozen news-agent run --task data/fixtures/task.json --corpus data/fixtures/corpus.json --mock
```

Run the included regression tests, which use fictional data and simulated provider responses:

```bash
uv run --frozen pytest
uv run --frozen ruff check src tests
```

`src/news_agent/` contains the agent, evidence store, and local web workbench. `configs/` includes the default budget and opt-in diagnostics. The uncapped diagnostic removes aggregate limits and can consume significantly more API usage; it is intended for debugging. `data/fixtures/` contains synthetic examples. Historical experiment outputs and classroom materials are separate from this trial edition.
