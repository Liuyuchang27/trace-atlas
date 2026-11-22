"""Offline regressions: python -m unittest discover -s tests -v."""
import json
import unittest
from collections import deque
from unittest.mock import patch

import httpx

from atlas_kernel.providers import Services
from atlas_kernel.research import Investigation, new_view, new_work


def completion(content, finish="stop"):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"total_tokens": 7},
    }


def model_reply(content, finish="stop"):
    return "/chat/completions", completion(content, finish)


class ScriptedResponses:
    """All HTTP traffic terminates here; unexpected requests fail the test."""

    def __init__(self, steps):
        self.steps = deque(steps)
        self.requests = []

    def __call__(self, request):
        self.requests.append((request.url.path, json.loads(request.content)))
        if not self.steps:
            raise AssertionError("Unexpected HTTP request")
        path, body = self.steps.popleft()
        if request.url.path != path:
            raise AssertionError(f"Expected {path}, got {request.url.path}")
        return httpx.Response(200, json=body)


class GenerationRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Never read a developer's credentials, and never use a network transport.
        env = patch.dict("os.environ", {"DEEPSEEK_API_KEY": "offline-placeholder"})
        env.start()
        self.addCleanup(env.stop)

    async def service(self, steps):
        script = ScriptedResponses(steps)
        client = httpx.AsyncClient(transport=httpx.MockTransport(script))
        self.addAsyncCleanup(client.aclose)
        return Services(client, fallback=False), script

    async def test_repair_can_restart_with_the_same_truncated_prefix(self):
        svc, script = await self.service([
            model_reply('{"answer":', "length"),
            model_reply('not json}'),
            model_reply('{"answer":', "length"),
            model_reply('"ok"}'),
        ])
        self.assertEqual(await svc.generate("Answer", {}, structured=True), {"answer": "ok"})
        self.assertFalse(script.steps)
        self.assertEqual(dict(svc.usage), {"model_calls": 4, "total_tokens": 28})

    async def test_repeated_continuation_within_a_repair_still_fails(self):
        svc, script = await self.service([
            model_reply('{"answer":', "length"),
            model_reply('not json}'),
            model_reply('{"answer":', "length"),
            model_reply('{"answer":', "length"),
        ])
        with self.assertRaisesRegex(ValueError, "continuation stopped making progress"):
            await svc.generate("Answer", {}, structured=True)
        self.assertFalse(script.steps)
        self.assertEqual(svc.usage["model_calls"], 4)

    async def test_malformed_json_gets_only_one_full_repair(self):
        svc, script = await self.service([
            model_reply('{"answer":', "length"),
            model_reply('not json}'),
            model_reply('{"answer":', "length"),
            model_reply('still not json}'),
        ])
        with self.assertRaisesRegex(ValueError, "Empty or malformed model response"):
            await svc.generate("Answer", {}, structured=True)
        self.assertFalse(script.steps)
        self.assertEqual(svc.usage["total_tokens"], 28)

    async def test_empty_continuation_still_fails(self):
        svc, script = await self.service([model_reply(" ", "length")])
        with self.assertRaisesRegex(ValueError, "continuation stopped making progress"):
            await svc.generate("Answer", {}, structured=True)
        self.assertFalse(script.steps)

    async def test_investigation_finishes_after_planner_repair_and_writer_continuation(self):
        quote = "观测站共记录 2,000 个样本。"
        page = "背景资料。" * 800 + "\n" + quote
        url = "https://example.org/record"
        self.assertGreater(len(page.encode("utf-8")), 10_000)
        svc, script = await self.service([
            model_reply('{"branches":', "length"),
            model_reply('not json}'),
            model_reply('{"branches":', "length"),
            model_reply('[{"label":"Sample count","query":"sample count record"}]}'),
            ("/v1/tools/search", {"results": [{"url": url, "title": "Sample record", "snippet": "Recorded samples"}]}),
            model_reply('{"open":[1]}'),
            ("/v1/tools/fetch", {"content": page}),
            model_reply(json.dumps({"findings": [{"claim": quote, "quote": quote}]})),
            model_reply('{"branches":[]}'),
            model_reply('{"title":"Sample report","sections":[{"heading":"Findings","sources":[1]}]}'),
            model_reply("观测站共记录 ", "length"),
            model_reply("2,000 个样本。 [[1]]"),
        ])
        events = []

        async def emit(name, data, task):
            events.append(name)

        view, work = new_view(), new_work()
        result = await Investigation(
            {"inputs": "观测站记录了多少个样本？", "lang": "zh", "mode": "report"},
            view, work, svc, svc, emit,
        ).run()

        self.assertEqual(result["article"], "# Sample report\n\n## Findings\n\n" + quote + " [[1]]")
        self.assertEqual(result["ref2url"]["1"]["url"], url)
        self.assertTrue(all(t["status"] == "completed" for t in view["tasks"].values()))
        self.assertEqual(work["sources"]["1"]["notes"][0]["quote"], quote)
        self.assertIn("research.partial", events)
        self.assertFalse(script.steps)
        self.assertEqual(result["usage"]["model_calls"], 10)
        self.assertEqual(result["usage"]["total_tokens"], 70)
        self.assertEqual(result["usage"]["search_calls"], 1)
        self.assertEqual(result["usage"]["page_reads"], 1)
        materials = [json.loads(body["messages"][1]["content"])
                     for path, body in script.requests if path == "/chat/completions"]
        self.assertEqual([m["passage"] for m in materials if "passage" in m], [page])


if __name__ == "__main__":
    unittest.main()
