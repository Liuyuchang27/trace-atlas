import json
from pathlib import Path

import pytest

from news_agent.schemas import ResearchTask

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def corpus_path():
    return ROOT / "data/fixtures/corpus.json"


@pytest.fixture
def task():
    return ResearchTask.model_validate(json.loads((ROOT / "data/fixtures/task.json").read_text()))
