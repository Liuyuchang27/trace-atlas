"""Attribute provider-reported usage at the call boundary, including concurrent branches."""
from contextlib import contextmanager
from contextvars import ContextVar


_actor = ContextVar("atlas_usage_actor", default=("coordinator", None))
_operation = ContextVar("atlas_usage_operation", default="other")


@contextmanager
def branch_scope(task):
    previous = _actor.set((task["id"], task["question"]))
    try:
        yield
    finally:
        _actor.reset(previous)


async def measured(operation, call, *args, **kwargs):
    previous = _operation.set(operation)
    try:
        return await call(*args, **kwargs)
    finally:
        _operation.reset(previous)


def increment(target, counts):
    for metric, value in counts.items():
        target[metric] = target.get(metric, 0) + value


class UsageLedger:
    def __init__(self, view=None):
        if view is None:
            view = {}
        self.total = view.setdefault("usage", {})
        self.rows = view.setdefault("usage_breakdown", {})
        if self.total and not self.rows:
            previous = self.total.copy()
            self.rows["prior_unattributed"] = {"total": previous, "operations": {"unknown": previous.copy()}}

    def add(self, actor, question, operation, counts):
        row = self.rows.setdefault(actor, {"total": {}, "operations": {}})
        if question is not None:
            row["question"] = question
        increment(row["operations"].setdefault(operation, {}), counts)
        increment(row["total"], counts)
        increment(self.total, counts)

    def record(self, **counts):
        actor, question = _actor.get()
        self.add(actor, question, _operation.get(), counts)

    def merge(self, other):
        for actor, row in other.rows.items():
            for operation, counts in row["operations"].items():
                self.add(actor, row.get("question"), operation, counts)
