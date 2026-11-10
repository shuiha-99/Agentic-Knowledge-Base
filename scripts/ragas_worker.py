"""Standalone JSONL worker for the pinned Ragas 0.3 environment.

Run with .venv-ragas/Scripts/python.exe, never import the LangGraph app here.
The application supplies its real dense embeddings; model-quality scores are
computed by Ragas itself, not by a look-alike formula in the application.
"""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import math
import os
import sys

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


def components():
    from langchain_core.embeddings import Embeddings
    from langchain_ollama import ChatOllama
    from ragas import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerCorrectness, Faithfulness, LLMContextRecall
    from ragas.run_config import RunConfig

    if importlib.metadata.version("ragas") != "0.3.3":
        raise RuntimeError("This worker requires ragas==0.3.3")
    return Embeddings, ChatOllama, SingleTurnSample, LangchainEmbeddingsWrapper, LangchainLLMWrapper, AnswerCorrectness, Faithfulness, LLMContextRecall, RunConfig


def build_metrics(config, cached, loaded):
    Embeddings, ChatOllama, SingleTurnSample, EmbeddingWrapper, LLMWrapper, AnswerCorrectness, Faithfulness, ContextRecall, RunConfig = loaded

    class CachedDense(Embeddings):
        def embed_documents(self, texts):
            # No invented fallback vector: exact sample strings must have been encoded.
            return [cached[text] for text in texts]

        def embed_query(self, text):
            return cached[text]

    keep_alive = config.get("score_keep_alive", 0 if config.get("low_memory_mode", True) else "5m")
    llm = ChatOllama(model=config["model"], base_url=config["base_url"], temperature=0, num_ctx=config["context_window_tokens"], num_predict=config["output_tokens"], keep_alive=keep_alive, client_kwargs={"timeout": config["request_timeout_seconds"], "trust_env": False})
    run_config = RunConfig(timeout=config["request_timeout_seconds"], max_retries=1, max_workers=1)
    wrapped_llm = LLMWrapper(llm, run_config=run_config)
    embeddings = EmbeddingWrapper(CachedDense(), run_config=run_config)
    metrics = [AnswerCorrectness(llm=wrapped_llm, embeddings=embeddings, weights=[0.75, 0.25]), ContextRecall(llm=wrapped_llm), Faithfulness(llm=wrapped_llm)]
    for metric in metrics:
        metric.init(run_config)
    return metrics


async def score(request, loaded):
    SingleTurnSample = loaded[2]
    config = request["config"]
    metrics = build_metrics(config, request["embeddings"], loaded)
    requested = request.get("metrics", ["answer_correctness", "context_recall", "faithfulness"])
    valid_names = {metric.name for metric in metrics}
    if not requested or any(name not in valid_names for name in requested):
        raise ValueError("Unknown or empty metric selection")
    metrics = [metric for metric in metrics if metric.name in requested]
    sample = SingleTurnSample(**request["sample"])
    scores, errors = {}, {}
    for metric in metrics:
        try:
            async with asyncio.timeout(config["metric_timeout_seconds"]):
                value = float(await metric.single_turn_ascore(sample))
            scores[metric.name] = value if math.isfinite(value) else None
            if not math.isfinite(value):
                errors[metric.name] = "Judge returned a non-finite score"
        except Exception as exc:
            scores[metric.name] = None
            errors[metric.name] = str(exc)
    return {"scores": scores, "errors": errors}


def main():
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    loaded = components()
    versions = {name: importlib.metadata.version(name) for name in ("ragas", "langchain-core", "langchain-community", "langchain-ollama")}
    if "--check" in sys.argv:
        config = {"model": "unused-in-import-check", "base_url": "http://127.0.0.1:11434", "context_window_tokens": 4096, "output_tokens": 256, "request_timeout_seconds": 30, "metric_timeout_seconds": 30}
        metrics = build_metrics(config, {}, loaded)
        print(json.dumps({"ready": True, "versions": versions, "metrics_initialized": [metric.name for metric in metrics]}))
        return
    # One event loop serves all requests; no nested event-loop monkey patching.
    with asyncio.Runner() as runner:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                result = runner.run(score(request, loaded))
            except Exception as exc:
                result = {"scores": {}, "errors": {"worker": str(exc)}}
            result["versions"] = versions
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
