# SPDX-License-Identifier: Apache-2.0
"""Minimal, reproducible vLLM V1 CPU inference smoke test."""

import os

# Set these before importing vLLM so platform discovery sees them.
os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")
os.environ.setdefault("VLLM_CPU_OMP_THREADS_BIND", "0-7")

from vllm import LLM, SamplingParams
from vllm.version import __version__


def main() -> None:
    model = "hmellor/tiny-random-LlamaForCausalLM"
    llm = LLM(
        model=model,
        dtype="float32",
        enforce_eager=True,
        max_model_len=128,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    params = SamplingParams(temperature=0, max_tokens=8)
    prompts = [
        "The Linux scheduler",
        "Paged virtual memory",
    ]

    print(f"vLLM={__version__}")
    print(f"model={model}")
    for output in llm.generate(prompts, params):
        generated = output.outputs[0]
        print(
            f"prompt={output.prompt!r} "
            f"token_ids={generated.token_ids} text={generated.text!r}"
        )


if __name__ == "__main__":
    main()
