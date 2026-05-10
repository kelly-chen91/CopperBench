# Models and Systems Used

## Stage 1: Model Benchmark

| # | Model configuration | Reasoning effort | Notes |
|---:|---|---|---|
| 1 | `gpt-4.1` | none | Base GPT-4.1 model |
| 2 | `gpt-4.1-mini` | none | Smaller GPT-4.1 variant |
| 3 | `gpt-4.1-nano` | none | Smallest GPT-4.1 variant |
| 4 | `gpt-4o` | none | GPT-4o baseline |
| 5 | `gpt-4o-mini` | none | Smaller GPT-4o variant |
| 6 | `gpt-5.4` | low | Reasoning model |
| 7 | `gpt-5.4` | medium | Reasoning model |
| 8 | `gpt-5.4` | high | Reasoning model |
| 9 | `gpt-5.4-mini` | low | Best Stage 1 MAE |
| 10 | `gpt-5.4-mini` | medium | Main model used for later stages |
| 11 | `gpt-5.4-mini` | high | Reasoning model |
| 12 | `gpt-5.4-nano` | low | Small reasoning model |
| 13 | `gpt-5.4-nano` | medium | Small reasoning model |
| 14 | `gpt-5.4-nano` | high | Small reasoning model |
| 15 | `gpt-5.4-pro` | high | Pro reasoning model |
| 16 | `gpt-5.5` | low | Reasoning model |
| 17 | `gpt-5.5` | medium | Reasoning model |
| 18 | `gpt-5.5` | high | Reasoning model |

## Stages 2-3: Prompt Variants

All Stage 2, Stage 2.5, and Stage 3 prompt experiments used:

| Base model | Reasoning effort |
|---|---|
| `gpt-5.4-mini` | medium |

| Prompt/system variant | Stage used | Purpose |
|---|---|---|
| Baseline | Stage 2 | Control prompt with no added mitigation |
| Persona | Stage 2, Stage 3 | Dietitian-style estimation protocol |
| Few-shot | Stage 2 | Examples showing how to estimate copper |
| Chain-of-thought / CoT | Stage 2 | Explicit reasoning-oriented prompt |
| Combined persona + few-shot | Stage 2.5, Stage 3 | Persona protocol plus worked examples |

## Short Presentation Summary

| Experimental phase | What was compared |
|---|---|
| Stage 1 | 18 model configurations across GPT-4.1, GPT-4o, GPT-5.4, GPT-5.4-mini, GPT-5.4-nano, GPT-5.4-pro, and GPT-5.5 |
| Stage 2 | Prompt variants using `gpt-5.4-mini` medium reasoning without ingredient proportions |
| Stage 3 | Targeted mitigation prompts using `gpt-5.4-mini` medium reasoning with ingredient proportions |
