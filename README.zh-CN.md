# Chunk-wise Data Synthesis

[English](./README.md) | [简体中文](./README.zh-CN.md)

这是一个受 Kimi-K2 启发的、带测试覆盖的最小化 chunk-wise 长文本合成实现，包含两条并行流水线：

1. `ChunkWiseRephrasePipeline`：忠实改写（rephrase）。
2. `ChunkWiseGenerationPipeline`：基于计划与状态表的纯生成（from scratch）。

## 功能特性

- 支持 overlap 的 chunk 切分与拼接。
- 基于滚动前缀窗口的自回归生成。
- rephrase 与 generation 并行共存、互不干扰。
- 纯生成支持 Plan + State + Consistency Pass。
- 可插拔质量控制（覆盖率/术语一致性/重复与漂移检测）。
- OpenAI 兼容后端，支持环境变量与脚本参数配置。

## 项目结构

```text
src/
  chunking.py             # chunk 切分与 overlap 逻辑
  pipeline.py             # rephrase 流程编排
  prompting.py            # rephrase 提示词
  generation_pipeline.py  # generation 流程编排
  generation_prompting.py # generation 提示词模板
  generation_state.py     # 状态表更新逻辑
  generation_quality.py   # 质量检查器
  generation_types.py     # generation 数据结构
  openai_backend.py       # OpenAI 兼容后端
  fidelity.py             # rephrase 保真度校验器
tests/
  test_*.py               # 基于 unittest 的确定性测试
scripts/
  run_live_openai_pipeline.py             # live rephrase 脚本
  run_live_openai_generation_pipeline.py  # live generation 脚本
```

## 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 运行测试

运行全部测试：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

迭代时运行单个模块：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_generation_pipeline.py' -v
```

## 真实流程运行（Rephrase）

```bash
export LLM_API_KEY=your_key_here
python3 scripts/run_live_openai_pipeline.py \
  --input tests/data/live_rephrase_input.txt \
  --output tests/data/rephrase_output.txt
```

## 真实流程运行（Generation）

```bash
export LLM_API_KEY=your_key_here
python3 scripts/run_live_openai_generation_pipeline.py \
  --topic "Chunk-wise autoregressive long-form generation" \
  --objective "Create long-context training text" \
  --target-tokens 1800 \
  --audience "ML engineers" \
  --tone "neutral technical" \
  --output tests/data/generation_output.txt
```

也可以传入手工计划（JSON）：

```bash
python3 scripts/run_live_openai_generation_pipeline.py \
  --manual-plan-path tests/data/manual_plan.json \
  --output tests/data/generation_output.txt
```

## Live 集成测试（默认不启用）

live 集成测试会发起真实 API 请求，默认跳过：

```bash
export LLM_API_KEY=your_key_here
export RUN_LIVE_LLM_TESTS=1
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_openai_backend_live.py' -v
```

## 配置项

环境变量：

- `LLM_API_KEY`（必填）：API Key。
- `LLM_MODEL`（可选）：覆盖模型 ID。
- `LLM_BASE_URL`（可选）：覆盖 API Base URL。

`/Users/H/Documents/workspace_python/chunk_wise_data_synthesis/src/openai_backend.py` 中当前默认值：

- `DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"`
- `DEFAULT_MODEL = "stepfun/step-3.5-flash:free"`

脚本也支持参数覆盖：

- `--model`
- `--base-url`
- `--temperature`
- `--top-p`
- `--max-new-tokens`

## 最小 API 用法

### Rephrase 流水线

```python
from pipeline import ChunkWiseRephrasePipeline, PipelineConfig
from prompting import RewriteRequest
from tokenizer import WhitespaceTokenizer


class EchoRewriteModel:
    def rewrite(self, request: RewriteRequest) -> str:
        return request.current_chunk


pipeline = ChunkWiseRephrasePipeline(
    model=EchoRewriteModel(),
    tokenizer=WhitespaceTokenizer(),
    config=PipelineConfig(chunk_tokens=256, overlap_tokens=64, prefix_window_tokens=1024),
)

rewritten = pipeline.run("Your long document here.", style_instruction="Rewrite for clarity.")
print(rewritten)
```

### Generation 流水线（manual plan）

```python
from generation_pipeline import ChunkWiseGenerationPipeline
from generation_types import GenerationConfig, GenerationPlan, SectionSpec
from model import LLMRequest
from tokenizer import WhitespaceTokenizer


class StubLLM:
    def generate(self, request: LLMRequest) -> str:
        if request.task == "section_generation":
            return "Section body with required entities and key points."
        if request.task == "consistency_pass":
            return "Section body with required entities and key points."
        raise ValueError("manual plan run should not call plan_generation")


plan = GenerationPlan(
    topic="Chunk-wise generation",
    objective="Teach the method",
    audience="ML engineers",
    tone="neutral technical",
    target_total_length=300,
    sections=[
        SectionSpec(
            title="Intro",
            key_points=["global anchor controls structure"],
            required_entities=["global anchor"],
            constraints=[],
            target_length=120,
        )
    ],
    terminology_preferences={"global anchor": "global anchor"},
    narrative_voice="third-person",
    do_not_include=[],
)

pipeline = ChunkWiseGenerationPipeline(
    model=StubLLM(),
    tokenizer=WhitespaceTokenizer(),
    config=GenerationConfig(prefix_window_tokens=800),
)

result = pipeline.run(manual_plan=plan)
print(result.final_text)
print(result.qc_report.coverage_missing)
```

## 常见问题

- 错误信息包含 `not a valid model ID`：
  请设置当前服务商可用模型，例如：
  `export LLM_MODEL=your_valid_model_id`。
- 提示缺少 API key：
  确认当前 shell 已导出 `LLM_API_KEY`。
