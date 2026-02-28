# Chunk-wise Data Synthesis

[English](./README.md) | [简体中文](./README.zh-CN.md)

这是一个受 Kimi-K2 启发、带测试覆盖的 chunk-wise 长文本合成实现，包含两条并行流水线：

1. `ChunkWiseRephrasePipeline`：忠实改写（rephrase）。
2. `ChunkWiseGenerationPipeline`：基于计划驱动的 chunk-wise 长文生成。

## 功能特性

- 分层无 overlap 切分 + 基于重叠检测的拼接去重。
- 基于滚动前缀窗口的自回归生成。
- rephrase 与 generation 并行共存、互不干扰。
- rephrase 支持重试与可插拔保真度校验。
- generation 支持分节重试与问题定向修复提示词。
- 长上下文 generation 支持可选提示词压缩。
- 纯生成支持 Plan + State + Consistency Pass 守卫。
- 内置质量检查（覆盖率、术语一致性、重复、漂移、必需实体）。
- OpenAI 兼容后端，支持环境变量与脚本参数配置。

## 重构后架构

仓库现在按明确的领域边界组织：

- `pipelines/`：只做流程编排（`rephrase.py`、`generation.py`，共用逻辑在 `base.py`）。
- `prompts/`：只做提示词渲染（`rephrase.py`、`generation.py`，语言共用逻辑在 `base.py`）。
- `quality/`：质量与保真校验（`fidelity.py`、`generation.py`，文本/token 共用工具在 `base.py`）。
- `backends/`：模型后端适配器（`openai.py`）。
- `core/`：稳定的分组 API 导出（`protocols.py`、`types.py`、`config.py`）。
- 顶层领域模块继续保留：`chunking.py`、`generation_state.py`、`generation_types.py`、`model.py`。

以下旧 wrapper 模块已移除，不应再导入：`pipeline.py`、`prompting.py`、`fidelity.py`、`openai_backend.py`、`generation_pipeline.py`、`generation_prompting.py`、`generation_quality.py`、`tokenizer.py`。

## 项目结构

```text
src/
  __init__.py             # 包级统一公开导出
  chunking.py             # chunk 切分与 overlap 逻辑
  generation_state.py     # 生成状态表更新逻辑
  generation_types.py     # generation 数据结构与结果类型
  model.py                # 模型请求/任务协议与适配定义
  pipelines/
    __init__.py
    rephrase.py           # rephrase 流程编排 + PipelineConfig
    generation.py         # generation 流程编排
    base.py               # 重叠检测与拼接
  prompts/
    __init__.py
    rephrase.py           # RewriteRequest 与 rephrase 提示词
    generation.py         # 计划/分节/修复/一致性提示词
    base.py               # 提示词语言共用逻辑
  quality/
    __init__.py
    fidelity.py           # 保真校验协议与实现
    generation.py         # generation 质量检查器与一致性守卫
    base.py               # token/文本匹配共用工具
  backends/
    __init__.py
    openai.py             # OpenAI 兼容后端与配置
  core/
    __init__.py
    protocols.py          # Tokenizer/LLMModel/RewriteModel/FidelityVerifier
    types.py              # LLMRequest、RewriteRequest、GenerationPlan、SectionSpec
    config.py             # PipelineConfig、GenerationConfig、OpenAIBackendConfig
  tokenization/
    __init__.py           # 分词协议与工具
tests/
  test_*.py               # 基于 unittest 的确定性测试 + 重构兼容性测试
scripts/
  run_live_openai_pipeline.py             # live rephrase 脚本
  run_live_openai_generation_pipeline.py  # live generation 脚本
```

## 环境准备

项目使用 `uv` 管理依赖与运行环境。

```bash
uv sync
```

## 运行测试

运行全部离线测试：

```bash
uv run python -m unittest discover -s tests -v
```

迭代时运行单个模块：

```bash
uv run python -m unittest tests.test_generation_pipeline -v
```

验证重构后的 API 边界与导出：

```bash
PYTHONPATH=src:tests uv run python -m unittest \
  tests.test_package_entrypoint \
  tests.test_core_api_compat \
  tests.test_pipelines_api -v
```

## 真实流程运行（Rephrase）

```bash
export LLM_API_KEY=your_key_here
uv run python scripts/run_live_openai_pipeline.py \
  --input tests/data/live_rephrase_input.txt \
  --output tests/data/rephrase_output.txt
```

## 真实流程运行（Generation）

```bash
export LLM_API_KEY=your_key_here
uv run python scripts/run_live_openai_generation_pipeline.py \
  --topic "Chunk-wise autoregressive long-form generation" \
  --objective "Create long-context training text" \
  --target-tokens 1800 \
  --audience "ML engineers" \
  --tone "neutral technical" \
  --output tests/data/generation_output.txt
```

也可以传入手工计划（JSON）：

```bash
uv run python scripts/run_live_openai_generation_pipeline.py \
  --manual-plan-path tests/data/manual_plan.json \
  --output tests/data/generation_output.txt
```

## Live 集成测试（默认不启用）

live 集成测试会发起真实 API 请求，默认跳过：

```bash
export LLM_API_KEY=your_key_here
export RUN_LIVE_LLM_TESTS=1
uv run python -m unittest tests.test_openai_backend_live -v
```

## 公共导入入口

推荐使用分层导出入口：

- `from pipelines import ChunkWiseRephrasePipeline, ChunkWiseGenerationPipeline, PipelineConfig`
- `from prompts import RewriteRequest, render_rewrite_prompt, render_plan_prompt`
- `from quality import FidelityVerifier, CompositeFidelityVerifier, NumericFactChecker`
- `from backends import OpenAIBackendConfig, OpenAILLMModel, OpenAIRewriteModel`
- `from core.protocols import Tokenizer, LLMModel, RewriteModel, FidelityVerifier`
- `from core.types import LLMRequest, RewriteRequest, GenerationPlan, SectionSpec`
- `from core.config import PipelineConfig, GenerationConfig, OpenAIBackendConfig`

兼容性包级入口依然可通过 `src` 使用：

- `from src import ChunkWiseRephrasePipeline, PipelineConfig, RewriteRequest, WhitespaceTokenizer`

## 最小 API 用法

### Rephrase 流水线

```python
from core.config import PipelineConfig
from core.types import RewriteRequest
from pipelines import ChunkWiseRephrasePipeline
from tokenization import WhitespaceTokenizer


class EchoRewriteModel:
    def rewrite(self, request: RewriteRequest) -> str:
        return request.current_chunk


pipeline = ChunkWiseRephrasePipeline(
    model=EchoRewriteModel(),
    tokenizer=WhitespaceTokenizer(),
    config=PipelineConfig(
        chunk_size=256,
        length_mode="token",
        prefix_window_tokens=1024,
        max_stitch_overlap_tokens=64,
    ),
)

rewritten = pipeline.run("Your long document here.", style_instruction="Rewrite for clarity.")
print(rewritten)
```

### Generation 流水线（manual plan）

```python
from core.config import GenerationConfig
from core.types import GenerationPlan, LLMRequest, SectionSpec
from pipelines import ChunkWiseGenerationPipeline
from tokenization import WhitespaceTokenizer


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

## 配置项

环境变量：

- `LLM_API_KEY`（必填）：API Key。
- `LLM_MODEL`（可选）：覆盖模型 ID。
- `LLM_BASE_URL`（可选）：覆盖 API Base URL。

`src/backends/openai.py` 中当前默认值：

- `DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"`
- `DEFAULT_MODEL = "stepfun/step-3.5-flash:free"`

实时 rephrase 脚本参数（`scripts/run_live_openai_pipeline.py`）：

- `--chunk-size`
- `--length-mode`（`auto` / `token` / `char`）
- `--prefix-window-tokens`
- `--style`
- `--prompt-language`（`en` / `zh`）
- `--model`
- `--base-url`
- `--temperature`
- `--top-p`
- `--max-new-tokens`
- `--verbose`

实时 generation 脚本参数（`scripts/run_live_openai_generation_pipeline.py`）：

- `--topic`
- `--objective`
- `--target-tokens`
- `--audience`
- `--tone`
- `--prompt-language`（`en` / `zh`）
- `--manual-plan-path`
- `--prefix-window-tokens`
- `--disable-consistency-pass`
- `--enable-reasoning`
- `--model`
- `--base-url`
- `--temperature`
- `--top-p`
- `--max-new-tokens`
- `--verbose`

## 常见问题

- 错误信息包含 `not a valid model ID`：
  请设置当前服务商可用模型，例如：
  `export LLM_MODEL=your_valid_model_id`。
- 提示缺少 API key：
  确认当前 shell 已导出 `LLM_API_KEY`。
