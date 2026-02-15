# Chunk-wise Data Synthesis

[English](./README.md) | [简体中文](./README.zh-CN.md)

这是一个受 Kimi-K2 启发的、带测试覆盖的最小化 chunk-wise 自回归改写实现：

1. 将长文本切分为多个 chunk。
2. 在已生成前缀 `y_<i>` 条件下改写第 `i` 个 chunk。
3. 将改写后的 chunk 拼接回完整文档。
4. 在启用校验器时，对低保真 chunk 进行重试。

## 功能特性

- 支持 overlap 的 chunk 切分与拼接。
- 基于滚动前缀窗口的自回归改写。
- 全局锚点控制（`head` 或 `none`）。
- 可插拔的保真度校验与重试机制。
- OpenAI 兼容后端，支持环境变量配置。

## 项目结构

```text
src/
  chunking.py         # chunk 切分与 overlap 逻辑
  pipeline.py         # 端到端流程编排
  prompting.py        # 提示词构造
  openai_backend.py   # 真实 API 后端
  fidelity.py         # 校验接口与默认实现
tests/
  test_*.py           # 基于 unittest 的确定性测试
scripts/
  run_live_openai_pipeline.py
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

迭代时运行单个测试模块：

```bash
PYTHONPATH=src python3 -m unittest tests.test_pipeline -v
```

## 真实流程运行（Live API）

当前真实端到端流程命令如下：

```bash
export LLM_API_KEY=your_key_here
python3 scripts/run_live_openai_pipeline.py --input tests/data/live_rephrase_input.txt --output tests/data/output.txt
```

说明：

- `--output` 可选；省略后会直接输出到 stdout。
- 示例输入文件：`tests/data/live_rephrase_input.txt`。
- 示例输出路径：`tests/data/output.txt`。

## Live 集成测试（默认不启用）

live 集成测试会发起真实 API 请求，默认跳过以避免意外成本：

```bash
export LLM_API_KEY=your_key_here
export RUN_LIVE_LLM_TESTS=1
PYTHONPATH=src python3 -m unittest tests.test_openai_backend_live -v
```

## 配置项

环境变量：

- `LLM_API_KEY`（必填）：API Key。
- `LLM_MODEL`（可选）：覆盖模型 ID。
- `LLM_BASE_URL`（可选）：覆盖 API Base URL。

`src/openai_backend.py` 中当前默认值：

- `DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"`
- `DEFAULT_MODEL = "stepfun/step-3.5-flash:free"`

也可通过脚本参数覆盖：

- `--model`
- `--base-url`
- `--temperature`
- `--top-p`
- `--max-new-tokens`

## 最小 API 用法

```python
from pipeline import ChunkWiseRephrasePipeline, PipelineConfig
from prompting import RewriteRequest
from tokenizer import WhitespaceTokenizer


class EchoModel:
    def rewrite(self, request: RewriteRequest) -> str:
        return request.current_chunk


pipeline = ChunkWiseRephrasePipeline(
    model=EchoModel(),
    tokenizer=WhitespaceTokenizer(),
    config=PipelineConfig(
        chunk_tokens=256,
        overlap_tokens=64,
        prefix_window_tokens=1024,
        fidelity_threshold=0.85,
        max_retries=2,
    ),
)

rewritten = pipeline.run(
    "Your long document here.",
    style_instruction="Rewrite for clarity while preserving facts.",
)
print(rewritten)
```

## 常见问题

- 错误信息包含 `not a valid model ID`：
  请设置当前服务商可用的模型，例如：
  `export LLM_MODEL=your_valid_model_id`。
- 提示缺少 API key：
  确认当前 shell 已导出 `LLM_API_KEY`。
