# 评测方法与结果边界

## 数据集

`data/eval/demo_cases.jsonl`含120条演示题：84条单文档、20条跨文档标签、6条歧义、6条无答案、4条多轮上下文题。当前按44个语义问题族分为32条dev与88条test；构建脚本显式合并相同核心关系的改写、跨文档组合及历史追问，再按整个问题族分配划分。开发调参使用dev，后续完整评测固定test。

本轮任务已扩展为完整固定88题问答、全部适用题的真实RAGAS评分，以及预先锁定小集的三次稳定性实验。完成情况以本次冻结运行的`manifest.json`、`journal.jsonl`和`summary.json`为准；已有单样本结果不代表完整评测已经完成。

104条单文档和跨文档题包含`reference_quotes`，构建脚本验证每个锚点确实存在于对应原文。**锚点存在不等于标准答案已经得到独立人工审查**。目前数据明确标注`ai_assisted_source_anchors_checked_not_independent_human_review`或`ai_assisted_manual_review_required`，应在人工核对答案完整性、歧义范围和参考文件后再建立正式gold版本。

语义归组也是AI辅助审阅，不能声称已穷尽所有近似题。`original_topic`保留修订前的主题名，`SEMANTIC_GROUPS`列出显式合并关系，方便人工复核。跨文档标签代表所列参考资料组合，并不都必须多文档才能回答；完成这些题不能单独证明多Agent的必要性或收益。详见`DATASET_REVIEW.md`。

重新生成并检查原文锚点：

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe scripts/build_demo_eval.py
```

长对话题会先回放`turns`中的多轮对话，再问最终问题。修订后的最终追问不再重复前文版本、写入状态或连接参数；连接预算题使用用户前文指定的非默认参数，不能直接套文档默认值。v1题的中间问题只涉及该版本适用的启动、健康检查和日志主题。它们是否触发压缩仍取决于实际token长度与窗口，不能仅凭“有9轮”就认定已测试到压缩；应检查逐条compression_records，并另建强制触发的压力集。

## RAGAS的独立环境

不要在应用主环境直接安装Ragas0.3.3：其导入的VertexAI等旧路径与LangChain1.x不兼容，依赖解析成功也不代表可以运行。项目使用独立评测环境，版本锁在`docs/requirements-ragas.lock.txt`。

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv venv .venv-ragas --python 3.11
uv pip install --python .venv-ragas\Scripts\python.exe -r docs/requirements-ragas.lock.txt
.venv-ragas\Scripts\python.exe scripts/ragas_worker.py --check
```

`--check`验证真实包导入、版本及三个指标的构造与初始化，不调用模型，也不能证明评判质量。实际评分仍需Ollama中的评判模型可用。可配置：

```powershell
$env:KB_EVAL_MODEL = 'qwen2.5:7b'
$env:KB_EVAL_BASE_URL = 'http://127.0.0.1:11434'
# 评测环境或worker放在别处时再设置：
# $env:KB_RAGAS_PYTHON = '绝对路径\python.exe'
# $env:KB_RAGAS_WORKER_PATH = '绝对路径\ragas_worker.py'
```

从项目根目录运行。常驻worker通过JSONL接收问题、标准答案、系统答案、实际上下文与应用计算的真实Dense向量。worker输出仅为协议JSON，诊断日志走stderr；超时与退出会关闭进程，不会用本地公式冒充RAGAS。低内存模式会设置Ollama keep_alive=0。

## 执行与消融

先确认索引和模型可用，再为每次运行使用新的输出目录：

```powershell
.venv\Scripts\python.exe -m agentic_kb evaluate --dataset data/eval/demo_cases.jsonl --split test --mode dense --output-dir runtime/evaluations/dense-test
.venv\Scripts\python.exe -m agentic_kb evaluate --dataset data/eval/demo_cases.jsonl --split test --mode agent --ragas --output-dir runtime/evaluations/agent-test-ragas
```

模式依次为`dense`、`hybrid`、`rerank`、`parent`、`agent`。前四种用于比较单问题检索链，agent再增加任务规划与有限补检。当前压缩在图的预算管理中统一执行，不能把上述模式误写成独立的“关闭压缩”实验；验证压缩收益需固定其他参数，单独构建窗口与压缩策略对照，并标明溢出样本。

主程序退出后才可另开同一本地Qdrant路径。短期流程检查可以用`--offline-test`，但离线模式禁止`--ragas`，得到的报告标记`quality_evaluation=false`。

## 产物与分母

- `manifest.json`：数据集哈希、配置、应用代码哈希、包版本、知识库文档版本、评判模型、指标权重及worker版本。
- `dataset_snapshot.jsonl`：本次选中题目的完整快照。
- `results.jsonl`：每一道题的答案、来源、上下文范围、事件、状态、耗时、压缩记录、逐指标得分及错误。失败题保留，不从结果文件移除。
- `summary.json`：完成/澄清/失败/超时数量，题型分布，检索诊断、RAGAS有效样本均值与失败数、压缩统计和延迟。

歧义、无答案题单独报告，不进入普通RAGAS分母。歧义题执行完毕不能自动说明澄清问得正确；无答案题完成执行也不等于正确拒答，均应人工核对。

评分失败保留为null。`mean_valid_only`只统计有效评分，旁边同时给出`valid_count/eligible_count/missing_or_failed_count`；缺失率高时不能单看均值。关键词覆盖仅为诊断，不能当作回答正确率。

## 指标口径

AnswerCorrectness使用Ragas0.3.3实现，事实与语义相似度权重为0.75/0.25；它不是“回答正确的问题数/总题数”。ContextRecall检查参考答案信息是否被上下文覆盖，不是文档Recall@K。Faithfulness检查回答是否受上下文支持。评判模型与生成模型相同时存在共同偏差，应结合人工审查，必要时使用独立更强评判模型。

RAGAS优先使用`generation_contexts`中的最终生成上下文，不用被压缩前的完整父块替代模型实际看到的证据。每题`context_scope`说明口径；旧引擎未提供该字段时会明确标注降级到原始检索来源。

额外的`document_recall_at_5/document_mrr`以最终来源中的**去重文档顺序**为单位，并仅检查题集指定的参考文件。当前参考文件不是所有等价来源的穷尽标注，故不能将结果称为全量相关性gold上的召回率；Agent模式又是跨任务汇总后的来源顺序，不应冒充单次检索器的chunk级Recall@5/MRR。

压缩比统一为after_tokens/before_tokens，报告样本均值和token总量加权比。未触发时比例可能为1。各阶段压缩对象的token合计不等于模型实际计费token，也不等于整段对话体积。0.55～0.7仅为原项目描述中的目标，必须结合触发比例、损失标记、约束与关键事实保留来判断。

AnswerCorrectness约0.87、ContextRecall约0.82均没有预先保证。若要使用“稳定”，需固定模型、知识库、题集、提示词和参数，至少重复多次实验并报告波动；不能把最好的一次当作稳定成绩。

官方定义：[Ragas0.3.3指标](https://docs.ragas.io/en/v0.3.3/references/metrics/)、[ContextRecall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)。

## 完整运行：冻结、分阶段、断点续跑

长时间正式实验使用`scripts/benchmark.py`，不要用旧的一次性`evaluate`命令承载可恢复的全量运行。正式runner只接受实际Ollama模式。开发集调优使用独立目录和`--split dev`；确定配置后再冻结test。查看test结果后修改参数需要新建实验目录，不能覆盖原运行或把调优后的结果混入旧报告。

当前固定test为88题：59条单文档、14条跨文档标签、5条歧义、6条无答案、4条多轮上下文题。其中77条符合普通RAGAS条件，主轮应有231个逐指标记录。歧义和无答案仍完成问答/交互并计入88题执行状态，只是不进入普通RAGAS分母。生成失败也保留在77题分母中，评分记为blocked，而不是被删掉。

下面各命令使用完全相同的冻结参数，按顺序执行；**停止使用该知识库的其他UI/CLI重模型任务后再运行**。正式运行目录只允许一个写进程，操作系统锁在崩溃后会自动释放。

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$taskRun = 'runtime/benchmarks/frozen-test-v1'
$taskCommon = @('--run-dir', $taskRun, '--metric-timeout', '900', '--score-keep-alive', '10m')

# 仅读取元数据，不推理；确认完整题集和稳定性范围后冻结。
.venv\Scripts\python.exe scripts/benchmark.py prepare @taskCommon

# 完整88题的第一轮。每个多轮题的中间turn也单独持久化。
.venv\Scripts\python.exe scripts/benchmark.py generate @taskCommon --scope main

# 锁定范围再追加两次；第一次直接复用main中同配置的真实结果。
.venv\Scripts\python.exe scripts/benchmark.py generate @taskCommon --scope stability

# 全部生成结束后，先集中准备真实answer/reference Dense向量。
.venv\Scripts\python.exe scripts/benchmark.py prepare-vectors @taskCommon

# 只读取已有生成与向量，逐指标评分；不会重新生成答案或临时加载BGE。
.venv\Scripts\python.exe scripts/benchmark.py score @taskCommon

# 全量检索消融：dense/hybrid/rerank/parent，不进行问答生成或LLM评判。
.venv\Scripts\python.exe scripts/benchmark.py retrieval @taskCommon

# 查看已写报告，既不推理也不重新计分。
.venv\Scripts\python.exe scripts/benchmark.py report --run-dir $taskRun
```

`prepare`会冻结完整数据集字节SHA、所选case与参考答案、稳定性样本ID和次数、全部应用配置、核心/runner/worker源码SHA、活动文档版本和字节SHA、Ollama实际模型digest、HF缓存snapshot revision和配置SHA、主环境与worker环境包版本、评判模型、权重及驻留策略。启用结构化辅助模型时，同样记录其实际Ollama digest、辅助tokenizer的HF revision，以及`structured_model_schemas`白名单；非空白名单只让其中命名的schema使用辅助模型，其他schema走主模型。后续每个执行阶段都会重新核对；任何差异都会拒绝续跑。模型必须先准备好，不能只按可变模型名称冻结。`report`展示已有历史报告，不宣称当前环境仍与历史一致。

默认稳定性小集在任何新分数出现前，按预定题型选择首题：`demo-001`、`demo-085`、`demo-111`，分别是单文档、跨文档标签、无答案题。主轮为第1次，再追加2次；因此合计94个生成slot，81个eligible生成slot对应243个指标slot。若需要其他更有代表性的固定小集，在**首次prepare前**用`--stability-ids ID1,ID2,ID3`显式确定，所有阶段保持一致。不能看完结果后再改选容易或高分样本。

本轮正式计划为**全88题各3轮**。必须在首次`prepare`之前，把全部test ID加入公共参数；之后所有执行阶段沿用同一个`$taskCommon`，再依次执行上面的命令。默认小集示例与本方案使用不同目录，不可给已冻结的小集目录追加这些参数。

```powershell
$taskRun = 'runtime/benchmarks/frozen-test-88x3-v1'
$taskTestIds = (.venv\Scripts\python.exe -c "from pathlib import Path; from agentic_kb.evaluation import load_dataset; print(','.join(c.id for c in load_dataset(Path('data/eval/demo_cases.jsonl'), 'test')))").Trim()
$taskCommon = @('--run-dir', $taskRun, '--metric-timeout', '900', '--score-keep-alive', '10m', '--stability-repeats', '3', '--stability-ids', $taskTestIds)
```

全量方案合计264个生成slot，231个eligible生成slot对应693个指标slot；第1轮直接复用main，追加第2、3轮，不另多生成一轮。`summary.json`按所选ID数量动态标注`full_dataset`或`subset`，给出main首轮、稳定性逐case、逐repeat，以及“每轮所选题目均值”的均值、总体标准差、最小/最大值。计划范围不代表已经完成，需查看实际执行计数。只有一个repeat的所有eligible指标均有效，才进入完整repeat均值统计；完整repeat数量不足3时必须说明缺失。小集三次不等于全88题三次。无答案题没有普通RAGAS分数，完成执行不自动代表拒答质量合格。

### CPU与模型驻留

应用和独立RAGAS worker调用配置的Ollama服务时使用`client_kwargs={"trust_env": False, ...}`直连，避免本机请求被Windows系统代理转发。这是模型调用连接设置，RAGAS指标、权重和超时定义保持一致。

执行顺序是生成全部完成→集中编码→卸载BGE→集中评分。向量阶段先对配置的Ollama服务执行仅卸载请求，再单独保持BGE驻留以避免反复装权重；向量写完后清除模型引用并回收内存。评分阶段不构造BGE，只加载已保存的向量。所有待评分项的向量会在首个judge请求前预检，缺失则明确要求先完成prepare-vectors。

benchmark的`--score-keep-alive 10m`只控制独立judge，不改变应用原有低内存行为；它被写进冻结清单。评分按指标批处理，便于相同模板前缀使用Ollama的KV缓存。没有缓存LLM输出，也没有改变RAGAS定义、提示词或0.75/0.25权重。检索消融阶段同样先卸载Ollama，只保持Dense和Cross-Encoder模型；不会在judge驻留时再装BGE。

### 中断与重试的真实边界

`journal.jsonl`使用唯一的case/repeat/attempt和metric键，每个terminal记录立即flush/fsync；最终JSON采用临时文件后原子替换。运行重启后，已提交turn、生成结果、向量和指标不会重做或重复计分。模型已给出done且事件已落盘、但生成汇总尚未提交时，会从durable事件恢复结果。未写完的journal尾行先留备份再移除，损坏的中间行直接拒绝读取。

如果进程在外部模型调用完成之前被强制终止，不能承诺该外部调用恰好一次。未提交且没有done事件的turn/metric保留为interrupted，默认不悄悄重试。明确决定重试后再传`--retry-failed`，runner增加attempt并保留旧失败记录。每个逻辑case/metric只使用当前attempt计分，同时报告历史失败attempt数，不能通过重试隐藏曾经失败的事实。生成重试使用新的thread_id，避免把失败尝试的半段历史带入新尝试。

`--limit N`只限制本次处理多少缺失case或metric，便于分批推进，不改变冻结样本与分母。重新运行同一命令会继续缺失项；不要删除journal或手工编辑分数。不同配置的新实验必须使用新目录。

每个case和指标都有开始/完成进度行。报告分别显示尚未开始、有效、失败/blocked，以及历史尝试数。`generation_attempted_all`和`eligible_metrics_attempted_all`表示已覆盖计划，并不表示所有结果正确；`all_metric_scores_valid`只表示每个eligible评分都有有效数值，也不表示达到目标或已经人工确认。

## 本机后台完整验证

`scripts/finish_validation.py`将上述阶段串行执行，另加40个长对话/局部证据开关条件，并在每步检查产物覆盖范围。当前机器已有独立语料副本`runtime/phase2_corpus`；原始受管文档继续保留，验证期间不要修改原文、索引、源码或模型。新的运行需先停止使用相同嵌入式Qdrant的服务，并准备独立副本，不直接指向正在使用的UI目录。

`data/eval/cpu_profile.example.json`是可调整的无密钥模板。本机实际profile保存在`work/validation_profile.json`，其中目录为当前机器的绝对路径。首次运行与恢复使用同一个profile和输出目录：

```powershell
.venv\Scripts\python.exe scripts\finish_validation.py --profile work\validation_profile.json --output runtime\final_validation_v3 --launch
# 仅在已有协调器和其子进程均已停止后恢复：
.venv\Scripts\python.exe scripts\finish_validation.py --profile work\validation_profile.json --output runtime\final_validation_v3 --launch --resume
# 仅读取已有产物并更新文字报告，不推理：
.venv\Scripts\python.exe scripts\finish_validation.py --output runtime\final_validation_v3 --report
```

任务使用独立追加日志、原子状态文件和操作系统互斥锁。后台子进程不会弹出控制台；本任务期间只临时阻止系统自动睡眠，屏幕仍可关闭，结束时恢复该请求。断电或主动休眠仍会中断，恢复后按上面命令继续。

验证期间`http://127.0.0.1:7860/`显示中文进度计数，不开放会与评测争用模型的问答。结束或执行异常时，协调器先确认自己的子进程树停止，再仅卸载profile中的模型服务；确认资源释放后，按原`.env`恢复真实Gradio界面。端口已被其他程序占用时不会杀掉该程序；资源释放未确认则不自动启动重模型界面，并写明原因。

当前正式产物在`runtime/final_validation_v3/`：`pipeline_state.json`记录阶段/PID/退出状态，`logs/`保留独立日志，`memory/`为压缩实验，`benchmark/`为三轮问答/消融/评分，`report.md`汇总实际数字与缺失。旧`runtime/final_validation/`中的系统代理502故障实验完整保留，不与v2均值混用。阶段执行完毕、指标有效以及质量达标分别解释，不互相替代。

在当前托管执行环境中，普通后台子进程会随启动命令退出被回收，本机实际采用持续会话：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.venv\Scripts\python.exe work\keep_validation_running.py --output runtime\final_validation_v3
# 已有同目录记录时加 --resume；不要同时启动第二个托管会话。
```

它先启动项目模型服务，再前台承载同一协调器；实验结束后继续承载本地问答服务。恢复后请保持电脑和任务会话运行，锁屏不影响；主动重启或结束任务需要再次恢复。

父块消融报告标签为parent，实际调用检索器full模式；四种模式均显式请求top5，记录search_mode与requested_k。独立真实Qdrant集成回归检查了父块/子块出处、长度及四模式无接口错误。v2的首组有效记忆对照原样保留，但最终完整统计使用v3，避免混入不同版本。
