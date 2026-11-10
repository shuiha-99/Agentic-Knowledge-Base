# 本地操作与排障

## 启动

在项目根目录使用已有`.venv`，先运行检查：

```powershell
.venv\Scripts\python.exe -m agentic_kb doctor
.venv\Scripts\python.exe -m agentic_kb start-ollama
.venv\Scripts\python.exe -m agentic_kb prepare-models
.venv\Scripts\python.exe -m agentic_kb ingest data/demo_docs
.venv\Scripts\python.exe -m agentic_kb serve
```

首次模型准备需要联网及磁盘空间。`start-ollama`依赖配置中的Ollama可执行文件，机器已启动服务时无需重复启动。问答模型应已存在于对应Ollama服务中。浏览器访问配置的本机地址，默认`http://127.0.0.1:7860`。

安装仍使用`./scripts/setup.ps1 -WithModels -WithEvaluation`。开发启动脚本显式读取`src/`；修改源码后若使用直接模块命令，需刷新非editable安装副本，保留主环境需要的依赖选项：

```powershell
uv sync --extra dev --extra models --no-editable --reinstall-package agentic-kb
```

长流程验证包含开发集校准、冻结评测与长对话检查。评测口径见[EVALUATION.md](EVALUATION.md)，实际完成状态、失败记录和成绩以本机 `runtime/` 下的实验产物为准；这些产物不随仓库发布，不要用题集规模或某条样例推断完整评测已经完成。

## 项目内模型与输出目录

`.env.example`提供独立服务的配置模板：Ollama地址为`http://127.0.0.1:11435`，模型目录为`runtime/local_ollama_models/`。`KB_OLLAMA_EXECUTABLE`填写本机已安装或便携程序路径；程序路径与模型存储路径分别配置。`KB_MODELS_CACHE_DIR=runtime/models`保存BGE-M3、Cross-Encoder和实际分词器。

首次准备Ollama模型时，先启动本项目服务，再让客户端连接该地址下载。以下程序路径需按本机安装填写：

```powershell
$taskOllama = 'C:\path\to\ollama.exe'
$env:OLLAMA_HOST = '127.0.0.1:11435'
& $taskOllama pull qwen2.5:7b
& $taskOllama pull qwen2.5:3b
```

客户端向本项目服务发送下载请求，模型写入该服务配置的项目内目录。主模型、辅助模型及分词器名称要与`.env`一致。`prepare-models`准备主分词器、嵌入和重排模型；已配置辅助模型时，也会显式准备对应的真实辅助分词器。若遗漏`KB_STRUCTURED_TOKENIZER_MODEL`，命令在加载模型前给出配置错误。离线运行前先完成这一步；该命令不安装Ollama程序，也不代替向Ollama服务准备主、辅助生成模型。

`doctor`只读查询配置、主环境依赖和Ollama状态，不下载模型、不创建runtime目录。输出中的`main_environment.dependencies`不要求安装旧版RAGAS；`isolated_ragas_environment`单独报告`.venv-ragas`或`KB_RAGAS_PYTHON`的存在情况及包版本，并标明是否匹配`ragas==0.3.3`。它通过有超时的独立解释器版本查询完成，不在主环境导入ragas，也不执行评分。`metadata_available`只表示版本信息可读取，完整评分链路仍按评测说明验证。

| 路径 | 用途 |
|---|---|
| `runtime/local_ollama_models/` | 本轮新增的独立Ollama模型库 |
| `runtime/models/` | 检索、重排与分词器的本地模型快照 |
| `runtime/documents/` | 导入原件及文档修订快照 |
| `runtime/qdrant/`、`runtime/documents.sqlite` | 默认知识库向量、父块和活动版本 |
| `runtime/checkpoints.sqlite`、`runtime/ui_sessions.sqlite` | 图恢复状态与界面会话历史 |
| `runtime/cache/inference.sqlite` | 默认真实向量和重排分数缓存 |
| `runtime/phase2_corpus/` | 当前长验证使用的隔离知识库及其缓存 |
| `runtime/development_v2/` | 开发集配置校准记录 |
| `runtime/verification/` | 已记录的实机样例和逐阶段事件 |
| `runtime/benchmarks/<run>/` | 按评测命令指定的冻结快照、日志和报告 |

设置独立`KB_DATA_DIR`后，索引、会话及推理缓存跟随该目录；`KB_MODELS_CACHE_DIR`可以继续指向项目的共享模型快照。普通`start-ollama`日志写入`KB_DATA_DIR/ollama/`；本轮独立验证服务日志位于`runtime/local_ollama_service/ollama/`。

## CPU调度与缓存

本机配置使用`KB_LOW_MEMORY_MODE=true`及`KB_MAX_PARALLEL_TASKS=1`，重模型操作按顺序执行。Send框架保留任务分发能力；当前CPU配置不同时运行多个重模型。批量导入通过受控批次复用嵌入模型，批次结束释放；普通检索、重排按需释放，Ollama每次调用使用`keep_alive=0`。本轮没有启用跨阶段短驻留策略。

缓存只保存真实嵌入模型产生的Dense/Sparse向量及Cross-Encoder计算的查询—片段分数。缓存键包含模型快照、精度、长度等推理约定；同批重复输入先去重，重复正文共享分数但保留各自出处。磁盘条数和内存LRU均有上限，当前上限为2048条嵌入、4096条pair分数及每适配器128条热缓存。默认缓存数据库有约128MiB页上限。

`embedding_inference`、`reranker_inference`和检索trace记录实际命中、未命中、计算条数与模型加载次数。计数以不同输入为单位，`batch_duplicates`另记同批重复。LLM答案、结构化判断和最终回答不进入该缓存；多轮稳定性检查仍重新调用生成模型。首次和后续缓存状态会影响耗时，报告延迟时应一并查看这些记录。

缓存异常会记录诊断并回到真实计算。切换嵌入模型或provider时仍按索引指纹检查决定是否重建知识库，不能将缓存命中视为不同向量空间可以混用。使用专用runtime目录保存各次验证数据。

## 规划、审核与辅助模型

`KB_EVIDENCE_REVIEW_POLICY=complex`下，问题分析认为无需拆解的简单题使用一个检索任务，跳过额外planner和模型完整性审核。执行记录中`planner_called=false`，对应`task_execution`项的`review_status=skipped`、`review_performed=false`，整体`completeness_confirmed=false`。这是明确的“完整性未经审核”，不会把检索成功写成审核通过。

复杂题仍进入规划与受限审核流程。需要每道agent问题都执行完整性审核时设置`KB_EVIDENCE_REVIEW_POLICY=always`；审核本身也受模型能力、证据与预算限制。其他检索消融模式不因此变成完整性评判器。

可选辅助模型配置如下：

```dotenv
KB_CHAT_MODEL=qwen2.5:7b
KB_TOKENIZER_MODEL=Qwen/Qwen2.5-7B-Instruct
KB_STRUCTURED_MODEL=qwen2.5:3b
KB_STRUCTURED_TOKENIZER_MODEL=Qwen/Qwen2.5-3B-Instruct
KB_STRUCTURED_MODEL_SCHEMAS=ConversationSummary,EvidenceDigest
```

CSV名单按schema类名精确匹配，允许逗号周围空格。上述配置只把累计摘要和证据摘要交给辅助模型，歧义判断、任务规划、证据审核以及文本和流式回答仍使用主模型。每个路由按实际选中模型的分词器检查输入预算，所有重模型调用仍经过同一互斥门禁。

白名单留空时，为兼容旧行为，所有结构化schema都会使用已配置的辅助模型。若希望关闭辅助路由，将辅助模型及辅助分词器两个字段都留空。不要只填写辅助模型而遗漏真实分词器。该角色配置依据开发集校准选择，能否改善总体效果以冻结评测记录为准。

仅验证流程时使用`--offline-test`，例如导入和serve均带该参数；页面会显示醒目的测试模式提示。该模式不是实际语义检索或大模型问答，不能用于报告质量。

## 界面

“文档管理”上传MD/TXT/PDF/DOCX并点击建库，可以查看每份文件进度、父子块数和版本。同名文件更新现有逻辑文档；同内容文件会去重。删除先移除活动元数据并使文档不可检索，再清理向量；清理失败会保留诊断，残留向量仍不参与检索。历史原件继续保留，供既有回答的引用回查。

“知识问答”展示答案、待补充信息、来源原文和阶段统计。澄清时直接在输入框回答即可，同一thread_id继续。每个新浏览器会话自动分配独立thread_id；本地单用户历史列表用于切换已有会话，不提供企业多用户权限隔离。

来源中的版本号是知识库文档修订版本，不应与业务文档正文的v1/v2产品版本混淆。PDF展示真实页码；其他格式展示标题路径，不伪造页码。

## 命令行澄清恢复

```powershell
.venv\Scripts\python.exe -m agentic_kb ask '这个版本怎么升级？' --thread-id demo-clarify
.venv\Scripts\python.exe -m agentic_kb ask --thread-id demo-clarify --clarification 'AtlasDesk v1.8升级到v2.0，尚未执行迁移。'
```

第二条命令沿原会话恢复，不开启新的问题。检查点持久化到SQLite，UI展示历史单独保存；两者不能相互替代。

## 常见问题

| 现象 | 处理 |
|---|---|
| 模块找不到或中文路径editable导入异常 | 使用项目提供的安装脚本；临时可设置`$env:PYTHONPATH='src'`后运行 |
| Ollama连接失败 | 检查`KB_OLLAMA_BASE_URL`、服务状态及配置的模型名 |
| 本机Ollama返回502，但服务本身正常 | 主、辅助模型客户端均设置`trust_env=False`，直连配置的Ollama地址，避免继承系统HTTP代理；无需修改系统代理。源码更新后刷新安装副本并重新启动应用 |
| BGE或Cross-Encoder未就绪 | 安装models依赖，运行prepare-models；检查缓存及网络 |
| 模型或provider改变后索引拒绝加载 | 使用新的`KB_DATA_DIR`重建，不混用不同向量空间 |
| 本地Qdrant路径已被占用 | 停止其他使用相同runtime的CLI/服务，或改用Qdrant服务端模式 |
| PDF没有可解析文字 | 先对扫描件OCR，再导入；OCR不在首版内 |
| 回答慢或超时 | 先用简单题检查链路，减少任务和候选数量，合理调整预算与超时 |
| RAGAS导入旧接口报错 | 参照EVALUATION.md使用独立`.venv-ragas`，不要在主环境降级LangChain |
| 恢复仍提示需要补充 | 确认thread_id与暂停时一致，明确产品、当前版本和目标版本 |

测试替身、实际模型试验与完整评测报告分别保存。遇到真实模型失败时应保留错误，不把测试模式结果替换为真实成绩。

CPU低内存模式会按需加载、释放检索和重排权重，并让Ollama生成后卸载模型。首次回答及复杂问题可能需要数分钟；具体耗时以验收记录为准。图中的任务仍可并行调度，但本机重模型计算按资源限制排队。

超时会结束图节点等待，不能强行终止已经在线程中运行的同步模型计算；后台计算结束后才会释放模型资源。因此停止等待后立即重试，仍可能需要等待前一次计算结束。
