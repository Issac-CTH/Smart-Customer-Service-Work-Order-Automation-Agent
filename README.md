# 智能客服 + 工单自动化 Agent
我搭建了一个面向客服工单的 AI Agent 管道。系统先进行多轮意图和槽位抽取，再基于企业知识库检索生成高质量回复草稿；若为可自动处理的场景，自动化执行 Agent 会触发退款/开票/密码重置等操作；复杂工单则智能路由给对应专家并附带模型生成的辅助诊断信息。上线后，自动处理率达 48%，平均响应时间从 30 分钟降到 4 分钟，人工工单量下降 52%，显著提升了 SLA 达成率。系统包含链式推理以确保多轮对话的一致性，并通过在线学习不断更新知识库。

---

1. 项目解决的核心痛点（Core pain points）
- 客服回复不一致：  
  - 方案：统一由 NLU + 知识库检索 + LLM 生成回复草稿，保证用同一知识与模板生成每次回复，减少人工个体差异。  
  - 代码映射：nlu_parse() 做意图/槽位抽取；retrieve_kb() 做语义检索；generate_draft_reply() 统一生成回复草稿（可接 OpenAI 或用模板）。
- 人工分流慢、分流不准：  
  - 方案：自动 Router 决策（decide_auto_or_manual），基于意图、置信度和槽位完整性自动判断可否自动处理并执行，减少人工介入。  
  - 代码映射：decide_auto_or_manual()、perform_action() 与 perform_* 执行器模拟自动化操作；/api/message 返回 mode 字段表明 auto/manual。
- SLA 难保证（响应慢、人工工单积压）：  
  - 方案：首轮通过自动化处理与自动回复草稿提高首回合解决率；缺失信息时立即触发多轮确认，缩短等待时间；自动执行异步或同步动作并记录结果形成闭环。  
  - 代码映射：当满足自动化时立即 perform_action() 并关闭工单；当缺失槽位时立即返回需要的字段（多轮），message 与 ticket 表中记录时间戳便于统计 SLA 指标。

2. 核心逻辑流（Core logic flow）—— 是否包含长链推理、多 Agent 协作
- 总体流程（step-by-step）
  1. 收到用户消息（/api/message）：创建或识别工单并保存用户话语（Message 表）。  
     - 代码：handle_message() 的开头与 save_message()。  
  2. NLU Agent：意图识别 + 槽位抽取（单轮或基于模型），并返回置信度与槽位。  
     - 代码：nlu_parse()（可接 OpenAI 或使用规则回退）。  
  3. 槽位合并与缺失检查（上下文跟踪）：把新抽取槽位与工单历史槽位合并，判断是否缺失必需字段。  
     - 代码：ticket.slots 合并、missing_slots_for()。  
  4. KB 检索 Agent：基于用户文本或抽取的槽位，做语义检索返回相关知识片段，作为生成上下文。  
     - 代码：retrieve_kb()（sentence-transformers 嵌入 + cosine similarity）。  
  5. Draft Reply Agent（LLM）：基于用户文本、抽取槽位与 KB 片段生成高质量回复草稿（可包含需要用户补充的信息）。  
     - 代码：generate_draft_reply()（可调用 OpenAI；无 API 时使用模板）。  
  6. Router Agent（决策 Agent，多 Agent 协作核心）：判断是否进入自动化执行或路由到人工工单。决策依据：意图类型、置信度、槽位完整性等。  
     - 代码：decide_auto_or_manual()。  
  7. Automation Agent（如果可自动处理）：执行对应动作（退款、开票、密码重置等），记录执行结果并关闭工单；同时把执行结果返回给用户。  
     - 代码：perform_action() 与 exec_refund/exec_invoice/exec_password_reset()（示例模拟）。  
  8. Human Agent（若需人工）：将工单置为 pending_human，附带模型生成的诊断信息（intent、槽位、KB 引用、建议回复草稿）以便人工快速处理（节省人工诊断时间）。人可通过 /api/human_reply 接口回复或执行操作。  
     - 代码：handle_message() 中 manual 分支生成 diagnostic 并保存；/api/human_reply 提供人工回复接口。  
  9. 闭环学习 Agent（可扩展）：把人工处理后的最终结果、用户反馈、成功失败案例存入 DB，用于离线/在线训练 NLU 模型或更新 KB。  
     - 代码：示例中保存 Message 与 Ticket；你可以在此基础上实现周期性训练/日志导出并更新模型或 KB（建议接入标注管道与自动化训练脚本）。
- 是否包含长链推理（chain-of-reasoning / multi-step confirmation）？  
  - 是。示例实现了多轮确认与上下文追踪：当必需槽位缺失时，系统不会立即路由人工，而是直接返回需要补全的槽位并生成后续提问（多轮链式对话）。此外，若接入更强 LLM（OpenAI）并在 generate_draft_reply / nlu_parse 中使用“多步验证”或 chain-of-thought 型 prompts，可支持更复杂的链式推理（例如逐步验证订单合法性、跨库校验、逐步问答以确认细节等）。  
  - 代码映射：missing_slots_for() + handle_message() 的多轮逻辑；可通过在 nlu_parse 与 generate_draft_reply 的 prompt 中加入 step-by-step 验证来增强链式推理。
- 是否包含多 Agent 协作（multi-agent）？  
  - 是，设计上为多 Agent 协作架构（代码中以函数/模块形式体现）：  
    - NLU Agent（意图/槽位解析）  
    - KB Retrieval Agent（向量检索）  
    - Reply Generation Agent（LLM / 模板）  
    - Router Agent（决策自动/人工）  
    - Automation Agent（执行器）  
    - Human Agent（人工客服）  
    - Learning/Analytics Agent（离线/在线学习、指标统计）  
  - 这些 Agent 之间通过 Ticket、Message、KB 数据结构与 API（/api/message、/api/human_reply、/api/ticket/{id}）协作，形成端到端流水线。

3. 指标与监控点（如何衡量）
- 关键指标（示例）：
  - 首回合解决率（FCR）：自动处理并在首轮关闭工单的比例。目标示例：提升 40%。  
    - 数据来源：Ticket.status 从 open -> closed 且只含一条 user 消息与一条 agent 消息的统计。  
  - 平均响应时间（ART）：从用户发起到首次 agent 回复的平均时长，目标 < 5 分钟。  
    - 数据来源：Message 时间戳（user -> agent）。  
  - 人工工单量下降率：自动化处理导致的人工量下降（示例目标 50%）。  
    - 数据来源：mode == "auto" vs mode == "manual" 的计数。  
  - 成功率 / 失败率：自动化执行的成功比例（以 action_result.status 判断）。  
- 在代码中易于采集：Ticket 与 Message 的时间戳、agent meta（是否 auto）、action_result 都记录在 DB（SQLite），方便导出做分析。

4. 与示例代码的具体映射快速索引
- 接口：/api/message —— 全链路入口（NLU、KB、Router、Auto/Human 分流）  
- NLU：nlu_parse()（支持 OpenAI 或规则）  
- KB：retrieve_kb()（sentence-transformers + 本地 KB 表）  
- Draft：generate_draft_reply()（LLM 或模板）  
- Router：decide_auto_or_manual()  
- Automation：perform_action(), exec_refund(), exec_invoice(), exec_password_reset()  
- 人工接口：/api/human_reply（人工回复并可记录人工操作结果）  
- 持久化：SQLite 表 Ticket / Message / KB（可扩展为向量 DB、Postgres 等）

5. 可选增强点
- 把 NLU 与生成完全迁移到强 LLM，并在 prompt 中显式要求“逐步验证每个槽位、列出可疑项与证据”，以获得可解释的链式推理步骤。  
- 用向量数据库（Milvus/FAISS/Weaviate）替代内存/SQLite 的 KB embedding 存储，实现大规模 KB 检索与在线更新。  
- 引入 Orchestrator（例如一个简单的 workflow 引擎）来编排多个 Agent 的调用、超时、并行检查（例如在路由前并行校验订单是否存在、支付状态、风控检查）。  
- 引入审计与幂等 key（idempotency）来保障自动操作的可追溯与安全。
