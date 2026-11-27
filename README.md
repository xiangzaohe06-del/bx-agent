# 安心保保险咨询 Agent

基于 **LangGraph多Agent协作与RAG条款检索**的保险咨询项目。前端使用React、Vite和Ant Design，后端使用FastAPI与LangChain，由Supervisor按任务委派保险顾问或闲聊子流程，结合PostgreSQL与Milvus获取产品和条款依据，并通过LangSmith持续测评效果与调用成本。

项目架构、Agent工作流、RAG检索、多轮记忆与测评原理见 [项目说明](项目说明.md)。

## 项目特点

- **多Agent分工协作**：主管负责路由与收尾，保险顾问负责产品、条款工具调用，闲聊子流程处理普通聊天；`respond`节点统一生成公开回复。协作采用顺序委派，不要求各角色使用不同基础模型。
- **按需调用，控制重复执行**：需求收集与信息更新走直接回复；子Agent完成或需用户补充时结束本轮，递归错误后停止自动重跑。
- **保险条款RAG**：产品范围过滤、稠密向量与BM25混合召回、Python侧聊天模型重排、父子块回查，提供较完整的条款章节作为回答依据。
- **持久化多轮咨询**：用检查点恢复会话状态，独立保存聊天原文，按阈值摘要较早对话，支持预算修改和保障对象切换等场景。
- **完整Agent测评闭环**：LangSmith保存题库和评分标准，本机SDK运行真实工作流，记录回答、工具证据、模型调用、token与耗时，再比较优化前后结果。

## 十题测评与优化结果

2026年11月26日，使用同一份十题合成题库、相同评委配置，每题在优化前后各运行一次。题库覆盖需求追问、多轮上下文、条款检索、推荐边界和能力边界，每类两题。

| 指标 | 优化前 | 优化后 | 变化 |
| --- | ---: | ---: | --- |
| 完整执行与清理 | 10/10 | 10/10 | 均完成 |
| 自动业务评分通过 | 9/10 | 10/10 | 多通过1题 |
| 聊天模型调用 | 74次 | 46次 | 减少37.8% |
| 聊天token | 119,180 | 75,683 | 减少36.5% |
| 十题累计耗时 | 240.3秒 | 150.9秒 | 减少37.2% |

其中，“已提供基础信息，只询问下一步”的题目，模型调用从18次降至2次，工具调用从8次降至0次，聊天token从39,545降至1,655。主要改进是让需求追问直接收尾，减少不必要的产品和条款查询。

以上为小样本单次对照结果，自动评分仍需人工复核，不能当作生产准确率。token统计为已采集的聊天模型用量，不包含评委、向量化等全部服务成本；累计耗时包含测试清理。实验方法、案例和已知问题见 [项目说明第10节](项目说明.md#10-已完成的调度优化与实测记录)。

## 仓库内容

- `frontend/`：产品展示与聊天界面。
- `agent-service/`：后端、Agent工作流、RAG和测评入口。
- `deploy/`：本地数据库与向量库的Compose配置、初始化SQL。

仓库仅保留代码、配置模板及两份合成测评题库。真实环境文件、聊天与会话数据、原始PDF、解析缓存、向量数据、日志和LangSmith实验报告不随源码发布。

## 本地准备

需要 Python 3.14或以上、uv、Node.js及Docker Compose。使用仓库锁文件安装依赖。

在仓库根目录执行：

```powershell
Copy-Item agent-service/.env.example agent-service/.env
Copy-Item deploy/.env.example deploy/.env
Copy-Item deploy/milvus/user.example.yaml deploy/milvus/user.yaml
```

填写自己的模型API Key和解析服务Token；自行生成数据库、MinIO密码。`agent-service/.env`中的`DB_PASSWORD`必须与`deploy/.env`中的`POSTGRES_PASSWORD`一致。不要覆盖已经配置好的本地环境文件。

启动本地基础服务：

```powershell
docker compose --env-file deploy/.env -f deploy/docker-compose.yml -f deploy/docker-compose.local.yml up -d
```

服务端口默认绑定本机；本地Milvus使用19531端口。现有数据库卷不会因为复制源码而迁移到新机器，初始化SQL只在新数据卷首次初始化时执行。

后端：

```powershell
cd agent-service
uv sync --frozen
uv run python -m app.main
```

前端在另一个终端启动：

```powershell
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

前端地址为`http://127.0.0.1:5173`，后端健康检查为`http://127.0.0.1:8001/health`。

## 知识库与测评

条款PDF由使用者自行准备，放入后端`data/raw/kb/`，文件名与产品目录中的`clause_name`一致。PDF及解析缓存只留本机，不上传仓库。导入会实际调用解析、整理、向量化等服务并产生相应费用。

在后端目录执行：

```powershell
uv run python -m scripts.ingest_rag
```

`data/evaluation/insurance_baseline_v1.json`保留早期工作流规则基线；其中对路由和委派次数的要求属于旧版设计，不能直接作为直接回复优化后的业务正确率。

`data/evaluation/insurance_langsmith_v1.csv`是合成十题数据，可导入自己的LangSmith Dataset。只将`messages`映射到Inputs，其余列映射到Reference Outputs。不会附带其他账号的实验结果或工作区标识。

通过SDK运行自己创建的远端题库：

```powershell
uv run python scripts/evaluate_langsmith.py --dataset-name YOUR_DATASET_NAME --check-only
uv run python scripts/evaluate_langsmith.py --dataset-name YOUR_DATASET_NAME
```

业务评分需在该Dataset上配置评估器。实际输入、输出和工具证据会上传到配置的工作区，测试使用合成信息，真实客户信息须先脱敏。模板默认关闭普通聊天的远端追踪，SDK测评则会主动创建实验。

## 源码发布与隐私

提交前检查文件清单和暂存差异，确认没有加入真实环境文件、个人资料或业务数据：

```powershell
git status --short
git diff --cached --stat
```

真实配置和业务数据通过`.gitignore`排除。它不会保护手动压缩的整个工作目录，也不能抹除已经提交的内容；不要直接把整个工作目录打包分享。

首次提交前，确认Git作者信息使用你愿意公开的名称和邮箱。可在GitHub设置中启用邮箱隐私，并按账户页面提供的noreply地址配置本仓库的提交邮箱，参见[GitHub邮箱隐私说明](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-email-preferences/setting-your-commit-email-address)。

建议启用[GitHub Push protection](https://docs.github.com/en/code-security/concepts/secret-security/push-protection)。如果密钥已经公开或被提交，先撤销并轮换，再处理历史，参见[GitHub敏感数据清理说明](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)。

## 使用范围

当前是本地演示项目，前端包含演示身份，不能直接作为真实用户认证方案。不要直接对公网开放数据库、管理面板或用演示身份承载真实客户聊天。生产部署需要独立完善认证授权和访问控制。
