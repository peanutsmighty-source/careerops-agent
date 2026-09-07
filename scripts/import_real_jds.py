from __future__ import annotations

from dataclasses import dataclass

from app.database import Base, SessionLocal, engine
from app.main import _archive_raw_content
from app.models import JobSource


@dataclass(frozen=True)
class RealJobSample:
    company: str
    title: str
    location: str
    source_name: str
    source_url: str
    raw_content: str


REAL_JOB_SAMPLES = [
    RealJobSample(
        company="Baidu",
        title="北京-Agent Harness 研发工程师",
        location="Beijing",
        source_name="Baidu Talent",
        source_url="https://talent.baidu.com/jobs/detail/GRADUATE/8683fc23-9bff-4d25-84b2-f5ddce782c5a",
        raw_content="""
Company: Baidu
Title: 北京-Agent Harness 研发工程师
Location: Beijing

负责 Agent Harness 产品技术架构设计与开发，建设支撑 LLM 能力在真实业务场景稳定释放的系统底座。
工作包括状态管理、沙箱环境、反馈闭环、执行约束、上下文交接、context compaction 优化，以及 LLM 与 Harness 的联合优化。
要求深入理解 AI Agent、ReAct、CoT、Plan-and-Execute，熟练使用 PyTorch、LangChain、AutoGen 等框架。
加分项包括 AI Agent 开源项目贡献、Harness 工程落地经验、智能体产品早期搭建经历。
""",
    ),
    RealJobSample(
        company="Baidu",
        title="Agent引擎开发工程师",
        location="Beijing",
        source_name="Baidu Talent",
        source_url="https://talent.baidu.com/jobs/detail/SOCIAL/d56ce9b0-296b-4615-9497-115968d4fc14",
        raw_content="""
Company: Baidu
Title: Agent引擎开发工程师
Location: Beijing

负责 Agent 运行引擎、工具/技能插件系统、长期记忆、多模型路由与调用等核心模块。
建设高可用、性能优化与稳定性能力，包括异步消息处理、状态持久化、限流、错误重试、trace、全链路日志和资源隔离。
构建 sandbox、工具调用权限校验、human-in-the-loop 审批、AGENTS.md 式规则引擎等安全防护能力。
要求 3 年以上后端经验，熟练 TypeScript/Node.js，熟悉 Go 或 Rust 优先，掌握 Redis、PostgreSQL、Kafka。
需要理解 Tool Calling、ReAct、Plan-and-Execute、RAG、Embedding、向量数据库等大模型 Agent 技术栈。
""",
    ),
    RealJobSample(
        company="Baidu",
        title="云原生 Agent Infra 高级研发工程师",
        location="Beijing, Shanghai",
        source_name="Baidu Talent",
        source_url="https://talent.baidu.com/jobs/detail/SOCIAL/4b326f6b-2c39-461c-ae64-69b4201c680e",
        raw_content="""
Company: Baidu
Title: 云原生 Agent Infra 高级研发工程师
Location: Beijing, Shanghai

参与智能体沙箱、网关、可观测等 Agent 运行平台产品设计研发，支撑 Agentic 应用和 Agentic RL 的规模化落地。
建设多租户隔离、资源管控、网络接入、安全防护、Multi-Agent 编排、海量 Agent 沙箱实例调度与治理能力。
构建面向 Agent 的 CLI、MCP、SDK、Skill 等接口与开发体验。
要求熟悉 Go/Python/C++ 至少一门，熟悉 Linux、Kubernetes、Docker/containerd、KataContainer、gVisor、Firecracker。
需要了解 Agent 执行模型、工具调用、上下文管理、Skill/MCP/Plugin、Context Engineering、Harness Engineering。
加分项包括 Coding Agent、Browser Use、Computer Use、LangChain、LangGraph、MCP、AutoGen、Prometheus、Grafana、Loki、OpenTelemetry 实践。
""",
    ),
    RealJobSample(
        company="Baidu",
        title="2027AIDU-智能体算法工程师",
        location="Beijing",
        source_name="Baidu Talent",
        source_url="https://talent.baidu.com/jobs/detail/GRADUATE/4f1cbc80-8332-4a92-b8fa-c0132b17d47e",
        raw_content="""
Company: Baidu
Title: 2027AIDU-智能体算法工程师
Location: Beijing

负责 AI Agent 的设计与研发，包括感知-决策-执行闭环、多智能体协作、长期记忆与推理机制。
研究 ReAct、AutoGPT、CoT 等范式，掌握 LangChain、AutoGen、CrewAI 等 Agent 开发框架。
优化大模型在 Agent 任务中的规划、工具调用、反思、代码生成等能力，探索 RAG 与 Agent 融合架构。
建设 Agent 评测体系，持续优化成功率、响应延迟、成本及用户体验。
要求熟悉大语言模型原理、Prompt Engineering、Fine-tuning，有 RAG、知识库、多 Agent 协同经验优先。
""",
    ),
    RealJobSample(
        company="Moonshot AI",
        title="Harness 研究工程师（Harness Engineer/Researcher）",
        location="Beijing",
        source_name="WatchJobs Moonshot Snapshot",
        source_url="https://watchjobs.net/zh/explore/job/00c1cb89-07b3-4002-a04c-9e12f291f2a7/Harness-%E7%A0%94%E7%A9%B6%E5%B7%A5%E7%A8%8B%E5%B8%88-Harness-Engineer%2FResearcher-%E6%9C%88%E4%B9%8B%E6%9A%97%E9%9D%A2",
        raw_content="""
Company: Moonshot AI
Title: Harness 研究工程师（Harness Engineer/Researcher）
Location: Beijing

负责 Kimi Agent 核心 Harness 的设计与迭代，包括执行循环、工具调用、上下文压缩、状态管理与容错重试。
构建 long-running agent harness，让 Agent 在浏览器、终端、桌面等环境中长时间运行并保持执行连贯性。
探索 meta-harness，让模型能够理解自身 harness 表现、识别瓶颈并提出改进。
要求熟悉 LLM、Agent Loop、Skills、MCP、Memory、Multi-Agent、Context Engineering、Harness Engineering。
需要理解沙箱隔离、状态管理、工具调用、上下文窗口、容错重试，并能从 agent trace 和长任务日志中分析系统问题。
""",
    ),
    RealJobSample(
        company="Moonshot AI",
        title="Agent 工程师",
        location="Beijing",
        source_name="DataHub Moonshot Snapshot",
        source_url="https://datahub.ac.cn/ai-jobs/companies/company-819fc19826.html",
        raw_content="""
Company: Moonshot AI
Title: Agent 工程师
Location: Beijing

参与 Kimi 产品研发，围绕真实场景持续迭代。
建设并优化 LLM/Agent 评估平台，形成实验、评测、回归与数据闭环。
打造可复用的工具与技能体系，提升扩展效率与规模化交付能力。
完善工程基建与效能体系，包括遥测、微服务治理、CI/CD、DevOps 和 AI+研发流程。
""",
    ),
    RealJobSample(
        company="DeepSeek",
        title="Agent Harness 团队",
        location="Beijing, Hangzhou",
        source_name="DeepSeek Talent",
        source_url="https://talent.deepseek.com/job/b8e62d9e-5cfb-4f24-bb6b-549bf45e1ee0",
        raw_content="""
Company: DeepSeek
Title: Agent Harness 团队
Location: Beijing, Hangzhou

DeepSeek 官方招聘页列出 Agent Harness 团队，职位类别为全栈开发/算法，工作地点为北京市、杭州市。
该官方列表同时列出 Agent Infra 研发工程师、服务端开发工程师（线上核心服务/Agent 后端/数据仓库）、Code Agent 数据工程师、通用 Agent 数据产品经理等 Agent 相关岗位。
当前公开列表页可确认岗位名称、公司和地点，但未展示完整职责正文。
""",
    ),
    RealJobSample(
        company="Tongyi",
        title="Token Foundry-大模型平台研发工程师-Agent Infra",
        location="Beijing, Hangzhou",
        source_name="JDWatch Tongyi Snapshot",
        source_url="https://www.jdwatch.work/jobs/jntc4wytj",
        raw_content="""
Company: Tongyi
Title: Token Foundry-大模型平台研发工程师-Agent Infra
Location: Beijing, Hangzhou

建设大规模数据生产和评测的 Agent 基础设施，增强大模型在 Coding 和 Agentic 领域的能力。
结合后训练框架优化强化学习效率，实现高效的大规模 Agentic RL。
要求具备 Python、Go、Shell 等编程和工程实现能力。
熟悉 Agent Harness 框架如 OpenCode、OpenClaw 者优先，熟悉大模型 Coding 与 Agentic 方向评测框架者优先。
参与过大规模分布式系统开发、设计和维护者优先。
""",
    ),
]


def main() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        imported = 0
        for sample in REAL_JOB_SAMPLES:
            source = _get_or_create_source(
                session, f"{sample.source_name} - {sample.title}", sample.source_url
            )
            archive = _archive_raw_content(
                session=session,
                source_url=sample.source_url,
                raw_content=sample.raw_content,
                source_id=source.id,
                should_parse=True,
                should_create_job=True,
            )
            imported += 1
            print(
                f"{imported}. {sample.company} - {sample.title} "
                f"-> archive={archive.id}, job={archive.job_id}, status={archive.status}"
            )


def _get_or_create_source(session, name: str, url: str) -> JobSource:
    source = session.query(JobSource).filter(JobSource.url == url).one_or_none()
    if source:
        return source
    source = JobSource(name=name, kind="public_job_board", url=url)
    session.add(source)
    session.commit()
    session.refresh(source)
    return source


if __name__ == "__main__":
    main()
