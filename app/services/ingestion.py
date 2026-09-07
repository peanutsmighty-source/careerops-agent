from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {"fbclid", "gclid", "msclkid", "spm"}
SKILL_CATALOG: dict[str, tuple[str, str]] = {
    "Python": ("programming_language", r"\bpython\b"),
    "Go": ("programming_language", r"\bgolang\b|\bgo\b"),
    "TypeScript": ("programming_language", r"\btypescript\b"),
    "Node.js": ("backend", r"\bnode\.?js\b"),
    "FastAPI": ("backend", r"\bfastapi\b"),
    "SQLAlchemy": ("backend", r"\bsqlalchemy\b"),
    "PostgreSQL": ("database", r"\bpostgres(?:ql)?\b"),
    "Redis": ("database", r"\bredis\b"),
    "Kafka": ("distributed_system", r"\bkafka\b"),
    "Docker": ("devops", r"\bdocker\b"),
    "Kubernetes": ("devops", r"\bkubernetes|\bk8s\b"),
    "PyTorch": ("ml_framework", r"\bpytorch\b"),
    "LangChain": ("llm_framework", r"\blangchain\b"),
    "LangGraph": ("agent_orchestration", r"\blanggraph\b"),
    "LlamaIndex": ("llm_framework", r"\bllamaindex\b"),
    "AutoGen": ("agent_orchestration", r"\bautogen\b"),
    "CrewAI": ("agent_orchestration", r"\bcrewai\b"),
    "RAG": ("llm_application", r"\brag\b|retrieval[- ]augmented generation"),
    "LLM Evaluation": ("llm_application", r"\beval(?:uation)?s?\b|评测|评估"),
    "Prompt Engineering": ("llm_application", r"prompt engineering|提示词"),
    "Context Engineering": ("agent_harness", r"context engineering|上下文工程|上下文管理|context compaction"),
    "Agent Harness": ("agent_harness", r"agent harness|harness engineering|harness"),
    "Agent Loop": ("agent_harness", r"agent loop|执行循环"),
    "Tool Calling": ("agent_harness", r"tool calling|tool use|工具调用"),
    "MCP": ("agent_harness", r"\bmcp\b"),
    "Memory": ("agent_harness", r"\bmemory\b|长期记忆|记忆机制"),
    "ReAct": ("agent_orchestration", r"\breact\b"),
    "Chain of Thought": ("agent_orchestration", r"\bcot\b|chain of thought|思维链"),
    "Plan-and-Execute": ("agent_orchestration", r"plan-and-execute|planning|任务规划|自主规划"),
    "Multi-agent Systems": ("agent_orchestration", r"multi[- ]agent|多智能体"),
}


class TextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "address",
        "article",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "p",
        "section",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        elif not self._ignored_depth and tag in self.BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif not self._ignored_depth and tag in self.BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self._chunks.append(data)

    def text(self) -> str:
        lines = [normalize_text(line) for line in "".join(self._chunks).splitlines()]
        return "\n".join(line for line in lines if line)


@dataclass(frozen=True)
class ParsedRequirement:
    name: str
    category: str
    importance: int
    requirement_type: str
    evidence_text: str


@dataclass(frozen=True)
class ParsedJobDescription:
    company: str
    title: str
    location: str
    description: str
    published_at: datetime | None
    requirements: list[ParsedRequirement]

    def as_dict(self) -> dict:
        return {
            "company": self.company,
            "title": self.title,
            "location": self.location,
            "description": self.description,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "requirements": [requirement.__dict__ for requirement in self.requirements],
        }


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key not in TRACKING_QUERY_KEYS
        and not any(key.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES)
    ]
    normalized_path = parts.path.rstrip("/") or "/"
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), normalized_path, urlencode(query), "")
    )


def content_fingerprint(raw_content: str) -> str:
    normalized = normalize_text(raw_content).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def html_to_text(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    return parser.text()


def fetch_public_text(url: str, timeout_seconds: int = 15) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "CareerOpsAgent/0.1 public-job-source-collector",
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        body = response.read()
        content_type = response.headers.get_content_charset() or "utf-8"
    text = body.decode(content_type, errors="replace")
    if "<html" in text[:500].lower() or "<body" in text[:1000].lower():
        return html_to_text(text)
    return normalize_text(text)


def parse_job_description(raw_content: str, fallback_url: str | None = None) -> ParsedJobDescription:
    description = normalize_text(raw_content)
    lines = [line.strip() for line in raw_content.splitlines() if line.strip()]
    title = _find_prefixed_value(lines, ("title", "职位", "岗位")) or _guess_title(lines)
    company = _find_prefixed_value(lines, ("company", "公司")) or "Unknown"
    location = _find_prefixed_value(lines, ("location", "地点", "城市")) or _guess_location(description)
    requirements = _extract_requirements(description)
    if fallback_url and company == "Unknown":
        company = urlsplit(fallback_url).netloc or company
    return ParsedJobDescription(
        company=company,
        title=title,
        location=location,
        description=description,
        published_at=None,
        requirements=requirements,
    )


def _find_prefixed_value(lines: list[str], prefixes: tuple[str, ...]) -> str | None:
    for line in lines[:20]:
        for prefix in prefixes:
            match = re.match(rf"^{re.escape(prefix)}\s*[:：]\s*(.+)$", line, re.IGNORECASE)
            if match:
                return match.group(1).strip()
    return None


def _guess_title(lines: list[str]) -> str:
    for line in lines[:8]:
        if 4 <= len(line) <= 120:
            return line
    return "Untitled Job"


def _guess_location(description: str) -> str:
    for location in ("Beijing", "北京", "Shanghai", "上海", "Shenzhen", "深圳", "Remote", "远程"):
        if location.lower() in description.lower():
            return "Beijing" if location == "北京" else location
    return "Beijing"


def _extract_requirements(description: str) -> list[ParsedRequirement]:
    sentences = re.split(r"(?<=[。.!?])\s+|[\n\r]+", description)
    requirements: list[ParsedRequirement] = []
    for skill_name, (category, pattern) in SKILL_CATALOG.items():
        match = re.search(pattern, description, re.IGNORECASE)
        if not match:
            continue
        evidence = _sentence_for_match(sentences, match.group(0)) or description[:240]
        lowered = evidence.lower()
        requirement_type = "preferred" if any(term in lowered for term in ("preferred", "plus", "加分", "优先")) else "required"
        importance = 5 if requirement_type == "required" else 3
        requirements.append(
            ParsedRequirement(
                name=skill_name,
                category=category,
                importance=importance,
                requirement_type=requirement_type,
                evidence_text=evidence[:500],
            )
        )
    return requirements


def _sentence_for_match(sentences: list[str], needle: str) -> str | None:
    for sentence in sentences:
        if needle.lower() in sentence.lower():
            return normalize_text(sentence)
    return None
