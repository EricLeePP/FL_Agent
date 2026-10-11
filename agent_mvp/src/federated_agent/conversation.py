"""Small, conservative conversational routing before research-task preparation."""

import re
import unicodedata
from datetime import datetime

from .models import AgentResponse


GREETINGS = {
    "hi", "hello", "hey", "hi there", "hello there", "hey there",
    "good morning", "good afternoon", "good evening", "how are you",
    "hi how are you", "hello how are you",
    "\u4f60\u597d", "\u60a8\u597d", "\u4f60\u597d\u5440", "\u4f60\u597d\u554a", "\u55e8",
    "\u65e9\u4e0a\u597d", "\u65e9\u5b89", "\u4e0b\u5348\u597d", "\u665a\u4e0a\u597d",
}
THANKS = {"thanks", "thank you", "thanks a lot", "thank you very much", "\u8c22\u8c22", "\u591a\u8c22"}

LOCAL_INFO_COMMANDS = {"/model": "model", "/date": "date", "/time": "time"}
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
SYSTEM_QUERIES = {
    "model": (
        r"(?:which|what) (?:(?:llm|ai|language) )?(?:model|provider|(?:agent )?framework) (?:(?:are|r) (?:you|u)|(?:you|u) are)(?: (?:using|running))?(?: (?:now|currently|today))?",
        r"(?:which|what) (?:llm|large language model) (?:are|r) (?:you|u)(?: (?:using|running))?(?: (?:now|currently))?",
        r"(?:what(?:'s| is)|tell me) (?:your|ur) (?:(?:current|llm|ai|language) ){0,2}(?:model(?: name| version)?|llm|(?:agent )?framework|model provider)",
        r"what (?:llm|model) do (?:you|u) use",
        r"(?:who|what) (?:are|r) (?:you|u)",
        r"\u4f60(?:\u73b0\u5728|\u76ee\u524d)?(?:\u7528|\u4f7f\u7528|\u6b63\u5728\u4f7f\u7528|\u5728\u7528)?(?:\u7684)?(?:\u662f)?(?:\u4ec0\u4e48|\u54ea\u4e2a|\u54ea\u79cd)(?:\u5927\u8bed\u8a00\u6a21\u578b|\u8bed\u8a00\u6a21\u578b|\u5927\u6a21\u578b|\u6a21\u578b|llm)(?:\u5462)?",
        r"\u4f60\u7684(?:\u6a21\u578b|\u6a21\u578b\u540d\u79f0|llm|\u6846\u67b6)\u662f\u4ec0\u4e48",
    ),
    "date": (
        r"what date is (?:it|today)(?: today)?",
        r"what(?:'s| is) (?:today(?:'s)? date|the (?:current )?date(?: today| now)?|today)",
        r"what day(?: of the week)? is (?:it|today)(?: today)?",
        r"what(?:'s| is) (?:today's|the) day(?: of the week)?",
        r"today'?s date",
        r"\u4eca\u5929(?:\u662f)?(?:\u51e0\u53f7|\u51e0\u6708\u51e0\u53f7|\u51e0\u6708\u51e0\u65e5|\u4ec0\u4e48\u65e5\u671f|\u661f\u671f\u51e0|\u5468\u51e0)(?:\u554a|\u5462)?",
        r"\u4eca\u5929\u7684\u65e5\u671f(?:\u662f\u4ec0\u4e48)?",
    ),
    "time": (
        r"what time is it(?: now)?",
        r"what(?:'s| is) (?:the )?(?:current |local )?time(?: now)?",
        r"\u73b0\u5728(?:\u662f)?(?:\u51e0\u70b9|\u4ec0\u4e48\u65f6\u95f4)(?:\u4e86|\u554a|\u5462)?",
    ),
}


def _system_query(text: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("\u2019", "'")
    normalized = " ".join(normalized.split()).strip(" !?,.;\u3002")
    if normalized in LOCAL_INFO_COMMANDS:
        return LOCAL_INFO_COMMANDS[normalized]
    normalized = re.sub(r"^(?:hi|hello|hey)[, ]+", "", normalized)
    normalized = re.sub(r"^(?:please |(?:can|could) (?:you|u) tell me )", "", normalized)
    normalized = re.sub(r",? please$", "", normalized)
    # Whole-turn matching is intentional: a second research instruction must not disappear.
    for kind, patterns in SYSTEM_QUERIES.items():
        if any(re.fullmatch(pattern, normalized) for pattern in patterns):
            return kind
    return None


def is_local_conversation(text: str) -> bool:
    """Recognize interjections even inside an interactive clarification menu."""
    return _system_query(text) is not None or social_response(text) is not None


def _model_message(planner, language):
    from .capabilities import PlannerCapabilities, capabilities_of

    if capabilities_of(planner, PlannerCapabilities).complete_requests_only:
        return ("\u5f53\u524d\u662f\u79bb\u7ebf\u89c4\u5219\u6f14\u793a\u6a21\u5f0f\uff0c\u6ca1\u6709\u4f7f\u7528 LLM\u3002"
                if language == "zh" else "This session uses the offline rule-based demo planner, not an LLM.")
    provider = getattr(planner, "provider", None)
    model = getattr(provider, "model", None)
    if not isinstance(model, str) or not model.strip():
        return ("\u5f53\u524d\u4f1a\u8bdd\u6ca1\u6709\u63d0\u4f9b\u53ef\u6838\u5b9e\u7684\u6a21\u578b\u914d\u7f6e\uff0c\u6211\u4e0d\u80fd\u731c\u6d4b\u578b\u53f7\u3002"
                if language == "zh" else "The model configuration is not available for this session, so I cannot reliably name its LLM.")
    name = getattr(provider, "provider_name", None)
    name = name if isinstance(name, str) and name else "OpenAI-compatible provider"
    runtime = getattr(provider, "runtime_name", None)
    version = getattr(provider, "runtime_version", None)
    if runtime == "openai-agents-sdk":
        framework = "OpenAI Agents SDK" + (f" {version}" if isinstance(version, str) else "")
    else:
        framework = "JSON-mode research planner"
    if language == "zh":
        return f"\u5f53\u524d\u4f1a\u8bdd\u914d\u7f6e\u7684 LLM \u662f {model}\uff08{name}\uff09\u3002Agent \u8fd0\u884c\u6846\u67b6\uff1a{framework}\u3002"
    return f"The configured LLM for this session is {model} ({name}). Agent runtime: {framework}."


def conversation_response(text: str, planner=None, language: str | None = None, *, now=None) -> AgentResponse | None:
    """Answer trusted local facts, never ask the model to guess its identity or date."""
    social = social_response(text, language)
    if social is not None:
        return social
    kind = _system_query(text)
    if kind is None:
        return None
    language = language or response_language(text)
    if kind == "model":
        message = _model_message(planner, language)
    else:
        local_now = now if now is not None else datetime.now().astimezone()
        if local_now.tzinfo is None:
            local_now = local_now.astimezone()
        offset = local_now.strftime("%z")
        zone = f"UTC{offset[:3]}:{offset[3:]}"
        date = local_now.date().isoformat()
        if kind == "time":
            value = local_now.isoformat(timespec="seconds")
            message = (f"\u672c\u673a\u5f53\u524d\u65f6\u95f4\uff1a{value}\u3002" if language == "zh"
                       else f"The current local system time is {value}.")
        elif language == "zh":
            weekday = "\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u65e5"[local_now.weekday()]
            message = f"\u4eca\u5929\u662f {date}\uff0c\u661f\u671f{weekday}\u3002\u4f9d\u636e\u672c\u673a\u65f6\u949f\uff08{zone}\uff09\u3002"
        else:
            message = f"Today is {WEEKDAYS[local_now.weekday()]}, {date}, based on this computer's local clock ({zone})."
    return AgentResponse(status="conversation", message=message)


def response_language(text: str, previous: str = "en") -> str:
    """Use only researcher-authored text, not model output or menu metadata."""
    if text.strip().startswith("/"):
        return previous
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002fa1f]", text):
        return "zh"
    if re.search(r"[A-Za-z]{2,}", text):
        return "en"
    return previous


def social_response(text: str, language: str | None = None) -> AgentResponse | None:
    # Match the WHOLE turn, never just a greeting prefix that could hide a task.
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    normalized = re.sub(r"[!?,.;\u3002\u3001]+", " ", normalized)
    normalized = " ".join(normalized.split())
    if normalized not in GREETINGS | THANKS:
        return None
    language = language or response_language(text)
    if normalized in THANKS:
        message = ("\u4e0d\u5ba2\u6c14\uff01\u51c6\u5907\u597d\u540e\uff0c\u53ef\u4ee5\u7ee7\u7eed\u63cf\u8ff0\u4f60\u7684\u7814\u7a76\u9700\u6c42\u3002"
                   if language == "zh" else "You're welcome! Continue with your research request whenever you're ready.")
    else:
        message = ("\u4f60\u597d\uff01\u8bf7\u544a\u8bc9\u6211\u4f60\u60f3\u505a\u4ec0\u4e48\u5206\u6790\uff0c\u6d89\u53ca\u54ea\u4e9b\u53d8\u91cf\u548c\u533b\u9662\u3002\u4e5f\u53ef\u4ee5\u8f93\u5165 /algorithms \u67e5\u770b\u5f53\u524d\u7b97\u6cd5\u3002"
                   if language == "zh" else "Hi! Tell me what you'd like to analyze, which variables to use, and which hospitals to include. You can also use /algorithms to see the current catalog.")
    return AgentResponse(status="conversation", message=message)
