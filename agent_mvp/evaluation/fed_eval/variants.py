"""Configuration variants: candidate product changes, measured before anyone adopts them.

A variant never edits product code. It patches the running process for the length
of a run, and the run records the variant and the resulting prompt hash.
"""

import json
import re
import tempfile
from contextlib import ExitStack, contextmanager
from difflib import SequenceMatcher
from pathlib import Path
from unittest.mock import patch

from federated_agent.planner import LLMResearchPlanner
from federated_agent.prompts import prompt_sha256

import federated_agent.requirements as requirement_rules
from federated_agent.resources import SEMANTIC_SCHEMA

from . import world
from .replay import grounding_filter
from .vocabulary import SITES, VARIABLES

# The most common failure in the first core runs: catalog limitations and validation
# messages written as the researcher's unmet requirements, which declines a supported request.
QUOTE_REQUIREMENTS = """
unsupported_requirements holds only what the researcher asked for that the catalog
cannot provide, each entry quoting the researcher's own words. Use [] when the
researcher asked for nothing beyond the catalog. Never copy catalog limitations,
constraints or validation messages into it. An unmatched hospital, dataset or
variable name is not a requirement: keep it in its field for the validators.
"""


@contextmanager
def quote_requirements():
    with patch.object(LLMResearchPlanner, "SYSTEM_PROMPT", LLMResearchPlanner.SYSTEM_PROMPT + QUOTE_REQUIREMENTS):
        yield


# The critical errors: hospitals or a dataset the researcher never named. A product check would
# use the deployment's alias table; the evaluation world's aliases are the same names.
SITE_FORMS = {org: [form for forms in names.values() for form in forms] for org, names in SITES.items()}


def _mentions(form, text):
    form = form.lower()
    if re.search(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", text):
        return True
    words = form.split()
    if len(words) != 2 or not words[0].isalpha():
        return False
    # Tolerate a misspelt "hospital" or "centre", never a different hospital number or letter.
    return any(SequenceMatcher(None, word, words[0]).ratio() >= 0.8
               for word in re.findall(rf"(?<![a-z])([a-z]+)\s+{re.escape(words[1])}(?![a-z0-9])", text))


def site_org(name):
    name = " ".join(str(name).lower().split())
    return next((org for org, forms in SITE_FORMS.items() if name in (form.lower() for form in forms)), None)


# Hospital-like phrases, known or not; a named hospital the plan lacks means the model dropped it.
HOSPITAL_LIKE = re.compile(r"\b(?:hospital|centre|center|clinic)\s+[a-z0-9]+\b|\b(?:st\.?|saint)\s+[a-z']+(?:'s)?\s+hospital\b"
                           r"|\d+\s*\u53f7(?:\u533b\u9662|\u4e2d\u5fc3)|\u533b\u9662\s*[a-z0-9]|\u7b2c[\u4e00-\u5341]+\u533b\u9662",
                           re.IGNORECASE)


def dropped_named_sites(plan, said):
    """Hospitals the researcher named in ``said`` that the plan neither contains nor keeps for the validators.

    The layers never add a value: a named hospital in "all hospitals except hospital 3" must not be put back.
    """
    said = " ".join(said.lower().split())
    planned = {site_org(site) for site in plan.sites}
    unknown_kept = sum(site_org(site) is None for site in plan.sites)
    dropped, unknown = [], []
    for match in HOSPITAL_LIKE.finditer(said):
        name = match.group(0)
        org = site_org(name) or next((o for o, forms in SITE_FORMS.items() if any(_mentions(f, name) for f in forms)), None)
        if org is None:
            unknown.append(name)
        elif org not in planned:
            dropped.append(name)
    return dropped + unknown[unknown_kept:]


def ground_plan(plan, text, previous=None, said=None):
    """Drop hospitals and a dataset the researcher never named, so that the validators ask for them, and ask
    again when a hospital the researcher just named is missing from the plan.

    Unknown names are kept: the validators already ask about those.
    """
    if plan.request_type != "analysis":
        return plan
    text = " ".join(text.lower().split())
    named = {org for org, forms in SITE_FORMS.items() if any(_mentions(form, text) for form in forms)}
    named |= {site_org(site) for site in (previous.sites if previous else [])}
    sites = [site for site in plan.sites if site_org(site) is None or site_org(site) in named]
    if len(sites) < len(plan.sites) or (said is not None and dropped_named_sites(plan, said)):
        # Never prepare a smaller task than the plan ("from hospital 1 to hospital 3" names no hospital 2):
        # a hospital that was taken away or left out means asking which hospitals are meant.
        sites = []
    database = plan.database_label
    inherited = previous is not None and database == previous.database_label
    if database and not inherited and not _mentions(database, text):
        database = None
    if said is not None and len({label.replace(" ", "_") for label in DATASET_LABEL.findall(_normal(said))}) > 1:
        database = None  # two datasets in one message contradict each other: ask which one
    return plan.model_copy(update={"sites": sites, "database_label": database})


@contextmanager
def ground_names():
    create, revise = LLMResearchPlanner.create_plan, LLMResearchPlanner.revise_plan

    def create_plan(self, request, catalog):
        return ground_plan(create(self, request, catalog), request, said=request)

    def revise_plan(self, reply, previous, catalog, questions=None):
        # An answer to a sites question must keep every hospital it names; edits are checked by edit_consistency.
        answering = not previous.sites
        return ground_plan(revise(self, reply, previous, catalog, questions), reply + " " + previous.purpose, previous,
                           said=reply if answering else None)

    with patch.object(LLMResearchPlanner, "create_plan", create_plan), patch.object(LLMResearchPlanner, "revise_plan", revise_plan):
        yield


@contextmanager
def site_aliases():
    """Add the Chinese and variant hospital names the test set uses to the deployment's aliases.

    The packaged sandbox deployment knows only "hospital A" and "centre 1" style aliases, so a
    planner that keeps "1号医院" in the researcher's words gets a needless clarification.
    """
    deployment = json.loads(Path(world.SANDBOX_DEPLOYMENT).read_text(encoding="utf-8"))
    for site in deployment["sites"]:
        forms = [form for form in SITE_FORMS[site["organisation_id"]] if form.lower() != site["name"].lower()]
        site["aliases"] = list(dict.fromkeys([*site["aliases"], *forms]))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / Path(world.SANDBOX_DEPLOYMENT).name
        path.write_text(json.dumps(deployment, ensure_ascii=False, indent=2), encoding="utf-8")
        with patch.object(world, "SANDBOX_DEPLOYMENT", path):
            yield


@contextmanager
def utf8_followups():
    """Send follow-ups to the planner as readable text.

    The planner serialises a follow-up with ensure_ascii=True, so a Chinese reply and the Chinese
    purpose reach the model as \\u escapes, and the response-language check reads them as English.
    """
    def revise_plan(self, reply, previous, catalog, questions=None):
        request = json.dumps({"previous_plan": previous.model_dump(), "questions": questions or [],
                              "researcher_reply": reply}, ensure_ascii=False)
        plan = self._generate(request, catalog, previous=previous)
        if plan.request_type == "analysis" and plan.purpose == previous.purpose:
            plan.purpose = previous.purpose + "\nResearcher follow-up: " + reply
        return plan

    with patch.object(LLMResearchPlanner, "revise_plan", revise_plan):
        yield


# Layer 2: a lexicon, not the model, decides which variable a researcher's term means. The evaluation's own
# vocabulary stands in for a curated multilingual lexicon, so results are an upper bound on coverage.
def _normal(text):
    return " ".join(str(text).lower().replace("_", " ").split())


VARIABLE_FORMS = {v.class_id: [_normal(form) for form in (v.canonical, *v.en, *v.zh, *v.en_hidden, *v.zh_hidden)]
                  for v in VARIABLES.values()}


def _schema_variables():
    names = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, dict) and str(value.get("@type", "")).endswith("Variable"):
                    names.add(key)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(json.loads(Path(SEMANTIC_SCHEMA).read_text(encoding="utf-8")))
    return names


SCHEMA_VARIABLES = _schema_variables()


def _same_word(token, word):
    """A misspelt word matches; a short word or one with digits must match exactly."""
    if len(word) <= 3 or any(ch.isdigit() for ch in word):
        return token == word
    return SequenceMatcher(None, token, word).ratio() >= 0.8


def _says(form, text):
    """The form occurs in the text, allowing misspelt words in a Latin-script phrase."""
    if re.search(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", text):
        return True
    if not form.isascii():
        return False
    words = form.split()
    tokens = re.findall(r"[a-z0-9][a-z0-9-]*", text)
    return any(all(_same_word(token, word) for token, word in zip(tokens[i:i + len(words)], words))
               for i in range(len(tokens) - len(words) + 1))


def variable_class(name):
    """The class a plan's variable name denotes; None for a name the validators must ask about."""
    key = _normal(name)
    found = next((cls for cls, forms in VARIABLE_FORMS.items() if key in forms), None)
    if found is None and str(name) in SCHEMA_VARIABLES:
        return "schema:" + str(name)  # a schema variable the lexicon never offers for these words
    return found


def mentioned_variables(text):
    text = _normal(text)
    return {cls for cls, forms in VARIABLE_FORMS.items() if any(_says(form, text) for form in forms)}


def ground_variable_plan(plan, said, context=""):
    """Ask instead of preparing when the plan's variables and the researcher's words disagree.

    ``said`` is what the researcher wrote in this turn; ``context`` is the study so far.
    """
    if plan.request_type != "analysis" or plan.unsupported_requirements or plan.analysis_type == "unsupported":
        return plan  # declined anyway
    named_now = mentioned_variables(said)
    named = named_now | mentioned_variables(context)
    planned = {variable_class(v) for v in plan.variables} - {None}
    accounted = planned | {variable_class(v) for v in plan.stratification_variables}
    if planned - named or named_now - accounted:
        return plan.model_copy(update={"variables": []})
    return plan


@contextmanager
def ground_variables():
    create, revise = LLMResearchPlanner.create_plan, LLMResearchPlanner.revise_plan

    def create_plan(self, request, catalog):
        return ground_variable_plan(create(self, request, catalog), request)

    def revise_plan(self, reply, previous, catalog, questions=None):
        plan = revise(self, reply, previous, catalog, questions)
        context = previous.purpose + " " + " ".join(previous.variables)
        return ground_variable_plan(plan, reply, context)

    with patch.object(LLMResearchPlanner, "create_plan", create_plan), patch.object(LLMResearchPlanner, "revise_plan", revise_plan):
        yield


# Layer 3: requirements are detected by rules as well as by the model. The product's backstop adds a
# condition it finds when the model listed none; these patterns widen what it finds. They were written
# after the test phrasings were known, so they are optimistic about coverage.
EXTRA_CONDITIONS = [
    r"\baged\s+\d+(?:\s*(?:to|-|\u2013)\s*\d+)?",
    r"\b(?:among|only\s+for|restricted\s+to|limited\s+to)\s+[^.;,\n]+",
    r"\bfor\s+(?:adolescents|children|adults|women|men|survivors|those|people|cases)\b[^.;\n]*",
    r"\bin\s+people\s+who\b[^.;\n]*",
    r"\b[a-z]+\s+only\b",
    r"\b(?:in|for)\s+the\s+(?:\d{4}(?:\s*[-\u2013]\s*\d{4})?)\s+(?:cohort|diagnosis\s+window)\b",
    r"\b(?:minimum|maximum|median|percentile|quartile|variance|trend|correlation|regression|t-test|p-values?|"
    r"kaplan-meier|hazards?|machine\s+learning|predict\w*)\b[^.;\n]*",
    r"\b(?:broken\s+down|split|stratified|grouped|separately)\s+(?:by|for)\b[^.;\n]*",
    r"\d+\s*\u5c81",
    r"(?:\u9752\u5c11\u5e74|\u513f\u7ae5|\u5973\u6027|\u7537\u6027|\u590d\u53d1|\u5e78\u5b58)[^\u3002\uff1b\n]*",
    r"(?:\u4ec5|\u53ea)(?:\u770b|\u7edf\u8ba1|\u9650|\u5305\u62ec)[^\u3002\uff1b\n]*",
    r"\u9650\u4e8e[^\u3002\uff1b\n]*",
    r"\d{4}\s*\u5e74(?:\u4ee5\u540e|\u4ee5\u524d|\u4e4b\u540e|\u4e4b\u524d|\u4ee5\u6765)[^\u3002\uff1b\n]*",
    r"(?:\u6700\u5c0f\u503c|\u6700\u5927\u503c|\u4e2d\u4f4d\u6570|\u767e\u5206\u4f4d|\u8d8b\u52bf|\u76f8\u5173|"
    r"\u56de\u5f52|\u751f\u5b58\u66f2\u7ebf|\u663e\u8457\u6027|\u9884\u6d4b)[^\u3002\uff1b\n]*",
    r"\u6309[^\u3002\uff1b\n]{1,8}?(?:\u5206\u522b|\u5206\u7ec4|\u5206\u5c42)",
    r"\b[a-z_]+\s*=\s*[0-9.]+",  # an explicit parameter such as alpha=0.01
    r"\u8bbe\u4e3a\s*[0-9.]+",
    r"\b(?:export|download|send|e-?mail|copy|print)\b[^.;\n]*",  # moving data out
    r"\b(?:raw|underlying)\s+(?:data|rows|records|tables?)\b[^.;\n]*",
    r"\b(?:names|addresses|dates\s+of\s+birth|patient\s+ids?)\b[^.;\n]*",
    r"(?:\u5bfc\u51fa|\u539f\u59cb\u6570\u636e|\u539f\u59cb\u75c5\u5386|\u59d3\u540d|\u51fa\u751f\u65e5\u671f|\u4f4f\u5740|\u8eab\u4efd\u8bc1)[^\u3002\uff1b\n]*",
]


@contextmanager
def conditions():
    pattern = requirement_rules.CONDITIONS.pattern + "|" + "|".join(f"(?:{extra})" for extra in EXTRA_CONDITIONS)
    with patch.object(requirement_rules, "CONDITIONS", re.compile(pattern, re.IGNORECASE)):
        yield


# Layer 4: an edit must do what the researcher said. "Replace A with B" must remove A and add B.
REPLACEMENTS = [
    (re.compile(r"\b(?:replace|swap)\s+(?P<old>.+?)\s+(?:with|for)\s+(?P<new>.+?)(?:[.;!?]|$)", re.IGNORECASE), "old", "new"),
    (re.compile(r"\buse\s+(?P<new>.+?)\s+instead\s+of\s+(?P<old>.+?)(?:[.;!?]|$)", re.IGNORECASE), "old", "new"),
    (re.compile(r"\u628a(?P<old>.+?)(?:\u6362\u6210|\u66ff\u6362\u6210|\u6539\u6210|\u6539\u4e3a)(?P<new>.+?)(?:[\u3002\uff1b\uff01]|$)"),
     "old", "new"),
]


REMOVAL = re.compile(r"\b(?:remove|drop|delete|exclude|without|replace|swap|instead|only|just)\b"
                     r"|\u53bb\u6389|\u5220\u9664|\u5220\u6389|\u4e0d\u8981|\u6362\u6210|\u66ff\u6362|\u6539\u6210|\u6539\u4e3a|\u53ea",
                     re.IGNORECASE)
CLEAR_DATASET = re.compile(r"\b(?:remove|clear|drop|undo|unset)\b[^.;\n]*\bdataset\b|\u6570\u636e\u96c6[^\u3002]*(?:\u53bb\u6389|\u5220\u9664|\u6e05\u9664)"
                           r"|(?:\u53bb\u6389|\u5220\u9664|\u6e05\u9664)[^\u3002]*\u6570\u636e\u96c6", re.IGNORECASE)


def _orgs(text):
    text = _normal(text)
    return {org for org, forms in SITE_FORMS.items() if any(_mentions(form, text) for form in forms)}


EDIT_REMOVE = re.compile(r"\b(?:remove|drop|delete|exclude|without|take\s+out)\b|\u53bb\u6389|\u5220\u9664|\u5220\u6389|\u4e0d\u8981",
                         re.IGNORECASE)
EDIT_ONLY = re.compile(r"\b(?:only|just)\b|\u53ea", re.IGNORECASE)
EDIT_ADD = re.compile(r"\b(?:add|also|include|plus)\b|\u52a0\u4e0a|\u4e5f", re.IGNORECASE)


def verify_edit(plan, previous, reply):
    """Check that a follow-up did what its words say; a field whose change does not match is asked again."""
    if REPLACEMENTS[0][0].search(reply) or REPLACEMENTS[1][0].search(reply) or REPLACEMENTS[2][0].search(reply):
        return plan  # replacements are checked separately
    said_sites, said_vars = _orgs(reply), mentioned_variables(reply)
    before_sites = {site_org(site) for site in previous.sites} - {None}
    before_vars = {variable_class(v) for v in previous.variables} - {None}
    after_sites = {site_org(site) for site in plan.sites} - {None}
    after_vars = {variable_class(v) for v in plan.variables} - {None}
    labels = set(DATASET_LABEL.findall(_normal(reply)))
    if labels and not CLEAR_DATASET.search(reply):
        wanted = {label.replace(" ", "_") for label in labels}
        if plan.database_label not in wanted:
            plan = plan.model_copy(update={"database_label": None})  # the named dataset was not taken
    if EDIT_REMOVE.search(reply):
        expected_sites, expected_vars = before_sites - said_sites, before_vars - said_vars
    elif EDIT_ONLY.search(reply):
        expected_sites = said_sites or after_sites
        expected_vars = said_vars or after_vars
    elif EDIT_ADD.search(reply) or said_sites or said_vars:
        expected_sites = before_sites | said_sites if before_sites else after_sites
        expected_vars = before_vars | said_vars if before_vars else after_vars
    else:
        return plan
    if said_sites or EDIT_REMOVE.search(reply):
        if after_sites != expected_sites:
            plan = plan.model_copy(update={"sites": []})
    if said_vars or EDIT_REMOVE.search(reply):
        if after_vars != expected_vars:
            plan = plan.model_copy(update={"variables": []})
    return plan


@contextmanager
def edit_consistency():
    revise = LLMResearchPlanner.revise_plan

    def revise_plan(self, reply, previous, catalog, questions=None):
        plan = revise(self, reply, previous, catalog, questions)
        if CLEAR_DATASET.search(reply) and plan.database_label is not None:
            plan = plan.model_copy(update={"database_label": None})  # the researcher withdrew the dataset
        if not REMOVAL.search(reply):
            # Nothing was asked to be removed, so a known hospital or variable must not silently disappear.
            kept_orgs = {site_org(site) for site in plan.sites}
            kept_vars = {variable_class(v) for v in plan.variables}
            if any(site_org(site) not in kept_orgs for site in previous.sites if site_org(site) is not None):
                plan = plan.model_copy(update={"sites": []})
            if any(variable_class(v) not in kept_vars for v in previous.variables if variable_class(v) is not None):
                plan = plan.model_copy(update={"variables": []})
        plan = verify_edit(plan, previous, reply)
        for pattern, old_group, new_group in REPLACEMENTS:
            match = pattern.search(reply)
            if not match:
                continue
            new_labels = {label.replace(" ", "_") for label in DATASET_LABEL.findall(_normal(match.group(new_group)))}
            if new_labels and plan.database_label not in new_labels:
                plan = plan.model_copy(update={"database_label": None})  # "use B instead of A" must take B
            old, new = _orgs(match.group(old_group)), _orgs(match.group(new_group))
            sites = {site_org(site) for site in plan.sites}
            kept = {site_org(site) for site in previous.sites} - old
            if old - new & sites or new - sites or kept - sites:
                return plan.model_copy(update={"sites": []})  # ask which hospitals rather than guess
            old_vars, new_vars = mentioned_variables(match.group(old_group)), mentioned_variables(match.group(new_group))
            planned = {variable_class(v) for v in plan.variables}
            if old_vars - new_vars & planned or new_vars - planned:
                return plan.model_copy(update={"variables": []})
        return plan

    with patch.object(LLMResearchPlanner, "revise_plan", revise_plan):
        yield


# Layer 6: coverage. Every word the researcher wrote must be understood: a name the plan carries (the
# validators handle those), a supported statistic, or ordinary request language. Anything else, such as
# "except", "not", "lowest", "each patient's" or an embedded instruction, declines the request and names
# the words that were not understood. An allowlist fails safe: on new phrasing it asks more, it never
# lets more through.
ALLOW_EN = set("""
a an the and of in at on for to from with using use used please can could would you me i i'm im we my our
give show tell report compute calculate run get provide do does what which how is are be describe description
summarise summarize summary summaries overview descriptive statistics statistic stats data dataset database
datasets hospitals hospital centres centre centers center sites site hi hello thanks thank across both mean means
average averages standard deviation deviations count counts frequency frequencies distribution distributions
missing value values variable variables number numbers look looks like want need interested about
""".split())
# Words that only a follow-up may use; their effect is checked by edit_consistency.
ALLOW_EDIT_EN = set("""
add also include plus remove drop delete replace swap instead switch change keep only just actually same as before
but sorry meant mean it them this that
""".split())
ALLOW_ZH = sorted("""
请描述 描述一下 描述 请 帮我统计一下 帮我 统计一下 统计 一下 的描述性统计 描述性统计 做描述性统计 计算 使用 数据集用 数据集
数据库 请用 用 中 的 和 与 及 以及 在 情况 给出 我想了解 了解 里
""".split(), key=len, reverse=True)
ALLOW_EDIT_ZH = sorted("""
再加上 加上 也 去掉 删除 删掉 换成 改成 改为 把 只 抱歉 我是指 先把 选择 我还没确定用哪个 还没确定用哪个
""".split(), key=len, reverse=True)
ALLOW_PHRASES = ["not sure"]
DATASET_LABEL = re.compile(r"synthetic[ _][a-z]+")  # after _normal, which turns "_" into a space


CONNECTORS = {"", "and", ",", ", and", "&", "、", "和", "及", "以及", "与", "，"}


def ambiguous_structure(text):
    """Joins that can change what is asked: variables linked by "with"/"for", or "both" before other than two hospitals."""
    text = _normal(text)
    spans = []
    for forms in VARIABLE_FORMS.values():
        for form in forms:
            for match in re.finditer(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", text):
                spans.append((match.start(), match.end()))
    spans.sort()
    kept = [span for i, span in enumerate(spans) if not any(o != span and o[0] <= span[0] and span[1] <= o[1] for o in spans)]
    problems = []
    for (_, end), (start, _) in zip(kept, kept[1:]):
        between = " ".join(text[end:start].split())
        if between not in CONNECTORS:
            problems.append(f"'{between}' between variables")
    if re.search(r"\bboth\b", text) and len(_orgs(text)) != 2:
        problems.append("'both' with other than two hospitals")
    return problems


def not_understood(text, plan, follow_up=False):
    """Words in ``text`` that nothing accounts for, and joins that make it ambiguous."""
    structure = ambiguous_structure(text)
    text = DATASET_LABEL.sub(" ", _normal(text).replace("’", "'"))
    text = re.sub(r"\bhosp[a-z]*\b", "hospital", re.sub(r"\bcent(?:er|re|r|e)\b", "centre", text))  # "hospitl 1"
    names = [*plan.variables, *plan.stratification_variables, *plan.sites, plan.database_label or ""]
    known = sorted({form for forms in VARIABLE_FORMS.values() for form in forms}
                   | {form.lower() for forms in SITE_FORMS.values() for form in forms}
                   | {_normal(name) for name in names if name} | set(ALLOW_PHRASES), key=len, reverse=True)
    for form in known:
        text = text.replace(form, " ")
    text = HOSPITAL_LIKE.sub(" ", text)
    for word in ALLOW_ZH + (ALLOW_EDIT_ZH if follow_up else []):
        text = text.replace(word, " ")
    tokens = re.findall(r"[a-z][a-z'\-]*|\d+(?:\.\d+)?", text)
    # A misspelt name counts only as a whole phrase ("survivl time"); its single words do not count alone.
    for form in (f for f in known if f.isascii() and " " in f):
        words = form.split()
        for i in range(len(tokens) - len(words) + 1):
            if all(_same_word(token, word) for token, word in zip(tokens[i:i + len(words)], words)):
                tokens[i:i + len(words)] = [""] * len(words)
    allowed = ALLOW_EN | (ALLOW_EDIT_EN if follow_up else set())
    single = {f for f in known if f.isascii() and " " not in f}
    left = [t for t in tokens if t and t not in allowed and not (
        t.isalpha() and len(t) >= 4 and any(SequenceMatcher(None, t, w).ratio() >= 0.85 for w in allowed | single))]
    left += re.findall(r"[一-鿿]+", text)
    return left + structure


@contextmanager
def coverage():
    create, revise = LLMResearchPlanner.create_plan, LLMResearchPlanner.revise_plan

    def flagged(plan, text, follow_up):
        if plan.request_type != "analysis" or plan.unsupported_requirements:
            return plan  # declined anyway
        left = not_understood(text, plan, follow_up)
        if not left:
            return plan
        return plan.model_copy(update={"unsupported_requirements": [f"not understood: {' '.join(left)}"]})

    def create_plan(self, request, catalog):
        return flagged(create(self, request, catalog), request, False)

    def revise_plan(self, reply, previous, catalog, questions=None):
        return flagged(revise(self, reply, previous, catalog, questions), reply, True)

    with patch.object(LLMResearchPlanner, "create_plan", create_plan), patch.object(LLMResearchPlanner, "revise_plan", revise_plan):
        yield


VARIANTS = {
    "quote_requirements": quote_requirements,  # prompt: requirements must quote the researcher
    "utf8_followups": utf8_followups,  # planner: follow-ups reach the model as readable text, not \u escapes
    "site_aliases": site_aliases,  # deployment: Chinese and variant hospital names become aliases
    "conditions": conditions,  # pipeline: a wider rule backstop for filters, statistics and grouping
    "grounding": grounding_filter,  # pipeline: drop requirements sharing no word with the researcher's text
    "ground_names": ground_names,  # pipeline: drop hospitals and a dataset the researcher never named
    "ground_variables": ground_variables,  # pipeline: ask when the plan's variables and the researcher's words disagree
    "edit_consistency": edit_consistency,  # pipeline: "replace A with B" must remove A and add B, or ask
    "coverage": coverage,  # pipeline: every word must be understood, or the request is declined with those words
}


@contextmanager
def applied(names):
    """Apply comma-separated variants, e.g. "quote_requirements,grounding"; empty means the product as it is.

    They are entered in the order of VARIANTS, so a pipeline check wraps a changed planner and not the reverse.
    """
    chosen = set(filter(None, (names or "").split(",")))
    unknown = chosen - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(sorted(unknown))}")
    with ExitStack() as stack:
        for name in VARIANTS:
            if name in chosen:
                stack.enter_context(VARIANTS[name]())
        yield


def current_prompt_sha256():
    return prompt_sha256(LLMResearchPlanner.SYSTEM_PROMPT, LLMResearchPlanner.UPDATE_PROMPT)
