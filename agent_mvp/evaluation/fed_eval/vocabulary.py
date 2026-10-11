"""Surface forms for variables, hospitals and datasets, in English and Chinese.

"core" forms build the main test set; "hidden" forms are disjoint paraphrases kept
for the held-out split, so a prompt tuned on the core set cannot memorise them.
Canonical names and class IDs come from the packaged semantic catalog.
"""

import json
from dataclasses import dataclass, field

from .world import INVENTORY


@dataclass(frozen=True)
class Variable:
    key: str
    canonical: str
    class_id: str
    datatype: str
    en: tuple
    zh: tuple
    en_hidden: tuple = field(default=())
    zh_hidden: tuple = field(default=())


VARIABLES = {v.key: v for v in [
    Variable("age", "age_at_initial_diagnosis", "ncit:C156420", "numerical",
             ("age at diagnosis", "age at initial diagnosis", "diagnosis age"), ("诊断年龄", "初次诊断年龄"),
             ("age when first diagnosed", "how old patients were at diagnosis"), ("确诊时的年龄", "首次确诊年龄")),
    Variable("survival", "survival_status", "ncit:C25717", "categorical",
             ("survival status", "vital status"), ("生存状态",),
             ("whether patients are still alive", "patient survival status"), ("存活状态", "生存结局")),
    # 性别 alone is ambiguous: the schema also has biological_sex (ncit:C28421).
    Variable("gender", "gender", "ncit:C158277", "categorical", ("gender",), ("社会性别",), ("gender identity",), ("性别认同",)),
    Variable("year", "year_of_initial_diagnosis", "ncit:C164339", "numerical",
             ("year of initial diagnosis", "year of diagnosis"), ("诊断年份",), ("diagnosis year", "the year patients were diagnosed"), ("确诊年份",)),
    Variable("survtime", "survival_event_time", "ncit:C125201", "numerical",
             ("survival time", "survival event time"), ("生存时间",), ("time to survival event", "follow-up time to the survival event"), ("生存事件时间",)),
    Variable("income", "annual_income", "ncit:C154890", "numerical", ("annual income",), ("年收入",), ("yearly household income", "income per year"), ("家庭年收入",)),
    Variable("ethnicity", "ethnicity", "ncit:C16564", "categorical", ("ethnicity",), ("民族",), ("ethnic background",), ("族裔",)),
    Variable("education", "educational_level", "ncit:C17953", "categorical",
             ("educational level", "education level"), ("教育程度",), ("highest level of education",), ("受教育水平", "学历")),
    Variable("stage", "overall_stage", "strongaya:overall_stage", "categorical", ("overall stage",), ("总体分期",), ("overall cancer stage",), ("肿瘤总体分期",)),
    Variable("chemo", "chemotherapy", "ncit:C15632", "categorical", ("chemotherapy",), ("化疗",), ("whether chemotherapy was given",), ("化学治疗",)),
    Variable("radio", "radiotherapy", "ncit:C15313", "categorical", ("radiotherapy",), ("放疗",), ("radiation therapy",), ("放射治疗",)),
    Variable("surgery", "surgery", "ncit:C15329", "categorical", ("surgery",), ("手术",), ("surgical treatment",), ("外科手术",)),
    Variable("anxiety", "hads_anxiety", "ncit:C118111", "numerical", ("HADS anxiety", "HADS anxiety score"), ("HADS焦虑评分",), ("anxiety score on the HADS",), ("HADS焦虑分数",)),
    Variable("depression", "hads_depression", "ncit:C118112", "numerical", ("HADS depression", "HADS depression score"), ("HADS抑郁评分",), ("depression score on the HADS",), ("HADS抑郁分数",)),
    Variable("fatigue", "eortc_qlq_c30_fatigue_FA", "ncit:C181148", "numerical",
             ("EORTC QLQ-C30 fatigue score",), ("EORTC QLQ-C30疲劳评分",), ("fatigue scale of the QLQ-C30",), ("QLQ-C30疲劳量表得分",)),
    Variable("qol", "eortc_qlq_c30_overall_qol_QL2", "ncit:C181142", "numerical",
             ("EORTC QLQ-C30 overall quality of life",), ("EORTC QLQ-C30总体生活质量",), ("global quality of life on the QLQ-C30",), ("QLQ-C30整体生活质量",)),
    Variable("pain", "eortc_qlq_c30_pain_PA", "ncit:C181150", "numerical", ("EORTC QLQ-C30 pain score",), ("EORTC QLQ-C30疼痛评分",), ("QLQ-C30 pain",), ("QLQ-C30疼痛得分",)),
    Variable("intent", "treatment_intent", "ncit:C210652", "categorical", ("treatment intent",), ("治疗意图",), ("intent of treatment",), ("治疗目的",)),
    Variable("metastases", "distant_metastases", "ncit:C18206", "categorical", ("distant metastases",), ("远处转移",), ("distant metastasis status",), ("远端转移情况",)),
    Variable("occupation", "occupational", "ncit:C25193", "categorical", ("occupational status",), ("职业状况",), ("employment situation",), ("就业状况",)),
]}

SITES = {
    1: {"en": ("hospital 1", "hospital A", "centre 1"), "zh": ("1号医院", "医院1", "医院A"),
        "en_hidden": ("Hospital One", "center 1"), "zh_hidden": ("第一医院", "1号中心")},
    2: {"en": ("hospital 2", "hospital B", "centre 2"), "zh": ("2号医院", "医院2", "医院B"),
        "en_hidden": ("Hospital Two", "center 2"), "zh_hidden": ("第二医院", "2号中心")},
    3: {"en": ("hospital 3", "hospital C", "centre 3"), "zh": ("3号医院", "医院3", "医院C"),
        "en_hidden": ("Hospital Three", "center 3"), "zh_hidden": ("第三医院", "3号中心")},
}
DATASETS = ("synthetic_clean", "synthetic_missing", "synthetic_rare", "synthetic_small")
ALGORITHM_ID = "strongaya_descriptive_statistics"


def _inventory():
    document = json.loads(INVENTORY.read_text(encoding="utf-8"))
    return {(site["organisation_id"], site["database_label"]): site for site in document["sites"]}


INVENTORY_BY_SITE = _inventory()


def availability(variable_key, organisation_id, dataset):
    entry = INVENTORY_BY_SITE[(organisation_id, dataset)]
    state = entry["variables"].get(VARIABLES[variable_key].class_id)
    if state:
        return state
    return "unavailable" if entry["complete_inventory"] else "unknown"


def all_usable(variable_keys, organisation_ids, dataset):
    """True when no requested variable is unavailable at any requested hospital."""
    return all(availability(v, o, dataset) != "unavailable" for v in variable_keys for o in organisation_ids)


def join_en(items):
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def join_zh(items):
    items = list(items)
    if len(items) == 1:
        return items[0]
    return "、".join(items[:-1]) + "和" + items[-1]
