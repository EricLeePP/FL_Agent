# Planner LLM evaluation

This directory evaluates which language models can serve as the Fed Agent planner.
Every case runs through the real Fed Agent pipeline: the planner, the deterministic
validators and the saved-study session. The scripted researcher never confirms a
task, so nothing is ever executed. The design and its rationale are in
[LLM_EVALUATION.md](../docs/LLM_EVALUATION.md).

| Path | Contents |
| --- | --- |
| `datasets/fed_agent_eval_core.jsonl` | 371 cases used to compare models |
| `datasets/fed_agent_eval_hidden.jsonl` | 123 held-out cases with different phrasings; use only to confirm finalists |
| `datasets/fed_agent_eval_smoke.jsonl` | 39 cases, three per category, for quick checks and CI |
| `datasets/fed_agent_eval_ablation.jsonl` | 126 core cases, ten per category, for slow configuration ablations; score it together with full core runs to compare on the same cases |
| `world/data_availability.json` | The synthetic data inventory of the evaluation world |
| `models.json` | The verified comparison list, with Ollama references and request options |
| `RESULTS_2026-10.md` | Results of the first evaluation and the resulting recommendation |
| `fed_eval/` | Generator, runner, oracle, scorer and report |
| `results/` | Run outputs (not committed) |

## Running

The runs below use a user-level Ollama whose models live on the external drive;
`~/.local/opt/ollama-0.40.1/serve-external.sh` starts it on port 11435 with a
16k default context.

```bash
cd agent_mvp/evaluation
python fedeval.py models                              # which models are present
python fedeval.py pull --model qwen3.5:9b             # download one, or omit --model for all
python fedeval.py validate --dataset datasets/fed_agent_eval_core.jsonl
python fedeval.py check                               # one real case per model: thinking off, no truncation
python fedeval.py run --dataset datasets/fed_agent_eval_core.jsonl --model qwen3.5-9b --out results/core
python fedeval.py run --dataset datasets/fed_agent_eval_core.jsonl --model qwen3.5-9b --seeds 1,2,3 --out results/core-repeats
python fedeval.py score --dataset datasets/fed_agent_eval_core.jsonl --results results/core
```

Always run `check` before a long run. Ollama switches thinking on by default for
some models, Gemma 4 included; a model that thinks spends its whole output budget
on reasoning and fails every case. `run` resumes where it stopped and writes one JSON line per case and repeat, with
every turn's status, issues, clarification questions, prepared task and model
calls (latency, tokens, finish reason, raw output). A model whose cases are all
done is skipped without loading it. A case lost to a connection failure, server
error or timeout is marked as an infrastructure error; the next `run` repeats it
and `score` ignores the failed attempt. `--wait` waits for a model
that is still downloading. `score` writes `report.md`, `summary.json` and one
scored file per model.

`replay` feeds recorded model outputs through a modified pipeline without calling
any model, so a product change that acts after the planner can be measured on
existing runs (multi-turn replays are approximate). A change that acts on the
model's input, such as a prompt rule, needs a new run with `run --variant`. The
variants live in `fed_eval/variants.py` and never edit product code:

| Variant | Kind | Change | Label-safe |
| --- | --- | --- | --- |
| `quote_requirements` | prompt | Unmet requirements must quote the researcher; catalog limitations and unmatched names are not requirements | yes |
| `ground_names` | pipeline | Hospitals and a dataset the researcher never named are removed, so the validators ask for them; misspelt "hospitl B" still counts | yes |
| `grounding` | pipeline | Model-written requirements that share no word with the researcher's text are dropped | no: drops real requirements written in another language |
| `site_aliases` | deployment | The Chinese and variant hospital names the test set uses ("1号医院", "center 1") become deployment aliases | yes, but it adds the test set's own names |
| `utf8_followups` | planner | Follow-ups reach the model as readable text instead of `\u` escapes | yes |
| `ground_variables` | pipeline | A lexicon decides which variable a researcher's term means; when the plan disagrees, the agent asks | yes, but its lexicon is the test vocabulary |
| `conditions` | pipeline | A wider rule backstop records filters, statistics, grouping, parameters and data export the planner dropped | yes, but written after the test phrasings were known |
| `edit_consistency` | pipeline | "Replace A with B" must remove A and add B, "remove the dataset" clears it, and an edit must not silently drop a hospital or variable | yes |

| `coverage` | pipeline | Every word must be understood (a known name, a supported statistic or ordinary request language) and variables may only be joined by "and"; otherwise the request is declined, naming the words | no: it declines 7% of perfect core plans and 22% of hidden ones |

Together, `ground_names,site_aliases,ground_variables,conditions,edit_consistency,coverage` form the
safeguard stack that the results report evaluates. The layers are applied in a fixed order whatever the
order on the command line.

`redteam` runs planner errors that must never become a wrong task (`fed_eval/redteam.py`): each attack
is a request, a plausible planner mistake and the intended task, run with and without the safeguards.
Controls carry correct planner outputs, so blocking them counts as a false alarm.

```bash
python fedeval.py redteam                      # the full safeguard stack
python fedeval.py redteam --layers ground_names,site_aliases --out results/redteam/two-layers.json
```

Label-safe means a perfect planner still passes every core and hidden label with
the variant applied.

```bash
python fedeval.py run --dataset datasets/fed_agent_eval_core.jsonl --variant quote_requirements --model qwen3.5-9b --out results/core-quote
python fedeval.py replay --dataset datasets/fed_agent_eval_core.jsonl --results results/core --change ground_names --out results/core-names
```

`validate` replays every case with its gold model outputs. A label passes only
if the real pipeline reproduces it, so the dataset cannot demand an outcome that
no planner could produce. The datasets are generated deterministically:

```python
from fed_eval.generate import build, smoke_subset, write
core = build("core", 20261009)
write("datasets/fed_agent_eval_core.jsonl", core)
write("datasets/fed_agent_eval_hidden.jsonl", build("hidden", 99117))
write("datasets/fed_agent_eval_smoke.jsonl", smoke_subset(core))
write("datasets/fed_agent_eval_ablation.jsonl", smoke_subset(core, per_category=10))
```

## Request settings

The evaluation client (`fed_eval/provider.py`) sends Fed Agent's own prompts to
an OpenAI-compatible endpoint with temperature 0, a 1,200-token output limit and
`response_format` set to the plan's JSON schema. That schema marks every field as
required and lists fields in the order the prompt names them: with Pydantic's
default schema, grammar-constrained decoding let small models skip `variables`
entirely. Thinking is switched off through each model's options in `models.json`.
A prompt that the server truncated is reported as an error rather than scored as
a model mistake.
