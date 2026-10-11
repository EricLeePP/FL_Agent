# Explicit Dataset Selection

The deployment's `database_label` field is not a researcher selection. Every
site must list it in the deployment configuration, but it is never bound during
hospital resolution, task compilation or session recording, and it is not sent to
the planner as a default to fill in.

## Conversation Contract

| Situation | Behavior |
| --- | --- |
| New study or `/new`, dataset omitted | `needs_clarification`, no compiled task or generated script |
| Only one eligible dataset | Still ask; never select it automatically |
| Explicit registered dataset | Validate coverage, algorithm compatibility and availability before review |
| Follow-up omits dataset | Retain the active study choice and display its source |
| Explicit clear/remove selection | Set `database_label=null`; require selection again |
| Explicit unknown dataset | Preserve the requested value and explain the problem; never substitute a default |
| No common registered dataset | Explain the catalog constraint; no automatic hospital, dataset or algorithm replacement |

Candidate labels are the intersection of declarations at all resolved selected
hospitals, the central execution organization and the selected algorithm's
deployment scope. Unknown hospitals must be clarified before reliable dataset
options can be offered. Catalog options are not proof of live data availability
or authorization. Existing data checks and policy hooks still apply.

The compiler also rejects missing labels when called directly. Dataset provenance
now points to `research_plan.database_label`, rather than suggesting the deployment
selected it. Supplying a dataset only prepares a task: a separate exact-task
confirmation is still required to execute.

Known unsupported analysis requirements remain blocking. A missing dataset does
not conceal them behind a dataset question or permit dropping a cohort filter.

## Example

Start a fresh study, particularly when testing omission after a previous request:

```text
/new
Describe age at diagnosis and survival status in hospital 1 and hospital 2.
```

The agent asks which dataset to use and shows registered candidates. Select one
with the interactive controls or reply `Use synthetic_clean`. The resulting task
must still be reviewed and confirmed. To clear the active choice:

```text
Clear the dataset selection. Keep the hospitals and variables unchanged.
```

Natural-language interpretation remains an LLM proposal. The deterministic
application distinguishes an omitted update field from explicit null; the model
is instructed to use null for clearing or requesting an unnamed replacement.

## Saved Sessions and Tasks

Sessions record `dataset_selection_version: "explicit-v1"`. `/recover` retrieves a
submitted task and does not recompile or resubmit it. A prepared task whose
research plan has no dataset cannot be submitted with an implicit default; it
requires an explicit selection, preparation and confirmation.

## Validation

Tests cover omission, empty values, sole candidates, coverage intersections,
direct compiler calls, unknown labels, confirmation, `/new`, clearing, resuming
and SDK streaming/non-streaming proposals. Test requests
whose purpose is successful execution now explicitly name their dataset rather
than depending on the old default. Benchmark B04 is the missing-dataset regression
and requires a `database_label` clarification, not merely any clarification.

The benchmark records the corpus hash; rerun it before claiming remote-model
performance.
