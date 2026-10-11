"""Joshi's fixed researcher-side submission call.

The trusted worker runs ``submit_task`` and the reviewable export embeds this exact
source, so the reviewed call and the executed call cannot drift apart.
"""


def submit_task(client, payload):
    """Create the Vantage6 task; only validated task values vary."""
    return client.task.create(
        collaboration=payload["collaboration"],
        organizations=payload["organizations"],
        name=payload["name"],
        image=payload["image"],
        description=payload["description"],
        input_=payload["input_"],
        databases=payload["databases"],
    )
