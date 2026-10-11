"""Reviewed runtime implementations. Registry JSON never supplies Python imports."""

class Vantage6UserClientV1:
    def configure(self, client):
        client.setup_encryption(None)

    def payload(self, task):
        return task.to_client_payload()

    def result(self, client, task_id):
        return client.result.from_task(task_id=int(task_id))


RUNTIME_ADAPTERS = {"vantage6_userclient_v1": Vantage6UserClientV1()}


def runtime_adapter(name):
    try:
        return RUNTIME_ADAPTERS[name]
    except KeyError:
        raise ValueError("Execution adapter is not installed") from None
