
class ParameterActor:

    def __init__(self):
        self.parameter = None

    def update(self, parameter):
        self.parameter = parameter

    def get(self):
        return self.parameter


class ParameterClient():

    def __init__(self, client):
        self._client = client
        self._actor_future = None
        self._interface = None

    def __enter__(self):
        self._actor_future = self._client.submit(ParameterActor, actor=True)
        return self._actor_future.result()

    def __exit__(self, type, value, traceback):
        self._actor_future.cancel()
