from . import builders


class Clusterizer:

    def __init__(self):
        self.builders = {}

    def register_builder(self, model, builder):
        self.builders[model] = builder

    def create(self, model, **kwargs):
        builder = self.builders[model]
        if builder is None:
            raise ValueError(f"Builder {builder} not registred!")
        return builder(**kwargs)


factory = Clusterizer()
factory.register_builder('fcm', builders.fcm)
factory.register_builder('fgmm', builders.fgmm)

