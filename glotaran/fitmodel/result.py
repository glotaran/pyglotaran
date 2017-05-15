from lmfit_varpro import SeparableModelResult


class Result(SeparableModelResult):
    def e_matrix(self, dataset, *args, **kwargs):
        data = self.model._model.datasets[dataset].data.data
        return super(Result, self).e_matrix(data, **{'dataset': dataset})

    def eval(self, dataset, *args, **kwargs):
        kwargs['dataset'] = dataset
        return super(Result, self).eval(*args, **kwargs)
