from . import builder
from . import learn


class ANFIS:

    def __init__(self, subset_size):
        self.subset_size = subset_size
        self.qtd_rules = 0
        self.qtd_inputs = 0
        self.fuzzysets = []
        self.cons_params = []
        self.prem_params = []
        self.linsys_coefs = []
        self.linsys_resul = []
        self.prem_mf = None
        self.regressor = None

    def fit_by_hybrid_learn(self, inputs, outputs, max_epochs):
        builder.configure_model(self, inputs.shape[0])
        epoch = 1
        while epoch <= max_epochs:
            for entry, output in zip(inputs, outputs):
                learn.hybrid_online(entry, output)
            epoch += 1

    def hybrid_offline(self, inp, out, max_epochs):
        builder.configure_model(self, inputs.shape[0])
        for x, y in zip(inp, out):
            l1 = learn._fuzzysets_membership_degrees(self, x)
            l2 = learn._rules_fire_strength(self, l1)
            l3 = learn._averaged_fire_strength(self, l2)

    def add_linsys_equation(self, coefs, result):
        self.linsys_coefs.append(coefs)
        self.linsys_resul.append(result)

    def l1size(self):
        return self.qtd_mfs * self.qtd_mfs


class Sugeno(ANFIS):

    def __init__(self, subset_size):
        super().__init__(subset_size)

