from load_model import Model1


class DTFT:
    def __int__(self, params, checkpoint=None):
        self.params = params
        self.model1 = Model1(params, checkpoint)

