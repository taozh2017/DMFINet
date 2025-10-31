
def create_model(opt):
    model = None
    print(opt.model)

    if opt.model == 'cyclegan_2D':
        assert(opt.dataset_mode == 'unaligned_2D')
        from .cycle_gan_model_2D import CycleGANModel_2D
        model = CycleGANModel_2D()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
