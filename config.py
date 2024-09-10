import torch

class ConfigLoader():
    def __init__(self, cfg, root):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for cfg_class in cfg.values():
            for attr, value in cfg_class.items():
                if attr == 'data_name':
                    attr = 'data_path'
                    value = root + '/data/' + value
                if attr == 'save_path':
                    value = root + value
                if attr == 'check_point':
                    if value != "":
                        value = root + value
                    else:
                        value = None
                self.__setattr__(attr, value)


class GenConfigLoader(ConfigLoader):
    def __init__(self, cfg, root):
        super().__init__(cfg, root)
        self.gen_seqs_path = root + "/generation/" + self.gen_seqs_path
        self.gen_model_path = root + "/save_models/" + self.gen_model_path
        self.current_gen_batch = self.gen_batch_size


class SweepConfigLoader(ConfigLoader):
    def __init__(self, cfg, root):
        super().__init__(cfg, root)
        pass
