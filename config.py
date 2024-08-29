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
                self.__setattr__(attr, value)


class SweepConfigLoader():
    pass
