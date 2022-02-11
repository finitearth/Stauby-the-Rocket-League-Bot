import wandb
from stable_baselines3.common.logger import make_output_format, KVWriter, Logger


class CustomLogger(Logger):
    def __init__(self,):
        super(CustomLogger, self).__init__("", [make_output_format("stdout", "./logs", "")])

    def dump(self, step=0):
        wandb.log(self.name_to_value)
        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()
