from typing import Type

from base.datasets_processing import DatasetManager


class Defender:
    name = "Defender"

    def __init__(
            self
    ):
        pass

    def defense_diff(
            self
    ):
        pass

    @staticmethod
    def check_availability(
            gen_dataset: DatasetManager,
            model_manager: Type
    ):
        return False


