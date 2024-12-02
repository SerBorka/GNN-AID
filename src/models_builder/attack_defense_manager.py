from typing import Type

from base.datasets_processing import GeneralDataset


for pack in [
    'defense.GNNGuard.gnnguard',
    'defense.JaccardDefense.jaccard_def',
]:
    try:
        __import__(pack)
    except ImportError:
        print(f"Couldn't import Explainer from {pack}")


class FrameworkAttackDefenseManager:
    """
    """

    def __init__(
            self,
            gen_dataset: GeneralDataset,
            gnn_manager,
            device: str = None
    ):
        self.files_paths = None
        if device is None:
            device = "cpu"
        self.device = device

        available_attacks = {
            "poison": True if gnn_manager.poison_attacker.name != "EmptyPoisonAttacker" else False,
            "evasion": True if gnn_manager.evasion_attacker.name != "EmptyEvasionAttacker" else False,
            "mi": True if gnn_manager.mi_attacker.name != "EmptyMIAttacker" else False,
        }

        available_defense = {
            "poison": True if gnn_manager.poison_defender.name != "EmptyEvasionDefender" else False,
            "evasion": True if gnn_manager.evasion_defender.name != "EmptyPoisonAttacker" else False,
            "mi": True if gnn_manager.mi_defender.name != "EmptyMIDefender" else False,
        }


