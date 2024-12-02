import warnings
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
        if device is None:
            device = "cpu"
        self.device = device
        self.gnn_manager = gnn_manager
        self.gen_dataset = gen_dataset

        self.available_attacks = {
            "poison": True if gnn_manager.poison_attacker.name != "EmptyPoisonAttacker" else False,
            "evasion": True if gnn_manager.evasion_attacker.name != "EmptyEvasionAttacker" else False,
            "mi": True if gnn_manager.mi_attacker.name != "EmptyMIAttacker" else False,
        }

        self.available_defense = {
            "poison": True if gnn_manager.poison_defender.name != "EmptyEvasionDefender" else False,
            "evasion": True if gnn_manager.evasion_defender.name != "EmptyPoisonAttacker" else False,
            "mi": True if gnn_manager.mi_defender.name != "EmptyMIDefender" else False,
        }

        self.start_attack_defense_flag_state = {
            "poison_attack": self.gnn_manager.poison_attack_flag,
            "evasion_attack": self.gnn_manager.evasion_attack_flag,
            "mi_attack": self.gnn_manager.mi_attack_flag,
            "poison_defense": self.gnn_manager.poison_defense_flag,
            "evasion_defense": self.gnn_manager.evasion_defense_flag,
            "mi_defense": self.gnn_manager.mi_defense_flag,
        }

    def set_clear_model(self):
        self.gnn_manager.poison_attack_flag = False
        self.gnn_manager.evasion_attack_flag = False
        self.gnn_manager.mi_attack_flag = False
        self.gnn_manager.poison_defense_flag = False
        self.gnn_manager.evasion_defense_flag = False
        self.gnn_manager.mi_defense_flag = False

    def return_attack_defense_flags(self):
        self.gnn_manager.poison_attack_flag = self.start_attack_defense_flag_state["poison_attack"]
        self.gnn_manager.evasion_attack_flag = self.start_attack_defense_flag_state["evasion_attack"]
        self.gnn_manager.mi_attack_flag = self.start_attack_defense_flag_state["mi_attack"]
        self.gnn_manager.poison_defense_flag = self.start_attack_defense_flag_state["poison_defense"]
        self.gnn_manager.evasion_defense_flag = self.start_attack_defense_flag_state["evasion_defense"]
        self.gnn_manager.mi_defense_flag = self.start_attack_defense_flag_state["mi_defense"]

    def evasion_attack_pipeline(
            self,
            metrics_attack,
            model_metrics,
            steps: int,
            save_model_flag: bool = True,
    ):
        metrics_values = {}
        if self.available_attacks["evasion"]:
            self.set_clear_model()
            from models_builder.gnn_models import Metric
            self.gnn_manager.train_model(
                gen_dataset=self.gen_dataset,
                steps=steps,
                save_model_flag=save_model_flag,
                metrics=[Metric("F1", mask='train', average=None)]
            )
            metric_clean_model = self.gnn_manager.evaluate_model(
                gen_dataset=self.gen_dataset,
                metrics=model_metrics
            )
            self.gnn_manager.evasion_attack_flag = True
            self.gnn_manager.train_model(
                gen_dataset=self.gen_dataset,
                steps=steps,
                save_model_flag=save_model_flag,
                metrics=[Metric("F1", mask='train', average=None)]
            )
            metric_evasion_attack_only = self.gnn_manager.evaluate_model(
                gen_dataset=self.gen_dataset,
                metrics=model_metrics
            )
            # TODO Kirill
            # metrics_values = evaluate_attacks(
            #     metric_clean_model,
            #     metric_evasion_attack_only,
            #     metrics_attack=metrics_attack
            # )
            self.return_attack_defense_flags()
            pass
        else:
            warnings.warn(f"Evasion attack is not available. Please set evasion attack for "
                          f"gnn_model_manager use def set_evasion_attacker")

        return metrics_values
