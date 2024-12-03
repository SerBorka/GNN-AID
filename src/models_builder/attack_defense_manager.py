import warnings
from typing import Type, Union, List

import torch

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
            steps: int,
            save_model_flag: bool = True,
            mask: Union[str, List[bool], torch.Tensor] = 'test',
    ):
        metrics_values = {}
        if self.available_attacks["evasion"]:
            self.set_clear_model()
            self.gnn_manager.modification.epochs = 0
            self.gnn_manager.gnn.reset_parameters()
            from models_builder.gnn_models import Metric
            self.gnn_manager.train_model(
                gen_dataset=self.gen_dataset,
                steps=steps,
                save_model_flag=save_model_flag,
                metrics=[Metric("F1", mask='train', average=None)]
            )
            y_predict_clean = self.gnn_manager.run_model(
                gen_dataset=self.gen_dataset,
                mask=mask,
                out='logits',
            )

            self.gnn_manager.evasion_attack_flag = True
            self.gnn_manager.modification.epochs = 0
            self.gnn_manager.gnn.reset_parameters()
            self.gnn_manager.train_model(
                gen_dataset=self.gen_dataset,
                steps=steps,
                save_model_flag=save_model_flag,
                metrics=[Metric("F1", mask='train', average=None)]
            )
            self.gnn_manager.call_evasion_attack(
                gen_dataset=self.gen_dataset,
                mask=mask,
            )
            y_predict_attack = self.gnn_manager.run_model(
                gen_dataset=self.gen_dataset,
                mask=mask,
                out='logits',
            )
            metrics_values = self.evaluate_attack_defense(
                y_predict_after_attack_only=y_predict_attack,
                y_predict_clean=y_predict_clean,
                metrics_attack=metrics_attack,
            )
            self.return_attack_defense_flags()

        else:
            warnings.warn(f"Evasion attack is not available. Please set evasion attack for "
                          f"gnn_model_manager use def set_evasion_attacker")

        return metrics_values

    def evaluate_attack_defense(
            self,
            y_predict_clean,
            y_predict_after_attack_only=None,
            y_predict_after_defense_only=None,
            y_predict_after_attack_and_defense=None,
            metrics_attack=None,
            metrics_defense=None,
    ):
        metrics_attack_values = {}
        if metrics_attack is not None and y_predict_after_attack_only is not None:
            for metric in metrics_attack:
                metrics_attack_values[metric.name] = metric.compute(y_predict_clean, y_predict_after_attack_only)

        return metrics_attack_values
