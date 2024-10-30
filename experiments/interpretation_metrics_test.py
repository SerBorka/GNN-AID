import json
import os
import random
import warnings

import torch

from aux.custom_decorators import timing_decorator
from aux.utils import EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH, EXPLAINERS_INIT_PARAMETERS_PATH, root_dir, \
    EVASION_DEFENSE_PARAMETERS_PATH, EVASION_ATTACK_PARAMETERS_PATH
from explainers.explainer_metrics import NodesExplainerMetric
from explainers.explainers_manager import FrameworkExplainersManager
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from src.aux.configs import ModelModificationConfig, ConfigPattern
from src.aux.utils import POISON_DEFENSE_PARAMETERS_PATH
from src.base.datasets_processing import DatasetManager
from src.models_builder.models_zoo import model_configs_zoo
from defense.JaccardDefense import jaccard_def
from attacks.metattack import meta_gradient_attack
from defense.GNNGuard import gnnguard

def load_result_dict(path):
    if os.path.exists(path):
        with open(path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
    return data


def save_result_dict(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(data, file)


@timing_decorator
def run_interpretation_test(dataset_full_name):
    # steps_epochs = 200
    # num_explaining_nodes = 1
    # explaining_metrics_params = {
    #     "stability_graph_perturbations_nums": 1,
    #     "stability_feature_change_percent": 0.05,
    #     "stability_node_removal_percent": 0.05,
    #     "consistency_num_explanation_runs": 1
    # }
    steps_epochs = 200
    num_explaining_nodes = 100
    explaining_metrics_params = {
        "stability_graph_perturbations_nums": 20,
        "stability_feature_change_percent": 0.05,
        "stability_node_removal_percent": 0.05,
        "consistency_num_explanation_runs": 20
    }
    explainer_name = 'GNNExplainer(torch-geom)'
    dataset_key_name = "_".join(dataset_full_name)
    metrics_path = root_dir / "experiments" / "explainers_metrics"
    dataset_metrics_path = metrics_path / f"{dataset_key_name}_{explainer_name}_metrics.json"

    restart_experiment = True
    if restart_experiment:
        result_dict = {}
    else:
        result_dict = load_result_dict(dataset_metrics_path)

    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=dataset_full_name,
        dataset_ver_ind=0
    )

    node_indices = random.sample(range(dataset.data.x.shape[0]), num_explaining_nodes)

    #
    # try:
    #     unprotected_key = "Unprotected"
    #     if unprotected_key not in result_dict:
    #         print(f"Calculation of explanation metrics with defence: {unprotected_key} started.")
    #         metrics = calculate_unprotected_metrics(
    #             dataset_full_name,
    #             explainer_name,
    #             steps_epochs,
    #             num_explaining_nodes,
    #             explaining_metrics_params,
    #             dataset,
    #             node_indices
    #         )
    #         result_dict[unprotected_key] = metrics
    #         print(f"Calculation of explanation metrics with defence: {unprotected_key} completed. Metrics:\n{metrics}")
    #         save_result_dict(dataset_metrics_path, result_dict)
    # except Exception:
    #     pass
    # try:
    #     jaccard_key = "Jaccard_defence"
    #     if jaccard_key not in result_dict:
    #         print(f"Calculation of explanation metrics with defence: {jaccard_key} started.")
    #         metrics = calculate_jaccard_defence_metrics(
    #             dataset_full_name,
    #             explainer_name,
    #             steps_epochs,
    #             num_explaining_nodes,
    #             explaining_metrics_params,
    #             dataset,
    #             node_indices
    #         )
    #         result_dict[jaccard_key] = metrics
    #         print(f"Calculation of explanation metrics with defence: {jaccard_key} completed. Metrics:\n{metrics}")
    #         save_result_dict(dataset_metrics_path, result_dict)
    # except Exception:
    #     pass
    

    adv_training_key = "AdvTraining_defence"
    if adv_training_key not in result_dict:
        print(f"Calculation of explanation metrics with defence: {adv_training_key} started.")
        metrics = calculate_adversial_defence_metrics(
            dataset_full_name,
            explainer_name,
            steps_epochs,
            num_explaining_nodes,
            explaining_metrics_params,
            dataset,
            node_indices
        )
        result_dict[adv_training_key] = metrics
        print(f"Calculation of explanation metrics with defence: {adv_training_key} completed. Metrics:\n{metrics}")
        save_result_dict(dataset_metrics_path, result_dict)

    gnnguard_key = "GNNGuard_defence"
    if gnnguard_key not in result_dict:
        print(f"Calculation of explanation metrics with defence: {gnnguard_key} started.")
        metrics = calculate_adversial_defence_metrics(
            dataset_full_name,
            explainer_name,
            steps_epochs,
            num_explaining_nodes,
            explaining_metrics_params,
            dataset,
            node_indices
        )
        result_dict[gnnguard_key] = metrics
        print(f"Calculation of explanation metrics with defence: {gnnguard_key} completed. Metrics:\n{metrics}")
        save_result_dict(dataset_metrics_path, result_dict)



@timing_decorator
def calculate_unprotected_metrics(
        dataset_full_name,
        explainer_name,
        steps_epochs,
        num_explaining_nodes,
        explaining_metrics_params,
        dataset,
        node_indices
):
    save_model_flag = False
    device = torch.device('cpu')

    data, results_dataset_path = dataset.data, dataset.results_dir

    gnn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')
    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "mask_features": [],
            "optimizer": {
                # "_config_class": "Config",
                "_class_name": "Adam",
                # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                # "_class_import_info": ["torch.optim"],
                "_config_kwargs": {},
            }
        }
    )
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )
    gnn_model_manager.gnn.to(device)
    data.x = data.x.float()
    # data.y = data.y.float()
    data = data.to(device)

    warnings.warn("Start training")
    try:
        raise FileNotFoundError
        gnn_model_manager.load_model_executor()
        print("Loaded model.")
    except FileNotFoundError:
        gnn_model_manager.epochs = gnn_model_manager.modification.epochs = 0
        train_test_split_path = gnn_model_manager.train_model(gen_dataset=dataset, steps=steps_epochs,
                                                              save_model_flag=save_model_flag,
                                                              metrics=[Metric("F1", mask='train', average=None)])

        if train_test_split_path is not None:
            dataset.save_train_test_mask(train_test_split_path)
            train_mask, val_mask, test_mask, train_test_sizes = torch.load(train_test_split_path / 'train_test_split')[
                                                                :]
            dataset.train_mask, dataset.val_mask, dataset.test_mask = train_mask, val_mask, test_mask
            data.percent_train_class, data.percent_test_class = train_test_sizes
    warnings.warn("Training was successful")

    metric_loc = gnn_model_manager.evaluate_model(
        gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro')])
    print(metric_loc)

    explainer_init_config = ConfigPattern(
        _class_name=explainer_name,
        _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
        _config_class="ExplainerInitConfig",
        _config_kwargs={}
    )
    explainer_metrics_run_config = ConfigPattern(
        _config_class="ExplainerRunConfig",
        _config_kwargs={
            "mode": "local",
            "kwargs": {
                "_class_name": explainer_name,
                "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                "_config_class": "Config",
                "_config_kwargs": explaining_metrics_params,
            }
        }
    )

    explainer = FrameworkExplainersManager(
        init_config=explainer_init_config,
        dataset=dataset, gnn_manager=gnn_model_manager,
        explainer_name=explainer_name,
    )

    explanation_metrics = explainer.evaluate_metrics(node_indices, explainer_metrics_run_config)
    return explanation_metrics


@timing_decorator
def calculate_jaccard_defence_metrics(
        dataset_full_name,
        explainer_name,
        steps_epochs,
        num_explaining_nodes,
        explaining_metrics_params,
        dataset,
        node_indices
):
    save_model_flag = False
    device = torch.device('cpu')

    data, results_dataset_path = dataset.data, dataset.results_dir

    gnn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')
    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "mask_features": [],
            "optimizer": {
                # "_config_class": "Config",
                "_class_name": "Adam",
                # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                # "_class_import_info": ["torch.optim"],
                "_config_kwargs": {},
            }
        }
    )
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )
    gnn_model_manager.gnn.to(device)
    data.x = data.x.float()
    data = data.to(device)

    poison_defense_config = ConfigPattern(
        _class_name="JaccardDefender",
        _import_path=POISON_DEFENSE_PARAMETERS_PATH,
        _config_class="PoisonDefenseConfig",
        _config_kwargs={
        }
    )

    gnn_model_manager.set_poison_defender(poison_defense_config=poison_defense_config)

    warnings.warn("Start training")
    try:
        print("Loading model executor")
        gnn_model_manager.load_model_executor()
    except FileNotFoundError:
        print("Training started.")
        gnn_model_manager.epochs = gnn_model_manager.modification.epochs = 0
        train_test_split_path = gnn_model_manager.train_model(gen_dataset=dataset, steps=steps_epochs,
                                                              save_model_flag=save_model_flag,
                                                              metrics=[Metric("F1", mask='train', average=None)])

        if train_test_split_path is not None:
            dataset.save_train_test_mask(train_test_split_path)
            train_mask, val_mask, test_mask, train_test_sizes = torch.load(train_test_split_path / 'train_test_split')[
                                                                :]
            dataset.train_mask, dataset.val_mask, dataset.test_mask = train_mask, val_mask, test_mask
            data.percent_train_class, data.percent_test_class = train_test_sizes
    warnings.warn("Training was successful")

    metric_loc = gnn_model_manager.evaluate_model(
        gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro')])
    print(metric_loc)

    explainer_init_config = ConfigPattern(
        _class_name=explainer_name,
        _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
        _config_class="ExplainerInitConfig",
        _config_kwargs={}
    )
    explainer_metrics_run_config = ConfigPattern(
        _config_class="ExplainerRunConfig",
        _config_kwargs={
            "mode": "local",
            "kwargs": {
                "_class_name": explainer_name,
                "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                "_config_class": "Config",
                "_config_kwargs": explaining_metrics_params,
            }
        }
    )

    explainer = FrameworkExplainersManager(
        init_config=explainer_init_config,
        dataset=dataset, gnn_manager=gnn_model_manager,
        explainer_name=explainer_name,
    )

    explanation_metrics = explainer.evaluate_metrics(node_indices, explainer_metrics_run_config)
    return explanation_metrics


@timing_decorator
def calculate_adversial_defence_metrics(
        dataset_full_name,
        explainer_name,
        steps_epochs,
        num_explaining_nodes,
        explaining_metrics_params,
        dataset,
        node_indices
):
    save_model_flag = False
    device = torch.device('cpu')

    data, results_dataset_path = dataset.data, dataset.results_dir

    gnn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')
    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "mask_features": [],
            "optimizer": {
                # "_config_class": "Config",
                "_class_name": "Adam",
                # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                # "_class_import_info": ["torch.optim"],
                "_config_kwargs": {},
            }
        }
    )
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )
    gnn_model_manager.gnn.to(device)
    data.x = data.x.float()
    data = data.to(device)

    fgsm_evasion_attack_config0 = ConfigPattern(
        _class_name="FGSM",
        _import_path=EVASION_ATTACK_PARAMETERS_PATH,
        _config_class="EvasionAttackConfig",
        _config_kwargs={
            "epsilon": 0.1 * 1,
        }
    )
    at_evasion_defense_config = ConfigPattern(
        _class_name="AdvTraining",
        _import_path=EVASION_DEFENSE_PARAMETERS_PATH,
        _config_class="EvasionDefenseConfig",
        _config_kwargs={
            "attack_name": None,
            "attack_config": fgsm_evasion_attack_config0  # evasion_attack_config
        }
    )

    from defense.evasion_defense import EvasionDefender
    from src.aux.utils import all_subclasses
    print([e.name for e in all_subclasses(EvasionDefender)])
    gnn_model_manager.set_evasion_defender(evasion_defense_config=at_evasion_defense_config)

    warnings.warn("Start training")
    try:
        gnn_model_manager.load_model_executor()
        print("Loaded model.")
    except FileNotFoundError:
        gnn_model_manager.epochs = gnn_model_manager.modification.epochs = 0
        train_test_split_path = gnn_model_manager.train_model(gen_dataset=dataset, steps=steps_epochs,
                                                              save_model_flag=save_model_flag,
                                                              metrics=[Metric("F1", mask='train', average=None)])

        if train_test_split_path is not None:
            dataset.save_train_test_mask(train_test_split_path)
            train_mask, val_mask, test_mask, train_test_sizes = torch.load(train_test_split_path / 'train_test_split')[
                                                                :]
            dataset.train_mask, dataset.val_mask, dataset.test_mask = train_mask, val_mask, test_mask
            data.percent_train_class, data.percent_test_class = train_test_sizes
    warnings.warn("Training was successful")

    metric_loc = gnn_model_manager.evaluate_model(
        gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro')])
    print(metric_loc)

    explainer_init_config = ConfigPattern(
        _class_name=explainer_name,
        _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
        _config_class="ExplainerInitConfig",
        _config_kwargs={}
    )
    explainer_metrics_run_config = ConfigPattern(
        _config_class="ExplainerRunConfig",
        _config_kwargs={
            "mode": "local",
            "kwargs": {
                "_class_name": explainer_name,
                "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                "_config_class": "Config",
                "_config_kwargs": explaining_metrics_params,
            }
        }
    )

    explainer = FrameworkExplainersManager(
        init_config=explainer_init_config,
        dataset=dataset, gnn_manager=gnn_model_manager,
        explainer_name=explainer_name,
    )

    explanation_metrics = explainer.evaluate_metrics(node_indices, explainer_metrics_run_config)
    return explanation_metrics


@timing_decorator
def calculate_gnnguard_defence_metrics(
        dataset_full_name,
        explainer_name,
        steps_epochs,
        num_explaining_nodes,
        explaining_metrics_params,
        dataset,
        node_indices
):
    save_model_flag = False
    device = torch.device('cpu')

    data, results_dataset_path = dataset.data, dataset.results_dir

    gnn = model_configs_zoo(dataset=dataset, model_name='gcn_gcn')
    manager_config = ConfigPattern(
        _config_class="ModelManagerConfig",
        _config_kwargs={
            "mask_features": [],
            "optimizer": {
                # "_config_class": "Config",
                "_class_name": "Adam",
                # "_import_path": OPTIMIZERS_PARAMETERS_PATH,
                # "_class_import_info": ["torch.optim"],
                "_config_kwargs": {},
            }
        }
    )
    gnn_model_manager = FrameworkGNNModelManager(
        gnn=gnn,
        dataset_path=results_dataset_path,
        manager_config=manager_config,
        modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
    )
    gnn_model_manager.gnn.to(device)
    data.x = data.x.float()
    data = data.to(device)

    gnnguard_poison_defense_config = ConfigPattern(
        _class_name="GNNGuard",
        _import_path=POISON_DEFENSE_PARAMETERS_PATH,
        _config_class="PoisonDefenseConfig",
        _config_kwargs={
            "lr": 0.01,
            "train_iters": 100,
            # "model": gnn_model_manager.gnn
        }
    )

    gnn_model_manager.set_poison_defender(poison_defense_config=gnnguard_poison_defense_config)

    warnings.warn("Start training")
    try:
        gnn_model_manager.load_model_executor()
        print("Loaded model.")
    except FileNotFoundError:
        gnn_model_manager.epochs = gnn_model_manager.modification.epochs = 0
        train_test_split_path = gnn_model_manager.train_model(gen_dataset=dataset, steps=steps_epochs,
                                                              save_model_flag=save_model_flag,
                                                              metrics=[Metric("F1", mask='train', average=None)])

        if train_test_split_path is not None:
            dataset.save_train_test_mask(train_test_split_path)
            train_mask, val_mask, test_mask, train_test_sizes = torch.load(train_test_split_path / 'train_test_split')[
                                                                :]
            dataset.train_mask, dataset.val_mask, dataset.test_mask = train_mask, val_mask, test_mask
            data.percent_train_class, data.percent_test_class = train_test_sizes
    warnings.warn("Training was successful")

    metric_loc = gnn_model_manager.evaluate_model(
        gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro')])
    print(metric_loc)

    explainer_init_config = ConfigPattern(
        _class_name=explainer_name,
        _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
        _config_class="ExplainerInitConfig",
        _config_kwargs={}
    )
    explainer_metrics_run_config = ConfigPattern(
        _config_class="ExplainerRunConfig",
        _config_kwargs={
            "mode": "local",
            "kwargs": {
                "_class_name": explainer_name,
                "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                "_config_class": "Config",
                "_config_kwargs": explaining_metrics_params,
            }
        }
    )

    explainer = FrameworkExplainersManager(
        init_config=explainer_init_config,
        dataset=dataset, gnn_manager=gnn_model_manager,
        explainer_name=explainer_name,
    )

    explanation_metrics = explainer.evaluate_metrics(node_indices, explainer_metrics_run_config)
    return explanation_metrics


if __name__ == '__main__':
    random.seed(777)
    # dataset_full_name = ("single-graph", "Planetoid", 'Cora')
    # run_interpretation_test(dataset_full_name)
    dataset_full_name = ("single-graph", "Amazon", 'Photo')
    run_interpretation_test(dataset_full_name)
