import random
import warnings

import torch

from aux.custom_decorators import timing_decorator
from aux.utils import EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH, EXPLAINERS_INIT_PARAMETERS_PATH, root_dir
from explainers.explainers_manager import FrameworkExplainersManager
from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from src.aux.configs import ModelModificationConfig, ConfigPattern
from src.base.datasets_processing import DatasetManager
from src.models_builder.models_zoo import model_configs_zoo


@timing_decorator
def run_interpretation_test():
    metrics_path = root_dir / "experiments" / "explainers_metrics"
    random.seed(777)
    steps_epochs = 200
    save_model_flag = False
    device = torch.device('cpu')
    num_explaining_nodes = 100
    explaining_metrics_params = {
        "stability_graph_perturbations_nums": 20,
        "stability_feature_change_percent": 0.05,
        "stability_node_removal_percent": 0.05,
        "consistency_num_explanation_runs": 20
    }
    explainer_name = 'GNNExplainer(torch-geom)'
    dataset_full_name = ("single-graph", "Planetoid", 'Cora')
    dataset_key_name = "_".join(dataset_full_name)
    dataset_metrics_path = metrics_path / f"{dataset_key_name}_{explainer_name}_metrics.json"

    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=dataset_full_name,
        dataset_ver_ind=0
    )
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

    warnings.warn("Start training")
    try:
        gnn_model_manager.load_model_executor()
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
    # explainer_metrics_run_config = ConfigPattern(
    #     _config_class="ExplainerRunConfig",
    #     _config_kwargs={
    #         "mode": "local",
    #         "kwargs": {
    #             "_class_name": explainer_name,
    #             "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
    #             "_config_class": "Config",
    #             "_config_kwargs": explaining_metrics_params,
    #         }
    #     }
    # )

    explainer_metrics_run_config = ConfigPattern(
        _config_class="ExplainerRunConfig",
        _config_kwargs={
            "mode": "local",
            "kwargs": {
                "_class_name": "GNNExplainer(torch-geom)",
                "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
                "_config_class": "Config",
                "_config_kwargs": {
                    "stability_graph_perturbations_nums": 10,
                    "stability_feature_change_percent": 0.05,
                    "stability_node_removal_percent": 0.05,
                    "consistency_num_explanation_runs": 10
                },
            }
        }
    )

    explainer = FrameworkExplainersManager(
        init_config=explainer_init_config,
        dataset=dataset, gnn_manager=gnn_model_manager,
        explainer_name=explainer_name,
    )

    node_indices = random.sample(range(dataset.data.x.shape[0]), num_explaining_nodes)
    # explanation_metrics = explainer.evaluate_metrics(node_indices, explainer_metrics_run_config)
    # print(explanation_metrics)


if __name__ == '__main__':
    run_interpretation_test()

    print()