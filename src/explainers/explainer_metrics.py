import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch_geometric.utils import subgraph

from aux.configs import ConfigPattern
from aux.custom_decorators import timing_decorator


class NodesExplainerMetric:
    def __init__(self, explainers_manager, explaining_metrics_params=None):
        self.node_id_to_explainer_run_config = None
        self.explanation_metrics_path = None
        if explaining_metrics_params is None:
            explaining_metrics_params = {}
        self.explainers_manager = explainers_manager
        self.model = explainers_manager.gnn
        self.gen_dataset = explainers_manager.gen_dataset
        self.explainer = explainers_manager.explainer
        self.graph = explainers_manager.gen_dataset.data
        self.x = self.graph.x
        self.edge_index = self.graph.edge_index
        self.kwargs_dict = {
            "stability_graph_perturbations_nums": 10,
            "stability_feature_change_percent": 0.05,
            "stability_node_removal_percent": 0.05,
            "consistency_num_explanation_runs": 10
        }
        self.kwargs_dict.update(explaining_metrics_params)

        self.dictionary = {
            "explaining_metrics_params": self.kwargs_dict,
            "perturbed_explanations": {}
        }
        print(f"NodesExplainerMetric initialized with kwargs:\n{self.kwargs_dict}")

    def get_explanation_path(self, run_config):
        self.explainers_manager.explanation_result_path(run_config)
        explainer_result_file_path, files_paths = \
            self.explainers_manager.explainer_result_file_path, self.explainers_manager.files_paths
        self.explanation_metrics_path = files_paths[0].parent / Path('explanation_metrics.json')
        return explainer_result_file_path

    def save_dictionary(self):
        with open(self.explanation_metrics_path, "w") as f:
            json.dump(self.dictionary, f, indent=2)

    def evaluate(self, node_id_to_explainer_run_config: dict):
        self.node_id_to_explainer_run_config = node_id_to_explainer_run_config
        target_nodes_indices = sorted(node_id_to_explainer_run_config.keys())

        self.get_explanations(target_nodes_indices[0])
        if os.path.exists(self.explanation_metrics_path):
            with open(self.explanation_metrics_path, "r") as f:
                self.dictionary = json.load(f)

        sparsity = []
        stability = []
        consistency = []
        for node_ind in target_nodes_indices:
            print(f"Processing explanation metrics calculation for node id {node_ind}.")
            self.get_explanations(node_ind)
            sparsity += [self.calculate_sparsity(node_ind)]
            stability += [self.calculate_stability(
                node_ind,
                graph_perturbations_nums=self.kwargs_dict["stability_graph_perturbations_nums"],
                feature_change_percent=self.kwargs_dict["stability_feature_change_percent"],
                node_removal_percent=self.kwargs_dict["stability_node_removal_percent"]
            )]
            consistency += [self.calculate_consistency(
                node_ind,
                num_explanation_runs=self.kwargs_dict["consistency_num_explanation_runs"]
            )]
        fidelity = self.calculate_fidelity(target_nodes_indices)
        self.dictionary["sparsity"] = process_metric(sparsity)
        self.dictionary["stability"] = process_metric(stability)
        self.dictionary["consistency"] = process_metric(consistency)
        self.dictionary["fidelity"] = process_metric(fidelity)
        self.save_dictionary()
        return self.dictionary

    @timing_decorator
    def calculate_fidelity(self, target_nodes_indices):
        original_answer = self.model.get_answer(self.x, self.edge_index)
        same_answers_count = []
        for node_ind in target_nodes_indices:
            node_explanation = self.get_explanations(node_ind)[0]
            new_x, new_edge_index, new_target_node = self.filter_graph_by_explanation(
                self.x, self.edge_index, node_explanation, node_ind
            )
            filtered_answer = self.model.get_answer(new_x, new_edge_index)
            matched = filtered_answer[new_target_node] == original_answer[node_ind]
            print(f"Processed fidelity calculation for node id {node_ind}. Matched: {matched}")
            same_answers_count.append(int(matched))

        return same_answers_count

    @timing_decorator
    def calculate_sparsity(self, node_ind):
        explanation = self.get_explanations(node_ind)[0]
        num = 0
        den = 0
        # TODO: fix me by NeighborLoader
        if explanation["data"]["nodes"]:
            num += len(explanation["data"]["nodes"])
            den += self.x.shape[0]
        if explanation["data"]["edges"]:
            num += len(explanation["data"]["edges"])
            den += self.edge_index.shape[1]

        sparsity = 1 - num / den
        print(f"Sparsity calculation for node id {node_ind} completed.")
        return sparsity

    @timing_decorator
    def calculate_stability(
            self,
            node_ind,
            graph_perturbations_nums=10,
            feature_change_percent=0.05,
            node_removal_percent=0.05
    ):
        print(f"Stability calculation for node id {node_ind} started.")
        base_explanation = self.get_explanations(node_ind)[0]
        run_config = self.node_id_to_explainer_run_config[node_ind]
        stability = []
        if node_ind not in self.dictionary["perturbed_explanations"]:
            self.dictionary["perturbed_explanations"][node_ind] = []

        for i in range(graph_perturbations_nums):
            if i < len(self.dictionary["perturbed_explanations"][node_ind]):
                perturbed_explanation = self.dictionary["perturbed_explanations"][node_ind][i]
            else:
                new_dataset = self.perturb_graph(
                    self.gen_dataset, node_ind, feature_change_percent, node_removal_percent
                )
                perturbed_explanation = self.calculate_explanation(run_config, new_dataset)
                self.dictionary["perturbed_explanations"][node_ind] += [perturbed_explanation]
                self.save_dictionary()

            base_explanation_vector, perturbed_explanation_vector = \
                NodesExplainerMetric.calculate_explanation_vectors(base_explanation, perturbed_explanation)

            stability += [euclidean_distance(base_explanation_vector, perturbed_explanation_vector)]

        # stability = stability / graph_perturbations_nums
        print(f"Stability calculation for node id {node_ind} completed.")
        return stability

    @timing_decorator
    def calculate_consistency(self, node_ind, num_explanation_runs=10):
        print(f"Consistency calculation for node id {node_ind} started.")
        explanations = self.get_explanations(node_ind, num_explanations=num_explanation_runs+1)
        explanation = explanations[0]
        consistency = []
        for ind in range(num_explanation_runs):
            perturbed_explanation = explanations[ind+1]
            base_explanation_vector, perturbed_explanation_vector = \
                NodesExplainerMetric.calculate_explanation_vectors(explanation, perturbed_explanation)
            consistency += [cosine_similarity(base_explanation_vector, perturbed_explanation_vector)]
            explanation = perturbed_explanation

        # consistency = consistency / num_explanation_runs
        print(f"Consistency calculation for node id {node_ind} completed.")
        return consistency

    @timing_decorator
    def calculate_explanation(self, run_config, gen_dataset, save_explanation_flag=False):
        # print(f"Processing explanation calculation for node id {node_idx}.")
        # self.explainer.run('local', {'element_idx': node_idx}, finalize=True)
        # self.explainer.evaluate_tensor_graph(x, edge_index, node_idx, **kwargs)
        # print(f"Explanation calculation for node id {node_idx} completed.")
        # return self.explainer.explanation.dictionary
        return self.explainers_manager.conduct_experiment_by_dataset(
            run_config,
            gen_dataset,
            save_explanation_flag=save_explanation_flag
        )

    def get_explanations(self, node_ind, num_explanations=1):
        node_explanations = []
        for explanation_index in range(num_explanations):
            run_config = self.node_id_to_explainer_run_config[node_ind]
            self.explainers_manager.modification_config = ConfigPattern(
                _config_class="ExplainerModificationConfig",
                _config_kwargs={"explainer_ver_ind": explanation_index}
            )
            explainer_result_file_path = self.get_explanation_path(run_config)
            if os.path.exists(explainer_result_file_path):
                with open(explainer_result_file_path, "r") as f:
                    node_explanation = json.load(f)
            else:
                node_explanation = self.calculate_explanation(run_config, self.gen_dataset, save_explanation_flag=True)
            node_explanations += [node_explanation]
        return node_explanations

    @staticmethod
    def parse_explanation(explanation):
        important_nodes = {
            int(node): float(weight) for node, weight in explanation["data"]["nodes"].items()
        }
        important_edges = {
            tuple(map(int, edge.split(','))): float(weight)
            for edge, weight in explanation["data"]["edges"].items()
        }
        return important_nodes, important_edges

    @staticmethod
    def filter_graph_by_explanation(x, edge_index, explanation, target_node):
        important_nodes, important_edges = NodesExplainerMetric.parse_explanation(explanation)
        all_important_nodes = set(important_nodes.keys())
        all_important_nodes.add(target_node)
        for u, v in important_edges.keys():
            all_important_nodes.add(u)
            all_important_nodes.add(v)

        important_node_indices = list(all_important_nodes)
        node_mask = torch.zeros(x.size(0), dtype=torch.bool)
        node_mask[important_node_indices] = True

        new_edge_index, new_edge_weight = subgraph(node_mask, edge_index, relabel_nodes=True)
        new_x = x[node_mask]
        new_target_node = important_node_indices.index(target_node)
        return new_x, new_edge_index, new_target_node

    @staticmethod
    def calculate_explanation_vectors(base_explanation, perturbed_explanation):
        base_important_nodes, base_important_edges = NodesExplainerMetric.parse_explanation(
            base_explanation
        )
        perturbed_important_nodes, perturbed_important_edges = NodesExplainerMetric.parse_explanation(
            perturbed_explanation
        )
        union_nodes = set(base_important_nodes.keys()) | set(perturbed_important_nodes.keys())
        union_edges = set(base_important_edges.keys()) | set(perturbed_important_edges.keys())
        explain_vector_len = len(union_nodes) + len(union_edges)
        base_explanation_vector = np.zeros(explain_vector_len)
        perturbed_explanation_vector = np.zeros(explain_vector_len)
        i = 0
        for expl_node_ind in union_nodes:
            base_explanation_vector[i] = base_important_nodes.get(expl_node_ind, 0)
            perturbed_explanation_vector[i] = perturbed_important_nodes.get(expl_node_ind, 0)
            i += 1
        for expl_edge in union_edges:
            base_explanation_vector[i] = base_important_edges.get(expl_edge, 0)
            perturbed_explanation_vector[i] = perturbed_important_edges.get(expl_edge, 0)
            i += 1
        return base_explanation_vector, perturbed_explanation_vector

    @staticmethod
    def perturb_graph(gen_dataset, node_ind, feature_change_percent, node_removal_percent):
        new_dataset = copy.deepcopy(gen_dataset)
        x = new_dataset.data.x
        edge_index = new_dataset.data.edge_index
        new_x = x.clone()
        num_nodes = x.shape[0]
        num_features = x.shape[1]
        num_features_to_change = int(feature_change_percent * num_nodes * num_features)
        indices = torch.randint(0, num_nodes * num_features, (num_features_to_change,), device=x.device)
        new_x.view(-1)[indices] = 1.0 - new_x.view(-1)[indices]

        neighbors = edge_index[1][edge_index[0] == node_ind].unique()
        num_nodes_to_remove = int(node_removal_percent * neighbors.shape[0])

        if num_nodes_to_remove > 0:
            nodes_to_remove = neighbors[
                torch.randperm(neighbors.size(0), device=edge_index.device)[:num_nodes_to_remove]
            ]
            mask = ~((edge_index[0] == node_ind) & torch.isin(edge_index[1], nodes_to_remove))
            new_edge_index = edge_index[:, mask]
        else:
            new_edge_index = edge_index
        new_dataset.data.x = new_x
        new_dataset.data.edge_index = new_edge_index
        return new_dataset


def process_metric(data):
    np_data = np.array(data)
    return {
        "mean": np.mean(np_data),
        "var": np.var(np_data),
        "data": data
    }


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)
