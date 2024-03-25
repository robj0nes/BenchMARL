import torch_geometric
from torch import nn
from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import SequenceModelConfig, GnnConfig, MlpConfig
experiment = Experiment(
    algorithm_config=MappoConfig.get_from_yaml(),
    model_config=GnnConfig(
        topology="full",
        self_loops=False,
        gnn_class=torch_geometric.nn.conv.GATv2Conv,
        gnn_kwargs={},
    ),
    critic_model_config=SequenceModelConfig(
        model_configs=[
            MlpConfig(num_cells=[8], activation_class=nn.Tanh, layer_class=nn.Linear),
            GnnConfig(
                topology="full",
                self_loops=False,
                gnn_class=torch_geometric.nn.conv.GraphConv,
            ),
            MlpConfig(num_cells=[6], activation_class=nn.Tanh, layer_class=nn.Linear),
        ],
        intermediate_sizes=[5,3],
    ),
    seed=0,
    config=ExperimentConfig.get_from_yaml(),
    task=VmasTask.PAINTING.get_from_yaml(),
)
experiment.run()