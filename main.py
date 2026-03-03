import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from einops import rearrange
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.eval_util.plot import plot_single, plot_next_multi
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

MODEL = "moirai2"  # model name: choose from {'moirai', 'moirai-moe', 'moirai2'}
SIZE = "small"  # model size: choose from {'small', 'base', 'large'} -> moirai2'de sadece small modeli var hugging face'de 
CTX = 256  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer


model = Moirai2Forecast(
    module=Moirai2Module.from_pretrained(
        f"Salesforce/moirai-2.0-R-small",
    ),
    prediction_length=100,
    context_length=1680,
    target_dim=1,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0,
)

module = Moirai2Module.from_pretrained(
    "Salesforce/moirai-2.0-R-small", 
    prediction_length= 1, 
    context_length= 512, 
    target_dim= 1, 
    feat_dynamic_real_dim= 0,
    past_feat_dynamic_real_dim= 0
)

