# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pandas as pd

filename = "datasets/air_passengers.csv"
df = pd.read_csv(filename)

# %%
import numpy as np
import pandas as pd
from noprophet import to_indices

values = np.random.randn(20)
indices = to_indices(values)

df = pd.DataFrame({"ds": indices, "y": values})
df

# %%
from noprophet import NoProphet

m = NoProphet()

ds, y = m.fit(df, epochs=20)

# %%
forecast = m.predict(df)
forecast

# %%
m.plot(to_indices(ds), y, forecast)
