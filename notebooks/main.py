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

filename = "https://github.com/ourownstory/neuralprophet-data/raw/main/datasets/air_passengers.csv"
df = pd.read_csv(filename)
df.head()

# %%
# import numpy as np
# import pandas as pd
# from noprophet import to_indices

# values = np.random.randn(50)
# indices = to_indices(values)

# df = pd.DataFrame({"ds": indices, "y": values})
# plt = df.plot(x="ds", y="y", kind="scatter")

# %%
from noprophet import NoProphet

m = NoProphet()

ds, y = m.fit(df, epochs=10, learning_rate=0.0001)

forecast = m.predict(df)
m.plot(ds, y, forecast)
