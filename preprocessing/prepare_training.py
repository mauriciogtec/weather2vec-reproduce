# %%
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

# %%
years = list(range(2000, 2015))
yms = []
for y in years:
    for m in range(12):
        yms.append(f"{y}{m + 1:02d}")
# print(yms)


# %%

rasts = []
msks = []
for ym in yms:
    fpath = f"./data/SO4/processed/{ym}.tif"
    with rasterio.open(fpath) as src:
        msk = src.read_masks()
        msk = (msk > 0).astype(np.float32)
        rast = src.read() * msk
        rasts.append(rast)
        msks.append(msk)

so4_rast = np.concatenate(rasts, 0)
so4_msk = np.squeeze(msks[0], 0)

# %%
X = so4_rast[0]
X = (X - X.min()) / (X.max() - X.min())
# plt.imshow(X)

# %%

rasts = []
msks = []
for ym in yms:
    fpath = f"./data/weather/processed/{ym}.tif"
    with rasterio.open(fpath) as src:
        msk = src.read_masks()
        msk = (msk > 0).astype(np.float32)
        rast = src.read() * msk
        rasts.append(rast)
        msks.append(msk)

cov_rast = np.stack(rasts, 0)
cov_msk = msks[0][0]


# %%
# plt.imshow(cov_rast[0, 3])



#%% plant locations

nrow = 128
ncol = 256
xmin = -135.0
xmax = -60.0
ymin = 20.0
ymax = 52.0

pp = pd.read_csv("data/so2_data_full.csv")
pp['ym'] = pp.year.astype(str) + pp.month.apply(lambda x: f"{x:02d}")
pp.loc[np.isnan(pp['so2_tons']), "so2_tons"] = 0
pp_idx = pp.set_index(["ym", "fid"])["so2_tons"]

pp_locs = pp.groupby(["fid", "lon", "lat"]).agg({"so2_tons": sum})
pp_locs = pp_locs.reset_index()
# pp_locs = pp[["fid", "lon", "lat"]].drop_duplicates()

delta_lon = (xmax - xmin) / ncol
delta_lat = (ymin - ymax) / nrow

pp_locs["col"] = ((pp_locs.lon - xmin) // delta_lon).astype(int)
pp_locs["row"] = ((pp_locs.lat - ymax) // delta_lat).astype(int)
pp_locs

# %%
pp_ymtot = (
    pp.groupby(["year", "ym", "month"])
      .agg({"so2_tons": sum})
      .reset_index()
      .query("2000 <= year <= 2014")
      .sort_values("ym")
)
# pp_ymtot.plot("ym", "so2_tons", figsize=(12, 6))
#%%
with open("data/weather/weather_names.csv", "r") as f:
    wn = f.read().splitlines()[1:]

traindata = dict(
    covars_names=wn,
    covars_rast=cov_rast,
    covars_mask=cov_msk,
    so4_rast=so4_rast,
    so4_mask=so4_msk,
    pp=pp,
    pp_locs=pp_locs,
    pp_ymtot=pp_ymtot
)

with open("data/training_data.pkl", "wb") as io:
    pkl.dump(traindata, io, protocol=pkl.HIGHEST_PROTOCOL)
