# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: worker_env
#     language: python
#     name: worker_env
# ---

# <div style="width: 100%;display: flex; align-items: top;">
#     <div style="float:left;width: 80%;text-align:left;position:relative">
#         <h1>Temporal chi visualization: Milvus milvus & Aves (iNaturalist)</h1>
#         <p><em><a href="mailto:a.dunkel@ioer.de">Alexander Dunkel</a>, Leibniz Institute of Ecological Urban and Regional Development, <br>
#         Transformative Capacities & Research Data Centre (IÖR-FDZ)</em></p>
#     </div>
# <div style="width:256px;text-align:right;margin-top:0px;margin-right:10px"><a href="https://gitlab.hrz.tu-chemnitz.de/ad/ephemeral_events"><img src="https://kartographie.geo.tu-dresden.de/ad/wip/ephemeral_events/version.svg" style="float:left"></a></div>
# </div>
#

# + jupyter={"source_hidden": true} tags=["hide_code"] editable=true slideshow={"slide_type": ""}
from IPython.display import Markdown as md
from datetime import date

today = date.today()
with open('/.version', 'r') as file: app_version = file.read().split("'")[1]
md(f"Last updated: {today.strftime('%b-%d-%Y')}, [Carto-Lab Docker](https://gitlab.vgiscience.de/lbsn/tools/jupyterlab) Version {app_version}")
# -

# <div style="width:500px">
#     
# Chi visualization of temporal patterns for ephemeral events. This notebook builds upon a  [a previous notebook](https://kartographie.geo.tu-dresden.de/ad/wip/ephemeral_events/html/01_temporal_chi_y.html).
#
# The chi equation requires a specific query (such as all photographs of the Red Kite - Milvus milvus - in Europe from iNaturalist) and a generic query to correct for system-wide sampling effects (such as growing number of iNaturalist contributors). However, different user groups follow different distribution patterns over time and space on geosocial media, which means that it is not easy to select a suitable generic query. Rapacciuolo et al. (2021) emphasize that comparison should be based on user selections from umbrella groups. Here, we explore whether using Chi for Milvus mulvus photographers compared to the umbrella group of "all bird photographers" on iNaturalist in Europe increases or decreases between 2010 to 2022. Given the increasing population of Milvus milvus, we would expect that chi also increases, as it is more likeley that bird photographers can take  photographs of the Red Kite.
#
# > Rapacciuolo, G., Young, A., & Johnson, R. (2021). Deriving indicators of biodiversity change from unstructured community‐contributed data. Oikos, 130(8), 1225–1239. https://doi.org/10.1111/oik.08215
#
# For selection of the generic query, we used the [emoji-taxa mappings](https://gitlab.vgiscience.de/lbsn/lbsntransform/-/blob/master/resources/mappings/field_mapping_inaturalist_gbif.py) that I made available for [lbsn.vgiscience.org](lbsn.vgiscience.org):
#
# The SQL syntax:
#
# <pre><code>
# SELECT *
# FROM topical.post t1
# WHERE t1.origin_id = 23 and '🐦'=ANY(emoji);
# </code></pre>
#
# <br>
# This selected <strong>9,147,013</strong> bird photographs from a total number of <strong>53,155,437</strong> nature and plant observations on iNaturalist (2010-2022). 
# <br><br>
# This dataset is filtered here based on space, used in the chi equation which is then visualized as a bar plot.
# </div>

# # Preparations

# + tags=["hide_code"] editable=true slideshow={"slide_type": ""}
import sys, os
import math
import numpy as np
import pandas as pd
import psycopg2
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import seaborn as sns
from matplotlib.axes import Axes 
from matplotlib import cm
from warnings import warn
from typing import Tuple, Dict, Any
from pathlib import Path
from python_hll.hll import HLL
from python_hll.util import NumberUtil
from shapely.geometry import box
module_path = str(Path.cwd().parents[0] / "py")
if module_path not in sys.path:
    sys.path.append(module_path)
from modules.base import tools, hll
# -

OUTPUT = Path.cwd().parents[0] / "out"       # output directory for figures (etc.)
WORK_DIR = Path.cwd().parents[0] / "tmp"     # Working directory

OUTPUT.mkdir(exist_ok=True)
(OUTPUT / "figures").mkdir(exist_ok=True)
(OUTPUT / "svg").mkdir(exist_ok=True)
WORK_DIR.mkdir(exist_ok=True)

# %load_ext autoreload
# %autoreload 2

# Select `M` for monthly aggregation, `Y` for yearly aggregation

AGG_BASE = "Y"

# First, define whether to study usercount or postcount

# + editable=true slideshow={"slide_type": ""} tags=["highlight"]
# METRIC = 'user'
METRIC = 'post'
# -

metric_col = 'post_hll'
if METRIC == 'user':
    metric_col = 'user_hll'

# Set global font

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Set global color

color_flickr = '#F89F5E'
color_inaturalist = '#ED6F53'

# ## Load HLL aggregate data
#
# Load the data from CSV, generated in the previous notebook. Data is stored as aggregate HLL data (postcount, usercount) for each month.

# + tags=["hide_code"] editable=true slideshow={"slide_type": ""}
INATURALIST_ALL = OUTPUT / "inaturalist_all_months.csv"
INATURALIST_ALL_MILVUSRANGE = OUTPUT / "milvus_range_inat_all_months.csv"
INATURALIST_ALL_AVES_MILVUSRANGE = OUTPUT / "inaturalist_birds_month.csv"
# -

# %%time
data_files = {
    "INATURALIST_ALL":INATURALIST_ALL,
    "INATURALIST_ALL_MILVUSRANGE":INATURALIST_ALL_MILVUSRANGE,
    "INATURALIST_ALL_AVES":INATURALIST_ALL_AVES_MILVUSRANGE,    
    }
tools.display_file_stats(data_files)

pd.read_csv(INATURALIST_ALL_AVES_MILVUSRANGE, nrows=10)

# ## Connect hll worker db

DB_USER = "hlluser"
DB_PASS = os.getenv('READONLY_USER_PASSWORD')
# set connection variables
DB_HOST = "127.0.0.1"
DB_PORT = "5452"
DB_NAME = "hllworkerdb"

# Connect to empty Postgres database running HLL Extension:

DB_CONN = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT ,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
)
DB_CONN.set_session(
    readonly=True)
DB_CALC = tools.DbConn(
    DB_CONN)
CUR_HLL = DB_CONN.cursor()

# test

# # Calculate HLL Cardinality per month and year

# Define additional functions for reading and formatting CSV as `pd.DataFrame`

# +
from datetime import datetime

def read_csv_datetime(csv: Path) -> pd.DataFrame:
    """Read CSV with parsing datetime index (months)
    
        First CSV column: Year
        Second CSV column: Month
    """
    date_cols = ["year", "month"]
    df = pd.read_csv(
        csv, index_col='datetime', 
        parse_dates={'datetime':date_cols},
        date_format='%Y %m',
        keep_date_col='False')
    df.drop(columns=date_cols, inplace=True)
    return df
    
def append_cardinality_df(df: pd.DataFrame, hll_col: str = "post_hll", cardinality_col: str = 'postcount_est'):
    """Calculate cardinality from HLL and append to extra column in df"""
    df[cardinality_col] = df.apply(
        lambda x: hll.cardinality_hll(
           x[hll_col], CUR_HLL),
        axis=1)
    df.drop(columns=[hll_col], inplace=True)
    return df

def filter_fill_time(
        df: pd.DataFrame, min_year: int, 
        max_year: int, val_col: str = "postcount_est",
        min_month: str = "01", max_month: str = "01", agg_base: str = None,
        agg_method = None):
    """Filter time values between min - max year and fill missing values"""
    max_day = "01"
    if agg_base is None:
        agg_base = "M"
    elif agg_base == "Y":
        max_month = "12"
        max_day = "31"
    min_date = pd.Timestamp(f'{min_year}-{min_month}-01')
    max_date = pd.Timestamp(f'{max_year}-{max_month}-{max_day}')
    # clip by start and end date
    if not min_date in df.index:
        df.loc[min_date, val_col] = 0
    if not max_date in df.index:
        df.loc[max_date, val_col] = 0
    df.sort_index(inplace=True)
    # mask min and max time
    time_mask = ((df.index >= min_date) & (df.index <= max_date))
    resampled = df.loc[time_mask][val_col].resample(agg_base)
    if agg_method is None:
        series = resampled.sum()
    elif agg_method == "count":
        series = resampled.count()
    elif agg_method == "nunique":
        series = resampled.nunique()
    # fill missing months with 0
    # this will also set the day to max of month
    return series.fillna(0).to_frame()


# -

# **Select dataset to process below**

# Apply functions to all data sets.
#
# - Read from CSV
# - calculate cardinality
# - merge year and month to single column
# - filter 2007 - 2018 range, fill missing values

def process_dataset(
        dataset: Path = None, metric: str = None, df_post: pd.DataFrame = None,
        min_year: int = None, max_year: int = None, agg_base: str = None) -> pd.DataFrame:
    """Apply temporal filter/pre-processing to all data sets."""
    if metric is None:
        metric = 'post_hll'
        warn(f"Using default value {metric}")
    if metric == 'post_hll':
        cardinality_col = 'postcount_est'
    else:
        cardinality_col = 'usercount_est'
    if min_year is None:
        min_year = 2007
    if max_year is None:
        max_year = 2022
    if df_post is None:
        df_post = read_csv_datetime(dataset)
    df_post = append_cardinality_df(df_post, metric, cardinality_col)
    return filter_fill_time(df_post, min_year, max_year, cardinality_col, agg_base=agg_base)


# %%time
df_post = process_dataset(INATURALIST_ALL_AVES_MILVUSRANGE, agg_base=AGG_BASE, metric='post_hll')

df_post.head(5)

# %%time
df_user = process_dataset(INATURALIST_ALL_AVES_MILVUSRANGE, metric=metric_col, agg_base=AGG_BASE)

df_user.head(5)


# # Visualize Cardinality

# Define plot function.

# +
def bar_plot_time(
        df: pd.DataFrame, ax: Axes, color: str,
        label: str, val_col: str = "postcount_est") -> Axes:
    """Matplotlib Barplot with time axis formatting

    If "significant" in df columns, applies different colors to fill/edge
    of non-significant values.
    """
    if color is None:
        # colors = sns.color_palette("vlag", as_cmap=True, n_colors=2)
        # color_rgba = colors([1.0])[0]
        # color = mcolor.rgb2hex((color_rgba), keep_alpha=True)
        color = color_inaturalist
    color_significant = color
    color_significant_edge = "white"
    if "significant" in df.columns:
        colors_bar = {True: color, False: "white"}
        color_significant = df['significant'].replace(colors_bar)
        colors_edge = {True: "white", False: "black"}
        color_significant_edge = df['significant'].replace(colors_edge)
    ax = df.set_index(
        df.index.map(lambda s: s.strftime('%Y'))).plot.bar(
            ax=ax, y=val_col, color=color_significant, width=1.0,
            label=label, edgecolor=color_significant_edge, linewidth=0.5, alpha=1.0)
    return ax

def plot_time(
        df: Tuple[pd.DataFrame, pd.DataFrame], title, color = None, filename = None, 
        output = OUTPUT, legend: str = "Postcount", val_col: str = None,
        trend: bool = None, seasonal: bool = None, residual: bool = None,
        agg_base: str = None):
    """Create dataframe(s) time plot"""
    x_ticks_every = 12
    fig_x = 15.7
    fig_y = 4.27
    font_mod = False
    x_label = "Month"
    linewidth = 3
    if agg_base and agg_base == "Y":
        x_ticks_every = 1
        fig_x = 3
        fig_y = 1.5
        font_mod = True
        x_label = "Year"
        linewidth = 1
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_x, fig_y)
    ylabel = f'{legend}'
    if val_col is None:
        val_col = f'{legend.lower()}_est'
    ax = bar_plot_time(
        df=df, ax=ax, color=color, val_col=val_col, label=legend)
    # x axis ticker formatting
    tick_loc = mticker.MultipleLocator(x_ticks_every)
    ax.xaxis.set_major_locator(tick_loc)
    ax.tick_params(axis='x', rotation=45, length=0)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    ax.set(xlabel=x_label, ylabel=ylabel)
    ax.spines["left"].set_linewidth(0.25)
    ax.spines["bottom"].set_linewidth(0.25)
    ax.spines["top"].set_linewidth(0)
    ax.spines["right"].set_linewidth(0)
    ax.yaxis.set_tick_params(width=0.5)
    # remove legend
    ax.get_legend().remove()
    ax.set_title(title)
    ax.set_xlim(-0.5,len(df)-0.5)
    if font_mod:
        for item in (
            [ax.xaxis.label, ax.title, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(8)
    if any([trend, seasonal, residual]):
        # seasonal decompose
        decomposition = sm.tsa.seasonal_decompose(
            df[val_col], model='additive')
        # plot trend part only
        if trend:
            plt.plot(list(decomposition.trend), color='black',
                label="Trend", linewidth=linewidth, alpha=0.8)
        if seasonal:
            plt.plot(list(decomposition.seasonal), color='black', linestyle='dotted',
                label="Seasonal", linewidth=1, alpha=0.8)
        if residual:
            plt.plot(list(decomposition.resid), color='black', linestyle='dashed',
                label="Residual", linewidth=1, alpha=0.8)
        # trend.plot(ax=ax)
    # store figure to file
    if filename:
        fig.savefig(
            output / "figures" / f"{filename}.png", dpi=300, format='PNG',
            bbox_inches='tight', pad_inches=1, facecolor="white")
        # also save as svg
        fig.savefig(
            output / "svg" / f"{filename}.svg", format='svg',
            bbox_inches='tight', pad_inches=1, facecolor="white")


# -

def load_and_plot(
        dataset: Path = None, metric: str = None, src_ref: str = "flickr", colors: cm.colors.ListedColormap = None,
        agg_base: str = None, trend: bool = None, return_df: bool = None, df_post: pd.DataFrame = None,):
    """Load data and plot"""
    if metric is None:
        metric = 'post_hll'
    if metric == 'post_hll':
        metric_label = 'postcount'
    else:
        metric_label = 'usercount'
    if colors is None:
        # colors = sns.color_palette("vlag", as_cmap=True, n_colors=2)
        # colors = colors([1.0])
        colors = color_inaturalist
    df = process_dataset(dataset, metric=metric, agg_base=agg_base, df_post=df_post)
    plot_time(
        df, legend=metric_label.capitalize(), color=colors,
        title=f'{src_ref.capitalize().replace(" ", "_")} {metric_label} over time', 
        filename=f"temporal_{metric_label}_{src_ref}_absolute", trend=trend, agg_base=agg_base)
    if return_df:
        return df


colors = sns.color_palette("vlag", as_cmap=True, n_colors=2)

load_and_plot(INATURALIST_ALL_AVES_MILVUSRANGE, src_ref=f"iNaturalist Aves + Milvus milvus range", agg_base=AGG_BASE, trend=False, metric=metric_col)

# Plot iNaturalist Milvus range

load_and_plot(INATURALIST_ALL_MILVUSRANGE, src_ref=f"iNaturalist All + Milvus milvus range", agg_base=AGG_BASE, trend=False, metric=metric_col)

# Repeat for all iNaturalist data

load_and_plot(INATURALIST_ALL, src_ref=f"inaturalist_{AGG_BASE}", agg_base=AGG_BASE, trend=False)
load_and_plot(INATURALIST_ALL, src_ref=f"inaturalist_{AGG_BASE}", metric="user_hll", colors=colors([0.0]), agg_base=AGG_BASE, trend=False)

# ### Calculate the Chi-value

# ### Prepare Chi
#
# This is adapted from [notebook three](https://ad.vgiscience.org/sunset-sunrise-paper/03_chimaps.html) of the original publication.

# First, define the input parameter:
#
# - **dof**: degrees of freedom (dof) is calculated: (variables - 1) with the variables being: observed posts, expected posts
# - **chi_crit_val**: given a dof value of 1 and a confidence interval of 0.05, the critical value to accept or neglect the h0 is 3.84
# - **chi_column**: we'll do the chi calculation based on the usercount-column, but this notebook can be run for postcount or userdays, too.

DOF = 1
CHI_CRIT_VAL = 3.84
CHI_COLUMN: str = f"{METRIC}count_est"


def calc_norm(
    df_expected: pd.DataFrame,
    df_observed: pd.DataFrame,
    chi_column: str = CHI_COLUMN):
    """Fetch the number of data points for the observed and 
    expected dataset by the relevant column
    and calculate the normalisation value
    """
    v_expected = df_expected[chi_column].sum()
    v_observed = df_observed[chi_column].sum()
    norm_val = (v_expected / v_observed)
    return norm_val


# +
def chi_calc(x_observed: float, x_expected: float, x_normalized: float) -> pd.Series:
    """Apply chi calculation based on observed (normalized) and expected value"""
    value_observed_normalised = x_observed * x_normalized
    a = value_observed_normalised - x_expected
    b = math.sqrt(x_expected)
    # native division with division by zero protection
    chi_value = a / b if b else 0
    return chi_value
    
def apply_chi_calc(
        df: pd.DataFrame, norm_val: float,
        chi_column: str = CHI_COLUMN, chi_crit_val: float = CHI_CRIT_VAL):
    """Calculate chi-values based on two GeoDataFrames (expected and observed values)
    and return new grid with results"""
    # lambda: apply function chi_calc() to each item
    df['chi_value'] = df.apply(
        lambda x: chi_calc(
           x[chi_column],
           x[f'{chi_column}_expected'],
           norm_val),
        axis=1)
    # add significant column, default False
    df['significant'] = False
    # calculate significance for both negative and positive chi_values
    df.loc[np.abs(df['chi_value'])>chi_crit_val, 'significant'] = True


# -

# # Visualize Chi for subqueries
#
# ## Proportion of "Milvus milvus" observations vs. all iNaturalist observations
#
# First load the worldwide dataset

src = Path.cwd().parents[0] / "00_data" / "milvus" / "observations-350501.csv"

load_inat_kwargs = {
    "filepath_or_buffer":src,
    "index_col":'datetime', 
    "parse_dates":{'datetime':["observed_on"]},
    "date_format":'%Y-%m-%d',
    "keep_date_col":'False',
    "usecols":["id", "observed_on", "user_id"]
}
df = pd.read_csv(**load_inat_kwargs)

df.drop(columns=['observed_on'], inplace=True)

df.head()

val_col = "id"
agg_method = "count"
metric_label="observations"
if METRIC == "user":
    val_col = "user_id"
    agg_method = "nunique"
    metric_label="observers"

df_milvus = filter_fill_time(
    df, 2007, 2022, val_col=val_col, agg_base=AGG_BASE, agg_method=agg_method)

df_milvus.rename(columns={val_col: metric_label}, inplace=True)

src_ref="iNaturalist Milvus milvus"

plot_time(
        df_milvus, legend=metric_label.capitalize(),
        title=f'{src_ref.capitalize()} {metric_label} over time\n worldwide', val_col=metric_label,
        filename=f"temporal_iNaturalist_milvusmilvus_absolute", trend=False, agg_base=AGG_BASE)

df_milvus.sum()

# ## Limit to Milvus milvus range

CRS_WGS = "epsg:4326" # WGS1984
CRS_PROJ = "esri:54009" # Mollweide

load_inat_kwargs["usecols"] = ["id", "user_id", "observed_on", "longitude", "latitude"]
df = pd.read_csv(**load_inat_kwargs)

load_inat_kwargs['usecols'] = None

df.dropna(subset=['longitude', 'latitude'], inplace=True)

df

# Intersect with the spatial range that is available as kml from iNaturalist.

import geopandas as gp
milvus_range = gp.read_file(
    OUTPUT/ 'Milvusmilvus_range.gpkg', layer='Milvus milvus')

gdf = gp.GeoDataFrame(
    df, geometry=gp.points_from_xy(df.longitude, df.latitude), crs=CRS_WGS)

# Intersect, keep only observations within range.

gdf_overlay = gp.overlay(
    gdf, milvus_range,
    how='intersection')

ax = gdf_overlay.plot(markersize=.1)
ax.axis('off')
plt.show()

# Calculate chi

gdf_overlay

gdf_overlay['datetime'] = pd.to_datetime(gdf_overlay["observed_on"], format=load_inat_kwargs.get("date_format"))
gdf_overlay.set_index('datetime', inplace=True)
gdf_cleaned = tools.drop_cols_except(gdf_overlay, [val_col], inplace=False)

df_milvus_range = filter_fill_time(
    gdf_cleaned, 2007, 2022, val_col=val_col, agg_base=AGG_BASE, agg_method=agg_method)

df_milvus_range.rename(columns={val_col: metric_label}, inplace=True)

src_ref="iNaturalist Milvus milvus"

df_milvus_range.sum()

# Conclusion: Majority of observations is within the spatial range kml (16k total vs 15k in spatial range)

plot_time(
        df_milvus_range, legend=metric_label.capitalize(),
        title=f'{src_ref.capitalize()} {metric_label} over time\n in Europe', val_col=metric_label,
        filename=f"temporal_iNaturalist_milvusmilvus_absolute", trend=False, agg_base=AGG_BASE)

# ### Calculate chi: Proportion to all iNaturalist photographs
#
# First: Calculate chi based on all iNaturalist observations within Europe

df_expected = load_and_plot(
    INATURALIST_ALL_MILVUSRANGE, src_ref=f"inaturalist_{AGG_BASE}",
    agg_base=AGG_BASE, trend=False, return_df=True, metric=metric_col)

norm_val = calc_norm(
    df_expected.rename(columns={f'{METRIC}count_est':metric_label}, inplace=False),
    df_milvus_range, chi_column = metric_label)
print(norm_val)

df_expected.rename(columns={f'{METRIC}count_est':f'{metric_label}_expected'}, inplace=True)

merge_cols = [f'{METRIC}count_est']
df_expected_observed_inat = df_expected.merge(
    df_milvus_range[metric_label],
    left_index=True, right_index=True)

# %%time
apply_chi_calc(
    df=df_expected_observed_inat,
    norm_val=norm_val, chi_column=metric_label)

plot_time(
    df_expected_observed_inat, legend="Signed Chi", val_col="chi_value",
    title=f'Chi for Red kite {metric_label} proportional to \nall iNaturalist {metric_label} in Europe over time', 
    filename=f"temporal_chi_inaturalist_{metric_label}_milvus_{AGG_BASE}_{metric_label}", trend=False, 
    seasonal=False, residual=False, agg_base=AGG_BASE)

# We can observe a proportional increase of Red kite observations between 2012 and 2019.

# ### Calculate chi: Proportion of Milvus milvus vs. "Aves" tax class observations

df_expected = load_and_plot(
    INATURALIST_ALL_AVES_MILVUSRANGE, src_ref=f"inaturalist_{AGG_BASE}_{metric_label}",
    agg_base=AGG_BASE, trend=False, return_df=True, metric=metric_col)

norm_val = calc_norm(
    df_expected.rename(columns={f'{METRIC}count_est':metric_label}, inplace=False),
    df_milvus_range, chi_column = metric_label)
print(norm_val)

df_expected.rename(columns={f'{METRIC}count_est':f'{metric_label}_expected'}, inplace=True)

merge_cols = [f'{METRIC}count_est']
df_expected_observed_inat = df_expected.merge(
    df_milvus_range[metric_label],
    left_index=True, right_index=True)

# %%time
apply_chi_calc(
    df=df_expected_observed_inat,
    norm_val=norm_val, chi_column=metric_label)

df_expected_observed_inat

plot_time(
    df_expected_observed_inat, legend="Signed Chi", val_col="chi_value",
    title=f'Chi for Red kite {metric_label} proportional to \nall bird {metric_label} in Europe over time', 
    filename=f"temporal_chi_inaturalist_{metric_label}_milvus_{AGG_BASE}", trend=False, 
    seasonal=False, residual=False, agg_base=AGG_BASE)

# ## Proportion of "Aves" tax class observations vs. all iNaturalist observations

# - both queries limited to Milvus milvus range in Europe

# ### Expected: All iNaturalist
#
# - limited to Europe

df_expected = load_and_plot(
    INATURALIST_ALL_MILVUSRANGE, src_ref=f"inaturalist_{AGG_BASE}_{metric_label}",
    agg_base=AGG_BASE, trend=False, return_df=True, metric=metric_col)

# ### Observed: All Aves tax class observations
#
# - limited to Europe

df_observed = load_and_plot(
    INATURALIST_ALL_AVES_MILVUSRANGE, src_ref=f"inaturalist_{AGG_BASE}_{metric_label}",
    agg_base=AGG_BASE, trend=False, return_df=True, metric=metric_col)

norm_val = calc_norm(
    df_expected.rename(columns={f'{METRIC}count_est':metric_label}), df_observed.rename(columns={f'{METRIC}count_est':metric_label}), chi_column = metric_label)
print(norm_val)

df_observed.head()

df_expected.rename(columns={f'{METRIC}count_est':f'{metric_label}_expected'}, inplace=True)
df_observed.rename(columns={f'{METRIC}count_est':metric_label}, inplace=True)

merge_cols = [f'{METRIC}count_est']
df_expected_observed_inat = df_expected.merge(
    df_observed[metric_label],
    left_index=True, right_index=True)

df_expected_observed_inat.head()

# %%time
apply_chi_calc(
    df=df_expected_observed_inat,
    norm_val=norm_val, chi_column=metric_label)

plot_time(
    df_expected_observed_inat, legend="Signed Chi", val_col="chi_value",
    title=f'Chi for all bird {metric_label} proportional to \nall iNaturalist {metric_label} in Europe over time', 
    filename=f"temporal_chi_inaturalist_{metric_label}_milvus_{AGG_BASE}", trend=False, 
    seasonal=False, residual=False, agg_base=AGG_BASE)

# Interestingly, there was also proportional peak of Aves observations on iNaturalist between 2024 and 2019. This suggests that the Red Kite peak was indeed part of a larger peak of bird observations on iNaturalist in these years.

# # Create notebook HTML

# + editable=true slideshow={"slide_type": ""}
# !jupyter nbconvert --to html_toc \
#     --output-dir=../resources/html/ ./01_temporal_chi.ipynb \
#     --output 01_temporal_chi_{AGG_BASE.lower()}_{metric_label} \
#     --template=../nbconvert.tpl \
#     --ExtractOutputPreprocessor.enabled=False >&- 2>&-

# + [markdown] editable=true slideshow={"slide_type": ""}
#
