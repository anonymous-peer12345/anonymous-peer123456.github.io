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

# <div style="width: 100%;text-align:right;display: flex; align-items: top;">
#     <div style="float: left;width: 80%;text-align:left">
#         <h1>Biodiversity Hotspots Exploration <a class="tocSkip">&#182;</a></h1>
#         <p><em><a href="mailto:alexander.dunkel@tu-dresden.de">Alexander Dunkel</a>, Institute of Cartography, TU Dresden</em></p></div>
#     <div style="width:256px;text-align:right;margin-top:0px;margin-right:10px"><a href="https://gitlab.hrz.tu-chemnitz.de/ad/ephemeral_events"><img src="https://kartographie.geo.tu-dresden.de/ad/wip/ephemeral_events/version.svg"></a></div>
# </div>

# + tags=["hide_code"]
from IPython.display import Markdown as md
from datetime import date

today = date.today()
with open('/.version', 'r') as file: app_version = file.read().split("'")[1]
md(f"Last updated: {today.strftime('%b-%d-%Y')}, [Carto-Lab Docker](https://gitlab.vgiscience.de/lbsn/tools/jupyterlab) Version {app_version}")
# -

# Visualization of temporal patterns for geo-social media posts for [Biodiversity Hotspots](https://www.bfn.de/bpbv-hotspots) in Germany.
#
# # Preparations

import os, sys
from pathlib import Path
import psycopg2
import geopandas as gp
import pandas as pd
import seaborn as sns
import calendar
import textwrap
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from typing import List, Tuple, Dict, Optional, Any
from IPython.display import clear_output, display, HTML
from datetime import datetime

module_path = str(Path.cwd().parents[0] / "py")
if module_path not in sys.path:
    sys.path.append(module_path)
from modules.base import tools, hll, preparations
from modules.base.hll import union_hll, cardinality_hll

# %load_ext autoreload
# %autoreload 2

OUTPUT = Path.cwd().parents[0] / "out"       # output directory for figures (etc.)
WORK_DIR = Path.cwd().parents[0] / "tmp"     # Working directory

(OUTPUT / "figures").mkdir(exist_ok=True)
(OUTPUT / "svg").mkdir(exist_ok=True)
WORK_DIR.mkdir(exist_ok=True)

# Set global font

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# ## Preview Hotspot regions

df_hotspots = gp.read_file(
    Path.cwd().parents[0] / "00_data" / "hotspots" / "hotspots2021.shp")

epsg_code = df_hotspots.crs.to_epsg()
print(epsg_code)

# Get Shapefile for Germany

source_zip = "https://daten.gdz.bkg.bund.de/produkte/vg/vg2500/aktuell/"
filename = "vg2500_12-31.utm32s.shape.zip"
shapes_name = "vg2500_12-31.utm32s.shape/vg2500/VG2500_LAN.shp"

# +
SHAPE_DIR = (OUTPUT / "shapes")
SHAPE_DIR.mkdir(exist_ok=True)

if not (SHAPE_DIR / shapes_name).exists():
    tools.get_zip_extract(uri=source_zip, filename=filename, output_path=SHAPE_DIR)
else:
    print("Already exists")
# -

shapes = gp.read_file(SHAPE_DIR / shapes_name)
shapes = shapes.to_crs(f"EPSG:{epsg_code}")

# Create overlay

ax = shapes.plot(color='none', edgecolor='black', linewidth=0.2, figsize=(2, 4))
ax = df_hotspots.plot(ax=ax, color='green')
ax.set_axis_off()

# ## Load HLL aggregate data

HOTSPOT_ALL = OUTPUT /  "hotspot_all_months.csv"
METRIC = "postcount"
# METRIC = "usercount"
UPPER_YEAR = 2023
LOWER_YEAR = 2007

# Preview

data_files = {
    "HOTSPOT_ALL":HOTSPOT_ALL,
    }
tools.display_file_stats(data_files)

df = pd.read_csv(HOTSPOT_ALL)

pd.options.display.width = 0
df.head(10)

# Get distinct hotspot_ids, to check completeness:

print(sorted(df['hotspot_id'].unique()))

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

db_conn = tools.DbConn(DB_CONN)

# ## Calculate HLL Cardinality per month

# Define additional functions for reading and formatting CSV as `pd.DataFrame`

# +
TIMESTAMP_FORMAT = '%Y %m'

def read_csv_datetime(csv: Path, timestamp_format: str = TIMESTAMP_FORMAT) -> pd.DataFrame:
    """Read CSV with parsing datetime index (months)
    
        First CSV column: Year
        Second CSV column: Month
    """
    date_cols = ["year", "month"]
    df = pd.read_csv(
        csv, index_col='datetime', 
        parse_dates={'datetime':date_cols},
        date_format=timestamp_format,
        keep_date_col='False')
    df.drop(columns=date_cols, inplace=True)
    return df
    
def append_cardinality_df(df: pd.DataFrame, hll_col: str = "post_hll", cardinality_col: str = 'postcount_est'):
    """Calculate cardinality from HLL and append to extra column in df"""
    df[cardinality_col] = df.apply(
        lambda x: cardinality_hll(
           x[hll_col], CUR_HLL),
        axis=1)
    df.drop(columns=[hll_col], inplace=True)
    return df

def filter_fill_time(
        df: pd.DataFrame, min_year: int, 
        max_year: int, val_col: str = "postcount_est",
        min_month: int = 1, max_month: int = 1):
    """Filter time values between min - max year and fill missing values"""
    min_date = pd.Timestamp(f'{min_year}-{min_month}-01')
    max_date = pd.Timestamp(f'{max_year}-{max_month}-01')
    # clip by start and end date
    if not min_date in df.index:
        df.loc[min_date, val_col] = 0
    if not max_date in df.index:
        df.loc[max_date, val_col] = 0
    df.sort_index(inplace=True)
    # mask min and max time
    time_mask = ((df.index >= min_date) & (df.index <= max_date))
    # fill missing months with 0
    # this will also set the day to max of month
    series = df.loc[time_mask][val_col].resample('M').sum().fillna(0)
    return series.to_frame()


# -

# Apply functions to all data sets.
#
# - Read from CSV
# - split by `origin_id`
# - calculate cardinality
# - merge year and month to single column
# - filter 2007 - 2021 range, fill missing values

df_post = read_csv_datetime(HOTSPOT_ALL)

source_names = {
    1: "Instagram", 2: "Flickr", 3: "Twitter", 23: "iNaturalist"}


# Split by data source, create copies of original dataframe., and process.

def process_df(
        df: pd.DataFrame, metric: str = METRIC, upper_year: int = UPPER_YEAR, lower_year: int = LOWER_YEAR):
    """Process df: append_cardinality() and filter_fill_time()"""
    if metric == "postcount":
        hll_col= "post_hll"
        cardinality_col = 'postcount_est'
    else:
        hll_col= "user_hll"
        cardinality_col = 'usercount_est'
    df = append_cardinality_df(df, hll_col, cardinality_col)
    df = filter_fill_time(df, lower_year, upper_year, cardinality_col)
    # fill where no data avilable
    df.fillna(0, inplace=True)
    return df


vis_df = []
for idx, name in source_names.items():
    sel_df = df_post[df_post["origin_id"]==idx].copy()
    vis_df.append(process_df(sel_df))

vis_df[0].head(5)


def plot_lines(
        df_list: List[pd.DataFrame],  
        xlegend: str = "Year", title: Optional[str] = None, metric: str = METRIC):
    """Plot lines from a list of DataFrames"""
    if metric == "postcount":
        hll_col: str = "post_hll"
        cardinality_col: str = 'postcount_est'
        ylegend: str = "Post count"
    else:
        hll_col: str = "user_hll"
        cardinality_col: str = 'usercount_est'
        ylegend: str = "User count"
    fig, ax = plt.subplots()
    fig.set_size_inches(15.7, 4.27)
    ylabel = f'{ylegend} (estimate)'
    for ix, df in enumerate(df_list):
        source_name = source_names.get(list(source_names)[ix])
        ax = df[cardinality_col].plot(ax=ax, kind='line', label=source_name)
    tick_loc = mticker.MultipleLocator(12)
    # x axis ticker formatting
    ax.xaxis.set_major_locator(tick_loc)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    ax.tick_params(axis='x', rotation=45, color='grey')
    ax.set(xlabel=xlegend, ylabel=ylegend)
    ax.spines["left"].set_linewidth(0.25)
    ax.spines["bottom"].set_linewidth(0.25)
    ax.spines["top"].set_linewidth(0)
    ax.spines["right"].set_linewidth(0)
    ax.yaxis.set_tick_params(width=0.5)
    # add legend
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, frameon=False, loc='best')
    if title:
        ax.set_title(title)


plot_lines(vis_df)

# ## Group temporal patterns by month

# First, merge into single dataframe

stacked_df = pd.DataFrame()
for ix, df in enumerate(vis_df):
    source_name = source_names.get(list(source_names)[ix])
    stacked_df[source_name] = df
    stacked_df.index = df.index

stacked_df.head()

# Total sums:

stacked_df.sum()

# +
BAR_PARAM = {
    "width":1.0,
    "label":f"Total {METRIC} aggregated for months",
    "edgecolor":"white",
    "linewidth":0.5,
    "alpha":0.7,
}

def plot_bars(
        df: pd.DataFrame, ax: matplotlib.axes = None, title: str = None, 
        ytitle: float = None, padtitle: float = None, legend: bool = None,
        bar_param: Dict[str, Any] = BAR_PARAM, title_legend: str = None,
        xlegend: float = None, ylegend: float = None, lang: str = None):
    """Plot stacked bars from a DataFrame with multiple columns"""
    colors = sns.color_palette("vlag", as_cmap=True, n_colors=2)
    if lang is None:
        lang = "en"
    # create figure
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(3, 1.5))
    colors = sns.color_palette("bright", as_cmap=True, n_colors=4)
    # plot
    # calculate mean:
    # exclude months with no data in mean calculation:
    df.replace(0, np.NaN) \
        .groupby(df.index.month, dropna=True) \
        .mean() \
        .plot.bar(ax=ax, stacked=True, color = colors, **bar_param)
    # format
    ax.set_xlim(-0.5,11.5)
    if lang == "en":
        month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    elif lang == "de":
        month_names = ['Jan','Feb','Mär','Apr','Mai','Jun',
                       'Jul','Aug','Sep','Okt','Nov','Dez']
    ax.set_xticklabels(month_names)
    ax.tick_params(axis='x', rotation=45, length=0) # length: of ticks
    ax.spines["left"].set_linewidth(0.25)
    ax.spines["bottom"].set_linewidth(0.25)
    ax.spines["top"].set_linewidth(0)
    ax.spines["right"].set_linewidth(0)
    ax.yaxis.set_tick_params(width=0.5)
    ax.set(xlabel="", ylabel="")
    if legend: 
        if xlegend is None:
            xlegend = 1.0
        if ylegend is None:
            ylegend = 1
        ax.legend(
            bbox_to_anchor=(xlegend, ylegend), loc='upper left', 
            fontsize=8, frameon=False, title=title_legend, title_fontsize=8)
    else:
        ax.legend().set_visible(False)
    if not title:
        title = f"{METRIC.capitalize()} per month (mean)"
    if ytitle is None:
        ytitle =-0.2
    if padtitle is None:
        padtitle=-14
    ax.set_title(title, y=ytitle, pad=padtitle)
    for item in (
        [ax.xaxis.label, ax.title, ax.yaxis.label] +
         ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)


# -

plot_bars(df=stacked_df, legend=True)

# Visualize for all Hotspots separately.

hotspots = df_post['hotspot_id'].unique()
print(len(hotspots))


# Get summary of total posts per Hotspots, for sorting:

def replace_datetime_with_month(df: pd.DataFrame, group_by: str = "hotspot_id"):
    """Extract month from datetime index, set as new composite index
    together with topic_group"""
    df.set_index([df.index.month, group_by], inplace=True)
    df.index.rename(("month", group_by), inplace=True)


# Union user count column, to get total user aggregate data

df_post = read_csv_datetime(HOTSPOT_ALL)
replace_datetime_with_month(df_post)
cardinality_series = tools.union_hll_series(
    hll_series=df_post["user_hll"],
    db_conn=db_conn, multiindex=True)
df = cardinality_series.to_frame()
df.rename(columns={"hll_cardinality":"user_count"}, inplace=True)
hotspots_sorted = df.unstack(level=1).fillna(0).droplevel(0, axis=1).sum().sort_values(ascending=False)

# Get hotspot labels

# Preview ranking order
#
# First: Get Hotspot names for x-labels
# - we also shorten a number of very long names

hotspot_names = {}
for idx in hotspots_sorted.index:
    hotspot_name = df_hotspots[df_hotspots["NUMMER"]==idx].NAME.iloc[0]
    hotspot_names[idx] = hotspot_name
    print(f'{idx} - {hotspot_name} - Postcount: {hotspots_sorted[idx]}')
hotspot_names[27] = "Schleswig-Holsteinische Ostseeküste"
hotspot_names[24] = "Untere Wümmeniederung mit Teufelsmoor"
hotspot_names[13] = "Saar-Ruwer-Hunsrück, Hoch- und Idarwald"
hotspot_names[23] = "Hunte-Leda-Moorniederung, Delmenhorster G."
hotspot_names[18] = "Südharzer Zechsteingürtel, Kyffhäuser und H."

ax = hotspots_sorted.plot.bar(figsize=(6, 2), fontsize=8, color="r", **BAR_PARAM)
ax.spines["left"].set_linewidth(0.25)
ax.spines["bottom"].set_linewidth(0.25)
ax.spines["top"].set_linewidth(0)
ax.spines["right"].set_linewidth(0)
ax.set(xlabel="Hotspot ID", ylabel="Total User Count")
ax.xaxis.label.set_fontsize(8)
ax.yaxis.label.set_fontsize(8)

df_post = read_csv_datetime(HOTSPOT_ALL)

SUBPLOTS = len(hotspots_sorted) # 30 Subplots

hotspots_sorted

hotspot_names

lang = "en"

# create figure object with multiple subplots
fig, axes = plt.subplots(nrows=int(round(SUBPLOTS/4)), ncols=4, figsize=(12, 18))
fig.subplots_adjust(hspace=.9) # adjust vertical space, to allow title below plot
# remove subplots beyond 30
fig.delaxes(axes[7][2])
fig.delaxes(axes[7][3])
# iterate hotspots
for ix, ax in enumerate(axes.reshape(-1)):
    if ix >= SUBPLOTS:
        break
    hotspot_id = list(hotspot_names.keys())[ix]
    hotspot_name = textwrap.fill(hotspot_names.get(hotspot_id), 20)
    # filter np_str and calculate cardinality
    df_post_filter = df_post[df_post["hotspot_id"]==hotspot_id].copy()
    stacked_df = pd.DataFrame()
    # process and stack into single df
    for idx, name in source_names.items():
        sel_df = df_post_filter[df_post_filter["origin_id"]==idx].copy()
        processed_df = process_df(sel_df)
        stacked_df[name] = processed_df
        stacked_df.index = processed_df.index        
    # plot bars individually
    plot_kwargs = {
        "ytitle":-0.65, 
        "padtitle":0, 
        "legend":False,
        "lang":lang
    }
    avrg_txt = "Average"
    if lang == "de":
        avrg_txt = "Durchschnitt"
    if ix == SUBPLOTS-1:
        # add legend on last subplot
        plot_kwargs["legend"] = True
        plot_kwargs["xlegend"] = 1.5
        plot_kwargs["title_legend"] = f'{METRIC.capitalize()} \n{avrg_txt} {LOWER_YEAR}-{UPPER_YEAR}'
    plot_bars(
        df=stacked_df, title=f'{hotspot_id}: {hotspot_name}', ax=ax, **plot_kwargs)

# save as svg and png

tools.save_fig(fig, output=OUTPUT, name=f"hotspots_months_{METRIC}")

# Create map with labels for paper Figure. Select Hotspots `2`, `14`, and `25`.

df_sel = df_hotspots[df_hotspots["NUMMER"].isin([2, 14, 25])]

df_sel

# manually create label offset

label_off = {
    2:(0, 200000),
    14:(200000, 0),
    25:(-200000, 0)}
label_rad = {
    2:0.1,
    14:0.5,
    25:-0.3}

fig, ax = plt.subplots(1, 1, figsize=(2, 4))
ax = shapes.plot(ax=ax, color='none', edgecolor='black', linewidth=0.2, figsize=(2, 4))
ax = df_hotspots.plot(ax=ax, color='green')
ax = df_sel.plot(ax=ax, color='red')
tools.annotate_locations(
    gdf=df_sel, ax=ax, label_off=label_off, label_rad=label_rad, 
    text_col="NUMMER", arrowstyle='-', arrow_col='black', fontsize=14)
ax.set_axis_off()

tools.save_fig(fig, output=OUTPUT, name=f"overview_hotspots")

# # Create Release File
#
# First convert all svg to pdf, for archive purposes and paper submission.

WEB_DRIVER = preparations.load_chromedriver()

# %%time
tools.convert_svg_pdf(in_dir=OUTPUT / "svg", out_dir=OUTPUT / "pdf")

# **Create release file with all results**

# Create a release file that contains ipynb notebooks, HTML, figures, svg and python converted files.
#
# Make sure that 7z is available (`apt-get install p7zip-full`)

# !cd .. && git config --system --add safe.directory '*' \
#     && RELEASE_VERSION=$(git describe --tags --abbrev=0) \
#     && 7z a -tzip -mx=9 out/release_$RELEASE_VERSION.zip \
#     md/* py/* out/*.csv resources/* out/pdf/* out/svg/* \
#     out/figures/* notebooks/*.ipynb \
#     README.md jupytext.toml nbconvert.tpl \
#     -x!py/__pycache__ -x!py/modules/__pycache__ -x!py/modules/.ipynb_checkpoints \
#     -y > /dev/null

# # Create notebook HTML

# !jupyter nbconvert --to html_toc \
#     --output-dir=../resources/html/ ./05_hotspots.ipynb \
#     --template=../nbconvert.tpl \
#     --ExtractOutputPreprocessor.enabled=False >&- 2>&-


