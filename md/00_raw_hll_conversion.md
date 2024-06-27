---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.7
  kernelspec:
    display_name: worker_env
    language: python
    name: worker_env
---

<!-- #region tags=["hide_code"] editable=true slideshow={"slide_type": ""} -->
<div style="width: 100%;text-align:right;display: flex; align-items: top;">
    <div style="float: left;width: 80%;text-align:left">
        <h1>Preparation: Raw->Hll Conversion <a class="tocSkip">&#182;</a></h1>
        <p><em><a href="mailto:a.dunkel@ioer.de">Alexander Dunkel</a>, Leibniz Institute of Ecological Urban and Regional Development, <br>
        Transformative Capacities & Research Data Centre (IÖR-FDZ)</em></p></div>
    <div style="width:256px;text-align:right;margin-top:0px;margin-right:10px"><a href="https://gitlab.hrz.tu-chemnitz.de/ad/ephemeral_events"><img src="https://kartographie.geo.tu-dresden.de/ad/wip/ephemeral_events/version.svg"></a></div>
</div>
<!-- #endregion -->

```python tags=["hide_code"] jupyter={"source_hidden": true}
from IPython.display import Markdown as md
from datetime import date

today = date.today()
with open('/.version', 'r') as file: app_version = file.read().split("'")[1]
md(f"Last updated: {today.strftime('%b-%d-%Y')}, [Carto-Lab Docker](https://gitlab.vgiscience.de/lbsn/tools/jupyterlab) Version {app_version}")
```

Abstract preparation for visualization of temporal patterns for ephemeral events.

## Prepare environment

<!-- #region -->
To run this notebook, as a starting point, you have two options:<br><br>

<div style="color: black;">
<details><summary style="cursor: pointer;"><strong>1.</strong> Create an environment with the packages and versions shown in the following cell.</summary>
   
As a starting point, you may use the latest conda <a href="https://gitlab.vgiscience.de/lbsn/tools/jupyterlab/-/blob/master-latest/environment_default.yml">environment_default.yml</a> from our CartoLab docker container.
<br><br>
</details>
</div>

<div style="color: black;">
<details><summary style="cursor: pointer;"><strong>2.</strong> If docker is available to to, we suggest to use the <a href="https://gitlab.vgiscience.de/lbsn/tools/jupyterlab">Carto-Lab Docker Container</a></summary>

Clone the repository and edit your <code>.env</code> value to point to the repsitory, where this notebook can be found, e.g.:
        
```bash
git clone https://gitlab.vgiscience.de/lbsn/tools/jupyterlab.git
cd jupyterlab
cp .env.example .env
nano .env
## Enter:
# JUPYTER_NOTEBOOKS=~/notebooks/ephemeral_events
# TAG=v0.12.3
docker network create lbsn-network
docker-compose pull && docker-compose up -d
```

</details>
</div>
<!-- #endregion -->

```python tags=["hide_code"]
import sys
from pathlib import Path

module_path = str(Path.cwd().parents[0] / "py")
if module_path not in sys.path:
    sys.path.append(module_path)
from modules.base import tools

root_packages = [
    'python', 'colorcet', 'holoviews', 'ipywidgets', 'geoviews', 'hvplot',
    'geopandas', 'mapclassify', 'memory_profiler', 'python-dotenv', 'shapely',
    'matplotlib', 'sklearn', 'numpy', 'pandas', 'bokeh', 'fiona',
    'matplotlib-venn', 'xarray']
tools.package_report(root_packages)
```

Load dependencies:

```python
import os, sys
from pathlib import Path
import psycopg2
import geopandas as gp
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from IPython.display import clear_output, display, HTML
```

To reduce the code shown in this notebook, some helper methods are made available in a separate file.

Load helper module from `../py/modules/base/tools.py`.

```python
module_path = str(Path.cwd().parents[0] / "py")
if module_path not in sys.path:
    sys.path.append(module_path)
from modules.base import tools
```

Activate autoreload of changed python files:

```python
%load_ext autoreload
%autoreload 2
```

### Parameters

Define initial parameters that affect processing

```python
WORK_DIR = Path.cwd().parents[0] / "tmp"     # Working directory                     
OUTPUT = Path.cwd().parents[0] / "out"       # Define path to output directory (figures etc.)
```

```python
for folder in [WORK_DIR, OUTPUT]:
    folder.mkdir(exist_ok=True)
```

Load dotfiles environment variables

```python
from dotenv import load_dotenv
load_dotenv(
    Path.cwd().parents[0] / '.env', override=True)
```

```python
DB_NAME_RAWDB = os.getenv("DB_NAME_RAWDB")    # lbsn-rawdb name
DB_HOST_RAWDB = os.getenv("DB_HOST_RAWDB")    # lbsn-rawdb name
```

## Raw to HLL conversion

```python
db_user = "postgres"
db_pass = os.getenv('POSTGRES_PASSWORD')
db_host = "127.0.0.1"
db_port = "25432"
db_name = "hlldb"
```

```python
db_connection_hll = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_pass
)
db_conn_hll = tools.DbConn(db_connection_hll)
cur_hll = db_connection_hll.cursor()
cur_hll.execute("SELECT 1;")
print(cur_hll.statusmessage)
```

Simplify query access:

```python
db_conn = tools.DbConn(db_connection_hll)
db_conn.query("SELECT 1;")
```

<div class="alert alert-success">
If any SQL results in an error, the cursor cannot be used again. In this case, run <code>db_connection.rollback()</code> once, to reset the cursor.
</div>

```python
db_connection_hll.rollback()
```

### Create Query Schema

Create a new schema called mviews and update Postgres search_path, to include new schema:

```python
sql_query = """
CREATE SCHEMA IF NOT EXISTS mviews;
ALTER DATABASE hlldb
SET search_path = "$user",
                  social,
                  spatial,
                  temporal,
                  topical,
                  interlinkage,
                  extensions,
                  mviews;"""
```

Since the above query will not return any result, we'll directly use the psycopg2 cursor object:

```python
cur = db_connection_hll.cursor()
cur.execute(sql_query)
print(cur.statusmessage)
```

By using Foreign Table, this step will establish the connection between hlldb to rawdb.

On hlldb, install [postgres_fdw extension](https://www.postgresql.org/docs/12/postgres-fdw.html):

```python
sql_query = """
CREATE EXTENSION IF NOT EXISTS postgres_fdw SCHEMA extensions;
"""
cur_hll.execute(sql_query)
print(cur_hll.statusmessage)
```

## iNaturalist Aves dataset


Check if foreign table has been imported already:

```python
raw_table_name = 'inaturalist_birds_reduced'
```

```python
result = tools.check_table_exists(db_conn_hll, raw_table_name)
print(result)
```

Conditional load password - this only need to be done once, if the server hasn't been added before.

```python
if not result:
    import getpass
    USER_KEY = getpass.getpass()
```

Create Foreign Server connection to rawdb, on hlldb:

```python
if not result:
    sql_query = f"""
    CREATE SERVER IF NOT EXISTS lbsnraw 
    FOREIGN DATA WRAPPER postgres_fdw
    OPTIONS (
        host '{DB_NAME_RAWDB}',
        dbname '{DB_HOST_RAWDB}',
        port '5432',
        keepalives '1',
        keepalives_idle '30',
        keepalives_interval '10',
        keepalives_count '5',
        fetch_size '500000');
    CREATE USER MAPPING IF NOT EXISTS for postgres
        SERVER lbsnraw 
        OPTIONS (user 'lbsn_reader', password '{USER_KEY}');
    """
    cur_hll.execute(sql_query)
    print(cur_hll.statusmessage)
```

Import foreign table definition on the hlldb.

```python
sql_query = f"""
IMPORT FOREIGN SCHEMA mviews
    LIMIT TO (
        {raw_table_name})
    FROM SERVER lbsnraw 
    INTO mviews;
"""
# only import table 
# if it hasn't been imported already
if not result:
    cur_hll.execute(sql_query)
    print(cur_hll.statusmessage)
```

test

```python
db_conn.query(f"SELECT * FROM mviews.{raw_table_name} LIMIT 10;")
```

Commit changes to `hlldb`

```python
db_connection_hll.commit()
```

### Prepare conversion of raw data to hll


**HyperLogLog parameters**

The HyperLogLog extension for Postgres from [Citus](https://github.com/citusdata/postgresql-hll) that we're using here, contains several tweaks, to optimize performance, that can affect sensitivity of data.

From a privacy perspective, for example, it is recommended to disable [explicit mode](https://github.com/citusdata/postgresql-hll/blob/master/REFERENCE.md#metadata-functions).

**Explicit mode?** When explicit mode is active, full IDs will be stored for small sets. In our case, any coordinates frequented by few users (outliers) would store full user and post IDs.

To disable explicit mode:

```python
db_conn_hll.query("SELECT hll_set_defaults(11, 5, 0, 1);")
```

```python
db_conn_hll.query("SELECT wkb_geometry from spatial.milvus_milvus_range_sub;")
```

### Aggregation step

- Convert data to Hll
- filter by space (Milvus milvus range)
- group by month, year
- order by year, month

```python
def materialized_view_hll(table_name_src: str, table_name_dest, schema: str = None, additional_cols: [str] = None) -> str:
    """Returns raw SQL for creating a materialized view with HLL aggregate"""
    if not schema:
        schema = 'mviews'
    if additional_cols is None:
        additional_cols = []
    return f"""
        DROP MATERIALIZED VIEW IF EXISTS {schema}.{table_name_dest};
        
        CREATE MATERIALIZED VIEW {schema}.{table_name_dest} AS
            WITH polies AS (SELECT wkb_geometry from spatial.milvus_milvus_range_sub)
            SELECT 
                EXTRACT(MONTH FROM post_create_date) AS "month",
                EXTRACT(YEAR FROM post_create_date) AS "year",
                hll_add_agg((hll_hash_text(post_guid))) AS "post_hll",
                hll_add_agg((hll_hash_text(user_guid))) AS "user_hll"
                {''.join([f",{x}" for x in additional_cols])}
            FROM {schema}.{table_name_src}, polies
            WHERE ST_Intersects(post_latlng, wkb_geometry)
            GROUP BY year, month{''.join([f",{x}" for x in additional_cols if len(additional_cols) > 0])}
            ORDER BY year ASC, month ASC;
        """
```

```python
db_connection_hll.rollback()
```

```python
%%time
destination_table = "inaturalist_birds_month"
origin_table = raw_table_name
sql_query = materialized_view_hll(
    table_name_src=origin_table, table_name_dest=destination_table)
cur_hll.execute(sql_query)
print(cur_hll.statusmessage)
```

Test:

```python
db_conn.query(f"SELECT * FROM mviews.{destination_table} LIMIT 10;")
```

```python
db_connection_hll.commit()
```

### Export data as CSV


Save hll data to CSV. The following records are available from table spatial.latlng:

- year     - distinct year
- month    - month
- post_hll - approximate post guids stored as hll set
- user_hll - approximate user guids stored as hll set

```python
sql_query = f"""
    SELECT  year,
            month,
            post_hll,
            user_hll
    FROM mviews.{destination_table};
    """
df = db_conn.query(sql_query)
# use type int instead of float
time_cols = ["year", "month"]
# drop where time cols are invalid 
df.dropna(subset=time_cols, inplace=True)
# turn float to int
for col in time_cols:
    df[col] = df[col].astype(int)
# we can also remove any rows where the year is < 2007
df.drop(df[df['year'] < 2007].index, inplace = True)
```

```python
df.head()
```

```python
usecols = ["year", "month", "post_hll", "user_hll"]
df.to_csv(
    OUTPUT / f"{destination_table}.csv",
    mode='w', columns=usecols,
    index=False, header=True)
```

## Flickr Cherry dataset


Check if foreign table has been imported already:

```python
raw_table_name = 'flickr_cherries_reduced'
```

```python
result = tools.check_table_exists(db_conn_hll, raw_table_name)
print(result)
```

Conditional load password - this only need to be done once, if the server hasn't been added before.


Import foreign table definition on the hlldb.

```python
sql_query = f"""
IMPORT FOREIGN SCHEMA mviews
    LIMIT TO (
        {raw_table_name})
    FROM SERVER lbsnraw 
    INTO mviews;
"""
# only import table 
# if it hasn't been imported already
if not result:
    cur_hll.execute(sql_query)
    print(cur_hll.statusmessage)
```

test

```python
db_conn.query(f"SELECT * FROM mviews.{raw_table_name} LIMIT 10;")
```

```python
db_conn.query(f"SELECT count(*) FROM mviews.{raw_table_name};")
```

Commit changes to `hlldb`

```python
db_connection_hll.commit()
```

### Prepare conversion of raw data to hll


**HyperLogLog parameters**

The HyperLogLog extension for Postgres from [Citus](https://github.com/citusdata/postgresql-hll) that we're using here, contains several tweaks, to optimize performance, that can affect sensitivity of data.

From a privacy perspective, for example, it is recommended to disable [explicit mode](https://github.com/citusdata/postgresql-hll/blob/master/REFERENCE.md#metadata-functions).

**Explicit mode?** When explicit mode is active, full IDs will be stored for small sets. In our case, any coordinates frequented by few users (outliers) would store full user and post IDs.

To disable explicit mode:

```python
db_conn_hll.query("SELECT hll_set_defaults(11, 5, 0, 1);")
```

### Aggregation step

- Convert data to Hll
- group by month, year
- order by year, month

```python
def materialized_view_hll(table_name_src: str, table_name_dest, schema: str = None, additional_cols: [str] = None) -> str:
    """Returns raw SQL for creating a materialized view with HLL aggregate"""
    if not schema:
        schema = 'mviews'
    if additional_cols is None:
        additional_cols = []
    return f"""
        DROP MATERIALIZED VIEW IF EXISTS {schema}.{table_name_dest};
        
        CREATE MATERIALIZED VIEW {schema}.{table_name_dest} AS
            SELECT 
                EXTRACT(MONTH FROM post_create_date) AS "month",
                EXTRACT(YEAR FROM post_create_date) AS "year",
                hll_add_agg((hll_hash_text(post_guid))) AS "post_hll",
                hll_add_agg((hll_hash_text(user_guid))) AS "user_hll"
                {''.join([f",{x}" for x in additional_cols])}
            FROM {schema}.{table_name_src}
            GROUP BY year, month{''.join([f",{x}" for x in additional_cols if len(additional_cols) > 0])}
            ORDER BY year ASC, month ASC;
        """
```

```python
db_connection_hll.rollback()
```

```python
%%time
destination_table = "flickr_cherries_hll"
origin_table = raw_table_name
sql_query = materialized_view_hll(
    table_name_src=origin_table, table_name_dest=destination_table)
cur_hll.execute(sql_query)
print(cur_hll.statusmessage)
```

Test:

```python
db_conn.query(f"SELECT * FROM mviews.{destination_table} LIMIT 10;")
```

```python
db_connection_hll.commit()
```

### Export data as CSV


Save hll data to CSV. The following records are available from table spatial.latlng:

- year     - distinct year
- month    - month
- post_hll - approximate post guids stored as hll set
- user_hll - approximate user guids stored as hll set

```python
sql_query = f"""
    SELECT  year,
            month,
            post_hll,
            user_hll
    FROM mviews.{destination_table};
    """
df = db_conn.query(sql_query)
# use type int instead of float
time_cols = ["year", "month"]
# drop where time cols are invalid 
df.dropna(subset=time_cols, inplace=True)
# turn float to int
for col in time_cols:
    df[col] = df[col].astype(int)
# we can also remove any rows where the year is < 2007
df.drop(df[df['year'] < 2007].index, inplace = True)
```

```python
df.head()
```

```python
usecols = ["year", "month", "post_hll", "user_hll"]
df.to_csv(
    OUTPUT / f"{destination_table}.csv",
    mode='w', columns=usecols,
    index=False, header=True)
```

## Create notebook HTML

```python editable=true slideshow={"slide_type": ""}
!jupyter nbconvert --to html_toc \
    --output-dir=../resources/html/ ./00_raw_hll_conversion.ipynb \
    --template=../nbconvert.tpl \
    --ExtractOutputPreprocessor.enabled=False >&- 2>&-
```

<!-- #region editable=true slideshow={"slide_type": ""} -->

<!-- #endregion -->
