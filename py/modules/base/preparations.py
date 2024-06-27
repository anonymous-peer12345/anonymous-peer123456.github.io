"""GridAgg Notebook Preparations"""


import logging
import holoviews as hv
import pandas as pd
import shapely.speedups as speedups
import pkg_resources
from typing import List, Tuple, Dict

def init_imports():
    """Initialize speedups and bokeh"""
    speedups.enable()
    hv.notebook_extension('bokeh')

def package_report(root_packages: List[str]):
    """Report package versions for root_packages entries"""
    root_packages.sort(reverse=True)
    root_packages_list = []
    for m in pkg_resources.working_set:
        if m.project_name.lower() in root_packages:
            root_packages_list.append([m.project_name, m.version])
    
    display(pd.DataFrame(
                root_packages_list,
                columns=["package", "version"]
            ).set_index("package").transpose())

def load_chromedriver():
    """Loads chromedriver (for bokeh svg export), of found

    Install dependencies first, see [1]:
        conda activate worker_env
        conda install selenium webdriver-manager -c conda-forge

    [1]: https://gist.github.com/Sieboldianus/8b257c7b7f46b26c92280d984ed98fb1
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=2000x2000")
        options.add_argument('--disable-dev-shm-usage')        
        
        service = Service(ChromeDriverManager().install())
        webdriver = webdriver.Chrome(service=service, options=options)
        print('Chromedriver loaded. Svg output enabled.')
    except:
        logging.warning('Chromedriver not found. Disabling svg output.')
        webdriver = None
    return webdriver
