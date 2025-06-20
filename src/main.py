from datetime import datetime
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from pandas import read_excel

from src.classes.data import Data
from src.classes.utilitaire import Utils
from src.classes.backtester import Portfolio
from src.classes.metrics import Metrics
from src.classes.visualization import Visualisation

# Pip install pour blpapi (à mettre dans un notebook)
# pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi


"""
Import des données
"""

# Définition des paramètres pour le backtester
start_date: datetime = datetime(1995, 1, 1)
end_date: datetime = datetime(2025, 1,1)
list_ticker_bench: list = ["RIY Index"]
list_ticker_rf: list = ["SOFR"]
data_loader: Data = Data(start_date=start_date, end_date=end_date, list_ticker_rf=list_ticker_rf,
                         list_ticker_bench=list_ticker_bench, use_api=False)

# Création du dataloader pour récupérer les informations depuis Bloomberg et calculer la métrique de valorisation
data_loader.import_all_data("book value") # prend bcp de temps et erreur sur les filtres par colonne, à checker

# Récupération du dataframe contenant la métrique de valorisation
df_valo: pd.DataFrame = data_loader.df_valo
list_dates: list = data_loader.calendar

# Import des secteurs
df_sector: pd.DataFrame = read_excel("data/Secteur des actifs.xlsx")
df_sector.replace(0, np.nan, inplace=True)

# Import de l'univers d'investissement
df_universe: pd.DataFrame = read_excel("data/Composition univers.xlsx")
df_universe = df_universe.rename(columns = {"Unnamed: 0": "dates"})

# Conversion avec les mêmes dates
df_universe_daily:pd.DataFrame = Utils.monthly_to_daily_dataframe(df_universe, list_dates)

# Réalisation du mapping sectoriel
dict_sector: dict = Utils.data_to_sector(df_valo, df_sector)

"""
Deuxième étape : Réalisation du backtest
"""

# Définition de la date à partir de laquelle on met en place la stratégie
strat_date: datetime = start_date + relativedelta(years=5)

# Définition du levier et de la taille du fonds (en $ par hypothèse ==> single currency)
leverage: int = 3
capital : int = 100000000

# Instanciation du portefeuille
ptf:Portfolio = Portfolio(df_val=df_valo,
                          df_prices=data_loader.df_prices,
                          universe=df_universe_daily,
                          dict_sector=dict_sector,
                          strat="Deep Value",
                          weighting="equalweight",
                          start_date=strat_date,
                          list_date=list_dates,
                          leverage=leverage,
                          capital=capital,
                          quantile=0.1
                          )

# Exécution du backtest
ptf.run_backtest()


"""
Troisième étape : Etude des performances
"""
nav_for_perf: pd.Series = ptf.df_nav["NAV"]

# Affichage graphique
ptf_ret: pd.DataFrame = Utils.compute_asset_returns(nav_for_perf, "daily", "discret")
Visualisation.plot_cumulative_returns(ptf_ret, "Rendements cumulatifs de la stratégie Deep Value")

# Initialisation du module de métrique
metriques: Metrics = Metrics(nav_for_perf, "discret", frequency="daily")
perf_stats_ptf: pd.DataFrame = metriques.display_stats("Deep Value")

"""
Quatrième étape : Export pour Bloomberg
"""
Utils.prepare_port_file(ptf.df_quantities, "Deep Value")

