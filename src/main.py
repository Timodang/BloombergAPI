from datetime import datetime
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from pandas import read_excel

from src.classes.data import Data
from src.classes.utilitaire import Utils

# Pip install pour blpapi (à mettre dans un notebook)
# pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi


"""
Import des données
"""

# Import des secteurs
df_sector: pd.DataFrame = read_excel("data/Secteur des actifs.xlsx", sheet_name="Secteurs")
df_sector.replace(0, np.nan, inplace=True)

# Définition des paramètres pour le backtester
start_date: datetime = datetime(1995, 1, 1)
end_date: datetime = datetime(2025, 1,1)
list_ticker_bench: list = ["RIY Index"]
list_ticker_rf: list = ["SOFR"]
data_loader: Data = Data(start_date=start_date, end_date=end_date, list_ticker_rf=list_ticker_rf,
                         list_ticker_bench=list_ticker_bench, use_api=False)

# Import des données
data_loader.import_all_data("book value") # prend bcp de temps et erreur sur les filtres par colonne, à checker

# Récupération du dataframe contenant la métrique de valorisation
df_valo: pd.DataFrame = data_loader.df_valo
list_dates: list = data_loader.calendar

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


"""
Troisième étape : Etude des performances
"""

"""
Quatrième étape : Export pour Bloomberg
"""

#old

"""
# Pour générer les compos (pb dans l'export bloom, fonction gère pas Equity)
# Import des données contenant les compositions par date / composition univers
df_compo_date: pd.DataFrame = pd.read_excel('data/Compo par date.xlsx', sheet_name="BDS VALUE FINAL AVEC DOUBLONS")
df_compo: pd.DataFrame = pd.read_excel("data/Composition univers.xlsx")
df_compo_date.dropna(axis="columns", how="all", inplace=True)
df_compo = df_compo.rename(columns = {"Unnamed: 0": "dates"})
df_compo.set_index("dates", inplace=True)



# Boucle pour construire le dataframe des compositions
for i, (index, row) in enumerate(df_compo.iterrows()):
    date = index
    # récupération de la liste des tickers
    list_current_ticker: list = df_compo_date.iloc[0:df_compo_date.shape[0],i]
    list_current_ticker_correct = [str(ticker) + " Equity" for ticker in list_current_ticker]

    #
    df_compo.loc[date, df_compo.columns.isin(list_current_ticker_correct)] = 1

df_compo.to_excel("Composition Russel 1000.xlsx")
a=3
"""

"""
# Import des données contenant les compositions mensuelles du S&P 500
df_compo:pd.DataFrame = pd.read_excel('data/Compo MSCI.xlsx', sheet_name="Composition MSCI World")
df_compo.set_index("Dates", inplace=True)
# Les valeurs manquantes sont remplacées par des 0
df_compo.fillna(0, inplace=True)

# Import des données contenant les prix des stocks du MSCI et retraitements
df_msci_stocks: pd.DataFrame = pd.read_excel('data/MSCI.xlsx')
df_msci_stocks.set_index("Dates", inplace=True)
df_msci_stocks = df_msci_stocks.apply(lambda series: series.loc[:series.last_valid_index()].ffill())
# Les valeurs manquantes sont remplacées par des 0 pour réaliser les traitements ultérieures
df_msci_stocks.replace(np.nan, 0, inplace=True)
df_msci_stocks.fillna(0, inplace=True)

# Import des données relatives au secteur de chaque ticker
df_sector: pd.DataFrame = pd.read_excel('data/Compo MSCI.xlsx', sheet_name="Secteurs")
df_sector.set_index("Ticker", inplace = True)

# Réalisation du mapping sectoriel
dict_tickers_sectors: dict = data_to_sector(df_msci_stocks, df_sector)
a=3
"""