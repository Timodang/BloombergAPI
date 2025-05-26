from datetime import datetime
import pandas as pd
import numpy as np

# Pip install pour blpapi


# Fonction permettant de réaliser un mapping sectoriel
def data_to_sector(prices: pd.DataFrame, sectors:pd.DataFrame) -> dict:
    """
    Fonction permettant de réaliser un mapping sectoriel pour construire des stratégies segmentées
    :param prices: DataFrame contenant les prix de tous les titres de l'univers d'investissement entre 2007 et 2024
    :param sectors: DataFrame contenant le secteur de chaque titre de l'univers d'investissement
    :return: Un dictionnaire qui associe à chaque secteur un dataframe contenant tous les titres qui lui sont rattachés
    """

    # création d'un dictionnaire vide pour stocker les tickers par secteur
    dict_sector: dict = dict()

    # Tous les tickers sans secteur reçoivent le ticker "other"
    sectors.replace(np.nan, "other", inplace=True)

    # Récupération des secteurs présents dans l'univers d'investissement
    sector_array: np.array = pd.unique(sectors.iloc[0])

    # Boucle sur chaque secteur
    for i in range(len(sector_array)):

        # Récupération du secteur qui sera utilisé comme clé
        sector_key: str = sector_array[i]

        # Récupération sous forme de booléen de tous les tickers qui sont rattachés à ce secteur
        sector_bool_array: np.array(bool) = sectors.iloc[0].eq(sector_key)

        # Filtre sur les tickers rattachés à ce secteurs
        df_prices_sector: pd.DataFrame = prices.loc[:, sector_bool_array]
        df_prices_sector.fillna(0, inplace=True)

        # Ajout au dictionnaire
        dict_sector[sector_key] = df_prices_sector

    # Suppression des others (et financières, à faire)
    del dict_sector["other"]
    return dict_sector

"""
Import des données
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
Première étape : Importation des données
"""

"""
Deuxième étape : Réalisation du backtest
"""

"""
Troisième étape : Etude des performances
"""

"""
Quatrième étape : Export pour Bloomberg
"""