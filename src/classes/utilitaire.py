from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from pyarrow import list_


class Utils:
    """
    Classe contenant des méthodes utilisées dans plusieurs autres classes
    Méthode :
    - compute_asset_returns : méthode utilisée pour calculer des rendements discrets ou continus
    - get_calendar : méthode permettant de construire un calendrier de jours ouvrés pour une date de début et une date de fin donnée
    """

    @staticmethod
    def compute_asset_returns(asset_prices: pd.DataFrame, periodicity_returns: str, method:str) -> pd.DataFrame:
        """
        Méthode permettant de calculer un ensemble de revenus à partir d'un jeu de données contenant des prix
        :param asset_prices: le dataframe contenant les prix des actifs pour lesquels on souhaite calculer le rendement
        :param periodicity_returns: la périodicité des données (monthly / quarterly / yearly
        :param method: la méthode pour le calcul des rendements (discret vs continu)
        :return: un dataframe contenant les rendements pour tous les actifs
        """

        # Test pour vérifier la périodicité des données
        if periodicity_returns not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise Exception(f"La périodicité {periodicity_returns} n'est pas implémentée. Veuillez modifier ce paramètre" )

        # Test pour vérifier la méthode de calcul des rendements
        if method not in ["discret","continu"]:
            raise Exception(f"La méthode {method} n'est pas implémentée pour le calcul des rendements. Veuillez modifier ce paramètre")

        # Les valeurs manquantes sont conservés pour prendre en compte des entrées / sorties de titres de l'indice
        returns: pd.DataFrame = asset_prices.pct_change() if (method == 'discret') else np.log(asset_prices).diff()

        # Retraitement des rendements  liés à l'entrée / sortie des valeurs de l'univers d'investissement
        returns.replace(-1, np.nan, inplace=True)
        returns.replace(np.inf, np.nan, inplace=True)

        # La première date n'est pas conservée
        return returns.iloc[1:returns.shape[0], ]

    @staticmethod
    def get_calendar(start_date: datetime, end_date:datetime)->list:
        """
        Méthode permettant de récupérer tous les jours ouvrés entre la date
        de début et la date de fin
        :param start_date:
        :param end_date:
        :return:
        """

        # Vérification sur les dates
        if start_date > end_date:
            raise Exception("La date de début ne peut pas être postérieure à la date de fin.")

        if end_date > datetime.now().replace(hour = 0, minute=0, second=0):
            raise Exception("La date de fin ne peut pas être postérieure à la date d'aujourd'hui")

        # Liste pour stocker le calendrier
        list_dates: list = [start_date]

        # Définition de l'incrément de jour
        deltat: timedelta = timedelta(1)

        # Calcul de la nouvelle date
        next_date: datetime = start_date + deltat

        # Boucle pour récupérer toutes les dates entre la date de début et de fin (jours ouvrés uniquement)
        while next_date < end_date:

            # test = next_date.weekday()

            # Gestion des jours ouvrés
            if next_date.weekday() == 5 or next_date.weekday() == 6:
                next_date += deltat
                continue

            # Ajout à la liste
            list_dates.append(next_date)

            # incrément de la date pour l'itération suivante
            next_date += deltat

        return list_dates

    @staticmethod
    def get_last_date_month(list_dates:list)->list:
        """
        Méthode permettant de récupérer le dernier jour ouvré de chaque mois
        (liste qui sera utilisée pour extraire les compositions d'indices)
        :param list_dates:
        :return:
        """

        # Création d'une liste pour stocker les résultat
        list_eom_dates: list = []

        # Récupération de la première date
        start_date: datetime = list_dates[0]

        # mois de la première période
        current_month: int = start_date.month

        # Boucle sur les dates de la liste
        for i in range (1, len(list_dates)):

            # Récupération de chaque date ultérieure
            new_date: datetime = list_dates[i]

            # comparaison des mois
            if new_date.month != current_month:
                # Récupération de la valeur précédente
                list_eom_dates.append(list_dates[i-1])
                current_month = new_date.month


        # Récupération  de la dernière date
        list_eom_dates.append(list_dates[-1])
        return list_eom_dates

    @staticmethod
    def get_lagged_calendar(list_date: list, nb_month:int)->list:
        """
        Méthode permettant de construire un calendrier laggé en ajoutant un nombre de mois
        défini par l'utilisateur
        :param list_date:
        :param nb_month:
        :return:
        """

        # Création d'une liste vide pour stocker les dates laggées
        lagged_list_date: list = []

        # Boucle pour appliquer le lag
        for date in list_date:

            # Récupération de la date en appliquant le lag
            lagged_date: datetime = date + relativedelta(months=nb_month)

            # Si ce n'est pas un jour ouvré, on prend le dernier jour ouvré associé
            if lagged_date.weekday() == 5:
                lagged_date -= timedelta(days=1)
            elif lagged_date.weekday() == 6:
                lagged_date -= timedelta(days=2)

            # Ajout à la liste
            lagged_list_date.append(lagged_date)

        return lagged_list_date



    @staticmethod
    def get_last_date_week(list_dates: list)->list:
        """
        Méthode permettant de récupérer un jour ouvré par semaine
        """

        # Création d'une liste pour stocker les résultat
        list_eow_dates: list = []

        # Boucle sur les dates de la liste
        for i in range(0, len(list_dates)):
            new_date: datetime = list_dates[i]
            # On récupère la date si c'est un vendredi
            if new_date.weekday() == 4:
                list_eow_dates.append(new_date)

        return list_eow_dates

    @staticmethod
    def data_to_sector(df_valo: pd.DataFrame, df_sectors: pd.DataFrame) -> dict:
        """
        Fonction permettant de réaliser un mapping sectoriel pour construire des stratégies segmentées
        :param df_valo: DataFrame contenant la métrique de valorisation de tous les titres ayant fait parti de l'univers
        :param df_sectors: DataFrame contenant le secteur de chaque titre de l'univers d'investissement
        :return: Un dictionnaire qui associe à chaque secteur un dataframe contenant tous les titres qui lui sont rattachés
        """

        # création d'un dictionnaire vide pour stocker les tickers par secteur
        dict_sector: dict = dict()

        # Tous les tickers sans secteur reçoivent le ticker "other"
        df_sectors.replace(np.nan, "other", inplace=True)

        # Seuls les titres pour lesquelles la métrique de valorisation est disponible sont conservés
        df_sectors = df_sectors.loc[:, df_sectors.columns.isin(df_valo.columns)]

        # Récupération des secteurs présents dans l'univers d'investissement
        sector_array: np.array = pd.unique(df_sectors.iloc[0])

        # Boucle sur chaque secteur
        for i in range(len(sector_array)):
            # Récupération du secteur qui sera utilisé comme clé
            sector_key: str = sector_array[i]

            # Récupération sous forme de booléen de tous les tickers qui sont rattachés à ce secteur
            sector_bool_array: np.array(bool) = df_sectors.iloc[0].eq(sector_key)

            # Filtre sur les tickers rattachés à ce secteurs
            df_prices_sector: pd.DataFrame = df_valo.loc[:, sector_bool_array]
            df_prices_sector.fillna(0, inplace=True)

            # Ajout au dictionnaire
            dict_sector[sector_key] = df_prices_sector

        # Suppression des others (et financières, à faire)
        del dict_sector["other"]
        del dict_sector["Financial Services"]
        del dict_sector["Banks"]
        del dict_sector["Insurance"]

        return dict_sector



