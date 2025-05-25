from datetime import datetime
import pandas as pd
from pandas import read_excel

from src.classes.blp import BLP
from src.classes.utilitaire import Utils

class Data:

    LIST_FUNDAMENTALS = ["book value"]

    """
    Classe permettant de récupérer / importer les données nécessaires pour réaliser
    le backtest
    """

    def __init__(self, start_date:datetime, end_date:datetime, list_fields: list,
                 ticker_rf:str, list_ticker_bench:list, use_api: bool)->None:
        # Récupération de la date de début et de la date de fin
        self.start_date:datetime = start_date
        self.end_date:datetime = end_date

        # Récupération du calendrier pour les extract
        self.calendar: list = Utils.get_calendar(self.start_date, self.end_date)
        self.eom_calendar: list = Utils.get_last_date_month(self.calendar)

        # Récupération de la liste de fields à récupérer sur bloomberg
        self.list_fields: list = list_fields

        # Récupération du nom du ticker pour le taux sans risque et le benchmark
        self.ticker_rf: str = ticker_rf
        self.ticker_bench: list = list_ticker_bench

        # booléen qui indique s'il faut importer les données depuis un terminal bloomberg ou non
        self.use_api: bool = use_api

    @staticmethod
    def _compute_valuation_metrics(str_metric: str) -> pd.DataFrame:
        """
        Méthode permettant de calculer plusieurs métriques de valorisation par date
        pour un ensemble de ticker : Book-to-price, ...
        :param str_metric:
        :return:
        """

        # test sur la métrique demandée
        if str_metric not in Data.LIST_FUNDAMENTALS:
            raise Exception("La métrique comptable souhaitée pour construire la métrique de valorisation n'est pas implémentée")

        # Chargement du dataframe contenant les cours des  actifs
        df_prices: pd.DataFrame = read_excel("src/data/Prix des actifs.xlsx")

        # Chargement du dataframe contenant la métrique de valorisation choisie (par défaut, book-value per share)
        if str_metric == "book value":
            # Récupération de la book value per share par date pour chaque actif
            df_fundamental: pd.DataFrame = read_excel("src/data/Book Value des actifs.xlsx")

        # Test sur la taille des dataframe
        if df_prices.shape != df_fundamental.shape:
            raise Exception("Les deux dataframe doivent avoir les mêmes dimensions")

        # A voir, alignement pour les dates

        # Lag des données comptables (décalage entre date associée à la valeur et date de publication), 21 jours par défaut
        df_prices_to_keep: pd.DataFrame = df_prices.iloc[21:df_prices.shape[0],]
        df_fundamental_to_keep: pd.DataFrame = df_fundamental.iloc[0 :df_fundamental.shape[0] - 21,]

        # Calcul du price-to-book
        df_price_to_book: pd.DataFrame = df_prices_to_keep / df_fundamental_to_keep

        return df_price_to_book


    def _import_all_data(self):
        """
        Méthode permettant d'importer et de sauvegarder toutes les données
        si ce n'a pas déjà été fait précédemment
        :return:
        """

        print("Début de l'import des données")

        # Cas où l'utilisateur souhaite importer les données :
        if self.use_api:

            # Ouverture d'une session bloomberg
            self.blp = BLP()

            # Première étape : Récupération des compositions historiques sous format dictionnaire
            self.dict_compo: dict = self._get_historical_composition()

            # Deuxième étape : Récupération de l'ensemble des tickers uniques et sauvegarde en excel
            self.unique_tickers: list = self._get_tickers_list()

            # Troisième étape : Création d'un dataframe contenant la composition par date et sauvegarde en excel
            self.universe: pd.DataFrame = self._get_investment_universe()

            # Quatrième étape : Récupération du secteur pour chaque ticker et sauvegarde en excel
            self.sector: pd.DataFrame = self._get_sector()

            # Cinquième étape : Récupération des métriques (prix, fondamentaux, ...) requis pour la stratégie et sauvegarde en excel
            self.dict_metrics: dict = self._get_company_metrics()

            # Fermeture de la session bloomberg
            self.blp.closeSession()

        print("Fin de l'import des données")

    def _get_tickers_list(self)->list:
        """
        Permet de récupérer la liste des tickers qui ont fait parti de l'univers d'investissement
        :return: list contenant l'ensemble des tickers
        """

        # Initialisation de la liste
        list_tickers: list = []

        # Boucle sur les éléments du dictionnaire contenant les compositions
        # (key = date, values = dict(indice)(compo))
        for key, values in self.dict_compo.items():
            # Récupération du dictionnaire contenant les compositions de l'indice
            dict_index: dict = self.dict_compo[key]
            for index, tickers in dict_index.items():
                # Récupération des compos au format liste
                list_tickers_date: list = tickers
                # Incrémentation de la liste globale
                list_tickers.extend(list_tickers_date)

        # Seuls les éléments uniques sont conservés
        list_tickers_unique: list = list(set(list_tickers))

        # Récupération de tous les actifs de l'univers
        pd.DataFrame(list_tickers_unique).to_excel("src/data/Tickers uniques.xlsx")

        # Récupération de la liste contenant les tickers uniques
        return list_tickers_unique


    def _get_historical_composition(self)->dict:
        """
        Méthode permettant de récupérer la composition de l'univers d'investissement.
        Les données sont récupérées à chaque fin de mois par défaut (à voir...)
        :return:
        """

        # Field utilisé pour récupérer les compositions historiques
        list_fields: list = ["INDX_MWEIGHT_HIST"]

        # Création d'un dictionnaire pour stocker les compositions à chaque date
        dict_compo: dict = {}

        # Boucle sur les fins de mois (/ dates de l'univers si veut récup les compos à chaque date)
        for date in self.eom_calendar:

            # Récupération des compositions
            df_date_compo: pd.DataFrame = self.blp.bds(strSecurity=self.ticker_bench, strFields=list_fields,
                                                       strOverrideField="END_DATE_OVERRIDE",strOverrideValue=date)

            # Création d'un dictionnaire pour stocker à la date concernée les compositions des benchmarks
            dict_compo[date] = {}
            for field in df_date_compo.columns:
                for index, ticker in df_date_compo.iterrows():
                    compo = ticker[field]
                    dict_compo[date][index] = compo

        return dict_compo

    def _get_sector(self)->pd.DataFrame:
        """
        Méthode permettant de récupérer le secteur pour chaque ticker ayant fait parti de l'univers d'investissement
        :return: un dataframe contenant le secteur pour chaque entreprise
        """

        # Field utilisé pour récupérer les secteurs
        field_sector:list = ["GICS_INDUSTRY_GROUP_NAME"]

        # Récupération du secteur pour tous les tickers de l'univers
        df_sector: pd.DataFrame = self.blp.bdp(strSecurity=self.unique_tickers,
                                               strFields=field_sector, strOverrideField="END_DATE_OVERRIDE")

        df_sector.to_excel("src/data/Secteur des actifs.xlsx")
        return df_sector

    def _get_investment_universe(self)->pd.DataFrame:
        """
        Fonction permettant de construire l'univers d'investissement
        :return: un DataFrame contenant des 0 et des 1 selon que le titre fasse parti de l'univers d'investissement ou non
        """

        # Création d'un dataframe vide pour déterminer à chaque fin de mois les titres présents dans l'indice
        df_universe: pd.DataFrame = pd.DataFrame(0, index = self.eom_calendar, columns=self.unique_tickers)

        # Boucle sur les dates pour récupérer les compositions par date
        for key, values in self.dict_compo.items():
            # Récupération du dictionnaire contenant les compositions de l'indice
            dict_index: dict = self.dict_compo[key]
            for index, tickers in dict_index.items():
                # Récupération des compos au format liste
                list_current_ticker: list = tickers

                # Si le ticker est présent dans l'indice pour le mois courant, on affecte un 1 dans le dataframe
                df_universe.loc[key, df_universe.columns.isin(list_current_ticker)] = 1

        # Sauvegarde en Excel
        df_universe.to_excel("src/data/Composition univers.xlsx")

        return df_universe

    def _get_company_metrics(self)->dict:
        """
        Méthode permettant de récupérer les métriques financières (prix, ...) ou comptables (EPS, ...)
        pour un ensemble d'entreprises entre une date de début et une date de fin
        :return: Dictionnaire contenant les métriques pour tous les tickers
        """

        # Field à récupérer : les prix, la book-value par action, les earnings, ... (==> les fonda qui sont utilisés en valo)
        list_fields:list = ["PX_LAST","BOOK_VAL_PER_SH"]
        list_nom_fichier: list = ["Prix des actifs.xlsx", "Book Value des actifs.xlsx"]
        i: int = 0

        # Récupération des prix / fondamentaux pour toutes les entreprises ayant fait parti de l'univers
        dict_metrics: dict = self.blp.bdh(strSecurity=self.unique_tickers, strFields=list_fields, startdate=self.start_date,
                                          enddate=self.end_date)

        # Boucle sur chaque dataframe du dictionnaire pour les sauvegarder en excel
        for df in dict_metrics:
            path: str = "src/data/"+list_nom_fichier[i]
            df.to_excel(path)
            i += 1

        return dict_metrics





