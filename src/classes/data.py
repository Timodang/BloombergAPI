from datetime import datetime
import pandas as pd
from pandas import read_excel

from src.classes.blp import BLP
from src.classes.utilitaire import Utils

class Data:

    LIST_FUNDAMENTALS = ["book value", "EPS"]

    """
    Classe permettant de récupérer / importer les données nécessaires pour réaliser
    le backtest
    """

    def __init__(self, start_date:datetime, end_date:datetime,
                 list_ticker_rf:list, list_ticker_bench:list, use_api: bool)->None:
        # Récupération de la date de début et de la date de fin
        self.start_date:datetime = start_date
        self.end_date:datetime = end_date

        # Récupération du calendrier pour les extract
        self.calendar: list = Utils.get_calendar(self.start_date, self.end_date)
        self.eom_calendar: list = Utils.get_last_date_month(self.calendar)

        # Récupération du nom du ticker pour le taux sans risque et le benchmark
        self.ticker_rf: list = list_ticker_rf
        self.ticker_bench: list = list_ticker_bench

        # booléen qui indique s'il faut importer les données depuis un terminal bloomberg ou non
        self.use_api: bool = use_api

        # Déclaration des arguments que l'on va remplir en important les données depuis l'API
        self.blp = None
        self.dict_compo: dict = dict()
        self.unique_tickers: list = []
        self.universe: pd.DataFrame = pd.DataFrame()
        self.sector: pd.DataFrame = pd.DataFrame()
        self.dict_metrics: dict = dict()
        self.df_valo: pd.DataFrame = pd.DataFrame()
        self.rf: pd.DataFrame = pd.DataFrame()
        self.df_prices: pd.DataFrame = pd.DataFrame()

        # Pour le mapping sectoriel
        self.dict_sector: dict = dict()

    @staticmethod
    def _compute_valuation_metrics(str_metric: str) -> (pd.DataFrame, pd.DataFrame):
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
        df_prices: pd.DataFrame = read_excel("data/Prix des actifs.xlsx")

        # Chargement du dataframe contenant la métrique de valorisation choisie (par défaut, book-value per share)
        if str_metric == "book value":
            # Récupération de la book value per share par date pour chaque actif
            df_fundamental: pd.DataFrame = read_excel("data/Book Value des actifs.xlsx")

        elif str_metric == "EPS":
            df_fundamental: pd.DataFrame = read_excel("data/EPS des actifs.xlsx")
        else:
            raise Exception("Pas de dataframe disponible")

        # Passage des dates en indice
        df_prices.set_index(df_prices["Unnamed: 0"], inplace=True)
        df_prices.drop("Unnamed: 0", axis=1, inplace=True)
        df_fundamental.set_index(df_fundamental["Unnamed: 0"], inplace=True)
        df_fundamental.drop("Unnamed: 0", axis=1, inplace=True)

        # Test sur les dimensions
        if df_prices.shape[1] != df_fundamental.shape[1]:
            print("Les deux dataframe n'ont pas le même nombre d'actif")
            # On aligne sur le nombre d'actifs disponible le plus petit (problème lors de l'import bloomberg)
            if df_prices.shape[1] < df_fundamental.shape[1]:
                df_fundamental = df_fundamental.loc[:, df_fundamental.columns.isin(df_prices.columns)]
            else:
                df_prices = df_prices.loc[:, df_prices.columns.isin(df_fundamental.columns)]

        # Alignement pour les dates si pas le même nombre de périodes
        df_prices_aligned, df_fundamental_aligned = df_prices.align(df_fundamental, join = "inner", axis=0)

        # Lag des données comptables (décalage entre date associée à la valeur et date de publication), 21 jours par défaut
        df_prices_to_keep: pd.DataFrame = df_prices_aligned.iloc[21:df_prices_aligned.shape[0],]
        df_fundamental_to_keep: pd.DataFrame = df_fundamental_aligned.iloc[0 :df_fundamental_aligned.shape[0] - 21,]

        # Calcul du price-to-book
        df_valo: pd.DataFrame = df_prices_to_keep / df_fundamental_to_keep

        return df_valo, df_prices


    def import_all_data(self, metric:str):
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

            # Sixième étape : Récupération du taux sans risque
            self.rf: pd.DataFrame = self._get_risk_free_rate()

            # Fermeture de la session bloomberg
            self.blp.closeSession()

        # Calcul et récupération des métriques de valorisation et des prix
        self.df_valo, self.df_prices = self._compute_valuation_metrics(metric)
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

        # Ajout d'Equity à la fin de chaque liste
        list_tickers_unique_correct = [ticker + " Equity" for ticker in list_tickers_unique]

        # Récupération de tous les actifs de l'univers
        pd.DataFrame(list_tickers_unique_correct).to_excel("data/Tickers uniques.xlsx")

        # Récupération de la liste contenant les tickers uniques
        return list_tickers_unique_correct


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
                                               strFields=field_sector)

        df_sector.to_excel("data/Secteur des actifs.xlsx")
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

                # Modification de la liste pour ajouter "Equity" à la fin de chaque ticker
                list_current_ticker_correct = [ticker + " Equity" for ticker in list_current_ticker]

                # Si le ticker est présent dans l'indice pour le mois courant, on affecte un 1 dans le dataframe
                df_universe.loc[key, df_universe.columns.isin(list_current_ticker_correct)] = 1

        # Sauvegarde en Excel
        df_universe.to_excel("data/Composition univers.xlsx")

        return df_universe

    def _get_company_metrics(self)->dict:
        """
        Méthode permettant de récupérer les métriques financières (prix, ...) ou comptables (EPS, ...)
        pour un ensemble d'entreprises entre une date de début et une date de fin
        :return: Dictionnaire contenant les métriques pour tous les tickers
        """

        # Field à récupérer : les prix, la book-value par action, les earnings, ... (==> les fonda qui sont utilisés en valo)
        list_fields:list = ["PX_LAST","BOOK_VAL_PER_SH", "EPS_FOR_RATIOS"]
        list_nom_fichier: list = ["Prix des actifs.xlsx", "Book Value des actifs.xlsx", "EPS des actifs.xlsx"]
        i: int = 0

        #tickers = self.unique_tickers
        #strFields = ["PX_LAST","BOOK_VAL_PER_SH","EPS_FOR_RATIOS"]
        #start_date = dt.datetime(2024,12,15)
        #end_date = dt.datetime(2024, 12, 20)
        #test = self.blp.bdh(strSecurity=tickers, strFields=strFields, startdate=start_date,enddate=end_date)

        # Récupération des prix / fondamentaux pour toutes les entreprises ayant fait parti de l'univers
        dict_metrics: dict = self.blp.bdh(strSecurity=self.unique_tickers, strFields=list_fields, startdate=self.start_date,enddate=self.end_date)

        # Boucle sur chaque dataframe du dictionnaire pour les sauvegarder en excel
        for field in dict_metrics:
            path: str = "data/"+list_nom_fichier[i]
            df: pd.DataFrame = dict_metrics[field]
            df.to_excel(path)
            i += 1

        return dict_metrics

    def _get_risk_free_rate(self)->pd.DataFrame:
        """
        Fonction permettant de récupérer le taux sans risqur
        :return:
        """

        # Field à récupérer
        fields: list = ["PX_LAST"]

        # Import du taux sans risque
        df_rf: pd.DataFrame = self.blp.bdh(strSecurity=self.ticker_rf, strFields=fields, startdate=self.start_date,enddate=self.end_date)

        # Export en excel pour réutiliser dans le code central
        path: str = "data/rf.xlsx"
        df_rf.to_excel(path)
        return df_rf





