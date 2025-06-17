import pandas as pd
import numpy as np
import datetime as datetime
from operator import add

from src.classes.utilitaire import Utils
from src.main import df_valo


class Portfolio:
    #Labels relatifs aux méthodes de calcul des rendements
    METHODS_LABELS = ["discret", "continu"]

    # Labels relatifs à la périodicité
    MONTHLY_LABEL = "monthly"
    QUARTERLY_LABEL = "quarterly"
    YEARLY_LABEL = "yearly"
    PERIODICITY_LABELS = [MONTHLY_LABEL, QUARTERLY_LABEL, YEARLY_LABEL]

    # Labels relatifs aux stratégies
    DEEP_VALUE_LABEL = "Deep Value"

    # Labels relatifs aux schémas de pondération
    WEIGHTING_LABELS = ["equalweight", "ranking"]

    # Labels relatifs à la segmentation
    SECTOR_LABEL = "sectorial"

    """
    Classe utilisée pour construire un portefeuille et effectuer un backtest sur 
    une stratégie donnée
    """

    def __init__(self, df_val: pd.DataFrame, df_prices: pd.DataFrame, universe:pd.DataFrame, dict_sector: dict,
                 strat: str, weighting: str, start_date: datetime,
                 list_date: list, quantile: float = None, segmentation:str = "sectorial",
                 capital: int = 100000000, leverage: int = 3):
        """
        :param df_val: DataFrame contenant la métrique de valorisation de tous les portefeuilles
        :param universe: DataFrame contenant la composition de l'univers d'investissement à chaque date
        :param dict_sector: Dictionnaire contenant la segmentation sectorielle du portefeuille
        :param strat: Nom de la stratégie à backtester
        :param weighting: Méthode de pondération utilisée
        :param quantile: quantile utilisé pour le calcul des poids dans un schéma de ranking
        :param segmentation: segmentation sectorielle du portefeuille
        """

        # Vérification sur les dimensions
        if df_prices.shape != df_val.shape:
            raise Exception("Les dimensions des dataframes de prix et de valorisation ne correspondent pas")

        # instance de la classe util
        self.utils: Utils = Utils()

        # récupération des prix et de la métrique de valorisation
        self.df_valo: pd.DataFrame = df_val
        self.df_prices: pd.DataFrame = df_prices

        # Récupération de la composition de l'univers d'investissement et du benchmark
        self.universe: pd.DataFrame = universe

        # Récupération de la segmentation sectorielle du portefeuille
        self.dict_sectors: dict = dict_sector
        self.segmentation: str = segmentation

        # Type de stratégie à implémenter et infos sur les périodes
        self.strategy: str = strat
        self.list_date: list = list_date
        self.start_date: datetime = start_date

        # Récupération des informations sur les schémas de pondération (quantile si ranking)
        self.quantile: float = quantile
        self.weighting: str = weighting

        # Capital et levier autorisé au sein du fonds
        self.capital: int = capital
        self.leverage: int = leverage

        # Séparation du capital en compte cash long et short
        self.cash_long: float = self.capital
        self.cash_short: float = 0

        # Exposition brute
        self.exposition: pd.Series = pd.Series()

        # Positions en action à chaque date
        self.positions: pd.DataFrame = pd.DataFrame()

        # NAV du portefeuille à chaque date
        self.portfolio_value: pd.Series = pd.Series()

        # Instanciation d'un dataframe pour stocker les trades
        self.trade_historic: pd.DataFrame = pd.DataFrame(columns = self.df_prices.columns)

    def run_backtest(self):
        """
        Méthode permettant de réaliser le backtest en calculant les positions de chaque titre
        et le valeur du portefeuille
        :return:
        """

        # 1ere étape : calcul des quantités en portefeuille
        portfolio_positions: pd.DataFrame = self.build_portfolio()
        self.positions = portfolio_positions

        # 3eme étape : Calcul de l'exposition brute du portefeuille

    def build_portfolio(self)->pd.DataFrame:
        """
        Méthode permettant de construire les poids du portefeuille dans
        le cas d'un backtest en euros
        :return:
        """

        # Initialisation d'un dataframe vierge pour stocker les positions par actifs
        positions: pd.DataFrame = pd.DataFrame(0.0, index=self.df_valo.index,
                                               columns=self.df_valo.columns)

        # Initialisation d'un portefeuille vierge pour déterminer la NAV par actif
        df_nav: pd.DataFrame = pd.DataFrame(0, index = self.df_valo.index, columns = ["NAV"])

        # Récupération du nombre de périodes pour le backtest
        nb_periods: int = df_valo.shape[0]

        # Récupération du nombre de dates en liste (modif l'import de la classe, mettre les dates en indice directement)
        list_date: list = self.df_valo.index.to_list()

        # Récupération de l'indice de la première date >= date de début
        index_start_date: int = next((d for d in list_date if d>=self.start_date), None)

        # Boucle sur les périodes
        for idx in range(index_start_date, self.df_valo.shape[0]):

            # récupération de la date courante
            current_date: datetime = list_date[idx]

            # Cas particulier : première date
            if idx == index_start_date:



                # Mise en place de la stratégie par portefeuille
                positions[idx, :] = self._compute_portfolio_position_v2(current_date, bool_first_date=True)
            # Autres cas
            else:
                positions[idx, :] = self._compute_portfolio_position_v2(current_date)

        # Seules les données à partir de la date de début de la stratégie sont conservées
        positions = positions.iloc[index_start_date:, ]
        return positions

    def _compute_portfolio_position(self, current_date:datetime,
                                       bool_first_date:bool = False)->list:
        """
        Méthode permettant de calculer les positions, date par date, pour le portefeuille Deep Value avec
        segmentation par industrie
        :return:
        """
        # Gestion des erreurs pour garantir que l'utilisateur réalise une segmentation sectorielle
        if self.segmentation != Portfolio.SECTOR_LABEL and self.segmentation is not None:
            raise Exception("Segmentation non implémentée")

        # Seule stratégie implémentée pour le moment : Deep Value
        if self.strategy != Portfolio.DEEP_VALUE_LABEL:
            raise Exception("Stratégie non implémentée")

        # Liste pour stocker les poids du portefueille
        list_weights_ptf: list = [0] * self.df_valo.shape[1]

        # Instanciation de la stratégie

        # Boucle par secteur
        for key, value in self.dict_sectors.items():

            # Récupération du secteur courant et du dataframe associé
            sector: str = key
            df_valo_sector: pd.DataFrame = value

            # Cas particulier de la première date
            if bool_first_date:

                # Appel de la méthode pour calculer la médiane sur les périodes précédentes (argument : sector, df_valo_sector, self.start_date)
                a = 3

            # Récupération de la métrique de valorisation pour tous les titres du portefeuille à la date courante
            vector_metrics: pd.DataFrame = df_valo_sector.loc[current_date, :]

            # Etape 1 : déterminer s'il faut prendre une position sur le secteur ou non (méthode take_positions)
            bool_take_position: bool = False

            # Si position à prendre (= bool_take_position) = True et pas de position actuellement en portefeuille sur le secteur, acquisition
            if bool_take_position and a==3:

                # Récupération des positions pour le secteur (renvoyer les poids par actif)
                list_weight_sector: list = []

                # Aggrégation avec les poids de tous les tickers sur lesquels une position est prise à cette date
                list_weights_ptf = list(map(add, list_weights_ptf, list_weight_sector))

            # Deuxième cas de figure : position à ne pas prendre (= False) et position actuellement en portefeuille : provoque une cession
            elif not bool_take_position and a==3:

                a = 3

            # Autres cas : aucune prise de position


    def _compute_nav(self, bool_first_date:bool = False)->float:
        """
        Méthode permettant de calculer la NAV actuel
        :return:
        """

        # Cas particulier : à la première date, la NAV correspond au capital du fonds
        if bool_first_date:
            return self.capital

        # Dans les autres cas : NAV de la période = compte cash long + positions détenues
