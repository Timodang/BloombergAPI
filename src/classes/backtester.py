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

        # Vérification sur la stratégie
        if strat != Portfolio.DEEP_VALUE_LABEL:
            raise Exception("Seula la stratégie Deep Value est implémentée pour l'instant")

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

        # Instanciation de la stratégie

        # Exposition brute
        self.exposition: pd.Series = pd.Series()

        # Positions en action à chaque date
        self.positions: pd.DataFrame = pd.DataFrame()

        # NAV du portefeuille à chaque date
        self.portfolio_value: pd.Series = pd.Series()

        # Instanciation d'un dataframe pour stocker les trades
        self.trade_historic: pd.DataFrame = pd.DataFrame(columns = self.df_prices.columns)

        # AJOUT : Stocker les positions par secteur sous forme de DataFrames
        self.sector_positions = {}  # {sector: DataFrame avec colonnes ['ticker', 'weight']}
        
        # MODIFICATION Instanciation de la stratégie
        if self.strategy == Portfolio.DEEP_VALUE_LABEL:
            from strategy import Strategy  # Import de la classe
            self.strategy_instance = Strategy(
                df_valo=self.df_val,
                df_prices=self.df_prices,
                universe=self.universe,
                dict_sectors=self.dict_sectors,
                weighting=self.weighting,
                quantile=self.quantile,
                start_date=self.start_date
            )

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

    def build_portfolio(self)->(pd.DataFrame, pd.DataFrame, pd.DataFrame) :
        """
        Méthode permettant de construire les poids du portefeuille dans
        le cas d'un backtest en euros
        :return:
        """

        # Initialisation d'un dataframe vierge pour stocker les positions par actifs
        positions: pd.DataFrame = pd.DataFrame(0.0, index=self.df_valo.index,
                                               columns=self.df_valo.columns)

        # Initialisation d'un dataframe vierge pour stocker les quantité par actids
        df_quantities: pd.DataFrame = positions.copy()

        # Initialisation d'un portefeuille vierge pour déterminer la NAV par actif
        df_nav: pd.DataFrame = pd.DataFrame(0, index = self.df_valo.index, columns = ["NAV"])

        # Récupération du nombre de périodes pour le backtest
        nb_periods: int = df_valo.shape[0]

        # Récupération du nombre de dates en liste (modif l'import de la classe, mettre les dates en indice directement)
        list_date: list = self.df_valo.index.to_list()

        # Récupération de l'indice de la première date >= date de début
        index_start_date: int = next((d for d in list_date if d>=self.start_date), None)

        # Boucle sur les périodes
        for idx in range(index_start_date, nb_periods):

            # récupération de la date courante
            current_date: datetime = list_date[idx]

            # Récupération des prix de la période précédentes
            prec_prices =  self.df_prices.iloc[idx-1, :]

            # Cas particulier : première date
            if idx == index_start_date:
                # Mise en place de la stratégie par portefeuille
                positions[idx, :] = self._compute_portfolio_position(current_date, bool_first_date=True)

                # NAV de la période = capital de la stratégie
                df_nav[idx, :] = self.capital

            # Autres cas
            else:

                # Calcul de la NAV de début de période

                # Mise à jour des poids pour l'ensemble des actifs
                positions[idx, :] = self._compute_portfolio_position(current_date)

            # Récupération des positions de la période et de la NAV de début de période
            current_positions = positions[idx, :]
            current_nav = df_nav[idx, :]

            # Calcul des quantités
            df_quantities[idx, :] = self.compute_quantity(current_positions, prec_prices, current_nav)

            # Mise à jour du compte cash

        # Seules les données à partir de la date de début de la stratégie sont conservées
        positions = positions.iloc[index_start_date:, ]
        df_quantities = df_quantities.iloc[index_start_date, :]
        df_nav = df_nav.iloc[index_start_date, :]
        return positions, df_quantities, df_nav

        # MODFICATIOIN
    def _compute_portfolio_position(self, current_date: datetime,
                                bool_first_date: bool = False) -> list:
        """
        Méthode adaptée pour utiliser la nouvelle interface Strategy
        qui retourne des actions string et des DataFrames
        """
        # Gestion des erreurs pour garantir que l'utilisateur réalise une segmentation sectorielle
        if self.segmentation != Portfolio.SECTOR_LABEL and self.segmentation is not None:
            raise Exception("Segmentation non implémentée")

        # Liste pour stocker les poids du portefeuille global
        list_weights_ptf: list = [0] * self.df_valo.shape[1]

        # Boucle par secteur
        for sector, df_valo_sector in self.dict_sectors.items():
            
            # Récupérer les positions existantes du secteur (DataFrame)
            existing_positions = self.sector_positions.get(sector, pd.DataFrame(columns=['ticker', 'weight']))
            
            # Étape 1 : Obtenir l'action à prendre ('buy', 'sell', ou 'neutral')
            action = self.strategy_instance.should_take_position(sector, current_date)
            
            # Étape 2 : Obtenir le DataFrame des positions pour ce secteur
            sector_positions_df = self.strategy_instance.get_sector_weights(
                sector, 
                current_date, 
                action,
                existing_positions
            )
            
            # Stocker les nouvelles positions du secteur
            self.sector_positions[sector] = sector_positions_df
            
            # Étape 3 : Convertir le DataFrame sectoriel en vecteur de poids global
            if not sector_positions_df.empty:
                for _, row in sector_positions_df.iterrows():
                    ticker = row['ticker']
                    weight = row['weight']
                    
                    # Trouver l'index du ticker dans le DataFrame global
                    if ticker in self.df_valo.columns:
                        ticker_index = self.df_valo.columns.get_loc(ticker)
                        list_weights_ptf[ticker_index] = weight
                    # else:
                    #     print(f"Warning: ticker {ticker} not found in global DataFrame columns")
        
        # # Vérification de l'exposition totale (optionnel)
        # total_exposure = sum(abs(w) for w in list_weights_ptf)
        # if total_exposure > 0:
        #     print(f"Date {current_date.strftime('%Y-%m-%d')} - Exposition totale: {total_exposure:.1%}")
            
            # # Compter les positions actives
            # long_count = sum(1 for w in list_weights_ptf if w > 0)
            # short_count = sum(1 for w in list_weights_ptf if w < 0)
            # if long_count > 0 or short_count > 0:
            #     print(f"  Positions: {long_count} longs, {short_count} shorts")
        
        # Retour de la liste des poids pour l'ensemble du portefeuille
        return list_weights_ptf

    @staticmethod
    def compute_quantity(current_positions: pd.DataFrame, prec_prices: pd.DataFrame, current_nav:pd.DataFrame)->list:
        """
        Méthode permettant de passer des poids aux quantités
        :param current_positions:
        :param prec_prices:
        :param current_nav:
        :return:
        """

        # récupération des quantités et des prix
        weights: pd.Series = pd.Series(current_positions, dtype=float)
        prices: pd.Series = pd.Series(prec_prices, dtype=float)

        # Montant à investir par actif
        amount_by_asset: pd.Series = weights * current_nav

        # Calcul des quantités
        quantities: list = [np.floor(amount_by_asset.values / prices.values).fillna(0).astype(int)]

        # Récupération des quantités
        return quantities

    # Méthode pour maj le compte cash

    # Méthode pour calculer la NAV
    def _compute_nav(self, bool_first_date:bool = False)->float:
        """
        Méthode permettant de calculer la NAV actuel
        :return:
        """

        # Cas particulier : à la première date, la NAV correspond au capital du fonds
        if bool_first_date:
            return self.capital

        # Dans les autres cas : NAV de la période = compte cash long + positions détenues
