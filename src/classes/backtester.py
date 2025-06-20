import pandas as pd
import numpy as np
import datetime as datetime

from src.classes.strategy import Strategy
from src.classes.utilitaire import Utils

class Portfolio:

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
        self.df_prices: pd.DataFrame = df_prices.fillna(0)

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

        # Méthode pour définir le montant à investir par industrie (arbitraire 10% pour le moment)
        self.amount_per_sector: float = 0.1

        # Instanciation de la stratégie
        self.strategy: str = strat
        if self.strategy == Portfolio.DEEP_VALUE_LABEL:
            self.strategy_instance = Strategy(
                df_valo=self.df_valo,
                universe=self.universe,
                dict_sectors=self.dict_sectors,
                weighting=self.weighting,
                quantile=self.quantile,
                start_date=self.start_date
            )

        # Stockage des positions par secteur sous forme de DataFrames
        self.sector_positions = {}  # {sector: DataFrame avec colonnes ['ticker', 'weight']}

        # Positions (poids) à chaque date
        self.df_positions: pd.DataFrame = pd.DataFrame(0.0, index=self.df_valo.index,
                                               columns=self.df_valo.columns)

        # Quantités à chaque date pour le portefeuille LS et Long Only
        self.df_quantities: pd.DataFrame = self.df_positions.copy()
        self.df_quantities_long: pd.DataFrame = self.df_quantities.copy()

        # NAV du portefeuille à chaque date / NAV du portefeuille Long Only
        self.df_nav: pd.DataFrame = pd.DataFrame(0, index = self.df_valo.index, columns = ["NAV"])
        self.df_nav_long: pd.DataFrame = self.df_nav.copy()

        # Exposition brute
        self.df_exposition:pd.DataFrame = pd.DataFrame(0, index = self.df_valo.index, columns = ["Exposition Brute"])
        self.df_exposition_long: pd.DataFrame = self.df_exposition.copy()

    def run_backtest(self):
        """
        Méthode permettant de réaliser le backtest d'un portefeuille en euros
        :return:
        """
        # Récupération du nombre de périodes pour le backtest
        nb_periods: int = self.df_valo.shape[0]

        # Récupération du nombre de dates en liste (modif l'import de la classe, mettre les dates en indice directement)
        list_date: list = self.df_valo.index.to_list()

        # Récupération de l'indice de la première date >= date de début
        start_date: datetime = next((d for d in list_date if d>=self.start_date), None)
        index_start_date: int = list_date.index(start_date)

        # Calcul de la NAV à la première période (= capital de la stratégie Deep Value)
        self.df_nav.iloc[index_start_date-1] = self.capital
        self.df_nav_long.iloc[index_start_date-1] = self.capital

        # Boucle sur les périodes
        for idx in range(index_start_date, nb_periods):

            # récupération de la date courante
            current_date: datetime = list_date[idx]

            # Début de la récupération pour la date
            print(f"Application de la stratégie pour la date {current_date}")

            # Récupération des prix de la période précédentes
            prec_prices =  self.df_prices.iloc[idx-1, :]

            # Récupération des poids
            self.df_positions.iloc[idx, :], nb_buy_sect = self._compute_portfolio_position(current_date)

            # NAV de début de période (utilisée pour le calcul des quantités) = NAV de fin de période précédente
            current_nav: float = self.df_nav.iloc[idx-1]['NAV']

            # Récupération des positions de la période
            current_positions = self.df_positions.iloc[idx, :]

            # Calcul des quantités (portefeuille LS et Long Only)
            self.df_quantities.iloc[idx, :] = self.compute_quantity(current_positions, prec_prices, current_nav)
            self.df_quantities_long.iloc[idx, :] = self.compute_quantity(current_positions, prec_prices, current_nav, bool_long_only=True)

            # Mise à jour du compte cash
            self._compute_operations(idx)

            # Calcul de la NAV de fin de période (==> NAV de la période suivante)
            self.df_nav.iloc[idx] = self._compute_nav(current_date)
            self.df_nav_long.iloc[idx] = self._compute_nav(current_date, bool_long_only=True)

            # Calcul de l'exposition brute
            self.df_exposition.iloc[idx] = self._compute_exposition(current_date)
            self.df_exposition_long.iloc[idx] = self._compute_exposition(current_date, bool_long_only=True)

            # Si breach, cessions partielles et maj du compte cash en conséquence
            self._check_and_reduce_exposure(current_date)
            self._compute_operations(idx)

        # Seules les données à partir de la date de début de la stratégie sont conservées
        self.df_positions = self.df_positions.iloc[index_start_date:, ]
        self.df_quantities = self.df_quantities.iloc[index_start_date:, :]
        self.df_nav = self.df_nav.iloc[index_start_date:, :]
        self.df_exposition = self.df_exposition.iloc[index_start_date:,:]

        self.df_quantities_long = self.df_quantities_long.iloc[index_start_date:, :]
        self.df_nav_long = self.df_nav_long.iloc[index_start_date:, :]
        self.df_exposition_long = self.df_exposition_long.iloc[index_start_date:,:]

    def _compute_portfolio_position(self, current_date: datetime) -> (list,int):
        """
        Méthode adaptée pour utiliser la nouvelle interface Strategy
        qui retourne des actions string et des DataFrames
        """
        # Gestion des erreurs pour garantir que l'utilisateur réalise une segmentation sectorielle
        if self.segmentation != Portfolio.SECTOR_LABEL and self.segmentation is not None:
            raise Exception("Segmentation non implémentée")

        # Liste pour stocker les poids du portefeuille global
        list_weights_ptf: list = [0] * self.df_valo.shape[1]

        # Nombre de secteurs sur lesquels des positions ont été prises au cours de la période
        nb_sect_buy:int = 0

        # Boucle par secteur
        for sector, df_valo_sector in self.dict_sectors.items():

            # Récupérer les positions existantes du secteur (DataFrame)
            # existing_positions = self.sector_positions.get(sector, pd.DataFrame(columns=['ticker', 'weight']))
            # has_position=False
            # Si au moins une position est positive, cela veut dire que le secteur est dans le portefeuille
            # if (existing_positions > 0).any().any():
            #    has_position: bool = True

            # Étape 1 : Obtenir l'action à prendre ('buy', 'sell', ou 'neutral')
            action = self.strategy_instance.should_take_position(sector, current_date)
            if action == "buy":
                nb_sect_buy+=1


            # Étape 2 : Obtenir le DataFrame des positions pour ce secteur
            sector_positions_df = self.strategy_instance.get_sector_weights(
                sector,
                current_date,
                action,
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

        # Rescaling de la liste
        list_weights_ptf = self.rescale_weights_long_short(list_weights_ptf)

        # Retour de la liste des poids pour l'ensemble du portefeuille
        return list_weights_ptf, nb_sect_buy

    def compute_quantity(self, current_positions: pd.DataFrame, prec_prices: pd.DataFrame, current_nav:float,
                         bool_long_only:bool = False)->list:
        """
        Méthode permettant de passer des poids aux quantités
        :param current_positions:
        :param prec_prices:
        :return:
        """

        # récupération des poids et des prix de la période précédente
        weights: pd.Series = pd.Series(current_positions, dtype=float).astype(float)

        # Cas où le calcul est réalisé pour la jambe longue
        if bool_long_only:

            # Tous les poids négatifs sont annulés
            weights = weights.clip(lower = 0)

            # Calcul de la somme des poids (= nombre de portefeuille sur lesquels on est actuellement long)
            sum_weights: float = np.sum(weights)

            # Rescaling (=> poids normalisés à 100)
            weights = weights / sum_weights

        # Récupération des prix
        prices: pd.Series = pd.Series(prec_prices, dtype=float).astype(float)

        # Montant à investir par actif (=> poids * part de la NAV allouée au secteur)
        amount_by_asset: pd.Series = weights * current_nav * self.amount_per_sector

        # Calcul des quantités (arrondis à l'entier inférieur pour ne pas détenir de fraction d'action)
        quantities: np.array = np.nan_to_num(amount_by_asset.values / prices.values, nan = 0.0)
        list_quantities: list = np.floor(quantities).astype(int).tolist()
        # quantities: list = [np.floor(amount_by_asset.values / prices.values).fillna(0).astype(int)]

        # Récupération des quantités
        return list_quantities

    # Méthode pour maj le compte cash
    def _compute_operations(self, current_date_idx:int):
        """
        Méthode permettant de mettre à jour le compte cash de la stratégie
        :return:
        """

        # Récupération des quantités de titres détenues sur la période et des quantités de titres détenues sur la période précédente
        df_current_quantities: pd.DataFrame = self.df_quantities.iloc[current_date_idx,:]
        df_prec_quantities: pd.DataFrame = self.df_quantities.iloc[current_date_idx-1,:]

        # Premier cas : prec < 0 et actuel = 0 ==> récupération des poids
        close_short_idx = (df_prec_quantities < 0) & (df_current_quantities == 0)
        tickers_close_short:list = df_current_quantities[close_short_idx].index.tolist()

        # Deuxième cas : actuel > prec (= buy)
        long_idx = df_current_quantities > df_prec_quantities
        tickers_long:list = df_current_quantities[long_idx].index.tolist()

        # Troisième cas :  prec > 0 et actuel = 0 (= vente)
        sell_idx = (df_prec_quantities > 0) & (df_current_quantities==0)
        tickers_sell:list = df_current_quantities[sell_idx].index.tolist()

        # Quatrième cas : prec = 0 et actuel < 0 (= short)
        short_idx = (df_prec_quantities == 0) & (df_current_quantities < 0)
        tickers_short:list = df_current_quantities[short_idx].index.tolist()

        # Mise à jour du compte de cash long
        self._update_long_account(tickers_long, tickers_sell, current_date_idx)

        # Mise à jour du compte de cash short
        self._update_short_account(tickers_short, tickers_close_short, current_date_idx)

        # Mise à jour de l'historique des trades

    # Méthode pour calculer la NAV
    def _compute_nav(self, current_date:datetime, bool_long_only: bool = False)->float:
        """
        Méthode permettant de calculer la NAV actuel
        :return:
        """
        # Récupération des prix
        current_prices: pd.DataFrame = self.df_prices.loc[current_date, :]

        if bool_long_only:
            # Récupération des quantités de titre pour chaque actifs détenus en t
            current_quantities: pd.DataFrame = self.df_quantities_long.loc[current_date, :]
            cash: float = self.cash_long
        else:
            # Récupération des quantités de titre pour chaque actifs détenus en t
            current_quantities: pd.DataFrame = self.df_quantities.loc[current_date, :]
            cash: float = self.cash_long + self.cash_short

        # Calcul de la valeur du portefeuille à la fin de la date actuelle (eop)
        ptf_value: float = (current_quantities.values * current_prices.values).sum()

        # Calcul de la nav (valeur du ptf + montant sur le compte de cash long, cash short bloqué)
        nav: float = ptf_value + cash
        return nav

    # Méthode permettant de calculer l'exposition brute par date
    def _compute_exposition(self, current_date:datetime, bool_long_only: bool = False):
        """
        Méthode permettant de calculer l'exposition brute du portefeuille
        :param current_date:
        :return:
        """
        if bool_long_only:
            # Récupération des quantités de titre pour chaque actifs détenus en t
            current_quantities: pd.DataFrame = self.df_quantities_long.loc[current_date, :]
        else:
            # Récupération des quantités de titre pour chaque actifs détenus en t
            current_quantities: pd.DataFrame = self.df_quantities.loc[current_date, :]

        # Récupération des prix en t
        current_prices: pd.DataFrame = self.df_prices.loc[current_date,:]

        # Récupération de la NAV eop en t
        current_nav: float = self.df_nav.loc[current_date, :]["NAV"]

        # Calcul de l'exposition brute du portefeuille
        exposition: float = np.sum(np.abs(current_quantities.values * current_prices.values)) / current_nav
        return exposition

    # Méthode pour mettre à jour le compte de cash pour la poche longue
    def _update_long_account(self,list_buy:list, list_sell, current_date_index: int):
        """
        Méthode permettant de mettre à jour le compte de cash de la poche longue
        :param list_buy:
        :param list_sell:
        :param current_date_index:
        :return:
        """

        # Récupération des prix de la période précédente et des quantités de la période
        current_quantity = self.df_quantities.iloc[current_date_index, :]
        current_prices = self.df_prices.iloc[current_date_index-1,:]

        # Récupération des prix et des quantités à acheter ==> Montant décaissé pour réaliser les achats de la période
        df_quantities_buy: pd.Series = current_quantity[list_buy]
        df_prices_buy: pd.Series = current_prices[list_buy]
        amount_spent: float = (df_quantities_buy * df_prices_buy).sum()

        # Récupération des prix et des quantités à vendre ==> Montant encaissé suite aux ventes
        # Pour les ventes : il faut récupérer les quantités de t-1 et non t (0 en t, positif en t-1)
        prec_quantity = self.df_quantities.iloc[current_date_index-1, :]
        df_quantities_sell: pd.Series = prec_quantity[list_sell]
        df_prices_sell: pd.Series = current_prices[list_sell]
        amount_received: float = (df_quantities_sell * df_prices_sell).sum()

        # Mise à jour du compte de cash
        delta_cash_long: float = amount_received - amount_spent
        self.cash_long += delta_cash_long

    # Méthode pour mettre à jour le compte de cash pour la poche short
    def _update_short_account(self, list_short: list, list_close_short:list, current_date_index: int):
        """
        Méthode permettant de mettre à jour le compte de cash de la poche short
        :param list_short:
        :param list_close_short:
        :param current_date_index:
        :return:
        """

        # Récupération des prix de la période précédente et des quantités de la période
        current_quantity = self.df_quantities.iloc[current_date_index, :]
        current_prices = self.df_prices.iloc[current_date_index-1,:]

        # Récupération des prix et des quantités à short ==> Montant encaissé suite au short (bloqué / placé)
        df_quantities_short: pd.Series = current_quantity[list_short]
        df_prices_short: pd.Series = current_prices[list_short]
        cash_received: float = (np.abs(df_quantities_short) * df_prices_short).sum()

        # Récupération des prix et des quantités des tickers pour lesquels on clos le short
        current_prices_for_close: pd.DataFrame = self.df_prices.iloc[current_date_index,:]
        df_prices_for_close = current_prices_for_close[list_close_short]
        df_prec_quantities = self.df_quantities.iloc[current_date_index -1, :]
        quantities_to_buy_back = abs(df_prec_quantities[list_close_short])
        cash_spent = (quantities_to_buy_back * df_prices_for_close).sum()

        # df_quantities_close_short: pd.Series = current_quantity[list_close_short]
        # df_prices_close_short: pd.Series = current_prices[list_close_short]
        # cash_spent: float = (np.abs(df_quantities_close_short) * df_prices_close_short).sum()

        # Mise à jour de la poche short du compte cash
        delta_cash_short: float = cash_received - cash_spent
        self.cash_short += delta_cash_short

    def _check_and_reduce_exposure(self, current_date: datetime, bool_long_only: bool = False):
        """Réduit l'exposition si elle dépasse le levier autorisé"""
        if bool_long_only:
            leverage = 1
            current_exposure = self._compute_exposition(current_date, bool_long_only)
        else:
            leverage = self.leverage
            current_exposure = self._compute_exposition(current_date, bool_long_only)

        if current_exposure > leverage:
            # Calculer le facteur de réduction
            reduction_factor = leverage / current_exposure
            # Réduire toutes les positions proportionnellement

            self.df_positions.loc[current_date, :] *= reduction_factor
            if bool_long_only:
                self.df_quantities_long.loc[current_date, :] = (
                        self.df_quantities_long.loc[current_date, :] * reduction_factor
                ).astype(int)
            else:
                self.df_quantities.loc[current_date, :] = (
                    self.df_quantities.loc[current_date, :] * reduction_factor
                ).astype(int)

    def rescale_weights_long_short(self, list_weights:list, target_gross_exposure: float = 1.0) -> list:
        """
        Rescaling pour garantir à chaque période que la somme des poids en LS est égale à 1
        """
        # Calcul de l'exposition brute actuelle en % (ex : 5 secteurs ==> 1000% d'expo, 100% par secteur)
        gross_exposure = sum(abs(w) for w in list_weights)

        # Vérfication pour ne pas diviser par 0
        if gross_exposure == 0:
            return [0.0 for _ in list_weights]

        # Appliquer le facteur de rescaling
        scale_factor = target_gross_exposure / gross_exposure
        rescaled_weights = [w * scale_factor for w in list_weights]
        return rescaled_weights