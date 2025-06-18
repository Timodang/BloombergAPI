import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
import datetime as datetime

class Strategy:
    """
    Stratégie Deep Value sur Russell 1000 - Interface compatible backtester Portfolio
    
    Architecture 
    1. Instanciation via Portfolio.__init__()
    2. Méthodes appelées directement par Portfolio._compute_portfolio_position()
    3. Gestion automatique warmup + historique continu

    Méthodes publiques pour Portfolio :
    - calculate_sector_median(sector, df_valo_sector, start_date) → médiane historique
    - should_take_position(sector, current_date, df_valo_sector) → bool
    - get_sector_weights(sector, df_valo_sector, current_date) → list des poids
    - update_sector_history(sector, current_date, df_valo_sector) → mise à jour historique
    """
    
    def __init__(self, df_valo: pd.DataFrame, df_prices: pd.DataFrame, universe: pd.DataFrame, 
                 dict_sectors: dict, weighting: str, quantile: float, start_date: datetime.datetime):
        """
        Initialisation de la stratégie - appelée par Portfolio.__init__()
        :param df_valo: DataFrame book-to-value (dates × actifs)
        :param df_prices: DataFrame des prix (dates × actifs) 
        :param universe: Composition Russell 1000 (dates × actifs avec 1/0)
        :param dict_sectors: Segmentation sectorielle {secteur: df_valo_secteur}
        :param weighting: Schéma de pondération ("equalweight", "ranking")
        :param quantile: Seuil quantile pour ranking
        :param start_date: Date de début de la stratégie
        """
        # Données financières
        self.df_valo = df_valo
        self.df_prices = df_prices
        self.universe = universe
        self.dict_sectors = dict_sectors  # contenant, pour chaque secteur, les book-to-value (ou métriques équivalentes) indexées par date.
        
        # Paramètres stratégie
        self.weighting = weighting
        self.quantile = quantile
        self.start_date = start_date
        
        # Calcul book-to-value si nécessaire
        self.book_to_value = (df_valo / df_prices).replace([np.inf, -np.inf], np.nan) 
        if self.book_to_value.empty and not df_prices.empty:
            # Fallback si df_valo est vide
            print("Calcul book-to-value à partir des prix...")
            # Ici, il faudrait les book values, mais utilisons df_valo comme fourni
        
        # Approximation capitalisation : prix relatifs
        if not df_prices.empty:
            self.market_caps = df_prices.iloc[-1] * 1000  # Base temporaire à chnger selon ce qu'on se dit
        else:
            self.market_caps = pd.Series()
        
        # Historique des spreads par secteur (warmup automatique)
        self.spread_history = {}
        
        # État des positions actuelles par secteur
        self.current_positions = {}  # {secteur: [weights_list]}
        
        # Paramètres warmup
        self.warmup_completed = False
        self.warmup_periods = 60  # 5 ans * 12 mois
        
        # Initialisation automatique
        self._initialize_strategy()
        
        # print(f"Strategy Deep Value initialisée :")
        # print(f"  Données : {df_valo.shape[0]} dates, {df_valo.shape[1]} actifs")
        # print(f"  Secteurs : {len(dict_sectors)} secteurs")
        # print(f"  Start date : {start_date}")
        # print(f"  Weighting : {weighting}")
    
    def _initialize_strategy(self):
        """Initialise la stratégie avec warmup automatique si nécessaire"""

        
        # Vérification si warmup nécessaire
        available_dates = self.df_valo.index.tolist()
        start_index = next((i for i, d in enumerate(available_dates) if d >= self.start_date), 0)
        
        if start_index >= self.warmup_periods:
            # Warmup sur les périodes précédant start_date
            self._perform_warmup(available_dates, start_index)
        else:
            print(f"Pas assez de données pour warmup complet ({start_index} < {self.warmup_periods})")
    
    def _perform_warmup(self, available_dates: list, start_index: int):
        """Effectue le warmup sur les périodes précédant start_date"""
        warmup_dates = available_dates[max(0, start_index - self.warmup_periods):start_index]
        
        print(f"Warmup sur {len(warmup_dates)} périodes : {warmup_dates[0]} → {warmup_dates[-1]}")
        
        for date in warmup_dates:
            # Calcul spreads pour tous les secteurs à cette date
            for sector, df_valo_sector in self.dict_sectors.items():
                spread = self._calculate_sector_spread_at_date(sector, df_valo_sector, date)
                if not np.isnan(spread):
                    if sector not in self.spread_history:
                        self.spread_history[sector] = []
                    self.spread_history[sector].append(spread)
        
        self.warmup_completed = True
        print(f"Warmup terminé - Historique initialisé pour {len(self.spread_history)} secteurs")
    
    # =================== MÉTHODES POUR PORTFOLIO ===================
    
    def calculate_sector_median(self, sector: str, df_valo_sector: pd.DataFrame, 
                               start_date: datetime.datetime) -> float:
        """
        Calcule la médiane historique d'un secteur (appelée par Portfolio pour première date)
        :param sector: Nom du secteur
        :param df_valo_sector: DataFrame valorisation du secteur
        :param start_date: Date de début de stratégie
        :return: Médiane des spreads historiques
        """
        if sector not in self.spread_history or len(self.spread_history[sector]) == 0:
            return np.nan
        
        historical_spreads = np.array(self.spread_history[sector])
        median = np.percentile(historical_spreads, 50)
        
        print(f"Médiane historique {sector}: {median:.3f} (sur {len(historical_spreads)} points)")
        return median
    
    def should_take_position(self, sector: str, current_date: datetime.datetime, 
                            df_valo_sector: pd.DataFrame) -> bool:
        """
        Détermine si une position doit être prise sur un secteur (remplace bool_take_position)
        :param sector: Nom du secteur
        :param current_date: Date courante
        :param df_valo_sector: DataFrame valorisation du secteur
        :return: True si position à prendre, False sinon
        """
        # Calcul du spread actuel
        current_spread = self._calculate_sector_spread_at_date(sector, df_valo_sector, current_date)
        
        if np.isnan(current_spread):
            return False
        
        # Mise à jour de l'historique
        if sector not in self.spread_history:
            self.spread_history[sector] = []
        self.spread_history[sector].append(current_spread)
        
        # Vérification historique suffisant
        if len(self.spread_history[sector]) < 24:  # Minimum 2 ans
            return False
        
        # Calcul des seuils
        historical_spreads = np.array(self.spread_history[sector])
        percentile_80 = np.percentile(historical_spreads, 80)
        median = np.percentile(historical_spreads, 50)
        
        # Décision selon les seuils  ## A MODIFIER POTENTIELLEMENT AVEC BUY / HOLD OU NEUTRAL 
        if current_spread > percentile_80:
            # ENTRÉE : Spread élevé = opportunité value
            print(f" {sector}: ENTRÉE - spread {current_spread:.3f} > P80 {percentile_80:.3f}")
            return True
        elif current_spread < median and sector in self.current_positions:
            # SORTIE : Spread faible + position existante = sortie
            print(f" {sector}: SORTIE - spread {current_spread:.3f} < médiane {median:.3f}")
            del self.current_positions[sector]  # Nettoyage position
            return False
        elif sector in self.current_positions:
            # MAINTIEN : Spread intermédiaire + position existante = maintien
            print(f" {sector}: MAINTIEN - spread {current_spread:.3f}")
            return True
        else:
            # ATTENTE : Spread intermédiaire + pas de position = attente
            return False
    
    def get_sector_weights(self, sector: str, df_valo_sector: pd.DataFrame, 
                          current_date: datetime.datetime) -> list:
        """
        Calcule les poids des actifs d'un secteur (remplace list_weight_sector)
        :param sector: Nom du secteur
        :param df_valo_sector: DataFrame valorisation du secteur
        :param current_date: Date courante
        :return: Liste des poids (taille = nombre total d'actifs)
        """
        # Initialisation vecteur de poids complet
        total_assets = self.df_valo.shape[1]
        weights_vector = [0.0] * total_assets
        
        # Vérification position active sur le secteur
        if sector not in self.current_positions:
            return weights_vector
        
        # Génération des signaux long/short pour le secteur
        sector_signals = self._generate_sector_signals(sector, df_valo_sector, current_date)
        
        # Application des poids selon le schéma de pondération
        for ticker, signal in sector_signals.items():
            if ticker in self.df_valo.columns:
                ticker_index = self.df_valo.columns.get_loc(ticker)
                
                if signal > 0:  # Long
                    weights_vector[ticker_index] = self._get_asset_weight(sector, "long")
                elif signal < 0:  # Short
                    weights_vector[ticker_index] = self._get_asset_weight(sector, "short")
        
        # Stockage pour maintien futur
        self.current_positions[sector] = weights_vector.copy()
        
        # Statistiques
        n_long = sum(1 for w in weights_vector if w > 0)
        n_short = sum(1 for w in weights_vector if w < 0)
        print(f"  → {sector}: {n_long} longs + {n_short} shorts générés")
        
        return weights_vector
    
    # def update_sector_history(self, sector: str, current_date: datetime.datetime, 
    #                          df_valo_sector: pd.DataFrame):
    #     """
    #     Met à jour l'historique d'un secteur (appelé à chaque période)
    #     :param sector: Nom du secteur
    #     :param current_date: Date courante
    #     :param df_valo_sector: DataFrame valorisation du secteur
    #     """
    #     # Cette mise à jour est déjà faite dans should_take_position()
    #     # Méthode fournie pour compatibilité si Portfolio l'appelle séparément
    #     pass
    
    # =================== MÉTHODES INTERNES ===================
    
    def _calculate_sector_spread_at_date(self, sector: str, df_valo_sector: pd.DataFrame, 
                                        date: datetime.datetime) -> float:
        """
        Calcule le spread de valorisation d'un secteur à une date donnée
        :param sector: Nom du secteur
        :param df_valo_sector: DataFrame valorisation du secteur
        :param date: Date pour le calcul
        :return: Spread de valorisation
        """

        # Si la date n'existe pas, on renvoie un nan
        if date not in df_valo_sector.index:
            return np.nan
        
        # Récupérer et nettoyer les ratios (éliminer tout ticker sans donnée à cette date.)
        sector_data = df_valo_sector.loc[date].dropna()
        
        # Filtrage Si Russell 1000 disponible
        if not self.universe.empty and date in self.universe.index:
            russell_tickers = self.universe.loc[date]
            russell_actifs = russell_tickers[russell_tickers == 1].index.tolist()
            sector_data = sector_data[sector_data.index.isin(russell_actifs)]  # isoler seulement les titres (du secteur) actifs dans l’indice à la date t.
        
        if len(sector_data) < 20:  # Pas assez d'actifs
            return np.nan
        
        # Approximation capitalisation
        if not self.market_caps.empty:
            caps_data = self.market_caps[sector_data.index]
            sector_df = pd.DataFrame({
                'BookToValue': sector_data,
                'MarketCap': caps_data
            }).dropna()
        else:
            # Fallback sans capitalisation
            sector_df = pd.DataFrame({
                'BookToValue': sector_data,
                'MarketCap': [1] * len(sector_data)  # Equal weight
            })
        
        if len(sector_df) < 20:
            return np.nan
        
        # Calcul du spread value/growth
        return self._compute_spread(sector_df)
    
    def _compute_spread(self, sector_data: pd.DataFrame) -> float:
        """
        Calcule le spread value/growth d'un secteur
        :param sector_data: DataFrame avec 'BookToValue' et 'MarketCap'
        :return: Spread (log ratio)
        """
        if len(sector_data) < 20:
            return np.nan
        
        # Tri par book-to-value (décroissant)
        sorted_data = sector_data.sort_values('BookToValue', ascending=False)
        n_assets = len(sorted_data)
        n_select = max(1, int(n_assets * 0.1))  # Top/Bottom 10%
        
        # Sélection déciles
        top_10 = sorted_data.head(n_select)     # Value (book-to-value élevés)
        bottom_10 = sorted_data.tail(n_select)  # Growth (book-to-value faibles)
        
        # Moyennes pondérées par capitalisation
        top_weighted = (top_10['BookToValue'] * top_10['MarketCap']).sum() / top_10['MarketCap'].sum()
        bottom_weighted = (bottom_10['BookToValue'] * bottom_10['MarketCap']).sum() / bottom_10['MarketCap'].sum()
        
        # Spread logarithmique 
        if bottom_weighted <= 0 or top_weighted <= 0:
            return np.nan
        
        spread = np.log(top_weighted / bottom_weighted)
        return spread
    
    def _generate_sector_signals(self, sector: str, df_valo_sector: pd.DataFrame, 
                                current_date: datetime.datetime) -> Dict[str, float]:
        """
        Génère les signaux long/short pour un secteur
        :param sector: Nom du secteur
        :param df_valo_sector: DataFrame valorisation du secteur
        :param current_date: Date courante
        :return: Dictionnaire avex un signal par ticker {ticker: signal} (uniquement pour les actifs sélectionnés; les autres tickers n’apparaissent pas)
        """
        sector_signals = {}
        
        if current_date not in df_valo_sector.index:
            return sector_signals
        
        # Données secteur à la date
        sector_data = df_valo_sector.loc[current_date].dropna()
        
        # Filtrage Russell 1000
        if not self.universe.empty and current_date in self.universe.index:
            russell_tickers = self.universe.loc[current_date]
            russell_actifs = russell_tickers[russell_tickers == 1].index.tolist()
            sector_data = sector_data[sector_data.index.isin(russell_actifs)]
        
        if len(sector_data) < 20:
            return sector_signals
        
        # Tri par book-to-value
        sorted_data = sector_data.sort_values(ascending=False)  # Décroissant
        n_assets = len(sorted_data)
        n_select = max(1, int(n_assets * 0.1))  # 10%
        
        # Sélection actifs
        top_10_tickers = sorted_data.head(n_select).index.tolist()    # Value → LONG
        bottom_10_tickers = sorted_data.tail(n_select).index.tolist() # Growth → SHORT
        
        # Génération signaux
        for ticker in top_10_tickers:
            sector_signals[ticker] = 1.0    # LONG
        
        for ticker in bottom_10_tickers:
            sector_signals[ticker] = -1.0   # SHORT
        
        return sector_signals
    
    def _get_asset_weight(self, sector: str, position_type: str) -> float:
        """
        JE DOIS PAS OUBLIER D INCLURE N_SELECT DANS LA LES ARGUMENTS POUR AVOIR LES VRAIS POIDS


        Calcule le poids d'un actif selon le schéma de pondération
        :param sector: Nom du secteur
        :param position_type: "long" ou "short"
        :return: Poids de l'actif
        """
        if self.weighting == "equalweight":



            # Equal weight : 10% par côté divisé par nombre d'actifs sélectionnés
            base_weight = 0.10 / 5  # 10% / 5 actifs = 2% par actif
            return base_weight if position_type == "long" else -base_weight
            
        elif self.weighting == "ranking":   ## A REVOIR
            # Ranking weight basé sur quantile
            if self.quantile:
                base_weight = self.quantile / 5  # Quantile divisé par nombre d'actifs
                return base_weight if position_type == "long" else -base_weight
            else:
                return 0.02 if position_type == "long" else -0.02  # Fallback 2%
        
        else:
            # Fallback equal weight
            return 0.02 if position_type == "long" else -0.02
    
    def get_current_positions(self) -> Dict[str, list]:
        """
        Retourne les positions actuelles par secteur
        :return: Dictionnaire {secteur: liste_poids}
        """
        return self.current_positions.copy()
    
    def get_sector_history(self, sector: str) -> List[float]:
        """
        Retourne l'historique des spreads d'un secteur
        :param sector: Nom du secteur
        :return: Liste des spreads historiques
        """
        return self.spread_history.get(sector, []).copy()

