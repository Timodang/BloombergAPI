import numpy as np
import pandas as pd
import datetime
from typing import Dict, List, Optional
from src.classes.weighting_scheme import EquallyWeighting, RankingWeightingSignals


class Strategy:
    """
    Stratégie Deep Value sectorielle sur Russell 1000

    Calcul de la "cherté" (cheapness) de chaque actif :
      cheapness = valorisation / prix

    Phases et logique :
    1. WARMUP (avant start_date) :
       - Pour chaque date et chaque secteur, calcul du spread historique
         (log(ratio haut décile / ratio bas décile)).
    2. TRADING (après start_date) :
       - Pour chaque date et secteur :
         - Calcul du spread courant (exclusion des extrêmes 5%/95%).
         - Signal d'entrée ('buy') si spread > 80e percentile historique.
         - Signal de sortie ('sell') si spread < médiane historique.
         - Sinon, 'neutral'.
       - Mise à jour de l'historique des spreads.
    3. POIDS :
       - _compute_equalweight_positions : pondération 10% long et 10% short,
         répartis également entre les actifs sélectionnés.
       - _compute_ranking_weights : pondération basée sur le rang de cheapness au start_date.
    """

    def __init__(
        self,
        df_valo: pd.DataFrame,
        universe: pd.DataFrame,
        dict_sectors: Dict[str, pd.DataFrame],
        weighting: str,
        quantile: float,
        start_date: datetime.datetime
    ):
        """
        :param df_valo: DataFrame valorisations (dates × tickers)
        :param df_prices: DataFrame prix (dates × tickers)
        :param universe: Composition Russell 1000 (dates × tickers binaires)
        :param dict_sectors: mapping secteur → df_valo sectoriel
        :param weighting: 'equalweight' ou 'ranking'
        :param quantile: fraction pour décile (ex. 0.1 pour 10%)
        :param start_date: début du backtest
        """
        # Stocker les paramètres
        self.universe = universe
        self.dict_sectors = dict_sectors
        self.weighting = weighting
        self.quantile = quantile
        self.start_date = start_date

        # Sélection du schéma de pondération
        if self.weighting == 'equalweight':
            self.scheme = EquallyWeighting(self.quantile)
        elif self.weighting == 'ranking':
            self.scheme = RankingWeightingSignals(self.quantile)
        else:
            raise ValueError(f"Unknown weighting: {self.weighting}")

        # df cheapness
        self.cheapness = df_valo.replace([np.inf, -np.inf], np.nan)

        # Historique des spreads par secteur
        self.spread_history: Dict[str, List[float]] = {}
        # Etat des positions par secteur (True si en portefeuille)
        self.current_positions: Dict[str, bool] = {}
        # Flag Warmup terminé
        self.warmup_completed: bool = False

        # Initialisation de la phase WARMUP
        self._initialize_strategy()

    def _initialize_strategy(self):
        """
        Prépare l'historique de spreads sur toutes les dates antérieures à start_date
        """
        if self.cheapness.empty:
            print("Warning: cheapness metrics empty, cannot perform warmup")
            return
        # Sélection des dates avant le début du backtest
        warmup_dates = [d for d in self.cheapness.index if d < self.start_date]
        if not warmup_dates:
            print(f"No data before {self.start_date} for warmup")
            return
        # Exécution du calcul d'historique
        self._perform_warmup(warmup_dates)

    def _perform_warmup(self, warmup_dates: List[datetime.datetime]):
        """
        Calcule et stocke les spreads historiques pour chaque secteur
        """
        print(f"Warmup over {len(warmup_dates)} periods: {warmup_dates[0]} → {warmup_dates[-1]}")
        for date in warmup_dates:
            for sector in self.dict_sectors:
                spread = self._calculate_sector_spread_at_date(sector, date)
                if not np.isnan(spread):
                    self.spread_history.setdefault(sector, []).append(spread)
        self.warmup_completed = True
        print("Warmup completed successfully")

    def should_take_position(
        self,
        sector: str,
        current_date: datetime.datetime
    ) -> str:
        """
        Détermine le signal pour un secteur à la date donnée :
          'buy', 'sell' ou 'neutral'
        """
        # Calcul du spread courant
        spread = self._calculate_sector_spread_at_date(sector, current_date)
        if np.isnan(spread):
            return 'neutral'
        # Mise à jour de l'historique des spreads
        self.spread_history.setdefault(sector, []).append(spread)
        hist = self.spread_history[sector]
        # Besoin d'au moins 24 points historiques (2 ans)
        if len(hist) < 24:
            return 'neutral'
        # Seuils : 80e percentile et médiane
        p80 = np.percentile(hist, 80)
        med = np.percentile(hist, 50)
        # Décision
        if spread > p80:
            self.current_positions[sector] = True
            return 'buy'
        if spread < med:
            self.current_positions.pop(sector, None)
            return 'sell'
        return 'neutral'

    def get_sector_weights(
            self,
            sector: str,
            current_date: datetime.datetime,
            action: str,
            existing_positions: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Retourne un DataFrame ['ticker','weight'] selon l'action :
        - 'neutral' : conserve existing_positions
        - 'sell'    : met tous les poids à 0
        - 'buy'     : calcule les poids via le schéma choisi
        """
        # Aucun changement
        if action == 'neutral':
            # Conserver les positions existantes, mais annuler celles qui ne sont plus valides
            df = existing_positions.copy() if existing_positions is not None else pd.DataFrame(
                columns=['ticker', 'weight'])

            if not df.empty:
                # Récupérer la composition Russell à la date
                if current_date in self.universe.index:
                    uni = self.universe.loc[current_date]
                    valid_tickers = set(uni[uni == 1].index)
                else:
                    valid_tickers = set()
                # filtrer également le secteur
                sector_tickers = set(self.dict_sectors[sector].columns)
                # Parcourir et mettre à zéro les poids invalides
                df['weight'] = df.apply(
                    lambda row: row['weight'] if (
                                row['ticker'] in valid_tickers and row['ticker'] in sector_tickers) else 0.0,
                    axis=1
                )
            return df

        # Entrée / ajustement de la position
        signals = self._generate_sector_signals(sector, current_date)
        if not signals:
            return pd.DataFrame(columns=['ticker', 'weight'])

        # Délégation au WeightingScheme
        signal_series = pd.Series(signals)
        weights_list = self.scheme.compute_weights(signal_series)
        # Construction du DataFrame final
        return pd.DataFrame({'ticker': list(signals.keys()), 'weight': weights_list})

    def _calculate_sector_spread_at_date(
        self,
        sector: str,
        date: datetime.datetime) -> float:
        """
        Calcule le log-spread de cheapness pour un secteur à une date :
          log(mean(top décile) / mean(bottom décile)),
        après exclusion des 5% extrêmes.
        """
        # 1) Vérifier disponibilité des données
        if date not in self.cheapness.index:
            return np.nan
        # 2) Extraire cheapness et filtrer sur univers
        series = self.cheapness.loc[date].dropna()
        if date in self.universe.index:
            members = self.universe.loc[date]
            allowed = members[members == 1].index
            series = series[series.index.isin(allowed)]
        # 3) Filtrer sur le secteur
        sector_tickers = list(self.dict_sectors[sector].columns)
        series = series[series.index.isin(sector_tickers)]
        # 4) Vérification d'un panel suffisant
        if len(series) < 20:
            return np.nan
        # 5) Exclusion des outliers (5%/95%)
        p5, p95 = series.quantile(0.05), series.quantile(0.95)
        filtered = series[(series > p5) & (series < p95)]
        if len(filtered) < 10:
            return np.nan
        # 6) Tri décroissant et sélection décile
        sorted_f = filtered.sort_values(ascending=False)
        k = max(1, int(len(sorted_f) * self.quantile))
        top = sorted_f.head(k)
        bot = sorted_f.tail(k)
        # 7) Moyennes simples
        top_avg = top.mean()
        bot_avg = bot.mean()
        # 8) Calcul du spread
        return np.log(top_avg / bot_avg) if bot_avg > 0 else np.nan

    def _generate_sector_signals(
        self,
        sector: str,
        current_date: datetime.datetime
    ) -> Dict[str, float]:
        """
        Génère un dict {ticker: signal} :
        - +1 pour les tickers du top décile (cheapness élevé ⇒ « value »)
        - -1 pour ceux du bottom décile (cheapness faible ⇒ « growth »)
        """
        # 1) Disponibilité et filtre univers
        if current_date not in self.cheapness.index:
            return {}
        s = self.cheapness.loc[current_date].dropna()
        if current_date in self.universe.index:
            mem = self.universe.loc[current_date]
            s = s[s.index.isin(mem[mem==1].index)]
        # 2) Filtrer sur le secteur
        sector_tickers = list(self.dict_sectors[sector].columns)
        s = s[s.index.isin(sector_tickers)]
        # 3) Exclusion outliers
        p5, p95 = s.quantile(0.05), s.quantile(0.95)
        s = s[(s > p5) & (s < p95)]
        if len(s) < 20:
            return {}
        # 4) Tri croissant et décile
        sorted_s = s.sort_values(ascending=True)
        k = max(1, int(len(sorted_s) * self.quantile))
        bottoms = sorted_s.head(k).index
        tops    = sorted_s.tail(k).index
        # 5) Construction des signaux
        signals = {t: 1.0 for t in tops}
        signals.update({t: -1.0 for t in bottoms})
        return signals