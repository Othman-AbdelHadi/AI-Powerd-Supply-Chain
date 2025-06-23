import numpy as np
import pandas as pd

class RiskAgent:
    def __init__(self, df):
        self.df = df

    def simulate_delay_risk(self, iterations=1000, threshold=5):
        """
        Perform Monte Carlo simulation to estimate delay risk.
        Returns metrics and distribution series for visualization.

        :param iterations: Number of Monte Carlo simulations
        :param threshold: Delay (in days) considered 'high risk'
        :return: dict with risk metrics and simulation results
        """
        try:
            if self.df is None or "delay_days" not in self.df.columns:
                return {"error": "Missing 'delay_days' column in the dataset."}

            delays = self.df["delay_days"].dropna()
            if delays.empty:
                return {"error": "No delay data available."}

            simulations = [
                np.mean(np.random.choice(delays, size=len(delays)))
                for _ in range(iterations)
            ]
            risk_series = pd.Series(simulations)

            return {
                "simulation_series": risk_series,
                "mean_delay": round(risk_series.mean(), 2),
                "max_delay": round(risk_series.max(), 2),
                "min_delay": round(risk_series.min(), 2),
                "std_delay": round(risk_series.std(), 2),
                "risk_probability": round((risk_series > threshold).mean() * 100, 2),
                "threshold": threshold,
                "original_sample": delays.describe().to_dict()
            }

        except Exception as e:
            return {"error": f"Simulation failed: {e}"}