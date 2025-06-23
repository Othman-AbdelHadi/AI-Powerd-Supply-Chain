# agents/shipment_agent.py

import pandas as pd
import numpy as np

class ShipmentAgent:
    def __init__(self, df):
        self.df = df.copy()

    def get_all_shipments(self):
        return self.df

    def get_eta_summary(self):
        return self.df.groupby("location")["delay_days"].mean().reset_index()

    def get_current_status_by_location(self):
        return self.df.groupby("location")["status"].value_counts().unstack(fill_value=0).reset_index()

    def get_delayed_shipments(self):
        return self.df[self.df["delay_days"] > 0]

    def get_today_shipments_count(self):
        if "date" in self.df.columns:
            today = pd.Timestamp.today().date()
            return len(self.df[pd.to_datetime(self.df["date"]).dt.date == today])
        return 0

    def get_total_delay_percentage(self):
        total = len(self.df)
        delayed = len(self.df[self.df["delay_days"] > 0])
        return (delayed / total * 100) if total > 0 else 0

    def get_best_and_worst_suppliers(self):
        """
        Returns the best (least average delay) and worst (most average delay) supplier.
        """
        delay_by_supplier = self.df.groupby("supplier")["delay_days"].mean()
        if delay_by_supplier.empty:
            return None, None
        best = delay_by_supplier.idxmin()
        worst = delay_by_supplier.idxmax()
        return best, worst

    def detect_delay_anomalies(self):
        """
        Detects anomalies in delay days using z-score.
        Returns a DataFrame with anomalies.
        """
        if "delay_days" not in self.df.columns or len(self.df) < 5:
            return pd.DataFrame()

        delay_mean = self.df["delay_days"].mean()
        delay_std = self.df["delay_days"].std()

        if delay_std == 0:
            return pd.DataFrame()

        self.df["z_score"] = (self.df["delay_days"] - delay_mean) / delay_std
        anomalies = self.df[np.abs(self.df["z_score"]) > 2]
        return anomalies[["location", "supplier", "delay_days", "z_score"]]

    def suggest_root_causes(self):
        """
        Suggests possible root causes for delay based on patterns in data.
        Returns a summary DataFrame.
        """
        delayed_df = self.df[self.df["delay_days"] > 2]  # Focus on notable delays
        if delayed_df.empty:
            return pd.DataFrame({"Insight": ["No significant delays found."]})

        common_locations = delayed_df["location"].value_counts().head(3).index.tolist()
        common_suppliers = delayed_df["supplier"].value_counts().head(3).index.tolist()

        insights = []
        if common_locations:
            insights.append(f"Most delays are from locations: {', '.join(common_locations)}")
        if common_suppliers:
            insights.append(f"Frequent delay from suppliers: {', '.join(common_suppliers)}")

        return pd.DataFrame({"Insight": insights})
