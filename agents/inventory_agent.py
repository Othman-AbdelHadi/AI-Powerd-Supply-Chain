import pandas as pd
import numpy as np
import plotly.express as px

class InventoryOptimizationAgent:
    """
    Agent for advanced inventory optimization.
    Supports EOQ, Reorder Point, Safety Stock, ABC Analysis, and visualizations.
    """

    def __init__(self, df: pd.DataFrame = None):
        # Use fallback inventory data if df is empty
        if df is None or df.empty:
            self.df = self._generate_mock_inventory()
            self.is_mock_data = True
        else:
            self.df = df.copy()
            self.is_mock_data = False

    def _generate_mock_inventory(self):
        """
        Provide default mock data if no inventory data is provided.
        """
        return pd.DataFrame({
            "item_name": ["Item A", "Item B", "Item C"],
            "annual_demand": [500, 300, 150],
            "unit_cost": [40, 60, 30],
            "ordering_cost": [50, 40, 30],
            "holding_cost": [10, 8, 5],
            "lead_time_days": [7, 10, 5]
        })

    def calculate_metrics(self) -> pd.DataFrame:
        """
        Compute EOQ, Reorder Point, Safety Stock, and Total Needed Inventory.
        """
        df = self.df.copy()

        # Fallback to supplier name if item_name is missing
        if "item_name" not in df.columns:
            df["item_name"] = df["supplier"] if "supplier" in df.columns else "Unknown"

        n = len(df)
        df["annual_demand"] = df.get("annual_demand", pd.Series([100] * n))
        df["unit_cost"] = df.get("unit_cost", pd.Series([50] * n))
        df["ordering_cost"] = df.get("ordering_cost", pd.Series([20] * n))
        df["holding_cost"] = df.get("holding_cost", pd.Series([5] * n))
        df["lead_time_days"] = df.get("lead_time_days", pd.Series([7] * n))

        df["daily_demand"] = df["annual_demand"] / 365
        df["EOQ"] = np.sqrt((2 * df["annual_demand"] * df["ordering_cost"]) / df["holding_cost"])
        df["Reorder_Point"] = df["daily_demand"] * df["lead_time_days"]
        df["Safety_Stock"] = df["daily_demand"] * 0.2 * np.sqrt(df["lead_time_days"])
        df["Total_Needed"] = df["EOQ"] + df["Safety_Stock"]

        return df

    def calculate_eoq(self, demand_rate, ordering_cost, holding_cost):
        """
        Return EOQ given demand rate, ordering cost, and holding cost.
        """
        try:
            eoq = ((2 * demand_rate * ordering_cost) / holding_cost) ** 0.5
            return round(eoq, 2)
        except Exception as e:
            return f"❌ EOQ Error: {str(e)}"

    def calculate_reorder_point(self, daily_demand, lead_time):
        """
        Return reorder point based on daily demand and lead time.
        """
        try:
            return round(daily_demand * lead_time, 2)
        except Exception as e:
            return f"❌ Reorder Point Error: {str(e)}"

    def calculate_safety_stock(self, std_dev_demand, service_factor, lead_time):
        """
        Return safety stock given demand variation, service level, and lead time.
        """
        try:
            safety_stock = service_factor * std_dev_demand * (lead_time ** 0.5)
            return round(safety_stock, 2)
        except Exception as e:
            return f"❌ Safety Stock Error: {str(e)}"

    def abc_analysis(self) -> pd.DataFrame:
        """
        Perform ABC classification based on annual cost.
        """
        df = self.df.copy()

        df["annual_demand"] = df.get("annual_demand", pd.Series([100] * len(df)))
        df["unit_cost"] = df.get("unit_cost", pd.Series([50] * len(df)))
        df["annual_cost"] = df["annual_demand"] * df["unit_cost"]

        df = df.sort_values("annual_cost", ascending=False)
        df["cumulative_cost"] = df["annual_cost"].cumsum()
        df["cumulative_perc"] = 100 * df["cumulative_cost"] / df["annual_cost"].sum()

        def classify(p):
            if p <= 80:
                return "A"
            elif p <= 95:
                return "B"
            else:
                return "C"

        df["ABC_Class"] = df["cumulative_perc"].apply(classify)
        return df

    def plot_abc(self, df: pd.DataFrame):
        """
        Return pie chart of ABC classification.
        """
        fig = px.pie(df, names="ABC_Class", title="ABC Inventory Classification")
        return fig

    def plot_eoq(self, df: pd.DataFrame):
        """
        Return bar chart showing EOQ by item.
        """
        fig = px.bar(df, x="item_name", y="EOQ", title="EOQ by Item", text="EOQ")
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        return fig