import pandas as pd

class SupplierAgent:
    """
    Agent responsible for evaluating supplier performance based on shipment delays.
    Supports filtering by country, product type, and service level.
    """

    def __init__(self, shipment_df: pd.DataFrame = None):
        self.df = shipment_df

    def apply_filters(self, filters: dict = None):
        """
        Apply optional filters like country, product_type, and service_level.
        """
        if self.df is None or filters is None:
            return self.df

        filtered_df = self.df.copy()

        if "country" in filters and "country" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["country"] == filters["country"]]

        if "product_type" in filters and "product_type" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["product_type"] == filters["product_type"]]

        if "service_level" in filters and "service_level" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["service_level"] == filters["service_level"]]

        return filtered_df

    def evaluate(self, filters: dict = None):
        """
        Calculates average delay days per supplier and returns performance metrics as a DataFrame.
        """
        if self.df is None:
            return pd.DataFrame([{"error": "Shipment data is missing."}])

        df_filtered = self.apply_filters(filters)

        if "supplier" not in df_filtered.columns or "delay_days" not in df_filtered.columns:
            return pd.DataFrame([{"error": "Required columns missing: 'supplier' or 'delay_days'."}])

        supplier_delay = df_filtered.groupby("supplier")["delay_days"].mean().reset_index()
        supplier_delay.columns = ["Supplier", "Average Delay (days)"]
        supplier_delay = supplier_delay.sort_values("Average Delay (days)")

        return supplier_delay

    def recommend_top_supplier(self, performance_df):
        """
        Recommends the supplier with the lowest average delay.
        """
        if not performance_df.empty and "Supplier" in performance_df.columns:
            return performance_df.iloc[0]["Supplier"]
        return "No suitable supplier found."

    def run(self, user_input, df=None, filters: dict = None):
        """
        Generate supplier performance insights for Streamlit chat assistant.
        :param user_input: Question asked by the user.
        :param df: Optional DataFrame override.
        :param filters: Optional filters (country, product_type, service_level).
        """
        if df is not None:
            self.df = df

        if self.df is None:
            return "⚠️ Please upload shipment data with 'supplier' and 'delay_days' columns."

        performance_df = self.evaluate(filters=filters)

        if "error" in performance_df.columns:
            return f"⚠️ {performance_df.iloc[0]['error']}"

        top_supplier = self.recommend_top_supplier(performance_df)

        is_arabic = any('\u0600' <= char <= '\u06FF' for char in user_input)
        response = "🏢 **Supplier Performance Evaluation**\n\n" if not is_arabic else "🏢 **تقييم أداء الموردين**\n\n"

        for _, row in performance_df.iterrows():
            supplier = row["Supplier"]
            delay = round(row["Average Delay (days)"], 2)
            response += f"- {supplier}: {delay} days avg. delay\n" if not is_arabic else f"- {supplier}: متوسط تأخير {delay} يوم\n"

        response += "\n✅ **Recommended Supplier**: " if not is_arabic else "\n✅ **المورد الموصى به**: "
        response += f"`{top_supplier}`"

        return response


# === Optional test run ===
if __name__ == "__main__":
    sample = pd.DataFrame({
        "supplier": ["A", "B", "A", "C", "B"],
        "delay_days": [1, 5, 2, 0, 6],
        "country": ["US", "US", "CN", "CN", "US"],
        "product_type": ["Electronics", "Electronics", "Furniture", "Furniture", "Electronics"],
        "service_level": ["Standard", "Express", "Standard", "Express", "Express"]
    })

    agent = SupplierAgent(sample)

    print(agent.run("من هو أفضل مورد؟", filters={"country": "US", "service_level": "Express"}))