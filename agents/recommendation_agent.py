import pandas as pd

class RecommendationAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def recommend(self):
        """
        Generate strategic logistics recommendations based on supplier delays,
        location bottlenecks, shipment patterns, and optionally product types.
        """
        recommendations = []

        required = {"delay_days", "supplier", "location", "status"}
        if not required.issubset(self.df.columns):
            missing = required - set(self.df.columns)
            return [f"❌ Missing required columns: {', '.join(missing)}"]

        # Supplier performance analysis
        supplier_stats = self.df.groupby("supplier")["delay_days"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        worst_suppliers = supplier_stats.head(3)
        for supplier, row in worst_suppliers.iterrows():
            recommendations.append(
                f"⛔ Audit supplier **{supplier}** — avg. delay: **{row['mean']:.1f}d**, shipments: {row['count']}."
            )

        # Location bottlenecks
        location_stats = self.df.groupby("location")["delay_days"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        for location, row in location_stats.head(3).iterrows():
            recommendations.append(
                f"📍 Delay hotspot: **{location}** — avg. delay: **{row['mean']:.1f}d**, affected: {row['count']} shipments."
            )

        # Optional: analyze product types if column exists
        if "product_type" in self.df.columns:
            top_products = self.df.groupby("product_type")["delay_days"].mean().sort_values(ascending=False).head(1)
            for product, delay in top_products.items():
                recommendations.append(
                    f"🔍 Product '{product}' is facing delays — avg: **{delay:.1f}d**. Consider specialized logistics."
                )

        # Delay rate analysis
        delayed_count = len(self.df[self.df["status"] == "Delayed"])
        total_shipments = len(self.df)
        if delayed_count / max(total_shipments, 1) > 0.2:
            recommendations.append("🚨 High delay rate detected — prioritize rerouting and faster carriers.")
        elif delayed_count > 5:
            recommendations.append("⚠️ Consider adding backup suppliers or regional distribution centers.")

        # Average delay recommendations
        avg_delay = self.df["delay_days"].mean()
        if avg_delay < 1:
            recommendations.append("✅ Logistics network operating efficiently — keep monitoring KPIs weekly.")
        elif avg_delay >= 3:
            recommendations.append("📦 Consider predictive demand planning to prevent future congestion.")
        if avg_delay >= 5:
            recommendations.append("🚚 Switch to faster transport modes for high-priority routes if possible.")

        return recommendations

    def suggest_distribution_plan(self):
        """
        Suggest a basic distribution plan based on most frequent locations.
        """
        if "location" not in self.df.columns:
            return "❌ Cannot generate distribution plan: 'location' column is missing."

        location_counts = self.df["location"].value_counts()
        if location_counts.empty:
            return "⚠️ No location data available for distribution plan."

        top_locations = location_counts.head(3).index.tolist()
        return f"📦 Suggested distribution plan: prioritize routes through {', '.join(map(str, top_locations))}."