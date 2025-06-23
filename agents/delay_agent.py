import pandas as pd

class DelayRootCauseAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def analyze(self):
        """
        Analyze potential root causes of shipment delays.
        Supports location, supplier, region, delay reason, product_type, and delay distribution.
        """
        if "delay_days" not in self.df.columns:
            return {"error": "Missing 'delay_days' column in dataset."}

        delayed_df = self.df[self.df["delay_days"] > 0]
        total_delays = len(delayed_df)
        if total_delays == 0:
            return {"note": "✅ No delayed shipments found."}

        causes = {}

        # 1. Top Delayed Locations
        if "location" in delayed_df.columns:
            loc_counts = delayed_df["location"].value_counts().head(3)
            causes["Top Delayed Locations"] = {
                loc: f"{count} delays ({(count/total_delays*100):.1f}%)" for loc, count in loc_counts.items()
            }

        # 2. Top Delayed Suppliers
        if "supplier" in delayed_df.columns:
            sup_counts = delayed_df["supplier"].value_counts().head(3)
            causes["Top Delayed Suppliers"] = {
                sup: f"{count} delays ({(count/total_delays*100):.1f}%)" for sup, count in sup_counts.items()
            }

        # 3. Delay Duration Distribution
        delay_bins = pd.cut(delayed_df["delay_days"], bins=[0, 2, 5, 10, 20, 100], include_lowest=True)
        dist = delay_bins.value_counts().sort_index()
        dist.index = dist.index.astype(str)
        causes["Delay Duration Distribution"] = dist.to_dict()

        # 4. Top Delay Reasons
        if "delay_reason" in delayed_df.columns:
            reason_counts = delayed_df["delay_reason"].value_counts().head(3)
            causes["Top Delay Reasons"] = {
                reason: f"{count} occurrences ({(count/total_delays*100):.1f}%)" for reason, count in reason_counts.items()
            }

        # 5. Top Delayed Regions
        if "region" in delayed_df.columns:
            region_counts = delayed_df["region"].value_counts().head(3)
            causes["Top Delayed Regions"] = {
                region: f"{count} delays ({(count/total_delays*100):.1f}%)" for region, count in region_counts.items()
            }

        # 6. Top Delayed Product Types
        if "product_type" in delayed_df.columns:
            product_counts = delayed_df["product_type"].value_counts().head(3)
            causes["Top Delayed Product Types"] = {
                prod: f"{count} delays ({(count/total_delays*100):.1f}%)" for prod, count in product_counts.items()
            }

        return causes


# Wrapper class to use inside chat system or UI
class DelayAgent:
    def __init__(self):
        self.agent_name = "Delay Analysis Agent"

    def run(self, user_input, df=None):
        if df is None or "delay_days" not in df.columns:
            return "⚠️ Please upload shipment data that includes a 'delay_days' column."

        root_agent = DelayRootCauseAgent(df)
        results = root_agent.analyze()

        if "error" in results:
            return f"❌ {results['error']}"
        if "note" in results:
            return results["note"]

        response = "📊 **Delay Root Cause Analysis**\n\n"

        # Build smart summary message
        section_labels = {
            "Top Delayed Locations": "🏙️ **Most Delayed Locations**",
            "Top Delayed Suppliers": "🏢 **Suppliers with Most Delays**",
            "Delay Duration Distribution": "⏱️ **Delay Duration Distribution**",
            "Top Delay Reasons": "⚠️ **Primary Delay Reasons**",
            "Top Delayed Regions": "🗺️ **Most Affected Regions**",
            "Top Delayed Product Types": "📦 **Affected Product Types**"
        }

        for key, label in section_labels.items():
            if key in results:
                response += f"{label}:\n"
                for k, v in results[key].items():
                    response += f"- {k}: {v}\n"
                response += "\n"

        return response