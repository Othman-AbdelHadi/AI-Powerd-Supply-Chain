import openai
import os
import pandas as pd

class VendorRecommenderAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def recommend_best_vendor(self, region: str = None, product: str = None):
        try:
            api_key = os.getenv("OPENAI_API_KEY")

            # ✅ Use fallback if no key is provided
            if not api_key or not api_key.startswith("sk-"):
                if "supplier" in self.df.columns and "delay_days" in self.df.columns:
                    best = self.df.groupby("supplier")["delay_days"].mean().sort_values().head(1)
                    return f"💡 Suggesting based on internal data: Best performing supplier is **{best.index[0]}** with avg delay of {best.values[0]:.1f} days."
                else:
                    return "❌ API Key is missing or invalid, and fallback cannot proceed due to missing 'supplier' or 'delay_days'."

            openai.api_key = api_key

            # ✅ Filter if region or product specified
            filtered_df = self.df.copy()
            if region and "location" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['location'].str.contains(region, case=False, na=False)]
            if product and "item_name" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['item_name'].str.contains(product, case=False, na=False)]

            # ✅ Validate required columns
            required_cols = ['supplier', 'location', 'delay_days', 'cost', 'status']
            missing_cols = [col for col in required_cols if col not in filtered_df.columns]
            if missing_cols:
                return f"⚠️ Missing required columns: {', '.join(missing_cols)}"

            if filtered_df.empty:
                return "📭 No vendor data available after filtering. Try adjusting region or product filters."

            sample = filtered_df[required_cols].dropna().head(10).to_dict(orient='records')

            # ✅ GPT Prompt
            prompt = f"""
You are an expert supply chain analyst.
Based on the shipment records below, recommend the most reliable supplier using MCDM principles (cost, delays, risk, etc.).
Focus on performance and give a clear justification.

Shipment Data:
{sample}

Provide your ranked supplier recommendation:
"""

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a smart vendor recommender system."},
                    {"role": "user", "content": prompt}
                ]
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"❌ Vendor recommendation failed: {e}"