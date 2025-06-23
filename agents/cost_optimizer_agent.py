import openai
import os
import pandas as pd

class CostOptimizerAgent:
    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    def analyze_cost_scenarios(self, purchase_cost: float, storage_cost: float, shipping_cost: float, delay_penalty: float, model="gpt-4"):
        """
        Use GPT to analyze cost breakdown and suggest improvements.
        """
        try:
            # ✅ Load API key securely
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or not api_key.startswith("sk-"):
                raise RuntimeError("❌ API Key is missing or invalid. Please check your credentials.")
            openai.api_key = api_key

            # ✅ Check required columns
            if self.df is None or "quantity_delivered" not in self.df.columns or "delay_days" not in self.df.columns:
                return "⚠️ Required columns 'quantity_delivered' and 'delay_days' not found."

            # ✅ Handle NaN safely
            df_safe = self.df.copy()
            df_safe["quantity_delivered"] = pd.to_numeric(df_safe["quantity_delivered"], errors="coerce").fillna(0)
            df_safe["delay_days"] = pd.to_numeric(df_safe["delay_days"], errors="coerce").fillna(0)

            # ✅ Calculate key values
            total_units = df_safe["quantity_delivered"].sum()
            delayed_units = df_safe[df_safe["delay_days"] > 0]["quantity_delivered"].sum()

            total_cost = (
                total_units * purchase_cost +
                total_units * storage_cost +
                total_units * shipping_cost +
                delayed_units * delay_penalty
            )

            # ✅ Prepare prompt for GPT
            prompt = f"""
You are a supply chain cost optimization assistant.
Given the following parameters:
- Total Units Delivered: {total_units}
- Delayed Units: {delayed_units}
- Purchase Cost per Unit: ${purchase_cost}
- Storage Cost per Unit: ${storage_cost}
- Shipping Cost per Unit: ${shipping_cost}
- Delay Penalty per Unit: ${delay_penalty}
- Total Estimated Cost: ${total_cost:,.2f}

Analyze the cost drivers and provide 3 actionable strategies to reduce the total cost while maintaining service level.
"""

            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert in supply chain financial optimization."},
                    {"role": "user", "content": prompt}
                ]
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"❌ GPT cost analysis failed: {e}"

    def optimize_costs(self):
        """
        Perform static cost optimization analysis without using GPT.
        """
        suggestions = []

        if self.df is None or self.df.empty:
            return ["⚠️ No data provided for analysis."]

        # Check average ordering cost
        if "ordering_cost" in self.df.columns:
            avg_order = self.df["ordering_cost"].mean()
            if avg_order > 100:
                suggestions.append("📉 Consider negotiating supplier contracts to reduce ordering cost.")

        # Check average holding cost
        if "holding_cost" in self.df.columns:
            avg_hold = self.df["holding_cost"].mean()
            if avg_hold > 20:
                suggestions.append("🏬 Optimize warehouse layout to reduce holding cost.")

        # Fallback if no risks found
        if not suggestions:
            suggestions.append("✅ No major cost risks found in static analysis.")

        return suggestions

    def run(self, user_input, df=None, params=None):
        """
        Unified interface for use inside AI chat assistant.
        - user_input: prompt entered by user
        - df: optional new dataframe
        - params: dict with cost inputs
        """
        if df is not None:
            self.df = df

        if self.df is None:
            return "⚠️ Please upload shipment data including 'quantity_delivered' and 'delay_days'."

        is_arabic = any('\u0600' <= c <= '\u06FF' for c in user_input)
        response = "💰 **Cost Optimization Analysis**\n\n" if not is_arabic else "💰 **تحليل تحسين التكلفة**\n\n"

        if params and all(k in params for k in ["purchase_cost", "storage_cost", "shipping_cost", "delay_penalty"]):
            analysis = self.analyze_cost_scenarios(
                purchase_cost=params["purchase_cost"],
                storage_cost=params["storage_cost"],
                shipping_cost=params["shipping_cost"],
                delay_penalty=params["delay_penalty"]
            )
            response += analysis
        else:
            # Static fallback
            suggestions = self.optimize_costs()
            for i, sug in enumerate(suggestions, 1):
                response += f"{i}. {sug}\n"

        return response