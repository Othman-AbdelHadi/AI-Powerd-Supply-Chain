import openai
import os

# Load API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

class GlobalComplexityAgent:
    def __init__(self, df):
        self.df = df

    def analyze_shipment_complexity(self):
        # Extract relevant data summary
        shipment_summary = self.df[['supplier', 'location', 'eta']].head(5).to_dict(orient='records')

        prompt = f"""
You are a global supply chain logistics expert.
You will be provided with sample shipment records including location and ETA.
Your task is to:
1. Identify any potential global logistics complexity (such as customs clearance, cold chain requirements, geopolitical issues).
2. For each case, propose a mitigation strategy.
3. Output should be expert-level but simplified for business stakeholders.

Shipment Data:
{shipment_summary}

Provide the complexity assessment and suggestions:
"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a supply chain AI expert."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Failed to fetch complexity analysis: {e}"
