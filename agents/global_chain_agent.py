import openai
import os

class GlobalChainAgent:
    def __init__(self, df):
        """
        Initialize the agent with a shipment dataframe.
        """
        self.df = df

    def analyze_complexity(self, shipment_id: str, model="gpt-4o"):
        """
        Analyze global risks and complexity for a given shipment ID using GPT.
        Returns a markdown-formatted analysis message.
        """
        try:
            # ✅ Ensure API Key is loaded
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or not api_key.startswith("sk-"):
                raise ValueError("❌ Missing or invalid OpenAI API key.")
            openai.api_key = api_key

            # ✅ Check if 'shipment_id' column exists
            if "shipment_id" not in self.df.columns:
                return "⚠️ Column 'shipment_id' is missing from the dataset."

            # ✅ Filter the shipment
            shipment = self.df[self.df["shipment_id"].astype(str) == shipment_id]
            if shipment.empty:
                return f"❌ Shipment ID `{shipment_id}` not found."

            data_text = shipment.to_string(index=False)

            # ✅ Detect language (Arabic or English)
            is_arabic = any('\u0600' <= c <= '\u06FF' for c in data_text)

            # ✅ Prompt template
            prompt = f"""
You are a global supply chain analyst. Analyze the following shipment and return risk assessment and mitigation:

Shipment Data:
{data_text}

Structure your response under:
- Potential Global Risks
- Recommended Mitigation Actions

{"Respond in Arabic." if is_arabic else "Respond in English."}
"""

            # ✅ Call GPT
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional global supply chain advisor."},
                    {"role": "user", "content": prompt}
                ]
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"❌ Error during global complexity analysis: {e}"