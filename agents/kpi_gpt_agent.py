from openai import OpenAI
import os

class KPIAgent:
    """
    GPT-powered agent that explains supply chain KPIs.
    Compatible with OpenAI >= 1.0.0
    """

    def __init__(self):
        self.agent_name = "KPI Explainer Agent"
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def explain_kpis(self, kpis: dict, language: str = "en") -> str:
        """
        Build prompt and send request to GPT to explain KPIs.
        :param kpis: Dictionary of calculated KPIs
        :param language: 'en' or 'ar' (auto-detected)
        :return: GPT explanation string
        """
        def val(key):
            return "N/A" if kpis.get(key) in [None, "N/A"] else kpis[key]

        prompt = f"""
You are a supply chain performance analyst. Analyze these KPIs:
- OTIF: {val('otif')}%
- Fill Rate: {val('fill_rate')}%
- Inventory Turnover: {val('inventory_turnover')}
- Avg Lead Time: {val('lead_time')} days

Explain in {"Arabic" if language == "ar" else "English"} what these KPIs indicate.
Suggest one improvement for each metric.
Be concise and helpful.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a supply chain KPI expert."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"⚠️ GPT Agent not active. ({e})"

    def run(self, user_input: str, kpis: dict):
        """
        Public method for external modules to call the KPI explanation.
        Handles language detection and output formatting.
        """
        is_arabic = any('\u0600' <= c <= '\u06FF' for c in user_input)
        language = "ar" if is_arabic else "en"

        explanation = self.explain_kpis(kpis, language=language)

        header = "📊 **KPI Analysis**\n\n" if language == "en" else "📊 **تحليل مؤشرات الأداء**\n\n"
        return header + explanation