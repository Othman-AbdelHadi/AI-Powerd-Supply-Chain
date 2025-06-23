# agents/forecast_agent.py

import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class AdvancedForecastAgent:
    def __init__(self, df):
        self.df = df.copy()
        self.external_data = None

    def add_external_factors(self, external_df: pd.DataFrame):
        if 'date' in external_df.columns:
            self.external_data = external_df.set_index("date")

    def _prepare_data(self):
        if "eta" not in self.df.columns or "delay_days" not in self.df.columns:
            return None
        df_grouped = self.df.groupby(self.df["eta"].dt.date)["delay_days"].mean().reset_index()
        df_grouped.columns = ["date", "value"]
        return df_grouped

    def predict(self, method="prophet"):
        data = self._prepare_data()
        if data is None or len(data) < 10:
            return None

        if method == "prophet":
            df_prophet = data.rename(columns={"date": "ds", "value": "y"})
            model = Prophet()

            # دمج العوامل الخارجية إن وجدت
            if self.external_data is not None:
                for col in self.external_data.columns:
                    model.add_regressor(col)
                df_prophet = df_prophet.set_index("ds").join(self.external_data, how="left").reset_index()

            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=7)
            if self.external_data is not None:
                future = future.set_index("ds").join(self.external_data, how="left").reset_index()
            forecast = model.predict(future)
            return forecast[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "Forecast"})

        elif method == "arima":
            model = ARIMA(data["value"], order=(2, 1, 2))
            fit = model.fit()
            forecast = fit.forecast(steps=7)
            future_dates = pd.date_range(start=data["date"].iloc[-1] + pd.Timedelta(days=1), periods=7)
            return pd.DataFrame({"date": future_dates, "Forecast": forecast})

        else:
            return None


# 👇 Wrapper class for integration with chat assistant
class ForecastAgent:
    def __init__(self):
        self.agent_name = "Forecast Agent"

    def run(self, user_input, df=None):
        if df is None or "eta" not in df.columns or "delay_days" not in df.columns:
            return "⚠️ Please upload shipment data including 'eta' and 'delay_days' columns."

        # Choose model based on user input
        method = "prophet"
        if "arima" in user_input.lower():
            method = "arima"

        model = AdvancedForecastAgent(df)
        forecast_df = model.predict(method=method)

        if forecast_df is None:
            return "⚠️ Not enough data to make a reliable forecast."

        # Format forecast to markdown response
        response = f"📈 **{method.upper()} Forecast for Delays (Next 7 Days)**\n\n"
        for _, row in forecast_df.iterrows():
            date = pd.to_datetime(row['date']).strftime("%Y-%m-%d")
            value = round(row['Forecast'], 2)
            response += f"- `{date}` → {value} delay days\n"

        return response