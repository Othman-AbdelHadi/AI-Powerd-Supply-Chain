import numpy as np
from sklearn.linear_model import LinearRegression

class DemandAgent:
    def __init__(self, sales_history: list = None):
        """
        Agent for forecasting future demand based on historical sales using linear regression.
        :param sales_history: A list of past daily sales numbers.
        """
        self.sales_history = sales_history

    def forecast_next_days(self, num_days: int = 7):
        """
        Predict demand for the next N days using a simple linear regression model.

        :param num_days: Number of future days to predict.
        :return: List of forecasted demand values.
        """
        if not self.sales_history or len(self.sales_history) < 2:
            return [0] * num_days  # Not enough data to forecast

        # Prepare training data
        x = np.arange(len(self.sales_history)).reshape(-1, 1)
        y = np.array(self.sales_history)

        # Train the regression model
        model = LinearRegression()
        model.fit(x, y)

        # Predict future demand values
        future_x = np.arange(len(self.sales_history), len(self.sales_history) + num_days).reshape(-1, 1)
        forecast = model.predict(future_x)

        # Return positive integer forecast values only
        return [max(0, round(value)) for value in forecast]

    def run(self, user_input: str, sales_history=None):
        """
        Run forecast logic from chat interface or Streamlit, returns a markdown string.
        :param user_input: User query or prompt text.
        :param sales_history: Optional list of historical sales data to override the current one.
        :return: Forecast message formatted in Arabic or English.
        """
        if sales_history:
            self.sales_history = sales_history

        if not self.sales_history:
            return "⚠️ Please provide sales history data for demand forecasting."

        forecast = self.forecast_next_days(num_days=7)

        # Detect Arabic language input
        is_arabic = any('\u0600' <= char <= '\u06FF' for char in user_input)
        response = "📈 **Demand Forecast (Next 7 Days)**\n\n" if not is_arabic else "📈 **توقعات الطلب (الأيام السبعة القادمة)**\n\n"

        for i, val in enumerate(forecast, start=1):
            if is_arabic:
                response += f"- اليوم {i}: {val} وحدة\n"
            else:
                response += f"- Day {i}: {val} units\n"

        return response