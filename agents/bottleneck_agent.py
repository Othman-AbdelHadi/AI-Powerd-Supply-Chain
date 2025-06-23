import pandas as pd  # ✅ Ensure pandas is available for datetime parsing and analysis

def detect_bottlenecks(df):
    try:
        # ✅ Basic validation
        if df.empty or "actual_delivery_date" not in df.columns or "planned_delivery_date" not in df.columns:
            return "⚠️ Missing required date columns to detect bottlenecks."

        # ✅ Parse dates if not already
        df["actual_delivery_date"] = pd.to_datetime(df["actual_delivery_date"], errors="coerce")
        df["planned_delivery_date"] = pd.to_datetime(df["planned_delivery_date"], errors="coerce")

        # ✅ Filter delayed shipments
        delayed = df[df["actual_delivery_date"] > df["planned_delivery_date"]]
        if delayed.empty:
            return "✅ No bottlenecks detected. All shipments were delivered on time."

        # ✅ Group by actual delivery day
        grouped = delayed.groupby("actual_delivery_date").size().sort_values(ascending=False)
        peak_day = grouped.idxmax()
        peak_count = grouped.max()

        # ✅ Identify worst suppliers / products if available
        insights = ""
        if "supplier" in delayed.columns:
            top_supplier = delayed["supplier"].value_counts().idxmax()
            insights += f"\n🔍 Top delayed supplier: **{top_supplier}**"

        if "product_type" in delayed.columns:
            top_product = delayed["product_type"].value_counts().idxmax()
            insights += f"\n📦 Most delayed product type: **{top_product}**"

        return (
            f"⏱️ Detected **{len(delayed)}** delayed shipments.\n"
            f"📅 Peak delay occurred on **{peak_day.date()}** with **{peak_count}** shipments delayed."
            f"{insights}"
        )

    except Exception as e:
        return f"❌ Bottleneck detection failed: {str(e)}"