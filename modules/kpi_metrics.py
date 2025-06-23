import pandas as pd

def calculate_kpis(df: pd.DataFrame) -> dict:
    """
    Calculate core supply chain KPIs from shipment data.
    
    :param df: Pandas DataFrame with shipment records
    :return: Dictionary with KPI values
    """
    # Convert dates safely
    df["planned_delivery_date"] = pd.to_datetime(df["planned_delivery_date"], errors='coerce')
    df["actual_delivery_date"] = pd.to_datetime(df["actual_delivery_date"], errors='coerce')

    # Calculate on-time and in-full delivery
    df["on_time"] = df["planned_delivery_date"] >= df["actual_delivery_date"]
    df["in_full"] = df["quantity_delivered"] >= df["quantity_ordered"]

    total_orders = len(df)
    total_ordered = df["quantity_ordered"].sum()
    total_delivered = df["quantity_delivered"].sum()

    # OTIF (On-Time In-Full)
    otif = ((df["on_time"] & df["in_full"]).sum() / total_orders) * 100 if total_orders > 0 else 0

    # Fill Rate
    fill_rate = (total_delivered / total_ordered) * 100 if total_ordered > 0 else 0

    # Inventory Turnover
    if "cost_of_goods_sold" in df.columns and "inventory" in df.columns and df["inventory"].mean() > 0:
        inventory_turnover = df["cost_of_goods_sold"].sum() / df["inventory"].mean()
    else:
        inventory_turnover = None

    # Lead Time (average delivery gap)
    if "actual_delivery_date" in df.columns and "planned_delivery_date" in df.columns:
        lead_time = (df["actual_delivery_date"] - df["planned_delivery_date"]).dt.days.mean()
    else:
        lead_time = None

    # Return all KPIs rounded and labeled
    return {
        "otif": round(otif, 2),
        "fill_rate": round(fill_rate, 2),
        "inventory_turnover": round(inventory_turnover, 2) if inventory_turnover else "N/A",
        "lead_time": round(lead_time, 2) if lead_time is not None else "N/A",
    }