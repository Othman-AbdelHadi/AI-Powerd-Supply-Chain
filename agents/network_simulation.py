import pandas as pd
import pydeck as pdk
import streamlit as st

class NetworkSimulationAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def visualize_network(self):
        required_cols = {"source", "destination", "lat", "lon", "type", "lead_time", "cost"}
        missing = required_cols - set(self.df.columns)
        if missing:
            st.warning(f"⚠️ Required columns for network simulation are missing: {', '.join(missing)}")
            return

        # 📍 Overview Map (if location available)
        if {"lat", "lon", "location"}.issubset(self.df.columns):
            st.subheader("📍 Locations Overview")
            st.pydeck_chart(
                pdk.Deck(
                    initial_view_state=pdk.ViewState(
                        latitude=self.df["lat"].mean(),
                        longitude=self.df["lon"].mean(),
                        zoom=5,
                        pitch=40,
                    ),
                    layers=[
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=self.df,
                            get_position='[lon, lat]',
                            get_color='[200, 30, 0, 160]',
                            get_radius=40000,
                        )
                    ],
                )
            )

        # 🧠 Build Arc Data
        route_data = self.df.groupby(["source", "destination"]).agg({
            "lead_time": "mean",
            "cost": "sum",
            "lat": "first",
            "lon": "first"
        }).reset_index()

        if route_data.empty:
            st.info("ℹ️ No route data found for simulation.")
            return

        arcs = []
        for _, row in route_data.iterrows():
            src = self.df[self.df["source"] == row["source"]]
            dst = self.df[self.df["destination"] == row["destination"]]
            if not src.empty and not dst.empty:
                arcs.append({
                    "from_lat": src["lat"].values[0],
                    "from_lon": src["lon"].values[0],
                    "to_lat": dst["lat"].values[0],
                    "to_lon": dst["lon"].values[0],
                    "lead_time": row["lead_time"],
                    "cost": row["cost"]
                })

        if not arcs:
            st.info("ℹ️ No valid arcs generated for simulation.")
            return

        arc_layer = pdk.Layer(
            "ArcLayer",
            data=arcs,
            get_source_position=["from_lon", "from_lat"],
            get_target_position=["to_lon", "to_lat"],
            get_width=2,
            get_tilt=15,
            get_source_color=[0, 255, 0, 100],
            get_target_color=[255, 0, 0, 100],
            pickable=True,
            auto_highlight=True
        )

        view_state = pdk.ViewState(latitude=31.9, longitude=35.9, zoom=5, pitch=45)

        st.subheader("🗭 Supply Chain Network Simulation")
        st.pydeck_chart(pdk.Deck(
            layers=[arc_layer],
            initial_view_state=view_state,
            tooltip={"text": "Lead Time: {lead_time} days\nCost: ${cost}"}
        ))

        st.success("✅ Network visualization completed")

        # 📊 Top Risky Routes
        top_routes = route_data.sort_values(by=["lead_time", "cost"], ascending=False).head(5)
        st.subheader("🔍 Top Risky Routes (High Cost or Lead Time)")
        st.dataframe(top_routes)