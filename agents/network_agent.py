# agents/network_agent.py

import pandas as pd
import pydeck as pdk
import streamlit as st

class NetworkAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def simulate_network(self):
        # Check for required columns
        required_cols = {"source", "destination", "lat", "lon", "lead_time", "cost"}
        if not required_cols.issubset(self.df.columns):
            st.warning(f"⚠️ Required columns for network simulation are missing: {required_cols - set(self.df.columns)}")
            return

        # Display locations on map (scatter layer)
        if {"lat", "lon", "location"}.issubset(self.df.columns):
            st.subheader("📍 Network Node Locations")
            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=self.df["lat"].mean(),
                    longitude=self.df["lon"].mean(),
                    zoom=5,
                    pitch=45,
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=self.df,
                        get_position='[lon, lat]',
                        get_color='[0, 150, 255, 160]',
                        get_radius=50000,
                        pickable=True,
                    )
                ]
            ))

        # Prepare arcs between source and destination
        routes = []
        for _, row in self.df.iterrows():
            source_df = self.df[self.df["location"] == row["source"]]
            dest_df = self.df[self.df["location"] == row["destination"]]

            if not source_df.empty and not dest_df.empty:
                routes.append({
                    "from_lat": source_df.iloc[0]["lat"],
                    "from_lon": source_df.iloc[0]["lon"],
                    "to_lat": dest_df.iloc[0]["lat"],
                    "to_lon": dest_df.iloc[0]["lon"],
                    "lead_time": row.get("lead_time", 0),
                    "cost": row.get("cost", 0)
                })

        if not routes:
            st.info("ℹ️ No valid source-destination pairs to draw arcs.")
            return

        arc_layer = pdk.Layer(
            "ArcLayer",
            data=routes,
            get_source_position=["from_lon", "from_lat"],
            get_target_position=["to_lon", "to_lat"],
            get_width=2,
            get_tilt=15,
            get_source_color=[0, 255, 0, 120],
            get_target_color=[255, 0, 0, 120],
            pickable=True,
            auto_highlight=True
        )

        view_state = pdk.ViewState(
            latitude=self.df["lat"].mean(),
            longitude=self.df["lon"].mean(),
            zoom=5,
            pitch=50,
        )

        st.subheader("🛍️ Supply Chain Network Flow")
        st.pydeck_chart(pdk.Deck(layers=[arc_layer], initial_view_state=view_state))

        # Show top risk routes
        st.subheader("🚨 Top Risky Routes")
        risky = self.df.sort_values(by=["cost", "lead_time"], ascending=False).head(5)
        st.dataframe(risky[["source", "destination", "cost", "lead_time"]], use_container_width=True)

        st.success("✅ Network simulation completed.")