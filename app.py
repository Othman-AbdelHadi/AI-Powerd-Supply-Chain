# ✅ Final: Streamlit App — All Features Fixed & Enhanced

import os
import io
import pandas as pd
import pydeck as pdk
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
import openai
from fpdf import FPDF
import re

# === Agents and Modules ===
from agents.shipment_agent import ShipmentAgent
from agents.demand_agent import DemandAgent
from agents.supplier_agent import SupplierAgent
from agents.delay_agent import DelayRootCauseAgent
from agents.recommendation_agent import RecommendationAgent
from agents.data_preprocessing_agent import DataPreprocessingAgent
from agents.network_agent import NetworkAgent
from agents.risk_agent import RiskAgent
from agents.forecast_agent import AdvancedForecastAgent
from agents.inventory_agent import InventoryOptimizationAgent
from agents.bottleneck_agent import detect_bottlenecks
from agents.global_complexity_agent import GlobalComplexityAgent
from agents.vendor_recommender_agent import VendorRecommenderAgent
from agents.cost_optimizer_agent import CostOptimizerAgent
from agents.network_simulation import NetworkSimulationAgent
from modules.kpi_metrics import calculate_kpis
from chat_memory import init_db
from utils.agent_validator import AgentValidator

# === Setup ===
st.set_page_config(page_title="Supply Chain AI", layout="wide")
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
init_db()


import streamlit as st
import openai
import os
from dotenv import load_dotenv
load_dotenv()

# === Load API Key from .env ===
default_key = os.getenv("OPENAI_API_KEY", "")

import streamlit as st
import openai
import os
from dotenv import load_dotenv
load_dotenv()

# === Load API Key securely from .env ===
default_key = os.getenv("OPENAI_API_KEY", "")

import streamlit as st
import openai
from io import BytesIO
from fpdf import FPDF
import pandas as pd
import zipfile

import streamlit as st
import openai
from fpdf import FPDF
from io import BytesIO
import pandas as pd
import zipfile

# === Sidebar Layout ===
with st.sidebar:
    st.markdown("## 🤖 AI Settings")

    # === GPT Model & API ===
    st.markdown("### 🧠 GPT Model")
    model = st.selectbox("Choose GPT Model", ["gpt-4o", "gpt-3.5-turbo"], index=0)

    st.markdown("### 🔐 OpenAI API Key")
    default_key = "sk-proj-..."  # Secure placeholder
    api_key = st.text_input("OpenAI API Key", type="password", value=default_key, key="api_key_input")

    if st.button("🔌 Connect"):
        if api_key.startswith("sk-"):
            openai.api_key = api_key
            st.session_state.api_connected = True
            st.success("✅ Connected to OpenAI.")
        else:
            st.session_state.api_connected = False
            st.warning("⚠️ Invalid API Key")

    # === Language Selection ===
    st.markdown("---")
    st.markdown("### 🌐 Language")
    if "lang" not in st.session_state:
        st.session_state.lang = "English"
    lang = st.selectbox("Select Language", ["English", "Arabic"], index=0 if st.session_state.lang == "English" else 1)
    st.session_state.lang = lang

    # === Tools Section ===
    st.markdown("---")
    st.markdown("### 🛠️ Tools")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧹 Reset"):
            st.session_state.messages = [{"role": "system", "content": "You are a supply chain expert assistant."}]
            st.success("✅ Chat reset.")

    with col2:
        if st.button("📤 Export"):
            try:
                df = st.session_state.get("df", pd.DataFrame([{"No": "data"}]))
                buffer = BytesIO()
                with zipfile.ZipFile(buffer, "w") as zip_file:
                    zip_file.writestr("report.csv", df.to_csv(index=False))
                    excel_io = BytesIO()
                    with pd.ExcelWriter(excel_io, engine="xlsxwriter") as writer:
                        df.to_excel(writer, index=False)
                    zip_file.writestr("report.xlsx", excel_io.getvalue())
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    for _, row in df.iterrows():
                        line = ", ".join([f"{k}: {v}" for k, v in row.items()])
                        pdf.multi_cell(0, 10, line)
                    pdf_bytes = pdf.output(dest='S').encode("latin1")
                    zip_file.writestr("report.pdf", pdf_bytes)
                st.download_button("⬇️ Download All", data=buffer.getvalue(), file_name="shipment_reports.zip", mime="application/zip")
            except Exception as e:
                st.error(f"❌ Export failed: {e}")

    # === Footer ===
    st.markdown(
        """
        <hr style="margin-top: 25px; margin-bottom: 5px;">
        <div style='text-align: center; color: #aaa; font-size: 12px;'>Othman AbdelHadi</div>
        """,
        unsafe_allow_html=True
    )

# === Styling Section ===
st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background-color: #1c1c1c !important;
        padding: 1rem;
        color: white;
    }
    .stTextInput > div > div > input {
        background-color: #2e2e2e !important;
        color: white !important;
    }
    div[data-testid="stForm"] button, button[kind="secondary"], .stButton>button {
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 0.84rem;
    }
    button {
        border: 1px solid rgba(255,255,255,0.15);
    }
    .stDownloadButton, .stButton {
        margin-top: 0.3rem;
        margin-bottom: 0.3rem;
    }
    hr {
        border-top: 1px solid #444;
    }
    </style>
""", unsafe_allow_html=True)

# === Title ===
st.markdown("""
    <h1 style='text-align: center; color: white; font-size: 2.5rem; margin-bottom: 1rem;'>
         AI-Powered Supply Chain
    </h1>
""", unsafe_allow_html=True)

# --- Agents ---
from agents.demand_agent import DemandAgent
from agents.supplier_agent import SupplierAgent
from agents.delay_agent import DelayAgent
from agents.forecast_agent import ForecastAgent
from agents.cost_optimizer_agent import CostOptimizerAgent




# --- Initialize Session ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a smart assistant specialized in Supply Chain."}
    ]

# --- Language Detection ---
def update_system_message(input_text):
    is_arabic = bool(re.search("[\u0600-\u06FF]", input_text))
    new_content = (
        "أنت مساعد ذكي متخصص في سلاسل الإمداد. أجب باللغة العربية وقدِم تحليلات دقيقة ومقترحات مهنية عند الحاجة."
        if is_arabic else
        "You are a smart assistant specialized in Supply Chain. Respond in English with deep insights and recommendations."
    )
    st.session_state.messages[0]["content"] = new_content
    return is_arabic

# --- Agent Routing ---
def route_to_agent(user_input: str):
    lower = user_input.lower()
    if "demand" in lower or "طلب" in lower:
        return demand_agent
    elif "supplier" in lower or "مورد" in lower:
        return supplier_agent
    elif "delay" in lower or "تأخير" in lower or "late" in lower:
        return delay_agent
    elif "forecast" in lower or "تنبؤ" in lower:
        return forecast_agent
    elif "cost" in lower or "تكلفة" in lower or "scenario" in lower:
        return cost_agent
    else:
        return None  # fallback to GPT

# --- Initialize Agents ---
demand_agent = DemandAgent()
supplier_agent = SupplierAgent()
delay_agent = DelayAgent()
forecast_agent = ForecastAgent()
cost_agent = CostOptimizerAgent()

# --- Chat Input Section ---
from openai import OpenAI
import re
import streamlit as st
from agents.demand_agent import DemandAgent
from agents.supplier_agent import SupplierAgent
from agents.delay_agent import DelayAgent
from agents.forecast_agent import ForecastAgent
from agents.cost_optimizer_agent import CostOptimizerAgent

# === Initialize Message Memory ===
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a smart assistant specialized in Supply Chain."}
    ]

# === Language Awareness ===
def update_system_message(input_text):
    is_arabic = bool(re.search("[\u0600-\u06FF]", input_text))
    new_content = (
        "أنت مساعد ذكي متخصص في سلاسل الإمداد. أجب باللغة العربية وقدِم تحليلات دقيقة ومقترحات مهنية عند الحاجة."
        if is_arabic else
        "You are a smart assistant specialized in Supply Chain. Respond in English with deep insights and recommendations."
    )
    st.session_state.messages[0]["content"] = new_content
    return is_arabic

# === Agent Routing ===
def route_to_agent(user_input: str):
    lower = user_input.lower()
    if "demand" in lower or "طلب" in lower:
        return demand_agent
    elif "supplier" in lower or "مورد" in lower:
        return supplier_agent
    elif "delay" in lower or "تأخير" in lower or "late" in lower:
        return delay_agent
    elif "forecast" in lower or "تنبؤ" in lower:
        return forecast_agent
    elif "cost" in lower or "تكلفة" in lower or "scenario" in lower:
        return cost_agent
    else:
        return None  # fallback to GPT

# === Initialize AI Agents ===
demand_agent = DemandAgent()
supplier_agent = SupplierAgent()
delay_agent = DelayAgent()
forecast_agent = ForecastAgent()
cost_agent = CostOptimizerAgent()

# === Title (only once) ===
st.subheader("🧠 AI Chat Assistant")

# === Chat Form UI ===
with st.form(key="chat_form"):
    chat_input = st.text_input("Ask about shipments, suppliers, delays, or performance...", key="chatbox")
    submit = st.form_submit_button("📩 Send")

# === Process User Input ===
if submit and chat_input:
    update_system_message(chat_input)
    st.session_state.messages.append({"role": "user", "content": chat_input})

    with st.chat_message("user"):
        st.markdown(chat_input)

    selected_agent = route_to_agent(chat_input)

    if selected_agent:
        try:
            reply = selected_agent.run(chat_input)
        except Exception as e:
            reply = f"⚠️ Agent Error: {str(e)}"
    else:
        try:
            client = OpenAI(api_key=st.session_state.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4",
                messages=st.session_state.messages,
                temperature=0.4
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f"⚠️ GPT Error: {str(e)}"

    # Append and display assistant reply
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)



# === Upload ===
st.subheader("📂 Upload a Shipment CSV file")
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df  # حفظ الداتا في session_state
        st.success("✅ File uploaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to read file: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error reading CSV file: {e}")
        st.stop()
    if openai.api_key:
        st.session_state.df = DataPreprocessingAgent(st.session_state.df).validate_and_format(api_key=openai.api_key)
        st.success("✅ Data processed with GPT and cleaned")
elif os.path.exists("data/shipments.csv"):
    st.session_state.df = pd.read_csv("data/shipments.csv")
    if openai.api_key:
        st.session_state.df = DataPreprocessingAgent(st.session_state.df).validate_and_format(api_key=openai.api_key)
else:
    st.session_state.df = pd.DataFrame({
        "supplier": ["Aramex", "DHL", "UPS"],
        "location": ["Amman", "Irbid", "Aqaba"],
        "delay_days": [0, 5, 3],
        "status": ["On Time", "Delayed", "Delayed"],
        "eta": pd.date_range(datetime.today(), periods=3),
        "lat": [31.95, 32.55, 29.53],
        "lon": [35.91, 35.85, 35.0],
        "cost": [100, 120, 110],
        "inventory": [50, 40, 60],
        "demand": [70, 80, 65]
    })

st.session_state.df = st.session_state.df.loc[:, ~st.session_state.df.columns.duplicated()]
df = st.session_state.df

# === Ensure ETA Format ===
if "eta" in df.columns:
    df["eta"] = pd.to_datetime(df["eta"], errors="coerce")
else:
    st.warning("⚠️ 'eta' column not found. Some forecasting features may not work.")


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from agents.kpi_gpt_agent import KPIAgent

# Optional fallback sample data
def load_sample_df():
    return pd.DataFrame({
        "planned_delivery_date": pd.date_range(start="2024-01-01", periods=10, freq="D"),
        "actual_delivery_date": pd.date_range(start="2024-01-02", periods=10, freq="D"),
        "quantity_ordered": [100, 150, 80, 200, 120, 90, 160, 130, 180, 110],
        "quantity_delivered": [90, 140, 80, 190, 120, 90, 150, 130, 170, 110],
        "cost_of_goods_sold": [5000]*10,
        "inventory": [1000]*10,
        "delay_days": [1, 2, 0, 1, 0, 0, 2, 1, 1, 0],
        "supplier": ["A", "B", "A", "B", "C", "C", "A", "C", "B", "A"],
        "route": ["R1", "R1", "R2", "R3", "R2", "R2", "R3", "R1", "R1", "R3"],
        "eta": pd.date_range(start="2024-01-01", periods=10, freq="D")
    })

# Load uploaded file or fallback
if "uploaded_df" in st.session_state:
    df = st.session_state.uploaded_df
else:
    df = load_sample_df()
import streamlit as st
import pandas as pd
from agents.kpi_gpt_agent import KPIAgent  # Ensure this is updated to support OpenAI 1.0+

# === Fallback demo data ===
def load_sample_df():
    return pd.DataFrame({
        "planned_delivery_date": pd.date_range(start="2024-01-01", periods=10, freq="D"),
        "actual_delivery_date": pd.date_range(start="2024-01-02", periods=10, freq="D"),
        "quantity_ordered": [100, 120, 90, 130, 110, 150, 95, 100, 105, 115],
        "quantity_delivered": [100, 110, 90, 120, 100, 150, 90, 100, 100, 110],
        "cost_of_goods_sold": [5000] * 10,
        "inventory": [1000] * 10
    })



# === Safe KPI calculation ===
from openai import OpenAI
from agents.kpi_gpt_agent import KPIAgent
from fpdf import FPDF
from io import BytesIO
import pandas as pd
import streamlit as st
import plotly.express as px

# === KPI Calculation ===
from openai import OpenAI
from agents.kpi_gpt_agent import KPIAgent
from fpdf import FPDF
from io import BytesIO
import pandas as pd
import streamlit as st
import plotly.express as px

# === KPI Calculation ===
def calculate_kpis(df: pd.DataFrame) -> dict:
    required = ["planned_delivery_date", "actual_delivery_date", "quantity_ordered", "quantity_delivered"]
    if any(col not in df.columns for col in required):
        return {"error": "Missing required columns for KPI calculation."}

    df["planned_delivery_date"] = pd.to_datetime(df["planned_delivery_date"], errors='coerce')
    df["actual_delivery_date"] = pd.to_datetime(df["actual_delivery_date"], errors='coerce')
    df["on_time"] = df["planned_delivery_date"] >= df["actual_delivery_date"]
    df["in_full"] = df["quantity_delivered"] >= df["quantity_ordered"]

    total_orders = len(df)
    total_ordered = df["quantity_ordered"].sum()
    total_delivered = df["quantity_delivered"].sum()

    otif = ((df["on_time"] & df["in_full"]).sum() / total_orders) * 100 if total_orders else 0
    fill_rate = (total_delivered / total_ordered) * 100 if total_ordered else 0

    inventory_turnover = "N/A"
    if "cost_of_goods_sold" in df.columns and "inventory" in df.columns and df["inventory"].mean() > 0:
        inventory_turnover = round(df["cost_of_goods_sold"].sum() / df["inventory"].mean(), 2)

    lead_time = (df["actual_delivery_date"] - df["planned_delivery_date"]).dt.days.mean()
    lead_time = round(lead_time, 2) if pd.notna(lead_time) else "N/A"

    return {
        "otif": round(otif, 2),
        "fill_rate": round(fill_rate, 2),
        "inventory_turnover": inventory_turnover,
        "lead_time": lead_time
    }

# === Load Data ===
df = st.session_state.get("uploaded_df", load_sample_df())
kpis = calculate_kpis(df)

# === Show KPI Metrics above the expander ===
if "error" in kpis:
    st.error(kpis["error"])
else:
    st.markdown("### 🔹 Key Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 OTIF", f"{kpis['otif']}%")
    col2.metric("🎯 Fill Rate", f"{kpis['fill_rate']}%")
    col3.metric("📊 Turnover", kpis["inventory_turnover"])
    col4.metric("⏱️ Lead Time", f"{kpis['lead_time']} days")

# === Executive KPI Dashboard with Charts, GPT and Export ===
with st.expander("📊 Executive KPI Dashboard", expanded=False):
    # OTIF Trend
    st.markdown("#### 📈 OTIF Monthly Trend")
    try:
        df["month"] = pd.to_datetime(df["actual_delivery_date"], errors="coerce").dt.to_period("M")
        trend_df = df.groupby("month").apply(
            lambda d: ((d["planned_delivery_date"] >= d["actual_delivery_date"]) &
                       (d["quantity_delivered"] >= d["quantity_ordered"])).sum() / len(d) * 100
        ).reset_index()
        trend_df.columns = ["Month", "OTIF"]
        trend_df["Month"] = trend_df["Month"].astype(str)
        fig = px.line(trend_df, x="Month", y="OTIF", markers=True, title="OTIF % by Month")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("📌 Not enough data to visualize OTIF trend.")

    # Recommendations
    st.markdown("#### 🧠 Auto Recommendations")
    if kpis["otif"] < 85:
        st.warning("🔸 OTIF is below 85%. Investigate delivery bottlenecks or supplier issues.")
    if kpis["fill_rate"] < 90:
        st.warning("🔸 Fill Rate is low. Improve order fulfillment accuracy.")
    if isinstance(kpis["inventory_turnover"], (int, float)) and kpis["inventory_turnover"] < 2:
        st.info("🔄 Low Inventory Turnover → Possible excess stock.")
    if isinstance(kpis["lead_time"], (int, float)) and kpis["lead_time"] > 5:
        st.info("⏱️ Long lead time. Check supply chain responsiveness.")

    # GPT KPI Explanation
    st.markdown("#### 🤖 GPT KPI Explanation")
    user_input = st.text_input("Ask GPT to explain the KPIs", value="Explain KPIs")
    if user_input:
        try:
            agent = KPIAgent()
            explanation = agent.run(user_input, kpis)
            st.markdown(explanation)

            # Export PDF
            if st.button("📥 Download KPI Report PDF"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 8, "📊 KPI Executive Summary\n\n", align="L")
                for key, value in kpis.items():
                    pdf.multi_cell(0, 8, f"{key.upper()}: {value}", align="L")
                pdf.ln(4)
                pdf.multi_cell(0, 8, explanation, align="L")
                buffer = BytesIO()
                st.download_button(
                    label="📄 Export KPI Summary PDF",
                    data=buffer.getvalue(),
                    file_name="kpi_summary.pdf",
                    mime="application/pdf"
                )
        except Exception as e:
            st.error(f"⚠️ GPT KPI Analysis Error: {e}")


# === 🧠 Multi-Agent Workflow Engine ===

from chains.agent_flow import get_agent_flow

with st.expander("🧠 Multi-Agent Supply Chain Flow", expanded=False):
    st.markdown("Automatically coordinate between agents to forecast, detect delays, evaluate suppliers, and recommend actions.")

    if st.button("🚀 Run Full Agent Flow"):
        try:
            state = {
                "df": df,
                "demand_data": [120, 130, 125, 128, 140],  # Sample demand history
            }
            flow = get_agent_flow()
            final_output = flow.invoke(state)

            st.success("✅ Multi-Agent Flow Completed")
            st.markdown("### 📈 Forecasted Demand")
            st.write(final_output.get("forecast"))

            st.markdown("### 🧾 Supplier Evaluation")
            st.write(final_output.get("supplier_eval"))

            st.markdown("### ⚠️ Delay Root Causes")
            st.write(final_output.get("delay_causes"))

            st.markdown("### 📌 Recommendations")
            for r in final_output.get("recommendations", []):
                st.markdown(f"- {r}")

        except Exception as e:
            st.error(f"❌ Flow execution failed: {e}")
            
# --- Root Cause Analysis of Delays ---
from fpdf import FPDF
from io import BytesIO
import pandas as pd
import streamlit as st
from agents.delay_agent import DelayRootCauseAgent

# --- Load DataFrame (real or fallback)
if "uploaded_df" in st.session_state:
    df = st.session_state.uploaded_df.copy()
else:
    df = pd.DataFrame({
        "location": ["A", "B", "A", "C", "A", "B", "C", "A", "B", "C"],
        "supplier": ["X", "Y", "X", "Z", "X", "Y", "Z", "X", "Y", "Z"],
        "delay_days": [2, 0, 5, 0, 3, 0, 6, 0, 1, 0],
        "delay_reason": ["Weather", "None", "Customs", "None", "Inventory", "None", "Inventory", "None", "Traffic", "None"],
        "product_type": ["Electronics", "Furniture", "Electronics", "Food", "Electronics", "Furniture", "Food", "Electronics", "Furniture", "Food"],
        "region": ["North", "South", "North", "West", "North", "South", "West", "North", "South", "West"]
    })

# --- Delay Root Cause Analysis ---

# --- Delay Root Cause Analysis ---
from fpdf import FPDF
from io import BytesIO
import pandas as pd
import streamlit as st
import plotly.express as px
from agents.delay_agent import DelayRootCauseAgent

# --- Load fallback or uploaded DataFrame ---
if "uploaded_df" in st.session_state:
    df = st.session_state.uploaded_df.copy()
else:
    df = pd.DataFrame({
        "location": ["A", "B", "A", "C", "A", "B", "C", "A", "B", "C"],
        "supplier": ["X", "Y", "X", "Z", "X", "Y", "Z", "X", "Y", "Z"],
        "delay_days": [2, 0, 5, 0, 3, 0, 6, 0, 1, 0],
        "delay_reason": ["Weather", "None", "Customs", "None", "Inventory", "None", "Inventory", "None", "Traffic", "None"],
        "product_type": ["Electronics", "Furniture", "Electronics", "Food", "Electronics", "Furniture", "Food", "Electronics", "Furniture", "Food"],
        "region": ["North", "South", "North", "West", "North", "South", "West", "North", "South", "West"]
    })

if "delay_days" not in df.columns:
    st.warning("⚠️ 'delay_days' column not found. Using fallback data.")
    df["delay_days"] = [2, 0, 5, 0, 3, 0, 6, 0, 1, 0]

# --- Agent Processing ---
st.markdown("### 🕵️ Delay Root Cause Analysis")

try:
    agent = DelayRootCauseAgent(df)
    cause_summary = agent.analyze()

    if not any(cause_summary.get(k) for k in cause_summary if k != "error"):
        st.info("✅ No delayed shipments found. Everything looks on time.")
    else:
        keys_to_show = {
            "Top Delayed Locations": "🏙️ Locations",
            "Top Delayed Suppliers": "🏢 Suppliers",
            "Top Delay Reasons": "⚠️ Reasons",
            "Top Delayed Regions": "🗺️ Regions",
            "Top Delayed Product Types": "📦 Product Types"
        }

        display_data = {}
        charts = {}

        for key, label in keys_to_show.items():
            if key in cause_summary:
                for sub_key, value in cause_summary[key].items():
                    display_data.setdefault(label, {})[sub_key] = value
                charts[label] = pd.Series(cause_summary[key])

        # === Main Executive Table ===
        main_table = pd.DataFrame(display_data).fillna("").astype(str)
        st.dataframe(main_table, use_container_width=True)

        # === Extra Charts + PDF ===
        with st.expander("📊 More Insights & PDF Export", expanded=False):
            for title, series in charts.items():
                chart_type = "pie" if "Reasons" in title or "Product" in title else "bar"
                df_chart = series.reset_index()
                df_chart.columns = ["Label", "Count"]

                st.markdown(f"**📈 {title} Breakdown**")
                if chart_type == "pie":
                    fig = px.pie(df_chart, values="Count", names="Label", title=title)
                else:
                    fig = px.bar(df_chart, x="Label", y="Count", title=title, text="Count")
                st.plotly_chart(fig, use_container_width=True)

            # === PDF Export ===
            if st.button("📥 Download PDF Summary"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)

                for label, series in charts.items():
                    pdf.cell(200, 10, txt=label, ln=True)
                    for k, v in series.items():
                        pdf.cell(200, 8, txt=f"- {k}: {v}", ln=True)
                    pdf.ln(4)

                buffer = BytesIO()
                pdf.output(buffer)
                st.download_button(
                    label="📄 Save Full Report as PDF",
                    data=buffer.getvalue(),
                    file_name="delay_root_cause_report.pdf",
                    mime="application/pdf"
                )

except Exception as e:
    st.error(f"❌ Delay analysis failed: {e}")

# ✅ Smart Descriptive Analytics & Intelligent Recommendations for Supply Chain
import streamlit as st
import pandas as pd
import numpy as np
from agents.recommendation_agent import RecommendationAgent
from agents.supplier_agent import SupplierAgent
from agents.delay_agent import DelayRootCauseAgent


df = st.session_state.df.copy()



# === GPT Summary ===
from openai import OpenAI
from fpdf import FPDF
from io import BytesIO
import pandas as pd
import streamlit as st

# === GPT Summary ===
st.subheader("📊 GPT Summary & Trends")

use_mock = False  # fallback switch

try:
    api_key = st.session_state.get("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        use_mock = True
        raise ValueError("No API key provided")

    client = OpenAI(api_key=api_key)

    filtered_shipments = df.copy()
    preview = filtered_shipments.head(10).to_string(index=False)

    summary_prompt = f"""
You are a supply chain analyst.
Given the first 10 rows of this shipment dataset, provide a short executive summary and highlight 2 key insights:

{preview}
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a supply chain analyst."},
            {"role": "user", "content": summary_prompt}
        ],
        temperature=0.4
    )

    summary_text = response.choices[0].message.content.strip()

except Exception:
    # Fallback summary if API fails or missing
    use_mock = True
    summary_text = """
📋 Sample GPT Summary:

- The dataset indicates consistent supplier activity with minor delays.
- Most delays are concentrated around a specific product and region.
- Consider forecasting demand better and rescheduling certain routes.
"""

# Display summary
st.success(summary_text.strip())

# Export to PDF
with st.expander("📤 Export GPT Summary"):
    if st.button("📄 Download GPT Summary as PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in summary_text.strip().split("\n"):
            pdf.multi_cell(0, 8, txt=line.strip())
        buffer = BytesIO()
        pdf.output(buffer)
        st.download_button(
            label="📥 Save Summary as PDF",
            data=buffer.getvalue(),
            file_name="gpt_summary_report.pdf",
            mime="application/pdf"
        )

# === Shipment Trends ===
st.subheader("📈 Shipment Trends")
try:
    df["month"] = pd.to_datetime(df["eta"]).dt.to_period("M")
    trend_data = df.groupby("month").size()
    st.line_chart(trend_data)
except:
    st.info("📌 No 'eta' column found — using fallback data.")
    fallback = pd.DataFrame({
        "eta": pd.date_range("2024-01-01", periods=6, freq="M"),
        "shipments": [12, 17, 10, 18, 21, 15]
    })
    fallback["month"] = fallback["eta"].dt.to_period("M")
    trend_data = fallback.groupby("month")["shipments"].sum()
    st.line_chart(trend_data)

    
# === Inventory Optimization Agent Demo ===
from fpdf import FPDF
from io import BytesIO
import pandas as pd
import streamlit as st
from agents.inventory_agent import InventoryOptimizationAgent

# === Inventory Optimization Section ===
with st.expander("📦 Inventory Optimization", expanded=False):
    try:
        if df is None or df.empty:
            st.warning("⚠️ No inventory data found. Using sample fallback data.")
            df = pd.DataFrame({
                "supplier": ["XCorp", "ZLog", "YTrans"],
                "location": ["Amman", "Irbid", "Aqaba"],
                "delay_days": [3, 5, 2],
                "status": ["Delivered", "Delayed", "Delayed"],
                "annual_demand": [1200, 800, 200],
                "ordering_cost": [40, 50, 60],
                "holding_cost": [10, 15, 12],
                "unit_cost": [30, 35, 25],
                "lead_time_days": [7, 10, 5],
                "item_name": ["Item A", "Item B", "Item C"]
            })

        inventory_agent = InventoryOptimizationAgent(df)

        # EOQ calculator
        st.subheader("📦 EOQ & Reorder Point Calculator")
        col1, col2, col3 = st.columns(3)
        with col1:
            demand_rate = st.number_input("📈 Annual Demand", min_value=1, value=1000)
        with col2:
            ordering_cost = st.number_input("📦 Ordering Cost", min_value=1.0, value=50.0)
        with col3:
            holding_cost = st.number_input("🏬 Holding Cost", min_value=1.0, value=10.0)

        if st.button("🔍 Calculate EOQ"):
            eoq = inventory_agent.calculate_eoq(demand_rate, ordering_cost, holding_cost)
            st.success(f"✅ EOQ: {eoq} units")

        # Summary table
        st.markdown("### 📊 Inventory Metrics Summary")
        metrics_df = inventory_agent.calculate_metrics()
        st.dataframe(metrics_df, use_container_width=True)

        # ABC table
        st.subheader("🔠 ABC Inventory Classification")
        abc_df = inventory_agent.abc_analysis()
        st.dataframe(abc_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Inventory Optimization Error: {e}")

# === Charts + Export (only if metrics_df exists) ===
if "metrics_df" in locals():
    inventory_agent = InventoryOptimizationAgent(df)
    abc_df = inventory_agent.abc_analysis()

    st.markdown("---")

    with st.expander("📈 EOQ Chart by Item"):
        fig_eoq = inventory_agent.plot_eoq(metrics_df)
        st.plotly_chart(fig_eoq, use_container_width=True)

    with st.expander("📊 ABC Pie Chart"):
        fig_abc = inventory_agent.plot_abc(abc_df)
        st.plotly_chart(fig_abc, use_container_width=True)

    with st.expander("📤 Export Inventory Reports"):
        st.download_button("📥 Download Metrics CSV", metrics_df.to_csv(index=False).encode("utf-8"), file_name="inventory_metrics.csv")
        st.download_button("📥 Download ABC CSV", abc_df.to_csv(index=False).encode("utf-8"), file_name="abc_classification.csv")

        # Dead / Overstock detection
        dead_stock = metrics_df[metrics_df["annual_demand"] < 300]
        over_stock = metrics_df[metrics_df["Total_Needed"] > 1000]

        if st.button("📝 Export Inventory Summary PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            # EOQ Section
            pdf.cell(200, 10, txt="📦 EOQ Summary", ln=True)
            for _, row in metrics_df.iterrows():
                pdf.cell(200, 8, txt=f"{row['item_name']}: EOQ = {row['EOQ']:.2f}, ROP = {row['Reorder_Point']:.2f}, Safety = {row['Safety_Stock']:.2f}", ln=True)

            # ABC Section
            pdf.ln(4)
            pdf.cell(200, 10, txt="🔠 ABC Classification", ln=True)
            for _, row in abc_df.iterrows():
                pdf.cell(200, 8, txt=f"{row['item_name']}: Class = {row['ABC_Class']}", ln=True)

            # Dead Stock Section
            pdf.ln(4)
            pdf.cell(200, 10, txt="🧊 Dead Stock Items (< 300 demand)", ln=True)
            for _, row in dead_stock.iterrows():
                pdf.cell(200, 8, txt=f"{row['item_name']}: {row['annual_demand']} units", ln=True)

            # Overstock Section
            pdf.ln(2)
            pdf.cell(200, 10, txt="📦 Overstock Items (> 1000 units needed)", ln=True)
            for _, row in over_stock.iterrows():
                pdf.cell(200, 8, txt=f"{row['item_name']}: Needed = {row['Total_Needed']:.2f}", ln=True)

            buffer = BytesIO()
            pdf.output(buffer)
            st.download_button(
                label="📄 Download Inventory PDF",
                data=buffer.getvalue(),
                file_name="inventory_summary.pdf",
                mime="application/pdf"
            )

        


from fpdf import FPDF
from io import BytesIO
import pandas as pd
import streamlit as st
from agents.vendor_recommender_agent import VendorRecommenderAgent

# === 🏅 Vendor Recommendation Engine ===
with st.expander("🏅 Vendor Recommendation Engine", expanded=True):
    try:
        st.markdown("Use AI-powered logic to find the most reliable vendor based on region, product, and performance.")

        # 🔍 Filters
        col1, col2 = st.columns(2)
        region_filter = col1.text_input("🌍 Filter by Region (optional)")
        product_filter = col2.text_input("📦 Filter by Product (optional)")

        # ⏳ GPT Recommendation
        if st.button("🤖 Recommend Best Vendor"):
            agent = VendorRecommenderAgent(df)
            result = agent.recommend_best_vendor(region=region_filter, product=product_filter)
            st.markdown("### 🧠 GPT Vendor Recommendation")
            st.success(result)

            # 📤 Export Recommendation as PDF
            with st.expander("📄 Export Recommendation"):
                if st.button("⬇️ Download Vendor PDF"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 8, "🏅 GPT Vendor Recommendation Summary\n\n", align="L")
                    pdf.multi_cell(0, 8, result.strip(), align="L")
                    buffer = BytesIO()
                    pdf.output(buffer)
                    st.download_button(
                        label="📥 Save Recommendation as PDF",
                        data=buffer.getvalue(),
                        file_name="vendor_recommendation.pdf",
                        mime="application/pdf"
                    )

    except Exception as e:
        st.error(f"❌ Vendor recommendation failed: {e}")


# === Forecasting with Prophet/ARIMA ===
from agents.forecast_agent import AdvancedForecastAgent
import plotly.express as px

with st.expander("📊 Forecast Delays with Prophet or ARIMA", expanded=False):
    try:
        st.markdown("Ensure your data includes: `eta`, `delay_days`.")

        method = st.selectbox("Choose forecasting method", ["prophet", "arima"])

        if st.button("🔮 Run Forecast"):
            if "eta" not in df.columns or "delay_days" not in df.columns:
                st.warning("⚠️ Missing required columns. Generating fallback data...")
                df = pd.DataFrame({
                    "eta": pd.date_range("2025-06-01", periods=20, freq="D"),
                    "delay_days": [2, 0, 1, 3, 2, 1, 0, 4, 1, 3, 2, 0, 0, 1, 2, 3, 1, 0, 2, 1]
                })

            forecast_agent = AdvancedForecastAgent(df)
            forecast_df = forecast_agent.predict(method=method)

            if forecast_df is not None:
                forecast_df["date"] = pd.to_datetime(forecast_df["date"])
                st.success("✅ Forecast completed:")
                st.dataframe(forecast_df)

                fig = px.line(forecast_df, x="date", y="Forecast", title="📈 Delay Forecast")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Not enough data to forecast. Make sure `eta` and `delay_days` exist and have enough rows.")

    except Exception as e:
        st.error(f"❌ Forecast error: {e}")

     


# --- Distribution & Rescheduling Plan Suggestion ---
# === 🚚 Distribution & Rescheduling Strategy ===
from agents.recommendation_agent import RecommendationAgent
from fpdf import FPDF
from io import BytesIO

with st.expander("🚚 Distribution & Rescheduling Strategy", expanded=False):
    st.caption("Leverage delay root cause analysis to craft a smart distribution plan and strategic recommendations.")

    try:
        # ✅ Initialize the recommendation agent
        recommendation_agent = RecommendationAgent(df)

        # ✅ Generate distribution plan
        plan_text = recommendation_agent.suggest_distribution_plan()

        # ✅ Display distribution plan
        st.markdown("### 📦 Suggested Distribution Plan")
        st.markdown(plan_text)

        # ✅ Generate smart recommendations
        smart_recs = recommendation_agent.recommend()
        if smart_recs:
            st.markdown("### 📌 Smart Operational Recommendations")
            for rec in smart_recs:
                st.markdown(f"- {rec}")
        else:
            st.info("✅ No major operational issues found.")

        # ✅ Export all recommendations to PDF
        st.markdown("### 📤 Export Recommendations")
        if st.button("📄 Download PDF Recommendations"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            # 📦 Distribution section
            pdf.multi_cell(0, 10, "🚚 Distribution Plan\n", align="L")
            for line in plan_text.strip().split("\n"):
                pdf.multi_cell(0, 8, line, align="L")

            # 📌 Recommendations
            if smart_recs:
                pdf.ln(4)
                pdf.multi_cell(0, 10, "📌 Operational Recommendations:\n", align="L")
                for line in smart_recs:
                    pdf.multi_cell(0, 8, f"- {line}", align="L")

            buffer = BytesIO()
            pdf.output(buffer)
            st.download_button(
                label="📥 Download Strategy PDF",
                data=buffer.getvalue(),
                file_name="distribution_strategy.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"❌ Distribution strategy generation failed: {e}")

# === Unified Cost Optimization Block ===
from agents.cost_optimizer_agent import CostOptimizerAgent
from fpdf import FPDF
from io import BytesIO
import pandas as pd
import streamlit as st

# Ensure df exists with fallback if necessary
if "df" not in locals() or df is None or df.empty:
    df = pd.DataFrame({
        "supplier": ["A", "B", "C"],
        "quantity_delivered": [100, 80, 90],
        "delay_days": [2, 5, 0],
        "ordering_cost": [120, 90, 140],
        "holding_cost": [30, 25, 35],
    })

# === Unified Cost Optimization Section ===
with st.expander("💲 Cost Optimization & Recommendations", expanded=False):
    try:
        # Initialize the agent
        cost_agent = CostOptimizerAgent(df)

        # Show static cost-saving suggestions
        st.subheader("📉 Static Cost Recommendations")
        static_tips = cost_agent.optimize_costs()
        static_df = pd.DataFrame({"Suggestions": static_tips})
        st.dataframe(static_df, use_container_width=True)

        # Toggle GPT analysis
        st.markdown("---")
        show_gpt = st.checkbox("🤖 Show GPT-4 Cost Analysis")

        if show_gpt:
            # Get cost scenario inputs
            st.markdown("Let GPT analyze cost drivers and suggest reduction strategies.")
            col1, col2 = st.columns(2)
            with col1:
                purchase_cost = st.number_input("Purchase Cost per Unit", min_value=0.0, value=10.0)
                storage_cost = st.number_input("Storage Cost per Unit", min_value=0.0, value=3.0)
            with col2:
                shipping_cost = st.number_input("Shipping Cost per Unit", min_value=0.0, value=5.0)
                delay_penalty = st.number_input("Delay Penalty per Delayed Unit", min_value=0.0, value=8.0)

            if st.button("📊 Analyze with GPT-4"):
                try:
                    api_key = st.session_state.get("OPENAI_API_KEY")
                    if not api_key or not api_key.startswith("sk-"):
                        st.warning("⚠️ Please connect a valid OpenAI API key.")
                        st.stop()

                    # Run GPT-based cost analysis
                    gpt_result = cost_agent.analyze_cost_scenarios(
                        purchase_cost=purchase_cost,
                        storage_cost=storage_cost,
                        shipping_cost=shipping_cost,
                        delay_penalty=delay_penalty
                    )

                    st.success("✅ GPT Cost Analysis Complete")
                    st.markdown(gpt_result)

                    # Export analysis as PDF
                    with st.expander("📤 Export GPT Summary as PDF"):
                        if st.button("📄 Download PDF Report"):
                            pdf = FPDF()
                            pdf.add_page()
                            pdf.set_font("Arial", size=12)
                            pdf.multi_cell(0, 8, txt=gpt_result)
                            buffer = BytesIO()
                            pdf.output(buffer)
                            st.download_button(
                                label="📥 Save Cost Analysis (PDF)",
                                data=buffer.getvalue(),
                                file_name="gpt_cost_analysis.pdf",
                                mime="application/pdf"
                            )

                except Exception as e:
                    st.error(f"❌ GPT Cost Analysis Failed: {e}")

    except Exception as e:
        st.error(f"❌ Cost Optimization Error: {e}")



# === ⚠️ Risk Simulation & Delay Probability ===
from agents.risk_agent import RiskAgent
from fpdf import FPDF
from io import BytesIO

with st.expander("⚠️ Risk Simulation & Delay Probability", expanded=False):
    try:
        risk_agent = RiskAgent(df)
        result = risk_agent.simulate_delay_risk(iterations=1000, threshold=5)

        if "error" in result:
            st.error(result["error"])
        else:
            st.markdown("### 📉 Risk Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Delay", f"{result['mean_delay']} days")
            col2.metric("Max Delay", f"{result['max_delay']} days")
            col3.metric(f"Risk > {result['threshold']}d", f"{result['risk_probability']}%")

            st.markdown("### 📊 Simulation Delay Distribution")
            st.line_chart(result["simulation_series"])

            # Export PDF
            if st.button("📄 Export Risk Report as PDF"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, "⚠️ Risk Simulation Summary\n\n")
                pdf.cell(200, 8, f"Mean Delay: {result['mean_delay']} days", ln=True)
                pdf.cell(200, 8, f"Max Delay: {result['max_delay']} days", ln=True)
                pdf.cell(200, 8, f"Risk Probability (> {result['threshold']} days): {result['risk_probability']}%", ln=True)

                buffer = BytesIO()
                pdf.output(buffer)
                st.download_button(
                    label="📥 Download PDF Report",
                    data=buffer.getvalue(),
                    file_name="risk_simulation_report.pdf",
                    mime="application/pdf"
                )

    except Exception as e:
        st.error(f"❌ Risk simulation failed: {e}")



# === Network Simulation ===
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from agents.network_simulation import NetworkSimulationAgent

# === Section Header ===
st.subheader("🛍️ Supply Chain Network Simulation")

try:
    # ✅ Fallback check
    if df is None or df.empty or not {"source", "destination", "quantity"}.issubset(df.columns):
        st.warning("⚠️ Required columns not found. Showing demo network instead.")

        demo_data = pd.DataFrame({
            "source": ["Factory", "Factory", "Warehouse A", "Warehouse B"],
            "destination": ["Warehouse A", "Warehouse B", "Retailer 1", "Retailer 2"],
            "quantity": [100, 120, 80, 60]
        })

        with st.expander("📍 Demo Supply Chain Network", expanded=True):
            fig = go.Figure()

            for _, row in demo_data.iterrows():
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[hash(row["source"]) % 10, hash(row["destination"]) % 10],
                    mode="lines+text",
                    line=dict(width=row["quantity"] / 10, color="#00BFFF"),  # 🟦 Soft blue
                    text=[row["source"], row["destination"]],
                    textposition="bottom center",
                    opacity=0.8
                ))

            fig.update_layout(
                title="Demo Supply Chain Network",
                showlegend=False,
                height=500,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="#111111",  # Dark background
                plot_bgcolor="#111111",
                font=dict(color="white")
            )

            st.plotly_chart(fig, use_container_width=True)

            # ✅ Export PNG
            try:
                png_bytes = fig.to_image(format="png")
                st.download_button(
                    label="📥 Export as PNG",
                    data=png_bytes,
                    file_name="demo_network_map.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"❌ PNG export failed: {e}")

            # ✅ Export PDF
            try:
                pdf_bytes = fig.to_image(format="pdf")
                st.download_button(
                    label="📄 Export as PDF",
                    data=pdf_bytes,
                    file_name="demo_network_map.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"❌ PDF export failed: {e}")

    else:
        # ✅ Real data case
        NetworkSimulationAgent(df).visualize_network()

except Exception as e:
    st.error(f"❌ Network simulation failed: {e}")
    
# === Supplier Search ===
# === Advanced Supplier Search ===
with st.expander("📋 Search Suppliers", expanded=False):
    try:
        st.markdown("Use the search box to filter suppliers by ID, name, region, or status.")

        # Search box
        query = st.text_input("🔍 Enter any supplier detail (name, ID, region, status):")
        filtered_suppliers = df.copy()

        # Dynamic filtering based on available columns
        if query:
            filter_mask = pd.Series([False] * len(df))
            for col in ["supplier", "location", "region", "status"]:
                if col in df.columns:
                    filter_mask |= df[col].astype(str).str.contains(query, case=False, na=False)
            filtered_suppliers = df[filter_mask]

        # If result is empty, notify user
        if filtered_suppliers.empty:
            st.info("🔎 No suppliers match your search query.")
        else:
            # Optional: Delay visualization
            if "supplier" in filtered_suppliers.columns and "delay_days" in filtered_suppliers.columns:
                st.markdown("### 📊 Average Delay by Supplier")
                delay_chart = filtered_suppliers.groupby("supplier")["delay_days"].mean()
                st.bar_chart(delay_chart)

            # Show results
            st.markdown("### 🧾 Matching Suppliers")
            st.dataframe(filtered_suppliers, use_container_width=True)

            # Export results to Excel
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                filtered_suppliers.to_excel(writer, index=False)
            st.download_button(
                label="📥 Download Excel",
                data=excel_buffer.getvalue(),
                file_name="filtered_suppliers.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"❌ Supplier search failed: {e}")

import pydeck as pdk
import pandas as pd
from io import BytesIO
from openai import OpenAI

with st.expander("🗺️ Track Shipments on Map", expanded=False):
    try:
        # Use fallback if no valid shipment data
        if df is None or df.empty or "lat" not in df.columns or "lon" not in df.columns:
            st.warning("⚠️ No valid shipment data found. Showing demo sample.")
            df_map = pd.DataFrame({
                "shipment_id": ["Sample-1", "Sample-2"],
                "lat": [31.9516, 32.0],
                "lon": [35.9230, 36.0],
                "status": ["In Transit", "Delayed"],
                "eta": ["2025-06-22", "2025-06-25"],
                "location": ["Amman", "Zarqa"],
                "supplier": ["Alpha Logistics", "Beta Transport"]
            })
        else:
            df_map = df.copy()
            df_map["shipment_id"] = (
                df_map["shipment_id"].astype(str)
                if "shipment_id" in df.columns
                else df_map.get("supplier", "Unknown").astype(str)
            )

        # 🔍 Flexible multi-field search (ID, supplier, location)
        query = st.text_input("🔍 Search by Shipment ID, Supplier, or Location")
        if query:
            mask = pd.Series([False] * len(df_map))
            for col in ["shipment_id", "supplier", "location"]:
                if col in df_map.columns:
                    mask |= df_map[col].astype(str).str.contains(query, case=False, na=False)
            shipment = df_map[mask]
        else:
            shipment = df_map

        # Show map if data exists
        if not shipment.empty:
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/dark-v9",
                initial_view_state=pdk.ViewState(
                    latitude=shipment["lat"].mean(),
                    longitude=shipment["lon"].mean(),
                    zoom=5,
                    pitch=45,
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=shipment,
                        get_position='[lon, lat]',
                        get_color='[0, 200, 255, 160]',
                        get_radius=5000,
                        pickable=True
                    )
                ]
            ))

            # Metrics for single shipment
            if len(shipment) == 1:
                row = shipment.iloc[0]
                col1, col2, col3 = st.columns(3)
                col1.metric("📦 Status", row.get("status", "N/A"))
                col2.metric("📅 ETA", row.get("eta", "N/A"))
                col3.metric("📍 Location", row.get("location", "N/A"))

                # 🧠 GPT Insight Button
                if st.button("🧠 Analyze Shipment with GPT"):
                    api_key = st.session_state.get("OPENAI_API_KEY", "")
                    if api_key and api_key.startswith("sk-"):
                        try:
                            client = OpenAI(api_key=api_key)
                            prompt = f"""
You are a supply chain logistics expert. Analyze the following shipment:

- ID: {row.get('shipment_id')}
- Supplier: {row.get('supplier')}
- Status: {row.get('status')}
- ETA: {row.get('eta')}
- Location: {row.get('location')}

Offer a short operational insight or recommendation.
"""
                            response = client.chat.completions.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": "You are a supply chain expert."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.5
                            )
                            st.success(response.choices[0].message.content.strip())
                        except Exception as e:
                            st.warning(f"GPT Error: {e}")
                    else:
                        st.info("💡 Sample Insight:\n\nThis shipment appears to be in progress. Consider checking the ETA against historical averages.")

            # Show shipment data table
            st.markdown("### 📄 Shipment Data")
            st.dataframe(shipment, use_container_width=True)

            # Export shipment results to CSV
            csv_data = shipment.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Shipment CSV",
                data=csv_data,
                file_name="tracked_shipments.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ No shipments matched your query or coordinates are missing.")

    except Exception as e:
        st.error(f"❌ Shipment tracking error: {e}")

# === 🌍 Global Supply Chain Complexity Tool ===
from fpdf import FPDF
from io import BytesIO
import pandas as pd
import streamlit as st

with st.expander("🌍 Global Supply Chain Complexity Tool", expanded=False):
    st.markdown("Analyze complexity & risks of any global shipment.")

    # ✅ Normalize column names to prevent KeyError
    if df is not None and not df.empty:
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    global_df = df.copy() if df is not None and not df.empty else pd.DataFrame({
        "shipment_id": ["GLOB123", "GLOB456"],
        "supplier": ["AsiaTrans", "EuroCargo"],
        "region": ["East Asia", "Western Europe"],
        "risk_factor": ["High Tariff", "Port Congestion"],
        "delay_days": [6, 3],
        "location": ["Shanghai", "Rotterdam"]
    })

    # ✅ Normalize fallback too
    global_df.columns = [col.strip().lower().replace(" ", "_") for col in global_df.columns]

    # === Shipment ID Input ===
    shipment_id = st.text_input("Enter Shipment ID to analyze", key="global_chain_id")
    gpt_output = ""

    if shipment_id:
        if "shipment_id" not in global_df.columns:
            st.error("❌ 'shipment_id' column not found after normalization.")
        else:
            target = global_df[global_df["shipment_id"].astype(str).str.contains(shipment_id, case=False)]

            if not target.empty:
                st.markdown("### 📦 Shipment Snapshot")
                st.dataframe(target, use_container_width=True)

                # === GPT Agent Analysis Button ===
                if st.button("🌍 Analyze Global Complexity"):
                    try:
                        from agents.global_chain_agent import GlobalChainAgent
                        agent = GlobalChainAgent(global_df)
                        result = agent.analyze_complexity(shipment_id, model=st.session_state.get("GPT_MODEL", "gpt-4"))
                        st.success("✅ Agent Analysis Complete")
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"❌ Failed to analyze shipment: {e}")

                # === GPT Manual Prompting ===
                if st.button("🧠 Generate GPT Insight"):
                    try:
                        api_key = st.session_state.get("OPENAI_API_KEY", "")
                        if api_key and api_key.startswith("sk-"):
                            from openai import OpenAI
                            client = OpenAI(api_key=api_key)
                            row = target.iloc[0]
                            prompt = f"""
You are a global supply chain strategist. Analyze the following shipment:

- ID: {row.get("shipment_id")}
- Supplier: {row.get("supplier")}
- Region: {row.get("region")}
- Risk Factor: {row.get("risk_factor")}
- Delay: {row.get("delay_days")} days
- Location: {row.get("location")}

What are the possible risks? Suggest mitigation strategies.
"""
                            response = client.chat.completions.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": "You are a global supply chain strategist."},
                                    {"role": "user", "content": prompt}
                                ]
                            )
                            gpt_output = response.choices[0].message.content.strip()
                            st.markdown("### 🧠 GPT Insight")
                            st.success(gpt_output)
                        else:
                            gpt_output = (
                                "This shipment may face delays due to high tariff exposure. "
                                "Consider splitting cargo across different ports or adjusting suppliers."
                            )
                            st.info(f"💬 Sample GPT Insight:\n\n{gpt_output}")
                    except Exception as e:
                        st.warning(f"⚠️ GPT Analysis Error: {e}")

                # === Export GPT Insight ===
                if gpt_output:
                    def export_gpt_insight_to_pdf(gpt_text: str) -> bytes:
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 8, txt="🌍 GPT Shipment Complexity Insight\n", align="L")
                        pdf.multi_cell(0, 8, txt=gpt_text.strip(), align="L")
                        buffer = BytesIO()
                        pdf.output(buffer)
                        return buffer.getvalue()

                    pdf_bytes = export_gpt_insight_to_pdf(gpt_output)
                    st.download_button(
                        label="📄 Export GPT Insight as PDF",
                        data=pdf_bytes,
                        file_name=f"{shipment_id}_gpt_insight.pdf",
                        mime="application/pdf"
                    )
            else:
                st.info("🔎 No shipment found with that ID.")
    else:
        st.info("💬 Please enter a shipment ID to analyze.")

# === 📋 Data Integrity Validation ===
with st.expander("📋 Data Integrity Validation", expanded=False):
    try:
        # ✅ Check if DataFrame exists
        if df is None or df.empty:
            st.error("❌ Uploaded data is empty.")
        else:
            # ✅ Define required columns
            required_columns = ["supplier", "location", "delay_days", "status"]
            missing_cols = [col for col in required_columns if col not in df.columns]

            # ✅ Validate presence of all required fields
            if missing_cols:
                st.error(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
            else:
                st.success("✅ Data is valid. All required fields are present.")

    except Exception as e:
        st.error(f"❌ Unexpected error during validation: {e}")





# === 📤 Export All Reports (Full Horizontal Layout + ZIP) ===
import io, zipfile
from fpdf import FPDF

st.subheader("📁 Export All Reports")

try:
    export_df = (
        filtered_shipments
        if "filtered_shipments" in locals() and not filtered_shipments.empty
        else df.copy()
        if df is not None and not df.empty
        else pd.DataFrame([{"Message": "No data available"}])
    )

    col1, col2, col3, col4 = st.columns(4)

    # === CSV Export ===
    with col1:
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Download  CSV",
            data=csv,
            file_name="shipment_report.csv",
            mime="text/csv"
        )

    # === Excel Export ===
    with col2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Report")
        st.download_button(
            label="📊 Download Excel",
            data=excel_buffer.getvalue(),
            file_name="shipment_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # === PDF Export ===
    with col3:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        for _, row in export_df.iterrows():  # ✅ Now includes all rows
            line = " | ".join(f"{col}: {val}" for col, val in row.items())
            pdf.multi_cell(0, 8, txt=line)
        pdf_bytes = pdf.output(dest="S").encode("latin1")
        st.download_button(
            label="🧾 Download PDF",
            data=pdf_bytes,
            file_name="shipment_report.pdf",
            mime="application/pdf"
        )

    # === ZIP Export for All Files Together ===
    with col4:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w") as zf:
            zf.writestr("shipment_report.csv", csv)
            zf.writestr("shipment_report.xlsx", excel_buffer.getvalue())
            zf.writestr("shipment_report.pdf", pdf_bytes)

        st.download_button(
            label="📦 Download All as ZIP",
            data=zip_buffer.getvalue(),
            file_name="shipment_reports_bundle.zip",
            mime="application/zip"
        )

except Exception as e:
    st.error(f"❌ Export failed: {e}")




# === Developer Footer ===
st.markdown("---")
st.caption("Version 2.9.3 · GPT Agents, Forecasting & Smart Insights · by Othman AbdelHadi") 





