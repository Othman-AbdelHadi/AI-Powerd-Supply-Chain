# ✅ Final Safe DataPreprocessingAgent (works with or without API)

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
from langdetect import detect
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataPreprocessingAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.schema_mapping = {}
        self.cleaned_df = pd.DataFrame()

    def detect_language(self, text):
        try:
            return detect(text)
        except:
            return "unknown"

    def get_column_mapping_from_gpt(self, api_key: str) -> dict:
        try:
            from openai import OpenAI
            if not api_key or not api_key.startswith("sk-"):
                raise RuntimeError("❌ API Key is missing or invalid.")
            client = OpenAI(api_key=api_key)
            columns = list(self.df.columns)
            prompt = (
                "You are a data preprocessing assistant. Map the following raw column names to one of "
                "these target names: supplier, location, delay_days, status, eta, lat, lon, cost, inventory, order_qty, lead_time.\n"
                f"Raw columns: {columns}\n"
                "Return only a JSON object like this: {\"RawName\": \"MappedName\"}"
            )
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for data transformation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            reply = response.choices[0].message.content.strip()
            mapping = json.loads(reply)
            if not isinstance(mapping, dict):
                raise ValueError("GPT response is not a dictionary.")
            seen = set()
            cleaned_mapping = {}
            for raw, mapped in mapping.items():
                if mapped in seen:
                    continue
                cleaned_mapping[raw] = mapped
                seen.add(mapped)
            return cleaned_mapping
        except Exception as e:
            print(f"⚠️ GPT mapping failed, using fallback. Reason: {e}")
            return {col: col for col in self.df.columns}

    def unify_column_names(self, api_key: str = None, verbose: bool = False):
        mapping = self.get_column_mapping_from_gpt(api_key)
        if verbose:
            print("🧠 Column Mapping:", mapping)
        self.df.rename(columns=mapping, inplace=True)

    def clean_values(self):
        for col in self.df.columns:
            if any(key in col.lower() for key in ["date", "eta"]):
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
        for col in self.df.select_dtypes(include="object").columns:
            self.df[col] = self.df[col].apply(lambda x: re.sub(r"[^\d\.-]", "", str(x)) if re.match(r"^[\d\.,]+$", str(x)) else x)
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        self.df.drop_duplicates(inplace=True)
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif pd.api.types.is_object_dtype(self.df[col]):
                self.df[col].fillna("unknown", inplace=True)
            else:
                self.df[col].fillna(method="ffill", inplace=True)

    def enrich_features(self):
        if set(["cost", "inventory"]).issubset(self.df.columns):
            self.df["cost_per_unit_moved"] = self.df["cost"] / self.df["inventory"].replace(0, np.nan)
        if "delay_days" in self.df.columns:
            self.df["delivery_efficiency"] = self.df["delay_days"].apply(lambda x: "on_time" if x <= 0 else "delayed")
        if set(["inventory", "order_qty"]).issubset(self.df.columns):
            self.df["stock_turnover"] = self.df["order_qty"] / self.df["inventory"].replace(0, np.nan)
        if set(["lead_time", "order_qty"]).issubset(self.df.columns):
            self.df["reorder_point"] = self.df["lead_time"] * self.df["order_qty"]

    def encode_categoricals(self):
        label_enc = LabelEncoder()
        for col in self.df.select_dtypes(include="object").columns:
            try:
                if self.df[col].nunique() < 50:
                    self.df[col] = label_enc.fit_transform(self.df[col].astype(str))
            except Exception as e:
                print(f"Encoding failed for {col}: {e}")

    def scale_numerics(self):
        scaler = StandardScaler()
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        try:
            scaled = scaler.fit_transform(self.df[num_cols])
            self.df[num_cols] = pd.DataFrame(scaled, columns=num_cols, index=self.df.index)
        except Exception as e:
            print(f"Scaling failed: {e}")

    def summarize_quality(self):
        summary = {}
        for col in self.df.columns:
            summary[col] = {
                "type": str(self.df[col].dtype),
                "nulls": int(self.df[col].isnull().sum()),
                "uniques": int(self.df[col].nunique()),
                "sample": self.df[col].dropna().unique()[:3].tolist()
            }
        return summary

    def validate_and_format(self, api_key: str = None):
        self.unify_column_names(api_key)
        self.clean_values()
        self.enrich_features()
        self.encode_categoricals()
        self.scale_numerics()
        self.cleaned_df = self.df.copy()
        return self.cleaned_df