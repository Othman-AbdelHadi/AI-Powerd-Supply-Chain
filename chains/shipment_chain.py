# chains/shipment_chain.py

import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

load_dotenv()

def get_shipment_analysis_chain(model_name="gpt-4", temperature=0.3):
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt = PromptTemplate.from_template("""
📦 Analyze the following shipment data:
{shipment_data}

Your response must include:
- 📌 Key Delay Findings
- 🔍 Root Causes
- ✅ Recommended Actions (logistics, routing, supplier)
Please respond in bullet points.
""")

    return LLMChain(llm=llm, prompt=prompt)

print("✅ Shipment Analysis Chain Loaded.")