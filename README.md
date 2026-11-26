---
title: Wafer Defect AI
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: ui/app.py
pinned: false
---

# Wafer‑Defect‑Detection AI Prototype 🚀

An end-to-end AI system for semiconductor wafer defect classification using ResNet-18, LangGraph, MCP, Grad-CAM, and Groq LLM.

## Highlights

- **94.88% validation accuracy** — ResNet-18 fine-tuned on 1,075 wafer images across 5 defect classes
- **Agentic pipeline** — LangGraph `StateGraph` with conditional routing (confidence-based human review, severity-based diagnosis)
- **Explainable AI** — Real Grad-CAM on `layer4[1].conv2` with severity scoring
- **MCP server** — FastMCP tools: `inspect_wafer`, `get_model_metrics`, `ask_wafer_expert`
- **Groq chatbot** — LLM-powered defect Q&A
- **5 defect classes**: deburring, inclusion, rolled_pit, silk_spot, waist_folding

## Architecture

```
Image → MCP Preprocess → ResNet-18 → Grad-CAM → LangGraph Pipeline → Groq Chat
```

## Model Performance

| Metric | Value |
|--------|-------|
| Val Accuracy | **94.88%** |
| Macro F1 | 0.888 |
| Weighted F1 | 0.949 |

## Tech Stack

ResNet-18 · PyTorch · LangGraph · FastMCP · Groq · Grad-CAM · Streamlit · FastAPI
