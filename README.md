# Indoor Environmental Quality Forecasting for Smart Aquaculture

This project focuses on **Indoor Environmental Quality (IEQ)** in closed aquaculture facilities.  
IEQ is crucial for:

- maintaining **fish welfare**,
- ensuring **worker comfort**, and  
- sustaining **high productivity**.

In many facilities, IEQ is still monitored **reactively**: operators only intervene **after** conditions have deteriorated. This project aims to change that by enabling **proactive**, data-driven control.

---

## Project Objectives

The main objective is to design and evaluate a **data-driven forecasting and control system** that:

1. **Forecasts IEQ variables** up to **1 hour in advance**.
2. **Uses these forecasts** to support **proactive ventilation and control decisions**.
3. **Operates under real conditions** in a smart aquaculture facility in **Morocco**.

The system performs **multi-variate forecasting**, predicting several IEQ variables simultaneously.

---

## Forecasted IEQ Variables

The model predicts the following IEQ variables:

- Temperature  
- Relative humidity  
- CO₂  
- VOCs (Volatile Organic Compounds)  
- PM2.5 (fine particulate matter)  
- PM10 (coarse particulate matter)  
- A **synthetic IEQ score** (aggregated indicator of overall indoor environmental quality)

Each prediction is generated **one hour ahead**, based on the previous **six hours** of data.

---

## Deep Learning Architectures

A key objective of this project is to **compare three deep-learning architectures** under **identical conditions**:

1. **LSTM (Long Short–Term Memory network)**
2. **TCN (Temporal Convolutional Network)**
3. **Transformer-based model**


---

Dashboard Link: https://ieqforecast.streamlit.app/
