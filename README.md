# Linear Regression Architecture Workshop

## 📌 Project Overview
This repository contains a comprehensive **Predictive Maintenance System** for industrial robotics, developed as part of the Foundations of Machine Learning Frameworks Workshop. The project implements a custom-built Linear Regression architecture to monitor and predict the health of robot axis currents, specifically targeting **Axis #1** to prevent mechanical failure through automated early warning alerts.

## 🛠️ Work Completed
### 1. Data Ingestion & Engineering
- **Hybrid Data Loader**: Developed a flexible pipeline in `src/data_loader.py` that supports seamless switching between local CSV files and remote **PostgreSQL (Neon)** databases.
- **Advanced Preprocessing**: Implemented data cleaning and **Moving Average Smoothing** to filter high-frequency noise from telemetry sensors.

### 2. Core Machine Learning
- **Manual Implementation**: Built a Linear Regression model from scratch using Gradient Descent (no high-level ML libraries used for the core solver).
- **Mathematical Validation**: Verified model parameters ($\theta_0$ intercept and $\theta_1$ slope) against industry-standard benchmarks.

### 3. MLOps Integration
- **Config-Driven Design**: Integrated YAML-based configurations to manage hyperparameters (Learning Rate, Iterations) and maintenance thresholds without modifying code.
- **Experiment Tracking**: Conducted hyperparameter sweeps to evaluate the impact of different smoothing windows on **Test RMSE**.
- **RUL Prediction**: Developed a logic to calculate **Remaining Useful Life (RUL)** and triggered automated alerts based on a 14-day lead time buffer.

## 📊 Key Design Decisions
- **Decoupling Concerns**: Separated data, configuration, and model logic to ensure high maintainability and scalability, following **Agile** development principles.
- **Noise Reduction**: Prioritized robust smoothing techniques (**Window Size: 500**) after empirical testing proved it significantly reduced **Test RMSE (0.7027)**, leading to more reliable failure predictions.
- **Defensive Programming**: Implemented automated directory management and model serialization (Pickle) to ensure the system is deployment-ready.

## 🚀 How to Run
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/yourusername/LinearRegressionArchitecture_Workshop.git](https://github.com/yourusername/LinearRegressionArchitecture_Workshop.git)