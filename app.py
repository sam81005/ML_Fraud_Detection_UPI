import tkinter as tk
import customtkinter as ctk
import pandas as pd
import joblib
from datetime import datetime
import numpy as np

# --- 1. Load the Ultimate ML Model (LightGBM) ---
print("🚀 Launching Ultimate ML Scam Detector...")
try:
    model = joblib.load('lgbm_upi_model.joblib')
    model_columns = joblib.load('lgbm_model_columns.joblib')
    print("✅ LightGBM prediction engine loaded successfully.")
except FileNotFoundError:
    print("\n❌ ERROR: Model files not found! Please run the 'train_model.py' script first.")
    exit()

# --- 2. The Direct Prediction Logic (CORRECTED) ---
def get_prediction(ui_elements):
    try:
        ui_elements['result_title'].configure(text="Analyzing...", text_color="#3498db")
        ui_elements['risk_meter_bar'].set(0)
        ui_elements['risk_meter_bar'].configure(progress_color="#3498db")
        root.update_idletasks()

        # --- A. Collect and Map GUI Inputs to Model Features ---
        amount = float(ui_elements['amount_var'].get())
        
        # Simulate user history for ratio calculation
        avg_tx = 4000
        max_tx = 50000
        
        # Map for transaction velocity
        freq_map = {"Low (1-5)": 3, "Medium (6-15)": 10, "High (16-30)": 23, "Anomalous Burst (30+)": 40}

        # THIS DICTIONARY NOW PERFECTLY MATCHES THE TRAINING DATA
        input_data = {
            'amount': amount,
            'amount_to_avg_ratio': amount / avg_tx,
            'is_new_beneficiary': ui_elements['new_ben_var'].get(), # Direct 0/1 mapping
            'is_new_device': ui_elements['new_dev_var'].get(), # Direct 0/1 mapping
            'tx_velocity_1h': freq_map[ui_elements['high_freq_var'].get()], # Better mapping
            'is_collect_request': 1 if ui_elements['type_var'].get() == 'COLLECT_REQUEST' else 0,
        }
        
        # --- B. Create DataFrame and Align Columns ---
        input_df = pd.DataFrame([input_data])
        final_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        probability = model.predict_proba(final_df)
        scam_prob = probability[0][1]
        
        # --- C. Update GUI with Rich Result ---
        update_risk_meter(scam_prob, ui_elements)
            
    except ValueError:
        ui_elements['result_title'].configure(text="Invalid Input", text_color="#f39c12")
        ui_elements['result_prob'].configure(text="Amount must be a valid number.")
    except Exception as e:
        ui_elements['result_title'].configure(text="Error", text_color="#e74c3c")
        ui_elements['result_prob'].configure(text=str(e))

# --- 3. 3-Tier Classification Logic ---
def update_risk_meter(scam_prob, ui_elements):
    HIGH_RISK_THRESHOLD = 0.70
    LOW_RISK_THRESHOLD = 0.30
    
    if scam_prob > HIGH_RISK_THRESHOLD:
        color = "#e74c3c" # Red
        title = "High Risk: Likely Scam"
        prob_text = f"Scam Probability: {scam_prob*100:.1f}%"
    elif scam_prob < LOW_RISK_THRESHOLD:
        color = "#2ecc71" # Green
        title = "Low Risk: Likely Legitimate"
        prob_text = f"Scam Probability: {scam_prob*100:.1f}%"
    else:
        color = "#f39c12" # Orange
        title = "Medium Risk: Needs Review"
        prob_text = f"Scam Probability: {scam_prob*100:.1f}% (Uncertain)"
        
    ui_elements['risk_meter_bar'].configure(progress_color=color)
    ui_elements['risk_meter_bar'].set(scam_prob)
    ui_elements['result_title'].configure(text=title, text_color=color)
    ui_elements['result_prob'].configure(text=prob_text)

# --- 4. The Redesigned Professional GUI ---
def create_main_app_widgets(container):
    ui_elements = {
        'amount_var': ctk.StringVar(),
        'type_var': ctk.StringVar(value="DEBIT"),
        'new_ben_var': ctk.IntVar(value=0),
        'new_dev_var': ctk.IntVar(value=0),
        'high_freq_var': ctk.StringVar(value="Low (1-5)"), # Changed to StringVar to match map
    }
    
    container.columnconfigure(0, weight=1)
    container.rowconfigure(1, weight=1)
    
    header_frame = ctk.CTkFrame(container, corner_radius=0)
    header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
    
    ctk.CTkLabel(header_frame, text="Ultimate UPI Scam Detector", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(10,5))
    ctk.CTkLabel(header_frame, text="Powered by an Advanced ML Classification Model", font=ctk.CTkFont(size=14)).pack(pady=(0,10))

    content_frame = ctk.CTkFrame(container, corner_radius=0, fg_color="transparent")
    content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
    content_frame.columnconfigure(0, weight=1)
    content_frame.rowconfigure(2, weight=1)

    input_frame = ctk.CTkFrame(content_frame)
    input_frame.grid(row=0, column=0, sticky="ew")
    input_frame.columnconfigure((0,1), weight=1)

    ctk.CTkLabel(input_frame, text="Amount (₹):", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, padx=20, pady=(20,5), sticky="w")
    ctk.CTkEntry(input_frame, textvariable=ui_elements['amount_var'], font=ctk.CTkFont(size=14)).grid(row=1, column=0, padx=20, pady=(0,20), sticky="ew")
    
    ctk.CTkLabel(input_frame, text="Transaction Type:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=1, padx=20, pady=(20,5), sticky="w")
    ctk.CTkOptionMenu(input_frame, variable=ui_elements['type_var'], values=["DEBIT", "CREDIT", "COLLECT_REQUEST"], font=ctk.CTkFont(size=14)).grid(row=1, column=1, padx=20, pady=(0,20), sticky="ew")
    
    # NEW WIDGET: OptionMenu for frequency
    ctk.CTkLabel(input_frame, text="24h Transaction Frequency:", font=ctk.CTkFont(size=14, weight="bold")).grid(row=2, column=0, padx=20, pady=(0,5), sticky="w")
    ctk.CTkOptionMenu(input_frame, variable=ui_elements['high_freq_var'], values=["Low (1-5)", "Medium (6-15)", "High (16-30)", "Anomalous Burst (30+)"], font=ctk.CTkFont(size=14)).grid(row=3, column=0, padx=20, pady=(0,20), sticky="ew")


    checkbox_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
    checkbox_frame.grid(row=1, column=0, pady=15)
    ctk.CTkCheckBox(checkbox_frame, text="New Beneficiary", variable=ui_elements['new_ben_var'], font=ctk.CTkFont(size=14)).pack(side="left", padx=15)
    ctk.CTkCheckBox(checkbox_frame, text="New Device", variable=ui_elements['new_dev_var'], font=ctk.CTkFont(size=14)).pack(side="left", padx=15)
    
    result_frame = ctk.CTkFrame(content_frame)
    result_frame.grid(row=2, column=0, sticky="nsew", pady=20)
    result_frame.columnconfigure(0, weight=1)
    
    ui_elements['result_title'] = ctk.CTkLabel(result_frame, text="Enter details to begin analysis", font=ctk.CTkFont(size=20, weight="bold"))
    ui_elements['result_title'].pack(pady=(20, 5))
    
    ui_elements['result_prob'] = ctk.CTkLabel(result_frame, text="", font=ctk.CTkFont(size=16))
    ui_elements['result_prob'].pack(pady=(0, 10))
    
    ui_elements['risk_meter_bar'] = ctk.CTkProgressBar(result_frame, height=20, corner_radius=10)
    ui_elements['risk_meter_bar'].set(0)
    ui_elements['risk_meter_bar'].pack(fill="x", padx=50, pady=(5, 20))
    
    button_frame = ctk.CTkFrame(container, fg_color="transparent")
    button_frame.grid(row=2, column=0, pady=20)

    ctk.CTkButton(button_frame, text="Analyze Transaction", font=ctk.CTkFont(size=16, weight="bold"), height=40,
                  command=lambda: get_prediction(ui_elements)).pack(side="left", padx=10)

    def reset_form():
        for key, var in ui_elements.items():
            if isinstance(var, ctk.StringVar): var.set("")
            if isinstance(var, ctk.IntVar): var.set(0)
        ui_elements['type_var'].set("DEBIT")
        ui_elements['high_freq_var'].set("Low (1-5)") # Reset frequency
        ui_elements['result_title'].configure(text="Enter details to begin analysis", text_color="white")
        ui_elements['result_prob'].configure(text="")
        ui_elements['risk_meter_bar'].set(0)

    ctk.CTkButton(button_frame, text="Reset", fg_color="#565B5E", hover_color="#6A7176",
                  font=ctk.CTkFont(size=16, weight="bold"), height=40, command=reset_form).pack(side="left", padx=10)
    
    return ui_elements

# --- 5. Main Application Setup ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
root = ctk.CTk()
root.title("Ultimate UPI Scam Detector")
root.geometry("800x700") # Made window a bit taller

ui_elements = create_main_app_widgets(root)
root.mainloop()