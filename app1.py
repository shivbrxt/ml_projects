import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Base */
  [data-testid="stAppViewContainer"] {
      background: #0d1117;
      color: #e6edf3;
  }
  [data-testid="stSidebar"] {
      background: #161b22;
      border-right: 1px solid #30363d;
  }

  /* Header */
  .header-band {
      background: linear-gradient(135deg, #1a1f35 0%, #0d1117 100%);
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 28px 36px;
      margin-bottom: 28px;
      display: flex;
      align-items: center;
      gap: 20px;
  }
  .header-band h1 {
      margin: 0;
      font-size: 2rem;
      font-weight: 700;
      color: #e6edf3;
      letter-spacing: -0.5px;
  }
  .header-band p {
      margin: 4px 0 0;
      color: #8b949e;
      font-size: 0.9rem;
  }

  /* Section labels */
  .section-label {
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #58a6ff;
      margin-bottom: 10px;
      margin-top: 24px;
  }

  /* Result cards */
  .result-card {
      border-radius: 10px;
      padding: 24px 28px;
      text-align: center;
      border: 1px solid #30363d;
  }
  .result-card.churn   { background: #1f1015; border-color: #f85149; }
  .result-card.no-churn { background: #0d1f1a; border-color: #3fb950; }
  .result-card .verdict {
      font-size: 1.6rem;
      font-weight: 800;
      margin-bottom: 4px;
  }
  .result-card.churn   .verdict { color: #f85149; }
  .result-card.no-churn .verdict { color: #3fb950; }
  .result-card .sub {
      font-size: 0.85rem;
      color: #8b949e;
  }

  /* Probability bar */
  .prob-wrap { margin-top: 18px; }
  .prob-label {
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      color: #8b949e;
      margin-bottom: 6px;
  }
  .prob-bar-bg {
      background: #21262d;
      border-radius: 6px;
      height: 12px;
      overflow: hidden;
  }
  .prob-bar-fill {
      height: 12px;
      border-radius: 6px;
      transition: width 0.4s ease;
  }

  /* Risk badge */
  .risk-badge {
      display: inline-block;
      padding: 3px 12px;
      border-radius: 20px;
      font-size: 0.78rem;
      font-weight: 600;
      margin-top: 10px;
  }
  .risk-low  { background: #0d4429; color: #3fb950; }
  .risk-med  { background: #341a00; color: #e3b341; }
  .risk-high { background: #3d0000; color: #f85149; }

  /* Streamlit overrides */
  div[data-testid="stSelectbox"] label,
  div[data-testid="stSlider"] label,
  div[data-testid="stNumberInput"] label { color: #c9d1d9 !important; }

  div.stButton > button {
      background: #1f6feb;
      color: #ffffff;
      border: none;
      border-radius: 8px;
      padding: 10px 28px;
      font-weight: 600;
      font-size: 0.95rem;
      width: 100%;
      transition: background 0.2s;
  }
  div.stButton > button:hover { background: #388bfd; }

  hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("telecom_customer_churn_model_XGB.pkl")

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-band">
  <div>
    <h1>📡 Telecom Churn Predictor</h1>
    <p>XGBoost · Telecom Customer Churn &nbsp;|&nbsp; Fill in customer details and click Predict</p>
  </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(
        "**Model file not found.** Place `telecom_customer_churn_model_XGB.pkl` "
        "in the same directory as `app.py` and restart."
    )
    st.stop()

# ── Sidebar – input form ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Customer Profile")
    st.markdown("---")

    # Demographics
    st.markdown('<p class="section-label">Demographics</p>', unsafe_allow_html=True)
    gender          = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen  = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner         = st.selectbox("Partner", ["No", "Yes"])
    dependents      = st.selectbox("Dependents", ["No", "Yes"])

    # Account
    st.markdown('<p class="section-label">Account</p>', unsafe_allow_html=True)
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless       = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method  = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    # Charges
    st.markdown('<p class="section-label">Charges</p>', unsafe_allow_html=True)
    monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total_charges   = st.number_input(
        "Total Charges ($)", 0.0, 9000.0,
        float(round(monthly_charges * tenure, 2)), step=1.0
    )

    # Phone services
    st.markdown('<p class="section-label">Phone Services</p>', unsafe_allow_html=True)
    phone_service   = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines  = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    # Internet services
    st.markdown('<p class="section-label">Internet Services</p>', unsafe_allow_html=True)
    internet        = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    no_inet         = "No internet service"
    inet_opts       = ["Yes", "No"] if internet != "No" else [no_inet]

    online_security  = st.selectbox("Online Security",   ["No", "Yes", no_inet] if internet != "No" else [no_inet])
    online_backup    = st.selectbox("Online Backup",     ["No", "Yes", no_inet] if internet != "No" else [no_inet])
    device_prot      = st.selectbox("Device Protection", ["No", "Yes", no_inet] if internet != "No" else [no_inet])
    tech_support     = st.selectbox("Tech Support",      ["No", "Yes", no_inet] if internet != "No" else [no_inet])
    streaming_tv     = st.selectbox("Streaming TV",      ["No", "Yes", no_inet] if internet != "No" else [no_inet])
    streaming_movies = st.selectbox("Streaming Movies",  ["No", "Yes", no_inet] if internet != "No" else [no_inet])

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Churn")

# ── Feature engineering (mirror notebook) ────────────────────────────────────
def build_input():
    avg_monthly = 0.0 if tenure == 0 else round(total_charges / tenure, 4)
    is_new      = int(tenure < 12)

    return pd.DataFrame([{
        "gender":           gender,
        "SeniorCitizen":    1 if senior_citizen == "Yes" else 0,
        "Partner":          partner,
        "Dependents":       dependents,
        "tenure":           tenure,
        "PhoneService":     phone_service,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet,
        "OnlineSecurity":   online_security,
        "OnlineBackup":     online_backup,
        "DeviceProtection": device_prot,
        "TechSupport":      tech_support,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_movies,
        "Contract":         contract,
        "PaperlessBilling": paperless,
        "PaymentMethod":    payment_method,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
        "AvgMonthlySpend":  avg_monthly,
        "IsNewCustomer":    is_new,
    }])

# ── Main area ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown("### Customer Summary")
    summary = {
        "Gender": gender, "Senior Citizen": senior_citizen,
        "Partner": partner, "Dependents": dependents,
        "Tenure": f"{tenure} months", "Contract": contract,
        "Monthly Charges": f"${monthly_charges:.2f}",
        "Total Charges": f"${total_charges:.2f}",
        "Internet Service": internet, "Payment Method": payment_method,
    }
    s_df = pd.DataFrame(summary.items(), columns=["Field", "Value"])
    st.dataframe(s_df, hide_index=True, use_container_width=True)

with col_right:
    st.markdown("### Prediction")

    if not predict_btn:
        st.info("Configure the customer profile in the sidebar and click **Predict Churn**.")
    else:
        customer_df = build_input()

        THRESHOLD = 0.40
        prob   = model.predict_proba(customer_df)[:, 1][0]
        churn  = prob >= THRESHOLD
        pct    = round(prob * 100, 1)

        if churn:
            card_cls, verdict, sub = "churn", "⚠️ Will Churn", "This customer is likely to leave"
        else:
            card_cls, verdict, sub = "no-churn", "✅ Will Stay", "This customer is likely to remain"

        if pct < 30:
            badge_cls, risk_label = "risk-low",  "Low Risk"
        elif pct < 55:
            badge_cls, risk_label = "risk-med",  "Medium Risk"
        else:
            badge_cls, risk_label = "risk-high", "High Risk"

        bar_color = "#f85149" if churn else "#3fb950"

        st.markdown(f"""
        <div class="result-card {card_cls}">
          <div class="verdict">{verdict}</div>
          <div class="sub">{sub}</div>
          <span class="risk-badge {badge_cls}">{risk_label}</span>
          <div class="prob-wrap">
            <div class="prob-label">
              <span>Churn Probability</span>
              <span><b>{pct}%</b></span>
            </div>
            <div class="prob-bar-bg">
              <div class="prob-bar-fill" style="width:{pct}%; background:{bar_color};"></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("&nbsp;")

        # Metric row
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn Prob",   f"{pct}%")
        m2.metric("Stay Prob",    f"{round(100 - pct, 1)}%")
        m3.metric("Threshold",    "40%")

        # Retention tip
        st.markdown("---")
        if churn:
            if contract == "Month-to-month":
                tip = "💡 Offer a discounted **annual or two-year contract** to reduce cancellation risk."
            elif tenure < 12:
                tip = "💡 Customer is still new — consider a **loyalty reward or onboarding check-in**."
            elif internet == "Fiber optic" and monthly_charges > 80:
                tip = "💡 High charges on Fiber — a **bundle discount or speed upgrade offer** may retain them."
            else:
                tip = "💡 Reach out proactively with a **personalised retention offer**."
            st.markdown(tip)
        else:
            st.markdown("✅ No immediate retention action needed. Monitor at next billing cycle.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#484f58; font-size:0.8rem;'>"
    "Telecom Churn Predictor · XGBoost · Threshold 0.40 · Built with Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
