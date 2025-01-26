import streamlit as st
from PIL import Image

# Page Configuration
st.set_page_config(page_title="X-POSE", layout="wide")

# CSS Styling
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to bottom, #a8edea, #fed6e3);
        font-family: 'Arial', sans-serif;
        color: #333;
    }
    .title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: bold;
        color: #FFFFFF;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #FFFFFF;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #7f8c8d;
        margin-top: 50px;
    }
    .sidebar {
        background-color: #ecf0f1;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.header("X-POSE")

st.subheader("How It Works")
st.markdown("""
1. **Upload or Capture Video**: Record your movement or use live video input.
2. **Pose Analysis**: Our AI model identifies key points and evaluates your posture.
3. **Feedback Generation**: Receive actionable insights to improve your form.
4. **Progress Tracking**: Monitor improvements over time with detailed analytics.
""")