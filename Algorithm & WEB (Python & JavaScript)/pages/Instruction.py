import streamlit as st

def instruction_page():
    # Embed the React app as a fullscreen iframe
    st.markdown(
        """
        <style>
            html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
                height: 100%;
                width: 100%;
                margin: 0;
                padding: 0;
                overflow: hidden;
            }

            .iframe-container {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                z-index: 1;
            }

            .iframe-container iframe {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                border: none;
            }
        </style>
        <div class="iframe-container">
            <iframe src="http://localhost:5173/instruction"></iframe> <!-- Update this URL with your React app's URL -->
        </div>
        """,
        unsafe_allow_html=True,
    )

# Run the credit page
instruction_page()
