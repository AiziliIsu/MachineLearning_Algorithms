import streamlit as st
from regression_page import regression_page
from classification_page import classification_page
from cross_validation_page import cross_validation_page


def main():
    st.set_page_config(
        page_title="ML Algorithms performance on different Datasets",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #2196F3;
        margin-bottom: 0.75rem;
    }
    .sidebar-header {
        font-size: 1.2rem;
        color: #FF9800;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<p class="sidebar-header">Navigation</p>', unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Select a page to explore:",
        ["Regression", "Classification", "Cross Validation"],  # <-- ADD NEW OPTION
        index=0
    )

    if page == "Regression":
        st.sidebar.markdown(
            "- Simple Linear Regression  \n"
            "- Multiple Linear Regression \n"
            "- Decision Tree  \n"
            "- Random Forest  \n"
            "- AdaBoost  \n"
            "- XGBoost")
        regression_page()
    elif page == "Classification":
        st.sidebar.markdown(
            "- Naive Bayes  \n"
            "- K-Nearest Neighbors  \n"
            "- Decision Tree  \n"
            "- Random Forest  \n"
            "- AdaBoost  \n"
            "- XGBoost")
        classification_page()
    elif page == "Cross Validation":
        st.sidebar.markdown("Cross Validation: Handle overfitting with k-fold validation.")
        cross_validation_page()


if __name__ == "__main__":
    main()

