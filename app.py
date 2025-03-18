from pathlib import Path
import streamlit as st

dir_path = Path(__file__).parent


def run():
    # Check for URL parameters to handle direct page access
    query_params = st.query_params

    # Map URL paths to page indices
    page_map = {
        "overview": 0,
        "data_exploration": 1,
        "data_preprocessing": 2,
        "feature_engineering": 3,
        "model_selections_and_training": 4,
        "hyperparameter_tuning": 5,
        "deployment": 6
    }

    # Define the pages
    pages = [
        # Overview
        st.Page(
            dir_path / "chapters/overview.py",
            title="1. Overview",
            icon=":material/overview_key:",
        ),
        # Data Exploration
        st.Page(
            dir_path / "chapters/data_exploration.py",
            title="2. Data Exploration",
            icon=":material/explore:",
        ),
        # Data Preprocessing
        st.Page(
            dir_path / "chapters/data_preprocessing.py",
            title="3. Data Preprocessing",
            icon=":material/room_preferences:",
        ),
        # Feature Engineering
        st.Page(
            dir_path / "chapters/feature_engineering.py",
            title="4. Feature Engineering",
            icon=":material/construction:",
        ),
        # Model Selections and Training
        st.Page(
            dir_path / "chapters/model_selections_and_training.py",
            title="5. Model Selections and Training",
            icon=":material/emoji_events:",
        ),
        # Hyperparameter Tuning
        st.Page(
            dir_path / "chapters/hyperparameter_tuning.py",
            title="6. Hyperparameter Tuning",
            icon=":material/tune:",
        ),
        # deployment
        st.Page(
            dir_path / "chapters/deployment.py",
            title="7. Deployment",
            icon=":material/slow_motion_video:",
        ),
    ]

    # Create navigation
    page = st.navigation(pages)

    # If 'page' parameter exists in URL, navigate to that page
    if "page" in query_params:
        page_name = query_params["page"]
        if page_name in page_map:
            # Set the current page using page.switch
            # Note: We need to do this after navigation is created
            st.session_state["navigation_selection"] = page_map[page_name]

    # Run the selected page
    page.run()


if __name__ == "__main__":
    run()