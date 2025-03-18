from pathlib import Path
import streamlit as st

dir_path = Path(__file__).parent

def run():
    page = st.navigation(
        [
            #Overview
            st.Page(
                dir_path / "chapters/overview.py",
                title="1. Overview",
                icon=":material/overview_key:",
            ),
            #Data Exploration
            st.Page(
                dir_path / "chapters/data_exploration.py",
                title="2. Data Exploration",
                icon=":material/explore:",
            ),
            #Data Preprocessing
            st.Page(
                dir_path / "chapters/data_preprocessing.py",
                title="3. Data Preprocessing",
                icon=":material/room_preferences:",
            ),
            #Feature Engineering
            st.Page(
                dir_path / "chapters/feature_engineering.py",
                title="4. Feature Engineering",
                icon=":material/construction:",
            ),
            #Model Selections and Training
            st.Page(
                dir_path / "chapters/model_selections_and_training.py",
                title="5. Model Selections and Training",
                icon=":material/emoji_events:",
            ),

            st.Page(
    dir_path / "chapters/hyperparameter_tuning.py",
    title="6. Hyperparameter Tuning",
    icon=":material/tune:",
)


        ]
    )
    page.run()

if __name__ == "__main__":
    run()