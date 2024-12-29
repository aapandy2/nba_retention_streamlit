import streamlit as st
import pandas as pd
import plotly.express as px

#--------------------Helper functions------------------------------------------#

@st.cache_data
def load_data():
    data = pd.read_csv("Data/full_data.csv")
    return data

def load_model_preds():
    preds = pd.read_csv("Data/model_predictions.csv")
    return preds

def visualize_data(year, x, y):
    df["Retained"] = "No"
    df.loc[df.IN_LEAGUE_NEXT == 1, "Retained"] = "Yes"
    
    fig = px.scatter(df.loc[df.SEASON_START==year], x=x, y=y, color="Retained",
                    hover_name="NAME", hover_data=["TEAMS_LIST"],
                    color_discrete_map={"Yes":"cornflowerblue", "No":"lightcoral"},
                    labels={"Retained":"Retained?"})
    st.plotly_chart(fig)

    return None

def visualize_preds(season, team):
    df_temp = preds_exp.loc[(preds_exp.TEAMS_AS_LIST == team)
            & (preds_exp.SEASON_START == season)].copy()

    df_temp["Retained"] = "No"
    df_temp.loc[df_temp.IN_LEAGUE_NEXT == 1, "Retained"] = "Yes"

    df_temp["Predict retention?"] = "No"
    df_temp.loc[df_temp.PRED == 1, "Predict retention?"] = "Yes"

    if(season != 2023):
        fig = px.bar(df_temp, x="NAME", y="PROB", color="Retained",
                    labels={"NAME":"Player name",
                            "PROB":"Model retention probability",
                            "Retained":"Retained?"},
                    color_discrete_map={"Yes":'cornflowerblue',
                                        "No":'lightcoral'})
    else:
        fig = px.bar(df_temp, x="NAME", y="PROB", color="Predict retention?",
                    labels={"NAME":"Player name",
                            "PROB":"Model retention probability"},
                    color_discrete_map={"Yes":'cornflowerblue', "No":'lightcoral'})

    st.plotly_chart(fig)

    return None

#---------------------Load the data--------------------------------------------#

#load full set of raw training/test data
df = load_data()

#load model prediction data
preds = load_model_preds()

#convert teams list from strings to actual lists
preds["TEAMS_AS_LIST"] = preds.apply(lambda x: eval(x.TEAMS_LIST), axis=1)
#explode out teams
preds_exp = preds.explode("TEAMS_AS_LIST", ignore_index=True)

#grab sorted list of teams
teams = preds_exp.TEAMS_AS_LIST.unique()
teams.sort()

#---------------Visualizing raw data-------------------------------------------#

st.title('Predicting NBA Player Retention')

st.markdown('''This page provides an interactive executive summary of the
            project _Predicting NBA Player Retention_, completed by Alex Pandya,
            Peter Johnson, Andrew Newman, Ryan Moruzzi, and Collin Litterell.
            It was named a Top Project for the Erdos Institute Data Science
            Bootcamp, Fall 2024.''')

st.markdown('''The project aims to answer the following question: _can we
            predict whether or not a given NBA player will still be in the
            league next season, given just present-season statistics and
            transaction data?_''')

st.subheader("Visualizing the data")

st.markdown('''The scatterplot below provides a visualization for the raw data,
               which includes counting stats, advanced stats, and salary data
               for seasons between 1990 and 2023.  The scatterplot shows
               whether or not the player was _retained_ (whether or not they
               played at least one game in the following season) in color.''')


features = df.select_dtypes(include='number').columns.drop(['PLAYER_ID', 
                                            'SEASON_START', 'IN_LEAGUE_NEXT'])

stats = list(features.drop(['WAIVED', 'RELEASED', 'TRADED',        
       'WAIVED_OFF', 'WAIVED_REG', 'WAIVED_POST', 'RELEASED_OFF',               
       'RELEASED_REG', 'RELEASED_POST', 'TRADED_OFF', 'TRADED_REG',             
       'TRADED_POST']))
stats.sort()

x_stat = st.selectbox("x-axis:", stats)
y_stat = st.selectbox("y-axis:", stats)
year   = st.slider(label="Season start year:", min_value=1990, max_value=2022)

visualize_data(year, x_stat, y_stat)

#----------------------------Model predictions---------------------------------#

st.subheader("Model predictions")

st.markdown('''The bar charts below summarize the predictions of our
               best-performing model (XGBoost with SMOTE-augmented training
               data, calibrated using Platt's method) trained on data from
               seasons before the one set in the slider.  The height of the bar
               gives the probability that the player will play at any point
               in the next season, and the color gives whether or not they did
               play in the following season.''')

pred_year = st.slider(label="Season start year:", min_value=2017,
                      max_value=2023) 
pred_team = st.selectbox("Team:", teams)

visualize_preds(pred_year, pred_team)

st.markdown('''Note: for the 2023 season we do not yet know whether a given
               player will play _at any point_, so the color of the bars comes
               from the classifier.''')
