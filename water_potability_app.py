#Importing the libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
sns.set_style('whitegrid')
from matplotlib import pyplot as plt
import joblib as jb
wpp = jb.load("water_potability_predictor.joblib")
model = wpp['model']
scaler = wpp['scaler']
input_cols = wpp['input_cols']


#setting page configuration
st.set_page_config(page_title='Water Potability', page_icon=':droplet:')


markdown='''
<style>
[data-testid="stAppViewContainer"]{
background-image: linear-gradient(to bottom right, #b3ffff, #009999);
color:#000000;
}
[data-testid="stHeader"]{
opacity:0.0;
}
[data-testid="stSidebar"]{
background-image: linear-gradient(to bottom right, #b3ffff, #009999);
}
[class="css-nqowgj edgvbvh3"]{
border: 1px solid rgb(12,208,219);
border-radius: 5px;
}
[data-testid="stTickBarMin"]{
color:#000000;
}
[data-testid="stTickBarMax"]{
color:#000000;
}
[data-testid="stThumbValue"]{
color:#000000;
}
[class="main-svg"]{
opacity:0.9
}
</style>

'''

st.markdown(markdown,unsafe_allow_html=True)


st.write("""# Water Potability""")

st.write("___")



#creating sidebar
st.sidebar.header('Water Quality Metrics')

#Taking user inputs form the side
def user_input_features():
    ph = st.sidebar.slider('ph',0.00, 14.00, 7.04)
    Hardness = st.sidebar.slider('Hardness',47.43,323.12,196.97)
    Solids = st.sidebar.slider('Solids',320.94,61227.20,20927.83)
    Chloramines = st.sidebar.slider('Chloramines',0.35,13.13,7.13 )
    Sulfate = st.sidebar.slider('Sulfate',129.00,481.03,333.39)
    Conductivity = st.sidebar.slider('Conductivity',181.48,753.34,421.88)
    Organic_carbon = st.sidebar.slider('Organic_carbon',2.20,28.30,14.22)
    Trihalomethanes = st.sidebar.slider('Trihalomethanes',0.74,124.00,66.54)
    Turbidity = st.sidebar.slider('Turbidity',1.45,6.74,3.96)
    data = {'ph': ph,
            'Hardness': Hardness,
            'Solids': Solids,
            'Chloramines': Chloramines,
            'Sulfate': Sulfate,
            'Conductivity': Conductivity,
            'Organic_carbon': Organic_carbon,
            'Trihalomethanes': Trihalomethanes,
            'Turbidity': Turbidity}
    features = pd.DataFrame(data, index=[0])
    return features

df1 = user_input_features()

st.write('### Water Quality Metrics :')
st.write(df1)
def predictor(df):
    df[input_cols] = scaler.transform(df[input_cols])
    predictions = model.predict(df[input_cols])
    return predictions[0]

prediction=predictor(df1)

st.write('### Prediction :')
if prediction==0:
    st.write("##### Water is not potabale :warning:")
else:
    st.write("##### Water is potable :innocent:")
st.write("___")
st.write('### Prediction Probability :')
prediction_proba = model.predict_proba(df1)
st.write(prediction_proba)
st.write("0 : Not Potable")
st.write("1 : Potable")

st.write('___')

df = pd.read_csv('water_potability_preprocessed.csv')
#Data Visualization Inputs
st.write('### Data Visualization :')
with st.container():
    l,m,r=st.columns(3)
    with l:
        option1=st.selectbox('On X Axis',('ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity'))
    with m:
        option2=st.selectbox('On Y Axis',('ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity'))
    with r:
        option4=st.selectbox('Type of Plot', ('Histplot (X Axis)','Scatterplot', 'Barplot (Y Axis)','Boxplot (Y Axis)' ))
#Plots
if option4=='Boxplot (Y Axis)':
    fig = px.box(data_frame=df, y=option2, x='Potability', color='Potability')
elif option4=='Scatterplot':
    fig = px.scatter(data_frame=df, x=option1, y=option2, color='Potability')
elif option4=='Histplot (X Axis)':
    fig = px.histogram(data_frame=df, x=option1, color='Potability')
elif option4=='Barplot (Y Axis)':
    fig = px.bar(data_frame=df, y=option2, x='Potability', color='Potability')
else:
    pass
st.plotly_chart(fig, use_container_width=True, theme=None)

st.write('___')
st.write('### Statistical Information :')
if st.button('Potable Water'):
    st.write(df[df.Potability==1].describe())
else:
    pass
if st.button('Not Potable Water'):
    st.write(df[df.Potability==0].describe())
else:
    pass
st.write("___")
st.write("My Profiles")

with st.container():
    left, middle, right = st.columns(3)
    with left:
        st.markdown("<a href='https://www.linkedin.com/in/prasad-posture-6a3a77215/' target='blank'><img align='center' src='https://img.shields.io/badge/-Prasad Posture-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/prasad-posture-6a3a77215/' alt='Prasad Posture' height='20' width='100' /></a>", unsafe_allow_html=True)
    with middle:
        st.markdown("<a href='https://github.com/prasadposture' target='blank'><img align='center' src='https://img.shields.io/badge/-prasadposture-black?style=flat-square&logo=GitHub&logoColor=white&link=https://github.com/prasadposture' alt='prasadposture' height='20' width='100' /></a>", unsafe_allow_html=True)
    with right:
        st.markdown("<a href='https://www.kaggle.com/prasadposture121' target='blank'><img align='center' src='https://img.shields.io/badge/-prasadposture121-blue?style=flat-square&logo=Kaggle&logoColor=white&link=https://www.kaggle.com/prasadposture121' alt='prasadposture121' height='20' width='100' /></a>", unsafe_allow_html=True)
