import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import streamlit as st
import datetime as dt
from streamlit_option_menu import option_menu
from PIL import Image
import json
from streamlit_lottie import st_lottie

#Changing App Name and Icon
img = Image.open("img/icon.png")
st.set_page_config(page_title="GoldCoast Stock Trend Predictor",page_icon=img)

#Removing header and Footer of the Web-App
hide_menu_style = '''
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility : hidden;}
    </style>'''
st.markdown(hide_menu_style, unsafe_allow_html=True)

#Importing json animation into project from file
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

#Bar on top of the screen
selected = option_menu(
    menu_title=None,
    options=["Home", "Project","FAQ","Contact"],
    icons=["house","book","question-circle","envelope"],
    default_index=0,
    orientation="horizontal",
    styles={
            "icon": {"color": "black", "font-size": "22px"},
            "nav-link": { "text-align": "center","font-size": "22px", "--hover-color": "#FFEF99"},
        },
)

#Home Page
if selected == "Home":
    st.title('GoldCoast Stock Trend Predictor')
    st.caption("")
    st.write('''The stock market is known for being volatile, dynamic and nonlinear. 
    Accurate stock price prediction is extremely challenging because of multiple 
    (macro and micro) factors such as politics, global economic conditions, 
    unexpected events, a company’s financial performance and so on.
    But, all of this also means that there’s a lot of data to find patterns in. 
    So financial analysts, researchers and data scientists keep exploring analytics 
    techniques to detect stock market trends. ''')

    lottie_coading = load_lottiefile("app-lottie\home.json")
    st_lottie(
        lottie_coading,
        speed = 1,
        reverse = False,
        loop=True,
        quality="high",
        height = "300px",
        width = "100%",
        key = None,
    )

#Main Project Page
if selected == "Project":
    st.title('Lets Begin:')
    user_input = st.text_input("Enter the symbol of the stock : ","MSFT")
    st.caption("For symbol [Click here](https://finance.yahoo.com/)")

    ## Range selector
    cols1, _ = st.columns((1, 2))  # To make it narrower
    format = 'MM, DD, YYYY'  # format output
    start_date = dt.date(year=2010, month=1, day=1)
    end_date = dt.datetime.now().date()
    max_days = end_date - start_date

    slider_start = cols1.slider('Select start date :', min_value=start_date, value=start_date, max_value=end_date, format=format)
    slider_end = cols1.slider('Select end date :', min_value=start_date, value=end_date, max_value=end_date, format=format)

    ## Sanity check
    st.table(pd.DataFrame([[slider_start, slider_end]],
                          columns=['Start',
                                   'End'],
                          index=['Date']))
    start = slider_start
    end = slider_end
    if end>start:
            df = data.DataReader(user_input, 'yahoo', start, end)

            #Describing Data
            st.subheader(f'Data from {start} to {end}')
            st.write(df.describe())

            #Visualizations
            st.subheader('Closing Price vs Time chart')
            fig = plt.figure(figsize=(12,6))
            plt.plot(df.Close)
            st.pyplot(fig)

            st.subheader('Closing Price vs Time chart with 100MA')
            ma100 = df.Close.rolling(100).mean()
            fig = plt.figure(figsize=(12,6))
            plt.plot(ma100)
            plt.plot(df.Close)
            st.pyplot(fig)

            st.subheader('Closing Price vs Time chart with 100MA & 200MA')
            ma100 = df.Close.rolling(100).mean()
            ma200 = df.Close.rolling(200).mean()
            fig = plt.figure(figsize=(12,6))
            plt.plot(ma100)
            plt.plot(ma200)
            plt.plot(df.Close)
            st.pyplot(fig)

            #Splitting Data into Training and Testing
            data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
            data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range = (0,1))
            data_training_array = scaler.fit_transform(data_training)

            #Load my model
            from keras.models import load_model
            model = load_model("keras_model.h5")

            #Testing Part
            past_100_days = data_training.tail(100)
            final_df = past_100_days.append(data_testing, ignore_index = True)
            input_data = scaler.fit_transform(final_df)

            x_test = []
            y_test = []

            for i in range (100, input_data.shape[0]):
                x_test.append(input_data[i-100: i])
                y_test.append(input_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)
            y_predicted = model.predict(x_test)
            scaler = scaler.scale_

            scale_factor = 1/scaler[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            #Final graph
            st.subheader('Predicted Price vs Original Price')
            fig2 = plt.figure(figsize=(12,6))
            plt.plot(y_test, 'b', label = 'Original Price')
            plt.plot(y_predicted, 'r', label = 'Predicted Price')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig2)

            lottie_coading = load_lottiefile("app-lottie\thanks.json")
            st_lottie(
                lottie_coading,
                speed=3,
                reverse=False,
                loop=True,
                quality="high",
                height="500px",
                width="100%",
                key=None,
            )
    else:
            lottie_coading = load_lottiefile("app-lottie\error.json")
            st_lottie(
                lottie_coading,
                speed=1,
                reverse=False,
                loop=True,
                quality="high",
                height="300px",
                width="100%",
                key=None,
            )
            st.caption("Start date should be before the end date.")

#FAQ Page
if selected == "FAQ":
    st.title(f"Frequently Asked Questions(FAQ)")

    lottie_coading = load_lottiefile("app-lottie\faq.json")
    st_lottie(
        lottie_coading,
        speed=1,
        reverse=False,
        loop=True,
        quality="high",
        height="300px",
        width="100%",
        key=None,
    )

    st.caption("#Question1")
    st.subheader("What is this Web App for?")
    st.write("""It helps us predict the future trends of any company 
            stock by analysing the past trends of the same.""")

    st.caption("#Question2")
    st.subheader("What is 100MA & 200MA?")
    st.write("""MA stands for Moving Average. 100MA and 200MA 
            denotes the average of closing prices of previous 
            100 days and 200 days respectively.""")

    st.caption("#Question3")
    st.subheader("What is the significance of  100MA?")
    st.write("""A moving average over 100 days helps investors 
            see how the stock has performed over previous 100 days 
            and to find the price trend if it was upward or downward.""")

    st.caption("#Question4")
    st.subheader("Data used in the aplication is real or fake?")
    st.write("""The data is perfectly real and it is fetching 
            information from Yahoo Finance website.""")

    st.caption("#Question5")
    st.subheader("Can I trade according to the app data?")
    st.write("""No, you cannot rely on the predictions made in this app because 
            predictions depend upon multiple factors such as politics, global economic 
            conditions, unexpected events, company's financial performance and so on and 
            this app data doesn't cover-up all the factors. Better you should take proper 
            advice from your financial advisor before investing.""")

#Contact Page
if selected == "Contact":
    st.header(f":mailbox: Get In Touch With Me!")

    contact = '''
    <form action="https://formsubmit.co/anishritolia6@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here"></textarea>
        <button type="submit">Send</button>
    </form>
    '''

    st.markdown(contact, unsafe_allow_html=True)

    lottie_coading = load_lottiefile("app-lottie/contact.json")
    st_lottie(
        lottie_coading,
        speed=0.9,
        reverse=False,
        loop=True,
        quality="high",
        height="150px",
        width="100%",
        key=None,
    )

    #Use Local CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("style/style.css")
