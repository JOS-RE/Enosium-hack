import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from PIL import Image 

image2=Image.open('assets/icon.png')
st.set_page_config(
    page_title="Loan Check",
    page_icon=image2,
)

image = Image.open('assets/banner2.png')
st.image(image)

st.title("Welcome to Runaha 🏦")

st.subheader('You have come to the :blue[mass data analyser]')

st.subheader('Instructions:')
st.markdown('''
    1. Upload your excel file here
    2. Click on the :blue[Get Results] button below to get the results
''')


spreadsheet = st.file_uploader("", type=["xlsx", "csv"], help="Upload your excel file here")
if spreadsheet is not None:
    spectra_df = pd.read_csv(spreadsheet)
    spectra = spectra_df[['Number of people who will provide maintenance',
       'Loan History', 'loan amount taken', 'Guarantor or Debtor',
       'Number of years of employment',
       'Number of loans taken from current bank',
       'Age of the applicant in Number of Years', 'amount in current account',
       'amount in savings account', '% of income paid as installment',
       'Other loans plans taken', 'Working abroad or not',
       'time duration for loan', 'Owned property', 'Type of job performed',
       'Type of Housing', 'Number of years of stay in current address']]
    
    ##label encoding 
    veri_status_mapping={"none":0,"co-applicant":1,"gaurantor":2}
    spectra["Guarantor or Debtor"]=spectra["Guarantor or Debtor"].map(veri_status_mapping)

    veri_status_mapping={"for free":0,"rent":1,"own":2}
    spectra["Type of Housing"]=spectra["Type of Housing"].map(veri_status_mapping)

    veri_status_mapping={"No":0,"Yes":1}
    spectra["Working abroad or not"]=spectra["Working abroad or not"].map(veri_status_mapping)

    veri_status_mapping={"No property":0,"car or other property":1,"building society savings agreement/life insurance":2,"Real Estate":3}
    spectra["Owned property"]=spectra["Owned property"].map(veri_status_mapping)

    veri_status_mapping={"unskilled - resident":0,"unemployed/ unskilled - non-resident":1,"skilled employee / official":2,"management/ self-employed/highly qualified employee/ officer":3}
    spectra["Type of job performed"]=spectra["Type of job performed"].map(veri_status_mapping)

    veri_status_mapping={"unemployed":0,"less than a year":1,"between 1 and 4 years":2,"greater than 4 years":3}
    spectra["Number of years of employment"]=spectra["Number of years of employment"].map(veri_status_mapping)

    veri_status_mapping={"less than 0":0,"no current account":1,"between 0 and 200":2,"greater than 200":3}
    spectra["amount in current account"]=spectra["amount in current account"].map(veri_status_mapping)

    veri_status_mapping={"lno savings account":0,"less than 100":1,"between 100 and 500":2,"between 500 and 1000":3,"greater than 1000":4}
    spectra["amount in savings account"]=spectra["amount in savings account"].map(veri_status_mapping)

    veri_status_mapping={'existing loans paid back duly till now':4,'critical account/other loans existing (not at this bank)':0,'delay in paying off loans in the past':1,
       'all loans at this bank paid back duly':3,
       'no loans taken/all loans paid back duly':2}
    spectra["Loan History"]=spectra["Loan History"].map(veri_status_mapping)

    veri_status_mapping={"none":2, "bank":1, "stores":0}
    spectra["Other loans plans taken"]=spectra["Other loans plans taken"].map(veri_status_mapping)

    imputer = KNNImputer(n_neighbors=3)
    main_train = imputer.fit_transform(spectra)

    spectra = pd.DataFrame(main_train, columns =spectra.columns) 

submit = st.button('Get Results')


if submit:
    
    #st.write(spectra_df)
    scaler = joblib.load('minmaxscaler.joblib')
    model = joblib.load('model.joblib')

    x= scaler.transform(spectra)
    result = model.predict(x)
    result = result.astype(int)

    # st.write(type(result) )
    result = result.tolist()
    # st.write(type(result) )

    for i in range(len(result)):
        if result[i] == 0:
            result[i] = 'Customer will not repay the loan, Not Eligible'
        else:
            result[i] = 'Eligible for Loan and will possibly repay'
    
    df = pd.DataFrame(
   
        np.column_stack([spectra_df['Unnamed: 0'], result]),
        columns=['User ID', 'User Repayment Status'],
    )

    st.table(df)

    # res = pd.DataFrame({'Customer ID': spectra_df['Unnamed: 0'], 'Loan Defaulted or not': result})
    # res['Loan Defaulted or not'] = res['Loan Defaulted or not'].astype(int)



    # st.write(res)