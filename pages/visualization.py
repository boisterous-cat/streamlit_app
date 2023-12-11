import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import json
import seaborn as sns
from PIL import Image
import numpy as np
import altair as alt

def show_page(df):
    st.header('_Exploratory data analysis_', divider='rainbow')
    st.subheader('Short description of dataset:')
    lst = ['AGE - client age', 'GENDER - client sex', 'MARITAL_STATUS - family status', 'CHILD_TOTAL - number of clients children', 'DEPENDANTS - number of clients dependents','SOCSTATUS_WORK_FL - client work status (working or not)', 'SOCSTATUS_PENS_FL - client pension status (pensioner or not)', 'AGREEMENT_RK - client unique identificator', 'TARGET - client reaction (responded or not)','PERSONAL_INCOME - client personal income', 'CREDIT_AMOUNT - number of all clie credits', 'CLOSED - at least one credit closed', 'PROPERTY - has auto or real estate in property']
    s = ''

    for i in lst:
        s += "- " + i + "\n"

    st.markdown(s)

    df=pd.DataFrame(df)

    df_vs = df.copy()
    df_vs = df_vs.astype({'AGE':float, 'AGREEMENT_RK':float, 'DEPENDANTS':float, 'CHILD_TOTAL':float, 'GENDER':float, 'CREDIT_AMOUNT': float})
    df_vs = df_vs.astype({'SOCSTATUS_WORK_FL': bool, 'SOCSTATUS_PENS_FL': bool, "TARGET": bool, "CLOSED": bool, "PROPERTY": bool})
    st.data_editor(
        df_vs,
        column_order=("TARGET", "PERSONAL_INCOME", "AGE", "GENDER", "EDUCATION", "MARITAL_STATUS", "CHILD_TOTAL", "DEPENDANTS", "SOCSTATUS_WORK_FL", "SOCSTATUS_PENS_FL", "CREDIT_AMOUNT","CLOSED", "PROPERTY","AGREEMENT_RK"),
        use_container_width = True,
        num_rows = "fixed",
        disabled = True,
        column_config={
            "PERSONAL_INCOME": st.column_config.ProgressColumn(
                "Personal income",
                help="Personal income in RUB",
                format="%f",
                min_value=df_vs['PERSONAL_INCOME'].min(),
                max_value=df_vs['PERSONAL_INCOME'].max(),
            ),
            "CREDIT_AMOUNT": st.column_config.NumberColumn(
                "Amount of credits",
                help="Amount of person's credits (both closed and open)",
                format="%d",
                #min_value=df_vs['CREDIT_AMOUNT'].min(),
                #max_value=df_vs['CREDIT_AMOUNT'].max(),
            ),
            "AGE": st.column_config.NumberColumn(
                "Age",
                help="Client age",
                format="%d",
            ),
            "AGREEMENT_RK": st.column_config.NumberColumn(
                "Uid",
                help="Client uid",
                format="%d",
            ),
            "SOCSTATUS_WORK_FL": st.column_config.CheckboxColumn(
                "Work status",
                help="Is client working or not",
                default=False,
                disabled = True,
            ),
            "EDUCATION": st.column_config.TextColumn(
                "Education",
                help="Client education",
                disabled=True,
            ),
            "MARITAL_STATUS": st.column_config.TextColumn(
                "Marital status",
                disabled=True,
            ),
            "GENDER": st.column_config.CheckboxColumn(
                "Male",
                help="Male client",
                default=False,
                disabled=True,
            ),
            "CHILD_TOTAL": st.column_config.NumberColumn(
                "Children",
                help="How many children client has",
                format="%d",
            ),
            "DEPENDANTS": st.column_config.NumberColumn(
                "Dependants",
                help="How many dependants client has",
                format="%d",
            ),
            "SOCSTATUS_PENS_FL": st.column_config.CheckboxColumn(
                "Pension status",
                help="Client has pension",
                default=False,
                disabled=True,
            ),
            "TARGET": st.column_config.CheckboxColumn(
                "Target",
                help="Positive client",
                default=False,
                disabled=True,
            ),
            "CLOSED": st.column_config.CheckboxColumn(
                "Closed credit",
                help="Has one closed credit",
                default=False,
                disabled=True,
            ),
            "PROPERTY": st.column_config.CheckboxColumn(
                "Property",
                help="Has property (auto/real estate)",
                default=False,
                disabled=True,
            )
        },
        hide_index=True,
    )
    st.divider()

    st.subheader('Feature correlation with target:')
    y_corr = df['TARGET']
    palette = ["#ea96a3", "#e4946a", "#c29a4b", "#a69f46", "#74c385", "#4aad8f", "#4baba4", "#4eabb8", "#55acd7",
               "#a0adea", "#c79fe9", "#e78ae0", "#e891c1"]

    numeric_data = df.drop(columns=['TARGET']).select_dtypes([np.number])
    numeric_data_mean = numeric_data.mean()
    numeric_features = numeric_data.columns
    correlations = df[numeric_features].corrwith(y_corr).sort_values(ascending=False)
    plot = sns.barplot(y=correlations.index, x=correlations, palette=palette)
    plot.set(xlabel='Correlation', ylabel='Feature')
    st.pyplot(plot.get_figure())
    st.caption('As you can see _Age_ has the most negative correlation with _Target_. That means the more is the age of a client the less is the probability that he will respond to the offer.\n The same situation is with _Pensioners_ and those who closed their credit.\n On the opposite _Personal income_ is the most positive correlation with _Target_. Thus, we can say that people with higher income more ikely respond to the offer.\nThe same is with working people who, oppoite to pensioners, are more possitive to the offer.')
    st.divider()

    st.subheader('Value count of credit amount:')
    value_counts = df['CREDIT_AMOUNT'].value_counts()
    st.bar_chart(value_counts)
    st.caption(
        'As you can see around 75% of the sample has just one credit.\n ')
    st.divider()

    st.subheader('Distribution of credits by education:')
    j_df = pd.DataFrame()
    j_df['Открытые кредиты'] = df[df['CREDIT_AMOUNT'] == 1]['EDUCATION'].value_counts()
    j_df['Закрытые кредиты'] = df[df['CLOSED'] >= 1]['EDUCATION'].value_counts()

    st.bar_chart(j_df)
    st.caption(
        'The most significant trend on the bar says the people with _Lower post-secondary vocational education_ has more open credits than closed.\n On the opposite only people with _Secondary school_ close credits more often.\n What is more intersting - people with _Second Higher Degree_ and _PhD_ practically do not have credits (niether open nor closed).')
    st.divider()

    st.subheader('Tendency to respond to an offer depending on age')
    age_lst = ['0.0 - from 21 to 24', '2.0 - from 26 to 27', '5.0 - from 31 to 32',
           '7.0 - from 35 to 36', '10.0 - from 40 to 41',
           '12.0 - from 44 to 45',
           '15.0 - from 51 to 52', '17.0 - from 56 to 57','19.0 - from 61 to 67']

    s2=' '
    for i in age_lst:
        s2 += "- " + i + "\n"

    st.markdown(s2)
    df_new = df.copy()
    df_new['age_buckets'] = pd.qcut(df_new['AGE'], 20, labels=False, duplicates='drop')

    # group by 'balance_buckets' and find average campaign outcome per balance bucket
    mean_age = df_new.groupby(['age_buckets'])['TARGET'].mean()
    st.line_chart(mean_age)
    st.caption(
        'As shown the younger person is, the higher probability of positive response to the offer will be.\n And starting from 51 years the average percantage of positive response tends to zero.')
    st.divider()

    st.subheader('Information on Offer Responses:')
    colors = ["#e891c1", "#a0adea"]
    labels = "Did not respond", "Responded"
    f, ax = plt.subplots(1, 2, figsize=(16, 8))
    f.tight_layout(pad=3.0)
    df["TARGET"].value_counts().plot.pie(explode=[0, 0.25], autopct='%1.2f%%', ax=ax[0], shadow=True, colors=colors,
                                         labels=labels, fontsize=12, startangle=125)

    palettes = ["#e4946a", "#4eabb8"]

    sns.barplot(x="EDUCATION", y="PERSONAL_INCOME", hue="TARGET", data=df, palette=palettes,
                estimator=lambda x: len(x) / len(df) * 100)
    ax[1].set(ylabel="(%)")
    ax[1].set_xticklabels(df["EDUCATION"].unique(), rotation=75)
    ax[1].legend(labels=['No response', 'Responded'])
    st.pyplot(f)
    st.caption(
        'As we can see only _12.13%_ positively responded to the bank offer.\n Most responses gave people with _Lower post-secondary vocational education_ and with _Secondary school_.')
    st.divider()


