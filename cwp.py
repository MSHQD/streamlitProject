import streamlit as st
import plotly.express as px
import plotly.graph_objs as pg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.subplots as sub
import numpy as np
from tabulate import tabulate
import dash_html_components as html
import plotly.subplots as sp
import matplotlib.colors as colors

#sunburst, marginal
#разобраться с медианой

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Title
st.title("Stress level research")
st.header('This dataset illustrates the stress level of students and shows the dependence of its level on some factors.')
st.text('First, let\'s look at the data in the dataset.')
st.code('''df = pd.read_csv('StressLevelDataset.csv')
df''')
df = pd.read_csv('StressLevelDataset.csv')
df_head = df.head()
df_tail = df.tail()
st.table(df_head)
st.table(df_tail)


st.header('Data cleanup')


st.text('''For more convenient and easy work, you should clean the dataset. 
This will prevent future errors and make the data more readable.
In order to clean the dataset, it is necessary to analyze the data
for unnecessary information such as undefined values and inaccurate values.''')


count_null = df.isnull().sum(axis=0)
st.code('count_null = df.isnull().sum(axis=0)')
st.metric(label="NaN", value=count_null.sum())


# Создаем DataFrame для вывода результатов
#result_df = pd.DataFrame({'Column Name': count_null.index, 'None Count': count_null.values})
result_df = pd.DataFrame({'Data': count_null.keys(), 'NaNs Count': count_null.values})
st.table(result_df)


'''As you can see, my dataset does not contain any data that could in any way
create inaccuracies in the analysis.'''




st.header('''Overview''')

col1, col2, col3 = st.columns(3)
with col1:
   st.metric(label="Stress level = 0", value=(df['stress_level'] == 0).sum())
with col2:
   st.metric(label="Stress level = 1", value=(df['stress_level'] == 1).sum())
with col3:
   st.metric(label="Stress level = 2", value=(df['stress_level'] == 2).sum())

'''As you can see, on average there is no particular stand out value: the number of students,
who rate their stress level at 0 out of 2 are slightly more than those who rate their stress level at 2 out of 2.'''

st.markdown('''Let's look at the minimum, mean, std, and maximum
measurement values that are contained in the dataset''')
#select the 21 columns for summary statistics
columns = df.iloc[:, :len(df)]

#calculate summary statistics
summary = columns.describe().transpose().round(2)[['min', 'mean', 'std', 'max']].transpose()
st.table(summary)


'''I have created a list that will hold the names of the aspects for which the dataset has information
available in the dataset.
'''


columns_names = list(df.columns)
columns_names = [items.replace('_', ' ') for items in columns_names]
st.code(columns_names)


st.markdown('''Let's explore the dataset and derive the main values''')

students = len(df)
above_avrg_stress = len(df.loc[df["stress_level"] > df["stress_level"].mean()])
depressed = len(df.loc[df["depression"] >= 10])
coun_ment_health = len(df.loc[df.mental_health_history == 1])
sleep_lvl_below_avrg = len(df.loc[df["sleep_quality"] < df["sleep_quality"].mean()])
peer_pres = len(df.loc[df["peer_pressure"] >= 4])
avrg_academic_performance = round(df.academic_performance.mean(), 2)
#creating an aray with avarage figures

main_values = {
    'Number of students': [students],
    'Mental health history': [coun_ment_health],
    'Stress level above average': [above_avrg_stress],
    'High level of depression': [depressed],
    'Feels peer pressure': [peer_pres],
    'Sleep problems': [sleep_lvl_below_avrg]
        }

main_inf = pd.DataFrame({'Statistics': main_values.keys(), 'Figures': main_values.values()})
st.write(main_inf)


'''We can clearly see that there are quite a lot of students experiencing a rather high level of stress.
A similar conclusion can be made about the level of stress, which is assessed as "high"." '''+str(depressed)+''' from 
''' + str(students) + ''' students'''

'''Let's take a look at the averages of each factor:'''
#------------------------------------------------------------------------------------------
#PLOT1


mean_values = df.mean()
fig = plt.figure(figsize=(10, 6))
col_map = plt.get_cmap('tab20c')
plt.bar(mean_values.index, mean_values.values, width=0.8, color=col_map.colors, edgecolor='maroon')
plt.title('Mean values of factors')
plt.xlabel('Factors')
plt.ylabel('Mean value')
plt.grid(True)
plt.xticks(rotation=90)
st.pyplot(plt)


#------------------------------------------------------------------------------------------
#PLOT2

'''Now, I would like to categorize the available factors into 5 groups:'''


types_of_factors0 = {
    'Pcycological': [columns_names[0:4]],
    'Physiological': [columns_names[4:8]],
    'Enviromental': [columns_names[8:12]],
    'Academic': [columns_names[12:16]+[columns_names[18]]],
    'Social': [columns_names[16:18]+[columns_names[19]]]
}
types_of_factors0 = pd.DataFrame(list(types_of_factors0.items()), columns=['Type', 'Factors'])

st.write(types_of_factors0)


'''In order to understand from what factors the level of stress of trainees depends on, let's look at the number of negative 
reports, categorizing them by factors:'''


factors = ['Psychological', 'Physiological', 'Environmental', 'Academic', 'Social']

negative = [
    (df[['anxiety_level', 'self_esteem', 'mental_health_history', 'depression']].apply(lambda x: x.lt(3).sum(),
    axis=1)).sum(),
    (df[['headache', 'blood_pressure', 'sleep_quality', 'breathing_problem']].apply(lambda x: x.gt(3).sum(), axis=1)).sum(),
    (df[['noise_level', 'living_conditions', 'safety', 'basic_needs']].apply(lambda x: x.gt(3).sum(), axis=1)).sum(),
    (df[['academic_performance', 'study_load', 'teacher_student_relationship', 'future_career_concerns']].apply(lambda x:
    x.lt(3).sum(), axis=1)).sum(),
    (df[['social_support', 'peer_pressure', 'extracurricular_activities', 'bullying']].apply(lambda x: x.gt(3).sum(),
    axis=1)).sum()
]


plt.figure(figsize=(10, 6))
sns.barplot(x=factors, y=negative, palette='Set3')
plt.title('Number of student\'s negative reports by factors')
plt.xlabel('Factor')
plt.ylabel('Number of reports')
plt.xticks(rotation=90)
st.pyplot(plt)

#PLOT3-----------------------------------------------------------------------------------------------

'''Since the largest number of negative reports relate to the academic factor, let us examine academic
reports in more detail. In addition, we know that each of the academic factor scores is a value
between 0 and 5, so we can look at the distribution of student responses'''
labels = ['0 level', '1 level', '2 level', '3 level', '4 level', '5 level']
plot3 = sub.make_subplots(2, 2, specs=[[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]])
plot3.add_trace(pg.Pie(labels=labels, values=df['academic_performance'].value_counts().values,
                       title='Academic performance'), 1, 1)
plot3.add_trace(pg.Pie(labels=labels, values=df['study_load'].value_counts().values, scalegroup='one',
                     title="Study load"), 1, 2)
plot3.add_trace(pg.Pie(labels=labels, values=df['future_career_concerns'].value_counts().values, scalegroup='one',
                     title="Future career concerns"), 2, 1)
plot3.add_trace(pg.Pie(labels=labels, values=df['teacher_student_relationship'].value_counts().values, scalegroup='one',
                     title="Relationship with teacher"), 2, 2)
plot3.update_layout(title_text='Academic factors')
st.write(plot3)

'''Most noticeable from the graph is that quite a large percentage of students rate each of the indicators related to the
to learning at a rather low level. Assessment "0" occupies the largest part of the circular graph.'''

#PLOT4-----------------------------------------------------------------------------------------

'''Let's take a look at how much a particular factor affects students' stress levels.'''



plot_correlations = df.corr()
fig = plt.figure(figsize=(8, 8))
colors = sns.diverging_palette(220, 20, s=80, as_cmap=True)
plot3 = sns.heatmap(plot_correlations.iloc[:-1, -1:], annot=True, cmap=colors)

st.pyplot(fig)


'''As you can see, bullying (social factor), anxiety level and depression (psycological factors), future career concerns (academic factor), etc. have the greatest impact on stress level. Since these indicators relate to completely different 
types of factors, let's try to identify some kind of trends.
'''

'''Let us look at each type individually'''


#PLOT5-----------------------------------------------------------------------------------------


st.title('Detailed overview')
st.subheader('Psycological factors correlations')

cor_anxiety = round(df["stress_level"].corr(df["anxiety_level"]), 2)
cor_self_esteem = round(df["stress_level"].corr(df["self_esteem"]), 2)
cor_depression = round(df["stress_level"].corr(df["depression"]), 2)
cor_history = round(df["stress_level"].corr(df["mental_health_history"]), 2)
psycological_factors = {
    'Anxiety': cor_anxiety,
    'Self esteem': cor_self_esteem,
    'Depression': cor_depression,
    'Mental health history': cor_history
}
psyco_factors = pd.DataFrame({'Factors': psycological_factors.keys(), 'Figures': psycological_factors.values()})
st.write(psyco_factors)


plot5 = px.scatter(df, x='depression', y='anxiety_level', trendline='lowess', color='stress_level',
                   color_continuous_scale="sunsetdark", title='Correlation between depression level and anxiety')
st.write(plot5)


#PLOT6-----------------------------------------------------------------------------------------


st.subheader('Physiological factors correlations')
cor_headache = round(df["stress_level"].corr(df["headache"]), 2)
cor_blood_pressure = round(df["stress_level"].corr(df["blood_pressure"]), 2)
cor_sleep = round(df["stress_level"].corr(df["sleep_quality"]), 2)
cor_breathing_problem = round(df["stress_level"].corr(df["breathing_problem"]), 2)
physiological_factors = {
    'Headache': cor_headache,
    'Blood pressure': cor_blood_pressure,
    'Sleep quality': cor_sleep,
    'Breathing problem': cor_breathing_problem
}
physio_factors = pd.DataFrame({'Factors': physiological_factors.keys(), 'Figures': physiological_factors.values()})
st.write(physio_factors)


plot_headache = px.bar(df, x='breathing_problem', y='headache', facet_col="stress_level", color='breathing_problem',
                       title='Dependence of stress levels, headaches and breathing problems')
st.write(plot_headache)

'''So, as you can see from the graph, those with high stress levels estimate headaches
and respiratory problems by more than half.'''



#PLOT7-----------------------------------------------------------------------------------------


st.subheader('Academic factors correlations')
cor_performance = round(df["stress_level"].corr(df["academic_performance"]), 2)
cor_study_load = round(df["stress_level"].corr(df["study_load"]), 2)
cor_relationship = round(df["stress_level"].corr(df["teacher_student_relationship"]), 2)
cor_career_concerns = round(df["stress_level"].corr(df["future_career_concerns"]), 2)
academic_factors = {
    'Academic performance': cor_performance,
    'Study load': cor_study_load,
    'Relationships with teachers': cor_relationship,
    'Future career concerns': cor_career_concerns
}
acad_factors = pd.DataFrame({'Factors': academic_factors.keys(), 'Figures': academic_factors.values()})
st.write(acad_factors)


st.subheader('Social factors correlations')
cor_social_support = round(df["stress_level"].corr(df["social_support"]), 2)
cor_peer_pressure = round(df["stress_level"].corr(df["peer_pressure"]), 2)
cor_bullying = round(df["stress_level"].corr(df["bullying"]), 2)
social_factors = {
    'Social support': cor_social_support,
    'Peer pressure': cor_peer_pressure,
    'Bullying': cor_bullying,
}

soc_factors = pd.DataFrame({'Factors': social_factors.keys(), 'Figures': social_factors.values()})
st.write(soc_factors)


st.subheader('Life quality factors correlations')
cor_noise = round(df["stress_level"].corr(df["noise_level"]), 2)
cor_living_conditions = round(df["stress_level"].corr(df["living_conditions"]), 2)
cor_safety = round(df["stress_level"].corr(df["safety"]), 2)
cor_basic_needs = round(df["stress_level"].corr(df["basic_needs"]), 2)
col1, col2, col3, col4 = st.columns(4)
life_qual__factors = {
    'Noise': cor_noise,
    'Living conditions': cor_living_conditions,
    'Safety': cor_safety,
    'Basic needs': cor_basic_needs
}
life_factors = pd.DataFrame({'Factors': life_qual__factors.keys(), 'Figures': life_qual__factors.values()})
st.write(life_factors)

'''Let's explore the dataset by analyzing some values. This will help us in the future
when making hypotheses. First, let's look at the number of people who rate their stress as 0, 1, and 2.'''




#--------------------------------------------------------------------------------------------------
st.header('New Data')


'''Since the level of depression significantly affects the level of stress, let's try to trace the relationship.
To do this, let's add a column "depression_experience" to our dataset, whose values will be "0" and "1" depending on the value that the stress level takes on. If a student rates their level of depression at more than 12,
we will consider that at the moment the student is in a state of depression'''

df['depression_experience'] = np.where(df['depression'] > 12, 1, 0).astype(int)

students_with_depression = (df['depression_experience'].sum() / students) * 100
st.metric(label="Percentage of students experiencing depression", value=students_with_depression.round(1))


'''Also, based on the "safety" column, we will create a new column. Since the "safety" and "noise" columns
items take values from 0 to 5, I'll assume that a student feels unsafe if he or she rates
their safety level less than 3 and their noise level more than 3'''
df['feel_unsafe'] = ((df['safety'] < 3) & (df['noise_level'] > 3)).astype(int)

students_feel_unsafe = (df['feel_unsafe'].sum() / students) * 100
st.metric(label="Percentage of students feeling unsafe", value=students_feel_unsafe.round(1))

'''Let's take a look at the added data:'''

st.table(df[['depression_experience', 'feel_unsafe']].head())


'''Suppose that students' stress level depends on psychological factors such as: anxiety level, self esteem,
 depression and mental health history'''

'''Let's make the following assumption:'''
st.subheader('''Those students who experience a depression or feel unsafe have higher level of stress''')
'''Also, since the level of stress largely depends on the indicators of anxiety and self esteem, let's check the correctness of the hypothesis on the graph
that shows the dependence between anxiety level and self esteem of different groups of students (those who experience depression and not; those who feel unsafe and not).
X axis represents the level of self esteem, Y axis shows the level of anxiety level. The color illustrates the level of a stress.
'''


plot_hypothesis = sub.make_subplots(rows=2, cols=2, subplot_titles=('Not depressed','Depressed', 'Feel safe',
'Feel unsafe'), shared_yaxes=True)
plot_hyp1_1 = px.scatter(df[df['depression_experience'] == 0], x='self_esteem', y='anxiety_level', trendline='lowess',
color='stress_level')
plot_hyp1_1.update_xaxes(title_text='self_esteem')
plot_hyp1_1.update_yaxes(title_text='anxiety_level')
trace_plot_hyp1_1 = plot_hyp1_1['data'][0]
plot_hypothesis.add_trace(trace_plot_hyp1_1, row=1, col=1)

plot_hyp1_2 = px.scatter(df[df['depression_experience'] == 1], x='self_esteem', y='anxiety_level', trendline='lowess',
color='stress_level')
plot_hyp1_2.update_xaxes(title_text='self_esteem')
plot_hyp1_2.update_yaxes(title_text='anxiety_level')
trace_plot_hyp1_2 = plot_hyp1_2['data'][0]
plot_hypothesis.add_trace(trace_plot_hyp1_2, row=1, col=2)

plot_hyp2_1 = px.scatter(df[df['feel_unsafe'] == 0], x='self_esteem', y='anxiety_level', trendline='lowess',
color='stress_level')
plot_hyp2_1.update_xaxes(title_text='self_esteem')
plot_hyp2_1.update_yaxes(title_text='anxiety_level')
trace_plot_hyp2_1 = plot_hyp2_1['data'][0]
plot_hypothesis.add_trace(trace_plot_hyp2_1, row=2, col=1)

plot_hyp2_2 = px.scatter(df[df['feel_unsafe'] == 1], x='self_esteem', y='anxiety_level', trendline='lowess',
color='stress_level')
plot_hyp2_2.update_xaxes(title_text='self_esteem')
plot_hyp2_2.update_yaxes(title_text='anxiety_level')
trace_plot_hyp2_2 = plot_hyp2_2['data'][0]
plot_hypothesis.add_trace(trace_plot_hyp2_2, row=2, col=2)

plot_hypothesis.update_yaxes(title_text="yaxis 1 title", row=1, col=1)
plot_hypothesis.update_yaxes(title_text="yaxis 2 title", row=1, col=2)
plot_hypothesis.update_yaxes(title_text="yaxis 3 title", row=2, col=1)
plot_hypothesis.update_yaxes(title_text="yaxis 4 title", row=2, col=2)

plot_hypothesis.update_layout(height=800, width=1000, title='Сorrelation between depression and anxiety of a student', showlegend=True)
st.plotly_chart(plot_hypothesis, theme=None)

'''Our assumption is correct:
the graph clearly shows a trend. Students who are depressed or feel insecure experience higher levels of stress.'''
