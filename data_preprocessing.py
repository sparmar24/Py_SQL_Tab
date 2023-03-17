'''Import libraries'''
import pandas as pd
import numpy 

df = pd.read_csv("Absenteeism_data.csv")
'''Create cpy of original file before preprocessing'''
raw = df.copy()

#print(df_data)
#print(df.head)
#print(raw.columns)
#print(df.info)

'''Drop ID and Age feature'''
df = df.drop(["ID"], axis=1)
df_no_age = df.drop(["Age"], axis=1)
# print(df_no_age.info)
# print(df_no_age.columns)

# print(df_no_age["Reason for Absence"])
# print(df_no_age["Reason for Absence"].min())
# print(df_no_age["Reason for Absence"].max())
# print(df_no_age["Reason for Absence"].unique())
# print(sorted(df_no_age["Reason for Absence"].unique()))

'''Create dummy variables for "Reason for Absence" and "Age"'''
reason_columns = pd.get_dummies(df_no_age["Reason for Absence"], drop_first=True)
# print(reason_columns)
age_dummies = pd.get_dummies(df["Age"])
# print(age_dummies)
# print(df.columns.values)
df = df.drop(["Reason for Absence"], axis=1)

'''Group different reson into 4 groups of similar reasons'''
reason_type_1 = reason_columns.loc[:, :14].max(axis=1) #-> operation horizontal axis
reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1) #-> operation horizontal axis
reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1) #-> operation horizontal axis
reason_type_4 = reason_columns.loc[:, 22:].max(axis=1) #-> operation horizontal axis

# print(df)
'''Concat data frame with 4 reasons dummy data'''
df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)
column_names = ['Date', 'Transportation Expense', 'Distance to Work',
 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
 'Pet', 'Absenteeism Time in Hours', 'reason_1', 'reason_2', 'reason_3', 'reason_4' ]
df.columns = column_names

# print(df["Date"])
column_names_reordered = ['reason_1', 'reason_2', 'reason_3', 'reason_4', 'Date', 'Transportation Expense', 'Distance to Work',
 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
 'Pet', 'Absenteeism Time in Hours']
df = df[column_names_reordered]
# print(df)
# print(df["Date"])

'''Concat'''
df_concatenated = pd.concat([df_no_age, age_dummies], axis=1)
# print(df_concatenated.columns.values)
# print(df_no_age.columns.values)
# print(age_dummies.columns.values)
# df_concatenated = df_concatenated.drop(["Reason for Absence"], axis=1)
column_names_rearranged = ['Reason for Absence', 'Date', 'Transportation Expense', 'Distance to Work',
 'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children', 'Pet',
 '27', '28', '29', '30', '31', '32', '33', '34', '36', '37', '38', '39', '40', '41', '43',
                '46', '47', '48', '49', '50', '58', 'Absenteeism Time in Hours']
# print(df_concatenated.columns.values)
df_concatenated.columns = column_names_rearranged
# print(df_concatenated)

'''copy the concatenated data frame'''
df_checkpoint = df_concatenated.copy()

'''Change format of date'''
df_reason_mod = df.copy()
df_reason_mod["Date"] = pd.to_datetime(df_reason_mod["Date"], format="%d/%m/%Y")
# print(df_reason_mod.info())

print(df_reason_mod["Date"][0])
print(df_reason_mod["Date"][0].month)

'''Extract the month value'''
list_months = []
#print(list_months)

for _ in range(700):
    list_months.append(df_reason_mod["Date"][_].month)

#print(len(list_months))

df_reason_mod["Month_value"] = list_months
print(df_reason_mod.head(5))

'''for exp:'''
#print(df_reason_mod["Date"][699].weekday())

'''extract days of the week'''
def date_to_weekday(date_value):
    return date_value.weekday()

df_reason_mod["Day_of_the_week"] = df_reason_mod["Date"].apply(date_to_weekday)
#print(df_reason_mod.head(5))

df_reason_mod = df_reason_mod.drop(["Date"], axis=1)
'''print(df_reason_mod.columns.values())
'''

column_names_upd = ['reason_1', 'reason_2', 'reason_3', 'reason_4', 'Month_value', 'Day_of_the_week',
       'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
       'Pet', 'Absenteeism Time in Hours']
df_reason_mod = df_reason_mod[column_names_upd]
#print(df_reason_mod.head(5))

df_reason_date_mod = df_reason_mod.copy()

print(type(df_reason_date_mod['Transportation Expense'][0]))
print(type(df_reason_date_mod['Distance to Work'][0]))
print(type(df_reason_date_mod['Age'][0]))
print(type(df_reason_date_mod['Daily Work Load Average'][0]))
print(type(df_reason_date_mod['Body Mass Index'][0]))


'''Education, Children, Pets'''
print(df_reason_date_mod)
print(df_reason_date_mod["Education"].unique())
print(df_reason_date_mod["Education"].value_counts())
df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1:0, 2:1, 3:1, 4:1})
print(df_reason_date_mod["Education"].unique())
print(df_reason_date_mod["Education"].value_counts())


'''Final check point'''
df_preprocessed = df_reason_date_mod.copy()
print(df_preprocessed.head(10))



