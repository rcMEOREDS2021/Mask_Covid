'''
Author: Group 5
'''
from detect_delimiter import detect
import io
import pandas as pd
import requests as r
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
pd.options.mode.chained_assignment = None

#replace the variables with the appropriate url, your local machine path
#and any file names you wish to pull in.
path = 'C:\\Users\\prasc\\Desktop\\final_new_new'
url_1 = 'https://download.bls.gov/pub/time.series/la/la.series'
url_2 = 'https://download.bls.gov/pub/time.series/la/la.data.64.County'
url_3 = 'https://data.cdc.gov/api/views/qz3x-mf9n/rows.csv?accessType=DOWNLOAD'
url_4 = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
url_5 = 'https://raw.githubusercontent.com/kjhealy/fips-codes/master/state_and_county_fips_master.csv'
#file1 = 'Provisional_COVID-19_Death_Counts_by_County_and_Race.csv'

#this function will create the get request, check the stauts code, detect the
#delimiter and create the DF.
def df_creator(file, type='csv'):
    res = r.get(file)
    #check status code before proceeding
    if res.status_code == 200:
        #delim will attempt to auto detect the delimiter and default to ','
        delim = detect(res.text, default=',')
        #if one isn't found.
        if type == 'flat':
            df = pd.read_csv(io.StringIO(res.text), delimiter=delim, engine='python')
            print("Data Frame has been created for %s"%file)
            return df
        elif file == url_3:
            df = pd.read_csv(file, low_memory=False)
            print("Data Frame has been created for %s"%file)
            return df
        else:
            df = pd.read_csv(file)
            print("Data Frame has been created for %s"%file)
            return df
    else:
        print("There was a problem. Please check the URL or Filename.")
        sys.exit()


#this function is designed to centralize all initial column changes for ease of
#access
def column_cleanup(df, df_name="default"):
    #change the case to all column headers to lowercase for ease of merging
    df.columns = map(str.lower, df.columns)
    #create a list of column headers to iterate through
    column_list = df.columns
    for i in column_list:
        #column removal list.
        if df_name == "ei_values" and i in ['footnote_codes']:
            df.drop(columns=i, axis=1, inplace=True)
        elif df_name == "employment_information" and i in ['area_type_code', 'measure_code', 'seasonal','srd_code', 'footnote_codes', 'begin_year', 'begin_period', 'end_year', 'end_period']:
            df.drop(columns=i, axis=1, inplace=True)
        elif df_name == "stay_at_home" and i in ['state_tribe_territory', 'county_name', 'fips_state', 'fips_county', 'date', 'effective_na_reason', 'expiration_na_reason', 'origin_dataset',
        'county', 'url', 'citation', 'flag']:
            df.drop(columns=i, axis=1, inplace=True)
        #drop columns from Kaggle data
        elif df_name == "kaggle" and i in ['state', 'county']:
            df.drop(columns=i, axis=1, inplace=True)
        #if space is at the end or beginning of a header title
        elif i[-1] == ' ' or i[1] == ' ':
            df.columns = df.columns.str.replace(' ', '')
        #transform column type to datetime
        elif i in ['effective_date', 'expiration_date']:
            df[i] =  pd.to_datetime(df[i])
    return df



#data frame creation section
employment_information = df_creator(url_1, 'flat')
#filtering out all non unemployment labor data and anything older than 2020
employment_information = employment_information[(employment_information['end_year'] == 2020) & (employment_information['measure_code'] == 3)]
column_cleanup(employment_information, "employment_information")
#remove all extra spaces from the series_id column values so the merge can be preformed without problems.
employment_information['series_id'] = employment_information['series_id'].str.strip()


ei_values = df_creator(url_2, 'flat')
#remove all values that aren't for 2020
ei_values = ei_values[ei_values['year'] == 2020]
#clean headers and drop columns
column_cleanup(ei_values, "ei_values")
#remove all extra spaces from the series_id column values so the merge can be preformed without problems.
ei_values['series_id'] = ei_values['series_id'].str.strip()


#create the stay at home policy by county dataframe.
stay_at_home = df_creator(url_3)
#remove all rows that indicate an stay at home order wasn't found.
stay_at_home = stay_at_home[(stay_at_home['Current_order_status'] != 'No order found')]
#adding leading zeros to the fips_state code
stay_at_home['FIPS_State'] = stay_at_home['FIPS_State'].astype(str).str.zfill(2)
#adding leading zeros to the fips_county code
stay_at_home['FIPS_County'] = stay_at_home['FIPS_County'].astype(str).str.zfill(3)
#combing the newly created state and county codes.
stay_at_home['FIPS_Code'] = stay_at_home['FIPS_State'] + stay_at_home['FIPS_County']
#clean headers and drop columns
column_cleanup(stay_at_home, "stay_at_home")

#splitting the stay_at_home dataframe into two: those without expiration date and those with an expiration date
no_expiration_months = stay_at_home[stay_at_home.expiration_date.isnull()]
#stripping the date down to the month, adding a leading zero to the month, and finishing it off by adding an M to the front
no_expiration_months["period"] = pd.to_datetime(no_expiration_months["effective_date"], format='%Y-%m-%d').apply(lambda x: x.strftime('%m'))
no_expiration_months['period'] = no_expiration_months['period'].astype(str).str.zfill(2)
no_expiration_months['period'] = 'M'+no_expiration_months['period'].astype(str)


expiration_months = stay_at_home[stay_at_home.expiration_date.notnull()]
#create a seperate row for each month a stay-at-home order was active between the two dates.
expiration_months = expiration_months.merge(expiration_months.apply(lambda s: pd.date_range(s.effective_date, s.expiration_date, freq='30D'), 1)\
                   .explode()\
                   .rename('period')\
                   .dt.strftime('M%m'),
                 left_index=True,
                 right_index=True)

#combine the two lists and overwrite the original dataframe
stay_at_home = expiration_months.append(no_expiration_months)

#create a subset of fips_code and period of counties that had a mandate
mandate_county_boolean = stay_at_home.loc[stay_at_home['order_code'].notnull(), ['fips_code']]

mandate_county_boolean['mandate_county'] = 'TRUE'


#create a dataframe with covid deaths and cases from kaggle source file
kaggle_df = df_creator(url_4)
#remove the trailing zeros
kaggle_df['fips'] = kaggle_df['fips'].astype(str).str[:-2]
#add leading zeroes to the fips codes
kaggle_df['fips'] = kaggle_df['fips'].astype(str).str.zfill(5)
#create a period key from the date
kaggle_df["period"] = pd.to_datetime(kaggle_df["date"], format='%Y-%m-%d').apply(lambda x: x.strftime('M%m'))
#clean headers and drop columns
column_cleanup(kaggle_df, "kaggle")
#create a percentage change by day
kaggle_df['cases_pct_change']= kaggle_df.groupby(['period', 'fips'])['cases'].pct_change()
kaggle_df['deaths_pct_change']= kaggle_df.groupby(['period', 'fips'])['deaths'].pct_change()
#summarize data by month
kaggle_df = kaggle_df.groupby(['period', 'fips'], as_index=False).agg({'deaths':'sum', 'cases':'sum','cases_pct_change':'mean', 'deaths_pct_change':'mean'})
#rename columsn so they are uniform with the rest of the data frames
kaggle_df.rename(columns={'fips': 'fips_code'}, inplace=True)



#create a dataframe with all the fips_codes so consistent county and state information can be moved over.
fips_codes = df_creator(url_5)
#rename the fips column for consistentency
fips_codes.rename(columns={'fips': 'fips_code'}, inplace=True)
#add leading zeroes to the fips code
fips_codes['fips_code'] = fips_codes['fips_code'].astype(str).str.zfill(5)


#creates an unemployment values dataframe.
final_df = pd.merge(employment_information, ei_values, how="inner", on='series_id')
#remove excess characters from the State/County FIPS code.
final_df['area_code'] = final_df['area_code'].str[2:7]
#remanem area_code column to FIPS_Code
final_df.rename(columns={'area_code': 'fips_code'}, inplace=True)
#merge the unemployment_df to the stay_at_home df using their FIPS codes and the period in which the
#covid mandates started
final_df = pd.merge(final_df, stay_at_home, how="left", on=['fips_code', 'period'])
#merge kaggle into the the main dataframe
final_df = pd.merge(final_df, kaggle_df, how="left", on=['fips_code', 'period'])
#merge FIPS info into the main dataframe
final_df = pd.merge(final_df, fips_codes, how="left", on=['fips_code'])
#merge the mandate county boolean
final_df = pd.merge(final_df, mandate_county_boolean, how="left", on=['fips_code'])
#rename columns in final_df
final_df.rename(columns={'value': 'unemployment_rate'}, inplace=True)
#fill na vales in the cases_deaths, pct_change and cases_pct_change
final_df[['deaths', 'cases', "cases_pct_change", "deaths_pct_change"]] = final_df[['deaths', 'cases', "cases_pct_change", "deaths_pct_change"]].fillna(value=0)
#mark duplicate columns for months that had more than one mandate active
final_df['duplicate'] = final_df.duplicated(subset=['fips_code', 'period'])
#replace inf values for the death percentage change with 0
final_df['deaths_pct_change'] = final_df['deaths_pct_change'].replace(np.inf, 0)
final_df.drop(columns='series_id', axis=1, inplace=True)

#create dummy variables
pd.get_dummies(final_df, columns=['current_order_status'])
encoder_stay_home = ce.OneHotEncoder(cols='current_order_status', handle_unknown='return_nan', return_df=True, use_cat_names=True)
#Fit and transform Data
data_encoded = encoder_stay_home.fit_transform(final_df)
#remove any duplicates based on state, county and order code status
data_encoded.drop_duplicates(subset=["fips_code", "period", "order_code"], inplace = True)

#create a csv file
data_encoded.to_csv(path+'final.csv', index = False)

#correlation plot
p_corr_all = data_encoded.corr()
mask = np.zeros_like(p_corr_all)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(p_corr_all, cmap = 'Spectral_r', mask = mask, square = True, vmin = -1, vmax = 1)
plt.title('Correlation Matrix')
