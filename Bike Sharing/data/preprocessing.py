from os.path import join, dirname
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


def rename_attributes(hour_df):
    hour_df.rename(columns={'instant': 'rec_id',
                            'dteday': 'datetime',
                            'holiday': 'is_holiday',
                            'workingday': 'is_workingday',
                            'weathersit': 'weather_condition',
                            'hum': 'humidity',
                            'mnth': 'month',
                            'cnt': 'total_count',
                            'hr': 'hour',
                            'yr': 'year'}, inplace=True)
    hour_df['datetime'] = pd.to_datetime(hour_df.datetime)
    hour_df['season'] = hour_df.season.astype('category')
    hour_df['is_holiday'] = hour_df.is_holiday.astype('category')
    hour_df['weekday'] = hour_df.weekday.astype('category')
    hour_df['weather_condition'] = hour_df.weather_condition.astype('category')
    hour_df['is_workingday'] = hour_df.is_workingday.astype('category')
    hour_df['month'] = hour_df.month.astype('category')
    hour_df['year'] = hour_df.year.astype('category')
    hour_df['hour'] = hour_df.hour.astype('category')
    return hour_df


def distribute_trend_pointplot(param):
    fig, ax = plt.subplots()
    sn.pointplot(data=hour_df[['hour',
                               'total_count',
                               '{}'.format(param)]],
                 x='hour', y='total_count',
                 hue='{}'.format(param), ax=ax)
    ax.set(title="Point plot {} wise hourly distribution of counts".format(param))
    fig.savefig("eda/{}_pointplot_distribute.png".format(param))


def distribute_trend_barplot(param):
    fig, ax = plt.subplots()
    sn.barplot(data=hour_df[['{}'.format(param),
                             'total_count']],
               x="{}".format(param), y="total_count")
    ax.set(title="Bar plot {} distribution of counts".format(param))
    fig.savefig("eda/{}_barplot_distribute.png".format(param))


if __name__ == '__main__':
    data_path = join(dirname(__file__), "corpus", "hour.csv")
    hour_df = pd.read_csv(data_path)
    hour_df = rename_attributes(hour_df)
    distribute_trend_pointplot(param="season")
    distribute_trend_pointplot(param="weekday")
    distribute_trend_barplot(param="month")
    distribute_trend_barplot(param="hour")
