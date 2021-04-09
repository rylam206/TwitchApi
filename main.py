"""
Ryan Lam and Shi Ming Kuang
CSE 163
Final Project

This is the main program where twitch.py is implemented (if needed)
to take and scrape data from Twitch and formatted into CSV and to be analyzed.
Additional datasets (e.g. Shroud) are manually taken in to be analyzed as well.
"""

import twitch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn import tree

sns.set()


def game_streamer_popularity():
    """
    This function is to answer Research Question 1
    which uses Twitch API implementation from twitch.py to scrape Twitch data
    and analyze how the game Valorant affects aceu, Sykkuno, and Shroud's
    viewership.
    """
    # get_videos -- game_id
    # 1. get game_id
    get_game_request = twitch.get_response(twitch.get_games_query('Valorant'))
    game_id = twitch.get_game_id(get_game_request)
    # 2. get_game_vids(game_id)
    game_vids_request = twitch.get_response(twitch.get_game_vids(game_id))
    twitch.save_json_to_csv(game_vids_request, 'val')
    # 3. Filter by user_login
    # opening & reading csv files:
    val_dataset = pd.read_csv('val.csv')
    # filter val_dataset for streamers instead:
    aceu_mask = val_dataset['user_login'] == 'aceu'
    sykkuno_mask = val_dataset['user_login'] == 'sykkuno'
    shroud_mask = val_dataset['user_login'] == 'shroud'
    filtered_val = val_dataset[aceu_mask | sykkuno_mask | shroud_mask]
    # top_views series contains the VOD w/ highest view_count for each streamer
    top_views = filtered_val.groupby('user_login')['view_count'].max()
    ace_top_views = filtered_val['view_count'] == top_views['aceu']
    syk_top_views = filtered_val['view_count'] == top_views['sykkuno']
    shrd_top_views = filtered_val['view_count'] == top_views['shroud']
    # filtered now only has the VOD info w/ highest view count for the streamer
    filtered = filtered_val[ace_top_views | syk_top_views | shrd_top_views]
    filtered.to_csv('filtered.csv')
    ace_only = streamer_vids('aceu', filtered)
    syk_only = streamer_vids('sykkuno', filtered)
    shrd_only = streamer_vids('shroud', filtered)
    plot_streams_views('aceu', ace_only, filtered)
    plot_streams_views('sykkuno', syk_only, filtered)
    plot_streams_views('shroud', shrd_only, filtered)
    # Compute avgs
    # find avgs before highest-viewed val stream -> id of vod -> <id
    ace_before = ace_only[ace_only['id'] < vid_id(filtered, 'aceu')]
    syk_before = syk_only[syk_only['id'] < vid_id(filtered, 'sykkuno')]
    shrd_before = shrd_only[shrd_only['id'] < vid_id(filtered, 'shroud')]
    avg_before_ace = compute_avg_views(ace_before)
    avg_before_syk = compute_avg_views(syk_before)
    avg_before_shrd = compute_avg_views(shrd_before)
    print('aceu before:', avg_before_ace)
    print('sykkuno before:', avg_before_syk)
    print('shroud before:', avg_before_shrd)
    print()
    # find avgs after highest-viewed val stream
    avg_after_ace = compute_avg_views(ace_only)
    avg_after_syk = compute_avg_views(syk_only)
    avg_after_shrd = compute_avg_views(shrd_only)
    print('aceu after:', avg_after_ace)
    print('sykkuno after:', avg_after_syk)
    print('shroud after:', avg_after_shrd)
    print()
    print('Differences in Averages:')
    diff_ace = avg_after_ace - avg_before_ace
    diff_syk = avg_after_syk - avg_before_syk
    diff_shrd = avg_after_shrd - avg_before_shrd
    print('aceu:', str(diff_ace))
    print('sykkuno:', str(diff_syk))
    print('shroud:', str(diff_shrd))
    print()
    print('Percentage Change:')
    print('aceu:', str((diff_ace / avg_before_ace) * 100))
    print('sykkuno:', str((diff_syk / avg_before_syk) * 100))
    print('shroud:', str((diff_shrd / avg_before_shrd) * 100))


def compute_avg_views(df):
    """
    Takes in DataFrame and returns the average view count
    """
    # filter df -> only view_count column
    view_count_column = df['view_count']
    return view_count_column.mean()


def vid_id(filtered_df, streamer_name):
    """
    Takes in the streamer's name and filtered dataset that contains the
    streamers' highest view count on Valorant.
    The function filters the dataset for the specific streamer
    then returns the id of the streamer's stream.
    """
    high_val_stream = filtered_df[filtered_df['user_login'] == streamer_name]
    val_id_series = high_val_stream.loc[:, 'id']
    stream_id = val_id_series[val_id_series.idxmax()]
    return stream_id


def plot_streams_views(streamer_name, streamer_only, filtered_df):
    """
    This function takes in streamer name, the dataset containing streams from
    the specific streamer and the filtered dataset that contains the streamer's
    highest view count and plots the view counts of each of their streams,
    taking account of their Valorant stream with the highest view count.
    """
    high_val_stream = filtered_df[filtered_df['user_login'] == streamer_name]
    val_view_count_series = high_val_stream.loc[:, 'view_count']
    view_count = val_view_count_series[val_view_count_series.idxmax()]
    # streamer_only have streams listed in chronological order
    # (more recent streams on top)
    sns.catplot(data=streamer_only, y='view_count', x='created_at', kind='bar')
    plt.xticks([])
    plt.xlabel('Streams (from most recent to oldest, left to right)')
    plt.ylabel('View Count')
    plt.title(streamer_name)
    plt.hlines(y=view_count, xmin=0, xmax=len(streamer_only))
    plt.savefig(streamer_name + '.png', bbox_inches='tight')


def streamer_vids(streamer, filtered_df):
    """
    This function takes in a streamer's name and the filtered dataset
    containing streams with the highest views for each streamer
    and returns a DataFrame
    containing the past streams from the streamer.
    """
    streamer_only_df = filtered_df[filtered_df['user_login'] == streamer]
    user_id_series = streamer_only_df['user_id']
    user_id = user_id_series[user_id_series.idxmax()]
    streamer_vods_endpt = twitch.get_user_videos_query(user_id)
    get_request = twitch.get_response(streamer_vods_endpt)
    twitch.save_json_to_csv(get_request, streamer)
    result = pd.read_csv(streamer + '.csv')
    return result


def male_vs_female(male_streamers, female_streamers):
    """
    This function answer research question 2. This function
    takes in a list of male streamers and female streamers
    and plots a graph with view count on the y axis and gender
    on x axis.
    """
    path = '/home/streamers'
    if not os.path.exists(path):
        os.makedirs(path)
    streamers = []
    for i in range(0, len(male_streamers)):
        query = twitch.get_user_query(male_streamers[i])
        response = twitch.get_response(query)
        data = twitch.save_json_to_csv(response, 'streamers/'
                                       + male_streamers[i])
        data = pd.read_csv('streamers/' + male_streamers[i] + '.csv')
        data = data[['display_name', 'view_count', 'id']]
        data['gender'] = 'male'
        streamers.append(data)
        query = twitch.get_user_query(female_streamers[i])
        response = twitch.get_response(query)
        data = twitch.save_json_to_csv(response, 'streamers/'
                                       + female_streamers[i])
        data = pd.read_csv('streamers/' + female_streamers[i] + '.csv')
        data = data[['display_name', 'view_count', 'id']]
        data['gender'] = 'female'
        streamers.append(data)
    streamers = pd.concat(streamers, axis=0, ignore_index=True)
    streamers.to_csv('streamers/streamers.csv')
    data = pd.read_csv('streamers/streamers.csv')
    sns.catplot(x='gender', y='view_count', data=data, kind='bar',
                hue='display_name')
    plt.savefig('male_vs_female.png', bbox_inches='tight')


def combine_csv(list_file):
    """
    This function takes in a list of csv files
    and return a csv file with all the csv files
    combined.
    """
    x = []
    for filename in list_file:
        x.append(pd.read_csv(filename))
    merged = pd.concat(x)
    merged
    return merged


def input_csv():
    """
    This function input csv files and calls the combine_csv
    function to combine multiple csv
    """
    shroud_2015 = combine_csv(['shroud/shroud2015.csv'])
    shroud_2015['year'] = 2015
    shroud_2015.to_csv('shroud/shroud_2015.csv')
    shroud_2016 = combine_csv(['shroud/shroud2016.csv',
                              'shroud/shroud2016(2).csv'])
    shroud_2016['year'] = 2016
    shroud_2016.to_csv('shroud/shroud_2016.csv')
    shroud_2017 = combine_csv(['shroud/shroud2017.csv',
                               'shroud/shroud2017(2).csv',
                               'shroud/shroud2017(3).csv'])
    shroud_2017['year'] = 2017
    shroud_2017.to_csv('shroud/shroud_2017.csv')
    shroud_2018 = combine_csv(['shroud/shroud2018.csv',
                               'shroud/shroud2018(2).csv',
                               'shroud/shroud2018(3).csv',
                               'shroud/shroud2018(4).csv'])
    shroud_2018['year'] = 2018
    shroud_2018.to_csv('shroud/shroud_2018.csv')
    shroud_2019 = combine_csv(['shroud/shroud2016.csv',
                               'shroud/shroud2016(2).csv',
                               'shroud/shroud2019(3).csv'])
    shroud_2019['year'] = 2019
    shroud_2019.to_csv('shroud/shroud_2019.csv')
    shroud_2020 = combine_csv(['shroud/shroud2020.csv',
                               'shroud/shroud2020(2).csv'])
    shroud_2020['year'] = 2020
    shroud_2020.to_csv('shroud/shroud_2020.csv')


def viewership_growth(data):
    """
    This function answer research question 3. This function
    takes in a dataframe and train a decision tree regressor
    model and a linear regression model. It would then plot
    a graph for each model showing you growth of viewership.
    """
    data = data[['Views', 'year']]
    data = data.groupby('year')['Views'].sum()
    data = pd.DataFrame({'year': data.index, 'Views': data.values})
    features = data.drop(columns=['Views'])
    labels = data['Views']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    model = DecisionTreeRegressor()
    model.fit(features_train, labels_train)
    # Test Accuracy Score
    predictions = model.predict(features_test)
    decision_score = accuracy_score(labels_test, predictions)
    print('DecisionTreeRegressor accuracy score: ' + str(decision_score))
    # Test Mean Square Error
    model.fit(features, labels)
    predictions = model.predict(features_test)
    decision_error = mean_squared_error(labels_test, predictions)
    print('DecisionTreeRegressor mean square error: ' + str(decision_error))
    # Visualizing Decision Tree and Predict subscription growth by year
    model.fit(features, labels)
    tree.export_graphviz(model, out_file='views.dot', feature_names=['year'],
                         class_names=sorted(labels.unique()), label='all',
                         rounded=True, filled=True)
    predictions = model.predict([[2017], [2018], [2021],
                                 [2022], [2023], [2024], [2025]])
    # Using LinearRegression to predict yearly growth
    model_2 = LinearRegression()
    model_2.fit(features, labels)
    predictions = model_2.predict(features)
    plt.scatter(features, labels, color='g')
    plt.plot(features, predictions, color='k')
    plt.savefig('viewership_growth.png')
    print('Regression Years: 2017, 2021-2025')
    print(model_2.predict([[2017], [2021], [2022], [2023], [2024], [2025]]))
    linear_score = model_2.score(features, labels)
    print('LinearRegression score: ' + str(linear_score))


def subscription_growth(data):
    """
    This function answer research question 4. This function
    takes in a dataframe and train a decision tree regressor
    model and a linear regression model. It would then plot
    a graph for each model showing you growth of subscribers.
    """
    data = data[['Followers gained', 'year']]
    data = data.groupby('year')['Followers gained'].sum()
    data = pd.DataFrame({'year': data.index, 'Followers gained': data.values})
    features = data.drop(columns=['Followers gained'])
    labels = data['Followers gained']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    model = DecisionTreeRegressor()
    model.fit(features_train, labels_train)
    # Test Accuracy Score
    predictions = model.predict(features_test)
    decision_score = accuracy_score(labels_test, predictions)
    print('DecisionTreeRegressor accuracy score: ' + str(decision_score))
    # Test Mean Square Error
    model.fit(features, labels)
    predictions = model.predict(features_test)
    decision_error = mean_squared_error(labels_test, predictions)
    print('DecisionTreeRegressor mean square error: ' + str(decision_error))
    # Visualizing Decision Tree and Predict subscription growth by year
    model.fit(features, labels)
    tree.export_graphviz(model, out_file='followers.dot',
                         feature_names=['year'],
                         class_names=sorted(labels.unique()), label='all',
                         rounded=True, filled=True)
    predictions = model.predict([[2021], [2022], [2023], [2024], [2025]])
    # Using LinearRegression to predict yearly growth
    model_2 = LinearRegression()
    model_2.fit(features, labels)
    predictions = model_2.predict(features)
    plt.scatter(features, labels, color='g')
    plt.plot(features, predictions, color='k')
    plt.savefig('subscription_growth.png')
    print(model_2.predict([[2017], [2021], [2022], [2023], [2024], [2025]]))
    linear_score = model_2.score(features, labels)
    print('LinearRegression score: ' + str(linear_score))


def main():
    male_streamers = ['ninja', 'tfue', 'shroud', 'rubius', 'xqcow']
    female_streamers = ['pokimane', 'amouranth', 'arigameplays',
                        'loserfruit', 'lilypichu']
    male_vs_female(male_streamers, female_streamers)
    input_csv()
    csv_file_list = ['shroud/shroud_2015.csv', 'shroud/shroud_2016.csv',
                     'shroud/shroud_2017.csv', 'shroud/shroud_2018.csv',
                     'shroud/shroud_2019.csv', 'shroud/shroud_2020.csv']
    merged = combine_csv(csv_file_list)
    print('Viewership: ')
    print()
    viewership_growth(merged)
    print('Subscription: ')
    print()
    subscription_growth(merged)
    game_streamer_popularity()


if __name__ == '__main__':
    main()
