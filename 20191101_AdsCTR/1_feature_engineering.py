import pandas as pd
import numpy as np
from datetime import datetime
import pickle
def get_time_of_day(hour):
    '''给定当前小时，返回1表示早上，2表示下午，3表示晚上'''
    if hour <= 12:
        return 1
    elif hour <= 18:
        return 2
    else:
        return 3

def get_user_df(input_, name):
    '''通过groupby给出的input_(一个df)，返回一个series'''
    index_names = ['click_interval_mean', 'click_interval_std', 'click_ratio', 'display_time',
                   'click_interval_min', 'click_interval_max', 'weekday_click_ratio',
                   'morning_click_ratio', 'afternoon_click_ratio']
    index_names = [name + i for i in index_names]
    df = input_.reset_index()
    if len(df) == 1:  # 如果长度为1
        if df.label[0] == 1:
            l = [0, 0, 1, 1, 0, 0]
            if df.time_weekday[0] == 1:
                l.append(1)
            else:
                l.append(0)
            if df.time_of_day[0] == 1:
                l.extend([1, 0])
            elif df.time_of_day[0] == 2:
                l.extend([0, 1])
            else:
                l.extend([0, 0])
        else:
            l = [0] * 9
    else:  # 如果长度不只1
        df_click = df.loc[df.label == 1]
        if len(df_click) > 1:
            df_click = df.loc[df.label == 1].sort_values(by = 'operTime').reset_index()  # 默认升序
            interval_time = ((df_click.operTime.values[1:] - df_click.operTime.values[:-1]) / 1e9).astype(int)  # 一定要转成整型，否则下一行的std就会报错，因为其内部用到了乘法
            l = [np.mean(interval_time), np.std(interval_time), len(df_click)/len(df), len(df), np.min(interval_time), np.max(interval_time), df_click.time_weekday.sum()/len(df), len(df_click.loc[df_click.time_of_day == 1])/len(df_click), len(df_click.loc[df_click.time_of_day == 2])/len(df_click)]
        elif len(df_click) == 1:
            df_click = df_click.reset_index()
            l = [0, 0, 1/len(df), len(df), 0, 0, df_click.time_weekday.sum() /len(df_click),
                 (df_click.time_of_day[0] == 1) * 1 /len(df_click), (df_click.time_of_day[0] == 2) * 1 /len(df_click)]
        else:
            l = [0] * 3 + [len(df)] + [0] * 5
    result = pd.Series(l, index = index_names)
    return result

def get_nonuser_df(input_, name):
    '''通过groupby给出的df，返回一个series'''
    index_names = ['click_ratio', 'display_num', 'click_user_ratio', 'display_user_num',
                   'click_interval_mean', 'click_interval_std',
                   'click_interval_min', 'click_interval_max', 'weekday_click_ratio',
                   'morning_click_ratio', 'afternoon_click_ratio']
    index_names = [name + i for i in index_names]
    df = input_.reset_index()
    if len(df) == 1:  # 如果长度为1
        if df.label[0] == 1:
            l = [1, 1, 1, 1, 0, 0, 0, 0]
            if df.time_weekday[0] == 1:
                l.append(1)
            else:
                l.append(0)
            if df.time_of_day[0] == 1:
                l.extend([1, 0])
            elif df.time_of_day[0] == 2:
                l.extend([0, 1])
            else:
                l.extend([0, 0])
        else:
            l = [0] * 11
    else:  # 如果长度不只1
        df_click = df.loc[df.label == 1]
        if len(df_click) > 1:
            df_click = df_click.sort_values(by = 'operTime').reset_index()  # 默认升序
            interval_time = ((df_click.operTime.values[1:] - df_click.operTime.values[:-1]) / 1e9).astype(int)
            l = [len(df_click)/len(df), len(df), len(df_click.uId.unique())/len(df.uId.unique()), len(df.uId.unique()), np.mean(interval_time), np.std(interval_time), np.min(interval_time), np.max(interval_time), df_click.time_weekday.sum()/len(df_click), len(df_click.loc[df_click.time_of_day == 1])/len(df_click), len(df_click.loc[df_click.time_of_day == 2])/len(df_click)]
        elif len(df_click) == 1:
            df_click = df_click.reset_index()
            l = [1/len(df), len(df), 1/len(df.uId.unique()), len(df.uId.unique()), 0, 0, 0, 0, df_click.time_weekday.sum() /len(df_click), (df_click.time_of_day[0] == 1) * 1/len(df_click), (df_click.time_of_day[0] == 2) * 1 /len(df_click)]
        else:
            l = [0, len(df), 0, len(df.uId.unique())] + [0] * 7
    result = pd.Series(l, index = index_names)
    return result

if __name__ == '__main__':
    ## 载入数据集
    update_data_path = './update_data/'
    intermediate_data_path = './small_small_data/'
    df_train = pd.read_csv(intermediate_data_path + 'df_train.csv')
    with open(update_data_path + 'user_info.pickle', 'rb') as f:
        user_info = pickle.load(f)
    with open(update_data_path + 'ad_info.pickle', 'rb') as f:
        ad_info = pickle.load(f)
    with open(update_data_path + 'content_info.pickle', 'rb') as f:
        content_info = pickle.load(f)
    print('Data read finished!')
    ## 开始构造
    df_train.operTime = pd.to_datetime(df_train.operTime, format = '%Y-%m-%d %H:%M:%S.%f')
    df_train['time_weekday'] = df_train.operTime.map(lambda x: 0 if x.isoweekday() >= 6 else 1)
    df_train['time_of_day'] = df_train.operTime.map(lambda x: get_time_of_day(x.hour))
    # 合并用户的所有信息
    t1 = datetime.now()
    df_user = df_train.groupby(['uId']).apply(lambda x: get_user_df(x, 'user_')).reset_index()
    df_user = pd.merge(df_user, user_info, how='left', on='uId')
    t2 = datetime.now()
    print('User_features created!Consumption time:%s' % ((t2 - t1).total_seconds()))
    # 合并广告的所有信息
    t1 = datetime.now()
    df_ad = df_train.groupby(['adId']).apply(lambda x: get_nonuser_df(x, 'ad_')).reset_index()
    df_ad = pd.merge(df_ad, ad_info, how='left', on='adId')
    t2 = datetime.now()
    print('Ad_features created!Consumption time:%s' % ((t2 - t1).total_seconds()))
    # 其他单表的信息
    t1 = datetime.now()
    df_site = df_train.groupby(['siteId']).apply(lambda x: get_nonuser_df(x, 'site_')).reset_index()
    t2 = datetime.now()
    print('Site_features created!Consumption time:%s' % ((t2 - t1).total_seconds()))
    t1 = datetime.now()
    df_slot = df_train.groupby(['slotId']).apply(lambda x: get_nonuser_df(x, 'slot_')).reset_index()
    t2 = datetime.now()
    print('Slot_features created!Consumption time:%s' % ((t2 - t1).total_seconds()))
    t1 = datetime.now()
    df_content = df_train.groupby(['contentId']).apply(lambda x: get_nonuser_df(x, 'content_')).reset_index()
    df_content = pd.merge(df_content, content_info, how='left', on='contentId')
    t2 = datetime.now()
    print('Content_features created!Consumption time:%s' % ((t2 - t1).total_seconds()))
    # 构造用户和其他对象交叉表
    t1 = datetime.now()
    df_user_ad = df_train.groupby(['uId', 'adId']).apply(lambda x: get_user_df(x, 'user_ad_')).reset_index()
    t2 = datetime.now()
    print('User_ad_features created!Consumption time:%s' % ((t2 - t1).total_seconds()))
    t1 = datetime.now()
    df_user_site = df_train.groupby(['uId', 'siteId']).apply(lambda x: get_user_df(x, 'user_site_')).reset_index()
    t2 = datetime.now()
    print('User_site_features created!Consumption time:%s' % ((t2 - t1).total_seconds()))
    t1 = datetime.now()
    df_user_slot = df_train.groupby(['uId', 'slotId']).apply(lambda x: get_user_df(x, 'user_slot_')).reset_index()
    t2 = datetime.now()
    print('User_slot_features created!Consumption time:%s' % ((t2 - t1).total_seconds()))
    t1 = datetime.now()
    df_user_content = df_train.groupby(['uId', 'contentId']).apply(lambda x: get_user_df(x, 'user_content')).reset_index()
    t2 = datetime.now()
    print('User_content_features created!Consumption time:%s' % ((t2 - t1).total_seconds()))
    ## 将特征都写出，留着之后备用
    df_user.to_csv(intermediate_data_path + 'df_user.csv', index = False)
    df_ad.to_csv(intermediate_data_path + 'df_ad.csv', index = False)
    df_content.to_csv(intermediate_data_path + 'df_content.csv', index = False)
    df_site.to_csv(intermediate_data_path + 'df_site.csv', index = False)
    df_slot.to_csv(intermediate_data_path + 'df_slot.csv', index = False)
    df_user_ad.to_csv(intermediate_data_path + 'df_user_ad.csv', index = False)
    df_user_content.to_csv(intermediate_data_path + 'df_user_content.csv', index = False)
    df_user_site.to_csv(intermediate_data_path + 'df_user_site.csv', index = False)
    df_user_slot.to_csv(intermediate_data_path + 'df_user_slot.csv', index = False)
print('All features created!')
