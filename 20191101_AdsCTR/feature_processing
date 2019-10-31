#深度学习特征预处理
import tensorflow as tf
import pandas as pd
import numpy as np
user_info = pd.read_csv('user_info.csv',header = None)
user_info.columns = columns = ['uId', 'age', 'gender', 'city', 'province','phoneType', 'carrier']
ad_info = pd.read_csv('ad_info.csv',header =  None)
ad_info.columns = ['adId', 'billId', 'primId', 'creativeType', 'intertype', 'spreadAppId']

# user_info 分类型变量onehot

label_ = LabelEncoder()
new_age = label_.fit_transform(afrer_user_info['age'])
age_dummy = pd.get_dummies(new_age,columns=['age1','age2','age3','age4','age5','age6'])

label_g = LabelEncoder()
new_gender = label_g.fit_transform(afrer_user_info['gender'])
gender_dummy = pd.get_dummies(new_gender,columns=['gender1','gender2','gender3','gender4'])

label_c = LabelEncoder()
new_c = label_g.fit_transform(afrer_user_info['carrier'])
c_dummy = pd.get_dummies(new_c,columns=['carrier1','carrier2','carrier3','carrier4'])

after_onehot_user_no_loc = pd.concat([age_dummy,gender_dummy,c_dummy],axis =1)

after_onehot_user_no_loc.columns = ['age1','age2','age3','age4','age5','age6','age7','gender1','gender2','gender3','gender4','carrier1','carrier2','carrier3','carrier4']
after_onehot_user_no_loc_1 = pd.concat([user_info.uId,after_onehot_user_no_loc],axis = 1)

after_onehot_user_no_loc_1.to_csv('df_user_basic_info.csv',index = None)

# ad_info onehot 处理
from sklearn.preprocessing import OneHotEncoder
one_hot_billId = OneHotEncoder()
one_hot_billId.fit(np.array(ad_info['billId']).reshape(-1,1))
s = one_hot_billId.transform(one_hot_billId.categories_[0].reshape(-1,1))
dict_billId = dict(zip(one_hot_billId.categories_[0],s))

one_hot_primId = OneHotEncoder()
one_hot_primId.fit(np.array(ad_info['primId']).reshape(-1,1))
s_prim = one_hot_primId.transform(one_hot_primId.categories_[0].reshape(-1,1))
dict_primId = dict(zip(one_hot_primId.categories_[0],s_prim))

one_hot_createId = OneHotEncoder()
one_hot_createId.fit(np.array(ad_info['creativeType']).reshape(-1,1))
s_create = one_hot_createId.transform(one_hot_createId.categories_[0].reshape(-1,1))
dict_createId = dict(zip(one_hot_createId.categories_[0],s_create))

one_hot_interId  = OneHotEncoder()
one_hot_interId.fit(np.array(ad_info['intertype']).reshape(-1,1))
s_inter = one_hot_interId.transform(one_hot_interId.categories_[0].reshape(-1,1))
dict_inter = dict(zip(one_hot_interId.categories_[0],s_inter))

one_hot_app = OneHotEncoder(categories='auto')
one_hot_app.fit(np.array(ad_info['spreadAppId'].fillna(-1)).reshape(-1,1))
s_app = one_hot_app.transform(one_hot_app.categories_[0].reshape(-1,1))
dict_app = dict(zip(one_hot_app.categories_[0],s_app))

def concat_adinfo(key,data = te, dict_app = dict_app, dict_billId = dict_billId, dict_primId  = dict_primId, dict_createId = dict_createId, dict_inter = dict_inter,one_hot_app = one_hot_app, ne_hot_billId=one_hot_billId, one_hot_createId = one_hot_createId, one_hot_interId = one_hot_interId,  one_hot_primId = one_hot_primId ):
    test = data.loc[key,:]
ad_info_one_hot = pd.concat([pd.DataFrame(dict_billId[test['billId']].todense(), columns = list(one_hot_billId.categories_[0])), pd.DataFrame(dict_primId[test['primId']].todense(),  columns=list(map(lambdax:'primId'+x, list(one_hot_primId.categories_[0].astype(int).astype(
str))))), pd.DataFrame(dict_createId[test['creativeType']].todense(), columns = list(map(lambda x:'creativeType'+x,list(one_hot_createId.categories_[0].astype(int).astype(str))))),  pd.DataFrame(dict_inter[test['intertype']].todense(), columns = list(map(lambda x: 'intertype' +x,list(one_hot_interId.categories_[0].astype(int).astype(str))))),pd.DataFrame(dict_app[test['spreadAppId']].todense(),columns=list(map(lambda x:'appId'+x, list(one_hot_app.categories_[
0].astype(int).astype(str)))))], axis = 1)
    return ad_info_one_hot

df_adinfo = pd.DataFrame()
for i in range(ad_info.shape[0]):
    new_d = concat_adinfo(ad_info.iloc[i,0])
df_adinfo = pd.concat([df_adinfo,new_d],axis=0)

df_adinfo.to_csv('df_ad_info_onehot.csv')

# contented onehot
content = pd.read_csv('content_info.csv',header = None)
content.columns = ['contentId', 'firstClass', 'secondClass']

onehot_content_1 = OneHotEncoder()
onehot_content_1.fit(np.array(content['firstClass']).reshape(-1,1))
s_1 = onehot_content_1.transform(onehot_content_1.categories_[0].reshape(-1,1)).todense()
dict_content1 = dict(zip(onehot_content_1.categories_[0],s_1))

all_word = list(content['firstClass'].unique())
cnt = []
ordered_word = []
for key in all_word:
    word_1 = []
for j in list(content.loc[content.loc[(content['firstClass']==key)  ].index , : ]['secondClass'
].unique()) :
        if len(str(j)) != 0 :
            ls = str(j).split('#')
            for word_ in ls:
                if word_ not in word_1:
                    word_1.append(word_)
        else:
            if 'nan' not in word_1:
                word_1.append('nan')
    ordered_word += word_1
cnt.append(len(word_1))

tt = []
for i in  range(len(all_word)):
tt +=  [all_word[i]] * cnt[i]

mergered_word = merge_word(tt,ordered_word)
dtmerged = pd.DataFrame(columns = mergered_word)

def clean_content_class12(data,dt_merged):
    dt_merged.loc[0,:] = 0
    cls1 = data['firstClass']
    if str(data['secondClass']) != 'nan':
        ls = data['secondClass'].split('#')
        for j in ls:
            newcol = cls1 + j
            dt_merged[newcol] = 1
    else:
        newcol = cls1 + 'nan'
        dt_merged[newcol] = 1
    return dt_merged

def concat_contentid(data,dt_merged = dtmerged):
    data1 = data['contentId']
#     print(data[['firstClass','secondClass']])
    data2 = clean_content_class12(data[['firstClass','secondClass']],dt_merged)
    return pd.concat([pd.DataFrame(list([data1]),columns=['contentId']),data2],axis = 1)
    
df_content = pd.DataFrame()
for i in range(content.shape[0]):
    added = concat_contentid(content.iloc[i,:])
df_content = pd.concat([df_content,added],axis =0)


df_content.to_csv('df_content_after_onehot.csv',index = None)

## 变量拼接
train = pd.read_csv('choose_sample_train.csv')
df_ad_info = pd.read_csv('df_ad_info_onehot.csv')
df_ad_info.head()
df_content = pd.read_csv('df_content_after_onehot.csv')
df_content.head()

merged0 = pd.merge(train,df_ad_info,on = 'adId')
merged1 = pd.merge(merged0,df_content,how='left',on = 'contentId')
## onehot 拼接后与人工特征继续拼接
mergerd_1 = pd.merge(mergerd1,df_ad[['adId', 'ad_click_ratio', 'ad_display_num', 'ad_click_user_ratio',
       'ad_display_user_num', 'ad_click_interval_mean',
       'ad_click_interval_std', 'ad_click_interval_min',
       'ad_click_interval_max', 'ad_weekday_click_ratio',
       'ad_morning_click_ratio', 'ad_afternoon_click_ratio']],how = 'left',on = 'adId')
mergerd_2 = pd.merge(mergerd_1,df_content,how = 'left',on = 'contentId')

mergerd_3 = pd.merge(mergerd_2,df_user[['uId', 'user_click_interval_mean', 'user_click_interval_std',
       'user_click_ratio', 'user_display_time', 'user_click_interval_min',
       'user_click_interval_max', 'user_weekday_click_ratio',
       'user_morning_click_ratio', 'user_afternoon_click_ratio']],how = 'left',on = 'uId')
mergerd_4 = pd.merge(mergerd_3,df_site,how = 'left',on = 'siteId')

mergerd_5 = pd.merge(mergerd_4,df_slot,how = 'left',on = 'slotId')
mergerd_6 = pd.merge(mergerd_5,df_user_ad,how = 'left',on = ['uId','adId'])
mergerd_7 = pd.merge(mergerd_6,df_user_content,how = 'left',on = ['uId','contentId'])
mergerd_8 = pd.merge(mergerd_7,df_user_site,how = 'left',on = ['uId','siteId'])
mergerd_9 = pd.merge(mergerd_8,df_user_slot,how = 'left',on = ['uId','slotId'])
