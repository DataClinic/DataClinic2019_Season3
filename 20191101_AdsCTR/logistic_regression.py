import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
plt.rc('figure', figsize= (25, 10))
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from scipy import interp

def calc_model_auc(X, y, model):
    '''X为自变量矩阵，y为标签，model为某个估计器'''
    skf = StratifiedKFold(n_splits=5)
    train_roc = []
    train_auc = []
    valid_roc = []
    valid_auc = []
    fpr_points = np.linspace(0, 1, 100)
    for train_index, valid_index in skf.split(X, y):
        X_train, X_valid = X.iloc[train_index, :], X.iloc[valid_index, :]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        model.fit(X_train, y_train)
        train_pred_prob = model.predict_proba(X_train)
        valid_pred_prob = model.predict_proba(X_valid)
        fpr, tpr, thresholds = roc_curve(y_train, train_pred_prob[:, 1])
        train_roc.append(interp(fpr_points, fpr, tpr))
        train_auc.append(auc(fpr, tpr))
        fpr, tpr, thresholds = roc_curve(y_valid, valid_pred_prob[:, 1])
        valid_roc.append(interp(fpr_points, fpr, tpr))
        valid_auc.append(auc(fpr, tpr))
    return train_roc,train_auc,valid_roc,valid_auc

def draw_roc(train_roc, train_auc, valid_roc, valid_auc):
    fpr_points = np.linspace(0, 1, 100)
    train_mean_tpr = np.mean(train_roc, axis = 0)
    train_mean_auc = np.mean(train_auc)
    std_auc = np.std(train_auc)
    plt.subplot(1,2,1)
    for i in range(len(train_auc)):
        use_data = train_roc[i]
        plt.plot(fpr_points, use_data, alpha = 0.3, label = 'ROC fold %s(auc = %0.2f)' % (i + 1, train_auc[i]))
    plt.plot([0,1], [0,1], linestyle = '--', lw = 2, color = 'r', label = 'Chance', alpha = 0.8)
    plt.plot(fpr_points, train_mean_tpr, label = r'ROC mean(auc = %0.2f $\pm$ %0.2f)' % (train_mean_auc, std_auc))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for train')
    plt.legend(loc="lower right")
    plt.subplot(1,2,2)
    valid_mean_tpr = np.mean(valid_roc, axis = 0)
    valid_mean_auc = np.mean(valid_auc)
    std_auc = np.std(valid_auc)
    for i in range(len(valid_auc)):
        use_data = valid_roc[i]
        plt.plot(fpr_points, use_data, alpha = 0.3, label = 'ROC fold %s(auc = %0.2f)' % (i + 1, train_auc[i]))
    plt.plot([0,1], [0,1], linestyle = '--', lw = 2, color = 'r', label = 'Chance', alpha = 0.8)
    plt.plot(fpr_points, valid_mean_tpr, label = r'ROC mean(auc = %0.2f $\pm$ %0.2f)' % (valid_mean_auc, std_auc))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for validation')
    plt.legend(loc="lower right")
plt.show()

with open('train_df.pickle', 'rb') as f:
    total_train_df = pickle.load(f)
with open('test_df.pickle', 'rb') as f:
    test_df = pickle.load(f)

df_ad = pd.read_csv('df_ad.csv')
df_content = pd.read_csv('df_content.csv')
df_site = pd.read_csv('df_site.csv')
df_slot = pd.read_csv('df_slot.csv')

df_ad_later = pd.read_csv('df_ad_later.csv')
df_content_later = pd.read_csv('df_content_later.csv')
df_site_later = pd.read_csv('df_site_later.csv')
df_slot_later = pd.read_csv('df_slot_later.csv')

with open('user_info.pickle', 'rb') as f:
    user_info = pickle.load(f)

def get_time_of_day(hour):
    '''给定当前小时，返回1表示早上，2表示下午，3表示晚上'''
    if hour <= 12:
        return 1
    elif hour <= 18:
        return 2
    else:
        return 3

#1.人工特征+LR
time_point = datetime.strptime('20190330000000', '%Y%m%d%H%M%S')
valid_df = total_train_df.loc[total_train_df.operTime >= time_point, :]  # 取出30号的数据
# 获得验证集上所有特征并创建所有可能特征
valid_df = pd.merge(valid_df, df_ad, how = 'left', on = 'adId')
valid_df = pd.merge(valid_df, df_content, how = 'left', on = 'contentId')
valid_df = pd.merge(valid_df, df_site, how = 'left', on = 'siteId')
valid_df = pd.merge(valid_df, df_slot, how = 'left', on = 'slotId')
valid_df = pd.merge(valid_df, user_info, how = 'left', on = 'uId')
valid_df['time_weekday'] = valid_df.operTime.map(lambda x: 0 if x.isoweekday() >= 6 else 1)
valid_df['time_of_day'] = valid_df.operTime.map(lambda x: get_time_of_day(x.hour))  # 创建两个时间相关的变量，然后就可以删掉时间了
valid_df = valid_df.drop(['uId', 'adId', 'operTime', 'slotId', 'contentId', 'phoneType', 'city'], axis = 1)
print('Finished feature engineering.')
# 进行必要的数据清理
valid_df = valid_df.drop(['spreadAppId', 'firstClass', 'secondClass', 'primId'], axis = 1)  # 删掉变量
missing_ = valid_df.apply(lambda x:sum(pd.isnull(x)) / valid_df.shape[0], axis = 0)
missing_ = missing_[missing_.values > 0]
# 填补缺失值
for col in missing_.index:
    if col.startswith('ad_') or col.startswith('content_'):
        valid_df.loc[pd.isnull(valid_df[col]), [col]] = np.median(valid_df.loc[pd.notnull(valid_df[col]), [col]])
    else:
        mode = valid_df[col].value_counts().sort_values().index[-1]  # 默认升序排列
        valid_df.loc[pd.isnull(valid_df[col]), [col]] = mode
print('Finished data cleaning.')
# 进行onehot处理
lr_onehot = OneHotEncoder()
trans = lr_onehot.fit_transform(valid_df[['siteId', 'netType', 'billId', 'creativeType', 'interType', 'age', 'gender', 'province', 'carrier', 'time_weekday', 'time_of_day']].values)
trans = pd.DataFrame(trans.toarray())
valid_df_final = pd.concat([valid_df.drop(['siteId', 'netType', 'billId', 'creativeType', 'interType', 'age', 'gender', 'province', 'carrier', 'time_weekday', 'time_of_day'], axis = 1),trans], axis = 1)
print('Finished onehot transformation.')
# 使用交叉验证的方法绘制图像
LR = LogisticRegression()
train_roc,train_auc,valid_roc,valid_auc = calc_model_auc(valid_df_final.drop(['label'], axis = 1), valid_df_final['label'], LR)
draw_roc(train_roc, train_auc, valid_roc, valid_auc)
# 使用所有验证集的数据进行LR拟合，用于测试集的测试
LR = LogisticRegression()
LR.fit(valid_df_final.drop(['label'], axis =1), valid_df_final['label'])
# 对测试集进行相对应的处理
test_df_lr = test_df
test_df_lr = pd.merge(test_df_lr, df_ad_later, how = 'left', on = 'adId')
test_df_lr = pd.merge(test_df_lr, df_content_later, how = 'left', on = 'contentId')
test_df_lr = pd.merge(test_df_lr, df_site_later, how = 'left', on = 'siteId')
test_df_lr = pd.merge(test_df_lr, df_slot_later, how = 'left', on = 'slotId')
test_df_lr = pd.merge(test_df_lr, user_info, how = 'left', on = 'uId')
test_df_lr['time_weekday'] = test_df_lr.operTime.map(lambda x: 0 if x.isoweekday() >= 6 else 1)
test_df_lr['time_of_day'] = test_df_lr.operTime.map(lambda x: get_time_of_day(x.hour))  # 创建两个时间相关的变量，然后就可以删掉时间了
test_df_lr = test_df_lr.drop(['uId', 'adId', 'operTime', 'slotId', 'contentId', 'phoneType', 'city'], axis = 1)
# 进行必要的数据清理
test_df_lr = test_df_lr.drop(['spreadAppId', 'firstClass', 'secondClass', 'primId'], axis = 1)  # 删掉变量
missing_ = test_df_lr.apply(lambda x:sum(pd.isnull(x)) / test_df_lr.shape[0], axis = 0)
missing_ = missing_[missing_.values > 0]
# 填补缺失值
for col in missing_.index:
    if col.startswith('ad_') or col.startswith('content_'):
        test_df_lr.loc[pd.isnull(test_df_lr[col]), [col]] = np.median(test_df_lr.loc[pd.notnull(test_df_lr[col]), [col]])
    else:
        mode = test_df_lr[col].value_counts().sort_values().index[-1]  # 默认升序排列
        test_df_lr.loc[pd.isnull(test_df_lr[col]), [col]] = mode
# 进行onehot处理
test_trans = lr_onehot.transform(test_df_lr[['siteId', 'netType', 'billId', 'creativeType', 'interType', 'age', 'gender', 'province', 'carrier', 'time_weekday', 'time_of_day']].values)
test_trans = pd.DataFrame(test_trans.toarray())
test_df_lr_final = pd.concat([test_df_lr.drop(['siteId', 'netType', 'billId', 'creativeType', 'interType', 'age', 'gender', 'province', 'carrier', 'time_weekday', 'time_of_day'], axis = 1), test_trans], axis = 1)
# 进行预测
fpr_points = np.linspace(0, 1, 100)
pred_prob = LR.predict_proba(test_df_lr_final.drop(['label'], axis = 1))
fpr, tpr, thresholds = roc_curve(test_df.label, pred_prob[:, 1])
roc = interp(fpr_points, fpr, tpr)
auc_ = auc(fpr, tpr)
# 绘制ROC曲线
plt.plot(fpr_points, roc, alpha = 0.3, label = 'ROC(auc=%s)' % (auc_))
plt.plot([0,1], [0,1], linestyle = '--', lw = 2, color = 'r', label = 'Chance', alpha = 0.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize = 20)
plt.ylabel('True Positive Rate', fontsize = 20)
plt.title('Receiver operating characteristic for LR', fontsize = 25)
plt.legend(loc="lower right", fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()

#2.人工特征+XGBoost
# 直接用前面处理好的验证集绘图
XGB = XGBClassifier()
train_roc,train_auc,valid_roc,valid_auc = calc_model_auc(valid_df_final.drop(['label'], axis = 1), valid_df_final['label'], XGB)
draw_roc(train_roc, train_auc, valid_roc, valid_auc)
# 拟合XGB
XGB = XGBClassifier()
XGB.fit(valid_df_final.drop(['label'], axis = 1), valid_df_final['label'])
# 进行预测
fpr_points = np.linspace(0, 1, 100)
pred_prob = XGB.predict_proba(test_df_lr_final.drop(['label'], axis = 1))
fpr, tpr, thresholds = roc_curve(test_df.label, pred_prob[:, 1])
roc = interp(fpr_points, fpr, tpr)
auc_ = auc(fpr, tpr)
# 绘制ROC曲线
plt.plot(fpr_points, roc, alpha = 0.3, label = 'ROC(auc=%s)' % (auc_))
plt.plot([0,1], [0,1], linestyle = '--', lw = 2, color = 'r', label = 'Chance', alpha = 0.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize = 20)
plt.ylabel('True Positive Rate', fontsize = 20)
plt.title('Receiver operating characteristic for XGB', fontsize = 25)
plt.legend(loc="lower right", fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()

#3.双GBDT+LR
with open('train_df.pickle', 'rb') as f:
    total_train_df = pickle.load(f)
with open('test_df.pickle', 'rb') as f:
test_df = pickle.load(f)
total_train_df = pd.merge(total_train_df, df_ad, how = 'left', on = 'adId')
total_train_df = pd.merge(total_train_df, df_content, how = 'left', on = 'contentId')
total_train_df = pd.merge(total_train_df, df_site, how = 'left', on = 'siteId')
total_train_df = pd.merge(total_train_df, df_slot, how = 'left', on = 'slotId')
total_train_df = pd.merge(total_train_df, user_info, how = 'left', on = 'uId')
total_train_df['time_weekday'] = total_train_df.operTime.map(lambda x: 0 if x.isoweekday() >= 6 else 1)
total_train_df['time_of_day'] = total_train_df.operTime.map(lambda x: get_time_of_day(x.hour))  # 创建两个时间相关的变量，然后就可以删掉时间了
total_train_df = total_train_df.drop(['spreadAppId', 'firstClass', 'secondClass'], axis = 1)  # 删掉变量
missing__ = total_train_df.apply(lambda x:sum(pd.isnull(x)) / total_train_df.shape[0], axis = 0)
missing__ = missing__[missing__.values > 0]
# 填补缺失值
for col in missing__.index:
    if col.startswith('ad_') or col.startswith('content_'):
        total_train_df.loc[pd.isnull(total_train_df[col]), [col]] = np.median(total_train_df.loc[pd.notnull(total_train_df[col]), [col]])
    else:
        mode = total_train_df[col].value_counts().sort_values().index[-1]  # 默认升序排列
        total_train_df.loc[pd.isnull(total_train_df[col]), [col]] = mode
test_df_gbdt = test_df
test_df_gbdt = pd.merge(test_df_gbdt, df_ad_later, how = 'left', on = 'adId')
test_df_gbdt = pd.merge(test_df_gbdt, df_content_later, how = 'left', on = 'contentId')
test_df_gbdt = pd.merge(test_df_gbdt, df_site_later, how = 'left', on = 'siteId')
test_df_gbdt = pd.merge(test_df_gbdt, df_slot_later, how = 'left', on = 'slotId')
test_df_gbdt = pd.merge(test_df_gbdt, user_info, how = 'left', on = 'uId')
test_df_gbdt['time_weekday'] = test_df_gbdt.operTime.map(lambda x: 0 if x.isoweekday() >= 6 else 1)
test_df_gbdt['time_of_day'] = test_df_gbdt.operTime.map(lambda x: get_time_of_day(x.hour))  # 创建两个时间相关的变量，然后就可以删掉时间了
# 进行必要的数据清理
test_df_gbdt = test_df_gbdt.drop(['spreadAppId', 'firstClass', 'secondClass'], axis = 1)  # 删掉变量
missing_ = test_df_gbdt.apply(lambda x:sum(pd.isnull(x)) / test_df_gbdt.shape[0], axis = 0)
missing_ = missing_[missing_.values > 0]
# 填补缺失值
for col in missing_.index:
    if col.startswith('ad_') or col.startswith('content_'):
        test_df_gbdt.loc[pd.isnull(test_df_gbdt[col]), [col]] = np.median(test_df_gbdt.loc[pd.notnull(test_df_gbdt[col]), [col]])
    else:
        mode = test_df_gbdt[col].value_counts().sort_values().index[-1]  # 默认升序排列
        test_df_gbdt.loc[pd.isnull(test_df_gbdt[col]), [col]] = mode
discrete_var_name = ['adId','siteId','slotId','contentId','netType','billId','primId','creativeType','interType','age','gender','city','province','phoneType','carrier','time_weekday','time_of_day']
# 获得编码方式
onehot_gbdt_in = OneHotEncoder()  # 它是输入到树中的向量
onehot_use = pd.concat([total_train_df.loc[:, discrete_var_name], test_df_gbdt.loc[:, discrete_var_name]], axis = 0)
onehot_gbdt_in.fit(onehot_use)  # 将所有数据集里面的离散变量都放进到onehot中fit
time_point = datetime.strptime('20190330000000', '%Y%m%d%H%M%S')
valid_df = total_train_df.loc[total_train_df.operTime >= time_point, :]  # 取出30号的数据
train_df = total_train_df.loc[total_train_df.operTime < time_point, :]
# 要把它们都切分为离散的部分和连续的部分
train_df_dis = train_df.loc[:, discrete_var_name]
train_df_dis = onehot_gbdt_in.transform(train_df_dis).toarray()
train_df_con = train_df.drop(discrete_var_name, axis = 1)
valid_df_dis = valid_df.loc[:, discrete_var_name]
valid_df_dis = onehot_gbdt_in.transform(valid_df_dis).toarray()
valid_df_con = valid_df.drop(discrete_var_name, axis = 1)
test_dis = test_df_gbdt.loc[:, discrete_var_name]
test_dis = onehot_gbdt_in.transform(test_dis).toarray()
test_con = test_df_gbdt.drop(discrete_var_name, axis = 1)
t1 = datetime.now()
gbdt_dis = GradientBoostingClassifier()
gbdt_dis.fit(train_df_dis, train_df['label'])
t2 = datetime.now()
print('Total used for fitting:%s s' % ((t2 - t1).total_seconds))
gbdt_con = GradientBoostingClassifier()
gbdt_con.fit(train_df_con.drop(['label', 'uId', 'operTime'], axis = 1), train_df['label'])
# 这里要把valid和test都丢进去
onehot_gbdt_out_dis = OneHotEncoder()  # 这些是将从树中输出出来的东西转为onehot
X_leaves_dis = onehot_gbdt_out_dis.fit_transform(gbdt_dis.apply(pd.concat([pd.DataFrame(valid_df_dis), pd.DataFrame(test_dis)], axis = 0))[:, :, 0]).toarray()
onehot_gbdt_out_con = OneHotEncoder()
X_leaves_con = onehot_gbdt_out_con.fit_transform(gbdt_con.apply(pd.concat([pd.DataFrame(valid_df_con.drop(['label', 'uId', 'operTime'], axis = 1)),
                                                                           pd.DataFrame(test_con.drop(['label', 'uId', 'operTime'], axis = 1))], axis = 0))[:, :, 0]).toarray()
# 画图看一下LR的效果
LR = LogisticRegression()
train_roc,train_auc,valid_roc,valid_auc = calc_model_auc(pd.concat([pd.DataFrame(X_leaves_dis[:valid_df_dis.shape[0], :]),
                                                                    pd.DataFrame(X_leaves_con[:valid_df_con.shape[0], :])], axis = 1), valid_df['label'], LR)
draw_roc(train_roc, train_auc, valid_roc, valid_auc)
# 使用整个验证集拟合一个LR
LR = LogisticRegression()
LR.fit(pd.concat([pd.DataFrame(X_leaves_dis[:valid_df_dis.shape[0], :]), pd.DataFrame(X_leaves_con[:valid_df_con.shape[0], :])], axis = 1), valid_df['label'])
pred_prob = LR.predict_proba(pd.concat([pd.DataFrame(X_leaves_dis[valid_df_dis.shape[0]:, :]), pd.DataFrame(X_leaves_con[valid_df_con.shape[0]:, :])], axis = 1))
fpr, tpr, thresholds = roc_curve(test_df.label, pred_prob[:, 1])
fpr_points = np.linspace(0, 1, 100)
roc = interp(fpr_points, fpr, tpr)
auc_ = auc(fpr, tpr)
# 绘制ROC曲线
plt.plot(fpr_points, roc, alpha = 0.3, label = 'ROC(auc=%s)' % (auc_))
plt.plot([0,1], [0,1], linestyle = '--', lw = 2, color = 'r', label = 'Chance', alpha = 0.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize = 20)
plt.ylabel('True Positive Rate', fontsize = 20)
plt.title('Receiver operating characteristic for two GBDT + LR', fontsize = 25)
plt.legend(loc="lower right", fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
