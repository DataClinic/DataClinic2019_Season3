#1.深度学习特征预处理
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

#2.DNN网络
import math
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.contrib.layers import fully_connected
import warnings
warnings.filterwarnings("ignore")

n_inputs =625*1#输入节点
n_hidden1 = 100#第一个隐藏层100个节点
n_hidden2 = 100#第二个隐藏层100个节点 
n_hidden3 = 40
n_outputs = 1 #输出节点

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.float32, shape=(None,n_outputs), name='y')
#用Tensorflow封装的函数
with tf.name_scope("dnn1"):
    # tensorflow使用这个函数帮助我们使用合适的初始化w和b的策略，默认使用ReLU激活函数
#     hidden1 = fully_connected(X, n_hidden1, scope="hd11",reuse=True)#构建第一层隐藏层 全连接
    hidden1 = fully_connected(X, n_hidden1, scope="hd11")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hd22")
#     hidden2 = fully_connected(hidden1, n_hidden2, scope="hd22",reuse=True)#构建第二层隐藏层 全连接
#     hidden3 = fully_connected(hidden2, n_hidden3, scope="hd33",reuse = True)#构建第二层隐藏层 全连接
    hidden3 = fully_connected(hidden2, n_hidden3, scope="hd33")
#     logits = fully_connected(hidden2, n_outputs, scope="oputs1",reuse= True, activation_fn= )#构建输出层 #注意输出层激活函数不需要
    W33 = tf.Variable(tf.truncated_normal([n_hidden3, n_outputs]))
    b = tf.Variable(tf.zeros([n_outputs]))
    logits = tf.sigmoid(tf.matmul(hidden3,W33)+b)
    predict = tf.arg_max(logits,1,name='predict')
with tf.name_scope("loss1"):
    # 定义交叉熵损失函数，并且求个样本平均
    # 函数等价于先使用softmax损失函数，再接着计算交叉熵，并且更有效率
    # 类似的softmax_cross_entropy_with_logits只会给one-hot编码，我们使用的会给0-9分类号
cross_entropy = -tf.reduce_mean(y * tf.log(tf.clip_by_value(logits,1e-10,1.0)) + (1-y) * tf.log(tf.clip_by_value(1-logits,1e-10,1.0)))

learning_rate = 0.0001

with tf.name_scope("train1"):
    optimizer = tf.train.AdamOptimizer(learning_rate)#创建梯度下降的优化器
    training_op = optimizer.minimize(cross_entropy)#最小化损失

with tf.name_scope("eval1"):#评估
    # 获取logits里面最大的那1位和y比较类别好是否相同，返回True或者False一组值
    pre = tf.arg_max(logits,1,name='pre')#logits返回是类别号 y也是类别号
accuracy = np.equal(tf.to_int32(pre),tf.to_int32(y))#转成1.0 0.0

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# 计算图阶段
# mnist = input_data.read_data_sets("MNIST_data_bak/")
n_epochs = 50 #运行400次
batch_size = 50 # 每次运行50个

pre_list  = []
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        print( 'epoch start :',epoch)
        for n in range(array_x.shape[0]// batch_size):#总共多少条/批次大小
            X_batch,  y_batch = array_x[n*batch_size: (n+1)*batch_size], array_y[n*batch_size : (n+1)*batch_size]  #每次传取一小批次数据
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})#传递参数
        acc_train = sess.run(cross_entropy,feed_dict={X: X_batch, y: y_batch})#每运行一次 看训练集准确率
        acc_test = sess.run(cross_entropy, feed_dict={X: test_array_x,#每运行一次 看测试集准确率
                                            y: test_array_y})
        
        if epoch % 50 == 0:
            print(epoch, "Train accuracy:", acc_train)
            print(epoch,'test accuracy:',acc_test)
            pre = sess.run(logits,feed_dict = {X: test_array_x} )
            pre_list.append(pre)
#     save_path = saver.save(sess, "./my_dnn_model_final.ckpt")

#3.DCN网络
with tf.name_scope("DCN_model"):
    he_init = tf.variance_scaling_initializer()
    with tf.name_scope("Embedding_layer"):
        embed_layer_res = tf.placeholder(tf.float32, shape=(None,624), name='X')
#         label = tf.placeholder(tf.float32, shape=(None), name='y')

    with tf.name_scope("Cross_Network"):
        x0 = embed_layer_res 
        cross_x = embed_layer_res
        for level in range(cross_layers):
            Cross_W = tf.get_variable(name='cross_W16%s' % level, shape=[num_col +  embedding_size, 1],
                                      initializer=he_init)  # (N + C * E) * 1
            Cross_B = tf.get_variable(name='cross_B16%s' % level, shape=[1,num_col + embedding_size],
                                      initializer=he_init)  # (N + C * E) * 1
            xtw = tf.matmul(cross_x, Cross_W)  # ? * 1
            cross_x = x0 * xtw + cross_x + Cross_B  # (N + C * E) * 1

    with tf.name_scope("Deep_Network"):
        deep_x = embed_layer_res
        for neurons in deep_layers:
            deep_x = tf.layers.dense(inputs=deep_x, units=neurons, name='deep16_%s' % neurons,activation=tf.nn.selu, kernel_initializer=he_init)

    with tf.variable_scope("Output-layer"):
        x_stack = tf.concat([cross_x, deep_x], 1)  # ? * ((N + C * E) + deep_layers[-1])
        logits = tf.layers.dense(inputs=x_stack, units=label_size, name="outputs16")
        z = tf.reshape(logits, shape=[-1])
        pred = tf.sigmoid(z)

with tf.name_scope("loss"):
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=z)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)
    
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope("eval"):
    acc, upacc = tf.metrics.accuracy(label, tf.math.round(pred))
    auc, upauc = tf.metrics.auc(label, pred)
    acc_summary = tf.summary.scalar('accuracy', upacc)
auc_summary = tf.summary.scalar('auc', upauc)


n_epochs = 2
batch_size = 64
nub_batch = x.shape[0]//batch_size
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    val_acc_list = []
    pre_list = []
    for i in range(n_epochs):
        print('epoch %s:' % i)
        for j in range(nub_batch):
            embed_layer_res,label = get_var(x1,x2,label)
            X_batch, y_batch = x.iloc[j*batch_size:(j+1)*batch_size,:],label[j*batch_size:(j+1)*batch_size]#每次传取一小批次数据
            if j == 0:
                print('model start train')
            loss_tr, loss_summary_str, up1, up2, acc_summary_str, auc_summary_str = \
            sess.run([loss, loss_summary, upacc, upauc, acc_summary, auc_summary],feed_dict={X: X_batch, y: y_batch})
            if j %2000 == 0:
                print("Epoch:", epoch, ",Batch_index:", j,
                      "\tLoss: {:.5f}".format(loss_tr),
                      "\tACC: ", up1,
                      "\tAUC", up2)
