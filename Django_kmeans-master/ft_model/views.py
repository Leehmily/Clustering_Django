from django.shortcuts import render, render_to_response
from django.http import JsonResponse, HttpResponse
# import featuretools
from django.template import RequestContext
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import collections
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift
from pyecharts import options as opts
from pyecharts.charts import WordCloud, Page
import os
import re

# #避免中文乱码
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# Create your views here.


# 用來接收无效URL的响应
def no_page(request):
    html = "<h1>There is no page referred to this response</h1>"
    return HttpResponse(html)


# 用来展现对于表和字段的选择
def select_tables(request):
    # 将接口改成对应CSV的api
    if not os.path.isdir(os.getcwd() + "/demo_data1"):
        os.mkdir(os.getcwd() + "/demo_data1")
    os.chdir(os.getcwd() + "/demo_data1")
    print(os.getcwd())
    print(os.listdir(os.getcwd()))
    regex = re.compile("csv")
    raw_dict = {}
    for file in os.listdir(os.getcwd()):
        if re.search(regex, file):
            raw_dict[file.split(".")[0]] = pd.read_csv(file, encoding = 'GBK')
    data = raw_dict
    os.chdir("..")

    columns_list = []
    name_list = []

    for k, v in data.items():
        columns_list.append(list(v.columns))
        name_list.append(k)
    columns_dict = {k: v for k, v in zip(name_list, columns_list)}
    # print('+++++++++')
    # print(name_list)
    # print('---------')
    # print(columns_list)
    response = render(request, "select_tables.html",
                      {"columns_dict": columns_dict})
    response.set_cookie('columns_dict', columns_dict)
    response.set_cookie('name_list', name_list)
    return response


# 用来展示已经选择的表和字段，并且可以选择对应的数据类型
def variables_type(request):
    columns_dict = request.COOKIES['columns_dict']
    name_list = request.COOKIES['name_list']
    # print(name_list)
    select_dict = {}
    for name in eval(name_list):
        column = request.POST.getlist(name)
        select_dict[name] = column

    response = render(request, "variables_type.html",
                  {"select_dict": select_dict})
    response.set_cookie('select_dict', select_dict)
    # columns_dict = request.POST.get('columns_dict')
    # print(select_dict)
    # print(columns_dict)
    # print("======================")
    return response


# 用来展示初始选择的特征和对应的数据类型
def model_parameters(request):
    # print(customers_types)
    # print(products_types)
    types_list = []
    name_list = []

    # target 是融合的对象
    columns_dict = request.COOKIES['columns_dict']
    select_dict = request.COOKIES['select_dict']
    # target = request.POST.get("target")
    # print("++++++")
    # print(target)
    # print("++++++")
    # print("++++++")
    # print(columns_dict)
    # print("++++++")
    for k, v in eval(columns_dict).items():
        types_list.append(request.POST.getlist(k))
        name_list.append(k)
    types_dict = {k: v for k, v in zip(name_list, types_list)}

    response = render(request, "model_parameters.html",
                      {"types_dict": types_dict,
                       "columns_dict": columns_dict, 
                       "select_dict": select_dict, })

    response.set_cookie('types_dict', types_dict)
    print('+++++++++++')
    print(types_dict)
    print('-----------')
    return response

def get_results(request):
    # 类别变量进行编码处理
    def data_encode(data, *args):
        data_new = data.copy()
        encode = LabelEncoder()
        for feature in args:
            feature_new = encode.fit_transform(data_new[feature])
            data_new[feature] = feature_new
        return data_new
    # 数值变量进行标准化处理
    def data_standard(data, *args):
        data_new = data.copy()
        zscore = StandardScaler()
        for feature in args:
            feature_new = zscore.fit_transform(data_new[feature])
            data_new[feature] = feature_new
        return  data_new
    # try:
    # 数据源相关参数
    types_dict = eval(request.COOKIES['types_dict'])
    # columns_dict = eval(request.COOKIES['columns_dict'])
    select_dict = eval(request.COOKIES['select_dict'])
    # 模型相关参数
    n_cluster = eval(request.POST['n_cluster'])
    max_iter = eval(request.POST['max_iter'])
    tol = eval(request.POST['tol'])
    # print(select_dict)
    # print(n_cluster)
    # print(max_iter)
    # print(tol)
    # context = {'n_cluster': n_cluster, 'max_iter': max_iter, 'tol': tol}
    # 数据接口改成处理csv结构
    import os
    import re
    if not os.path.isdir(os.getcwd() + "/demo_data1"):
        os.mkdir(os.getcwd() + "/demo_data1")
    os.chdir(os.getcwd() + "/demo_data1")
    regex = re.compile("csv")
    raw_dict = {}
    for file in os.listdir(os.getcwd()):
        if re.search(regex, file):
            raw_dict[file.split(".")[0]] = pd.read_csv(file, encoding = 'GBK')
    data = raw_dict
    os.chdir("..")
    if len(data) == 0:
        raise Exception("数据源为空，请检查数据源文件")
    else:
        # 这里暂时只讨论传输一个csv文件，如何进行join操作
        # data_df为一个data_frame
        data_df = list(data.values())[0]
        data_feature = [v for v in select_dict.values()][0]
        # 根据数据类型进行数据自动化处理
        type_feature = [v for v in types_dict.values()][0]
        feature_standard = [data_feature[i] for i in range(len(type_feature)) if type_feature[i] == 'numeric']
        feature_encode = [data_feature[i] for i in range(len(type_feature)) if type_feature[i] == 'category']
        data_df_kmeans = data_df[data_feature]

        # 经过处理之后的数据data_df_kmeans_processing
        data_df_kmeans_processing = data_standard(data_df_kmeans, feature_standard)
        data_df_kmeans_processing = data_encode(data_df_kmeans_processing, feature_encode)


    # print(data_feature)
    # print(data_df_kmeans.info())
    # 下面为Kmeans模型
    # 传入提交的参数
    km = KMeans(n_clusters = n_cluster, max_iter = max_iter, tol = tol)
    cluster = km.fit(data_df_kmeans_processing)
    cluster_label = list(cluster.labels_)
    # 绘制聚类手肘曲线图
    import scikitplot as skplt
    from scikitplot.cluster import plot_elbow_curve
    plot_elbow_curve(cluster, data_df_kmeans_processing, cluster_ranges=range(1,20), figsize=(8,8))
    address = './ft_model/static/images/'
    name = 'kmeans_elbow_curve.png'
    picname = address + name
    # print(picname)
    plt.savefig(picname)

    picnamenew = '/static/images/' + name
    images = [picnamenew]


    # 绘制分组变量分布图

    for i in range(n_cluster):
        km_cluster_group = data_df_kmeans_processing.loc[cluster.labels_ == i, :]
        p = km_cluster_group.plot(kind='kde', linewidth = 2, subplots = True, sharex = False)
        plt.legend()
        plt.title('类群 ' + str(i) + ' Feature Distribution Figure')
        name = '类群' + str(i) +'FeatureDistribution.png'
        picname = address + name
        plt.savefig(picname)
        picnamenew = '/static/images/' + name
        images.append(picnamenew)

    # 数据存储 （聚类之后的数据存储）
    # 创建data_result文件夹存储数据
    if not os.path.isdir(os.getcwd() + "/data_result"):
        os.mkdir(os.getcwd() + "/data_result")
    for i in range(n_cluster):
        df = data_df.loc[cluster.labels_ == i, :]
        name = os.getcwd() + "/data_result/" + "group" + str(i) + "_result.csv"
        df.to_csv(name, encoding = 'GBK', index = False)
    

    response = render(request, 'get_results.html', {'images': images, 'cluster':cluster, 'cluster_label':cluster_label})
    # response.set_cookie('cluster', cluster)
    response.set_cookie('cluster_label',n_cluster)
    response.set_cookie('cluster_label',cluster_label)
    return response
    # return render(request, 'get_results.html', {'images': images},)

# 用来展现对表和字段的选择，为了后续进行组群词云图绘制
def select_features(request):
    cluster = request.COOKIES['cluster']
    n_cluster = request.COOKIES['n_cluster']
    cluster_label = request.COOKIES['cluster_label']
    # 将接口改成对应CSV的api
    if not os.path.isdir(os.getcwd() + "/demo_data1"):
        os.mkdir(os.getcwd() + "/demo_data1")
    os.chdir(os.getcwd() + "/demo_data1")
    print(os.getcwd())
    print(os.listdir(os.getcwd()))
    regex = re.compile("csv")
    raw_dict = {}
    for file in os.listdir(os.getcwd()):
        if re.search(regex, file):
            raw_dict[file.split(".")[0]] = pd.read_csv(file, encoding = 'GBK')
    data = raw_dict
    os.chdir("..")

    columns_list_feature = []
    name_list_feature = []

    for k, v in data.items():
        columns_list_feature.append(list(v.columns))
        name_list_feature.append(k)
    columns_dict_feature = {k: v for k, v in zip(name_list_feature, columns_list_feature)}
    # print('+++++++++')
    # print(name_list)
    # print('---------')
    # print(columns_list)
    response = render(request, "select_features.html",
                      {"columns_dict_feature": columns_dict_feature, "cluster_label":cluster_label, "n_cluster":n_cluster,})
    response.set_cookie('columns_dict_feature', columns_dict_feature)
    response.set_cookie('name_list_feature', name_list_feature)
    response.set_cookie('cluster_label', cluster_label)
    response.set_cookie('n_cluster',n_cluster)
    return response









def plot_wordcloud(request):
    cluster = eval(request.COOKIES['cluster'])
    n_cluster = eval(request.COOKIES['n_cluster'])
    cluster_label = eval(request.COOKIES['cluster_label'])
    print('hi the type is ')
    print(type(cluster_label))
    columns_dict_feature = request.COOKIES['columns_dict_feature']
    name_list_feature = request.COOKIES['name_list_feature']
    # print(name_list)
    select_dict = {}
    for name in eval(name_list_feature):
        column = request.POST.getlist(name)
        select_dict[name] = column

    import os
    import re
    if not os.path.isdir(os.getcwd() + "/demo_data1"):
        os.mkdir(os.getcwd() + "/demo_data1")
    os.chdir(os.getcwd() + "/demo_data1")
    regex = re.compile("csv")
    raw_dict = {}
    for file in os.listdir(os.getcwd()):
        if re.search(regex, file):
            raw_dict[file.split(".")[0]] = pd.read_csv(file, encoding = 'GBK')
    data = raw_dict
    os.chdir("..")
    if len(data) == 0:
        raise Exception("数据源为空，请检查数据源文件")
    else:
        # 这里暂时只讨论传输一个csv文件，如何进行join操作
        # data_df为一个data_frame
        data_df = list(data.values())[0]
        data_feature = [v for v in select_dict.values()][0]
        data_df_wordcloud = data_df[data_feature]
    print(cluster_label)
    print(n_cluster)
    print(type(n_cluster))
    page = Page()
    for i in range(n_cluster):
        data_df_wordcloud_group = data_df_wordcloud.loc[np.array(cluster_label) == i, :] 
        word = []
        for colume in data_feature:
            word.extend(list(data_df_wordcloud_group[colume]))
        word_frequency = collections.Counter(word)
        word_frequency_common = word_frequency.most_common()
        wordcloud = WordCloud()
        wordcloud.add("", word_frequency_common)
        page.add(wordcloud)
    return HttpResponse(page.render_embed())