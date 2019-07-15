from django.shortcuts import render, render_to_response
from django.http import JsonResponse, HttpResponse
# import featuretools
from django.template import RequestContext
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import collections
from sklearn.preprocessing import LabelEncoder
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
    # import featuretools as ft
    # import pandas as pd
    # import numpy as np
    # from featuretools.primitives import make_trans_primitive, make_agg_primitive
    # from featuretools.variable_types import DatetimeTimeIndex, Numeric
    # import os
    # import re

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
    # print(types_dict)
    # target_id = ''
    # for type_i, column_i in zip(types_dict[target], columns_dict[target]):
    #     if 'Index' in type_i:
    #         target_id = column_i
    # print("=============")
    # print(target_id)
    # print("=============")
    response.set_cookie('types_dict', types_dict)
    # print('+++++++++++')
    # print(types_dict)
    # print('-----------')
    # print(columns_dict)
    # response.set_cookie('target_id', target_id)
    # response.set_cookie('target', target)
    return response

def get_results(request):
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
        data_df_kmeans = data_df[data_feature]
    # print(data_feature)
    # print(data_df_kmeans.info())
    # 下面为Kmeans模型
    # 传入提交的参数
    km = KMeans(n_clusters = n_cluster, max_iter = max_iter, tol = tol)
    cluster = km.fit(data_df_kmeans)
    cluster_label = list(cluster.labels_)
    # 绘制聚类手肘曲线图
    import scikitplot as skplt
    from scikitplot.cluster import plot_elbow_curve
    plot_elbow_curve(cluster, data_df_kmeans, cluster_ranges=range(1,20), figsize=(8,8))
    address = './ft_model/static/images/'
    name = 'kmeans_elbow_curve.png'
    picname = address + name
    # print(picname)
    plt.savefig(picname)

    picnamenew = '/static/images/' + name
    images = [picnamenew]


    # 绘制分组变量分布图

    for i in range(n_cluster):
        km_cluster_group = data_df_kmeans.loc[cluster.labels_ == i, :]
        p = km_cluster_group.plot(kind='kde', linewidth = 2, subplots = True, sharex = False)
        plt.legend()
        plt.title('类群 ' + str(i) + ' Feature Distribution Figure')
        name = '类群' + str(i) +'FeatureDistribution.png'
        picname = address + name
        plt.savefig(picname)
        picnamenew = '/static/images/' + name
        images.append(picnamenew)

    response = render(request, 'get_results.html', {'images': images, 'cluster':cluster, 'cluster_label':cluster_label})
    # response.set_cookie('cluster', cluster)
    response.set_cookie('cluster_label',n_cluster)
    response.set_cookie('cluster_label',cluster_label)
    return response
    # return render(request, 'get_results.html', {'images': images},)

# 用来展现对表和字段的选择，为了后续进行组群词云图绘制
def select_features(request):
    # import featuretools as ft
    # import pandas as pd
    # import numpy as np
    # from featuretools.primitives import make_trans_primitive, make_agg_primitive
    # from featuretools.variable_types import DatetimeTimeIndex, Numeric
    # import os
    # import re
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
#     wordcloud.render_notebook()
    
    # address = './ft_model/static/images/'
    # name = 'wordcloud.html'
    # picname = address + name
    # # print(picname)
    # picnamenew = '/static/images/' + name
    # images = [picnamenew]

    # page.render(picname)
    

    return HttpResponse(page.render_embed())
    # # 绘制分组变量分布图
    # from matplotlib.backends.backend_agg import FigureCanvasAgg
    # for i in range(n_cluster):
    #     km_cluster_group = data_df_kmeans.loc[cluster.labels_ == i, :]
    #     p = km_cluster_group.plot(kind='kde', linewidth = 2, subplots = True, sharex = False)
    #     plt.legend()
    #     canvas=FigureCanvasAgg(figure)
    #     response=HttpResponse(content_type='image/png')
    #     canvas.print_png(response)
    # return response
    # except Exception as e:
    #     response = render(request, 'erro.html', {'erro': e})
    # return response


        



# # 函数get_results用来处理模型相关参数提交后服务器响应的结果
# def get_results(request):
#     try:
#         import featuretools as ft
#         import pandas as pd
#         import numpy as np
#         from featuretools.primitives import make_trans_primitive, make_agg_primitive

#         # 数据源相关的参数
#         types_dict = eval(request.COOKIES['types_dict'])
#         columns_dict = eval(request.COOKIES['columns_dict'])
#         target = request.COOKIES['target']

#         # 如何决定 base entity?
#         # 目前思路是由 id 类型最多的 entity 来做 base entity
#         # 把对应的表和id个数封装成字典，然后根据个数给表名排逆序，然后按照这个顺序merge表，是为最终思路
#         base_entity = ''
#         base_index = ''

#         max_count = 0
#         sorted_dict = {}
#         for k, v in types_dict.items():
#             count = 0

#             index = ''
#             for i in v:
#                 if '.Id' in str(i):
#                     count += 1
#                 if '.Index' in str(i):
#                     index = i
#             sorted_dict[k] = count
#             if count > max_count:
#                 base_entity = k
#                 base_index = index
#                 max_count = count
#         sorted_list = sorted(sorted_dict.items(), key=lambda item: item[1], reverse=True)
#         sorted_table_name = [i[0] for i in sorted_list]

#         print("sorted_table_name\n", sorted_table_name)

#         # 把columns 和对应的 类型拼接成字典，存在一个列表中,并且找到base_index
#         types_dict_list = []
#         entity_name_list = []
#         for key, values1, values2 in zip(columns_dict.keys(), columns_dict.values(), types_dict.values()):
#             types_dict_list.append({k: eval(v) for k, v in zip(values1, values2)})
#             entity_name_list.append(key)
#             if key == base_entity:
#                 for k, v in zip(values2, values1):
#                     if '.Index' in k:
#                         base_index = v

#         # 自动识别标记为Index的特征，并作为抽取实体的index参数，传入模型
#         # 把所有的类型字典拼成一个大字典
#         index_list = []
#         total_type_dict = {}
#         for each_dict in types_dict_list:
#             total_type_dict.update(each_dict)
#             for k, v in each_dict.items():
#                 if '.Index' in str(v):
#                     index_list.append(k)
#         print(index_list)
#         # print(total_type_dict)

#         # 原表全部join在一起之后再抽取实体
#         # 数据接口改成处理CSV结构
#         import os
#         import re
#         if not os.path.isdir(os.getcwd() + "/demo_data"):
#             os.mkdir(os.getcwd() + "/demo_data")
#         os.chdir(os.getcwd() + "/demo_data")
#         regex = re.compile("csv")
#         raw_dict = {}

#         for file in os.listdir(os.getcwd()):
#             if re.search(regex, file):
#                 raw_dict[file.split(".")[0]] = pd.read_csv(file)

#         data = raw_dict
#         os.chdir("..")

#         # todo : merge的逻辑比较复杂，要如何执行join操作？？
#         if len(data) == 0:
#             raise Exception("数据源为空，请检查数据源文件")
#         elif len(data) > 1:
#             data_df = data.pop(sorted_table_name.pop(0))
#             # print(data_df)
#             for i in sorted_table_name:
#                 data_df = data_df.merge(data[i])
#             #
#             # for i in list(data.values()):
#             #     data_df = data_df.merge(i)

#         elif len(data) == 1:
#             data_df = list(data.values())[0]
#         es = ft.EntitySet()

#         # print("+++++++++++++++++++++++")
#         # print("data_df\n", data_df)
#         # print("entity_id\n", base_entity)
#         # print("base_index\n", base_index)
#         # print("total_type_dict\n", total_type_dict)
#         # print("+++++++++++++++++++++++")
#         # 构造base entity, 将第一个表名作为基础实体名称
#         es = es.entity_from_dataframe(entity_id=base_entity,
#                                       dataframe=data_df,
#                                       index=base_index,
#                                       # time_index="transaction_time",
#                                       variable_types=total_type_dict)

#         # 基于base entity抽取实体,逻辑比较复杂，基本逻辑是作为base entity的字段，跳过实体抽取，其余的将index 字段单独存储，设为index参数
#         for k, v in columns_dict.items():
#             if k == base_entity:
#                 continue
#             index = ''
#             for i in index_list:
#                 if i in v:
#                     v.remove(i)
#                     index = i
#             # print("=========")
#             # print(k)
#             # print(index)
#             # print(v)
#             # print("=========")
#             es = es.normalize_entity(base_entity_id=base_entity,
#                                      new_entity_id=k,
#                                      index=index,
#                                      # make_time_index="session_start",
#                                      additional_variables=v)

#         """
#         自定义agg_primitives:
#         改写time since last，原函数为秒，现在改为小时输出
#         """

#         def time_since_last_by_hour(values, time=None):
#             time_since = time - values.iloc[-1]
#             return time_since.total_seconds() / 3600

#         Time_since_last_by_hour = make_agg_primitive(function=time_since_last_by_hour,
#                                                      input_types=[ft.variable_types.DatetimeTimeIndex],
#                                                      return_type=ft.variable_types.Numeric,
#                                                      uses_calc_time=True)

#         """
#         自定义trans_primitives:
#         添加log e 的自然对数
#         """
#         import numpy as np

#         def log(vals):
#             return np.log(vals)

#         # def generate_name(self, base_feature_names):
#         #     return "-(%s)" % (base_feature_names[0])
#         log = make_trans_primitive(function=log,
#                                    input_types=[ft.variable_types.Numeric],
#                                    return_type=ft.variable_types.Numeric,
#                                    # uses_calc_time=True,
#                                    description="Calculates the log of the value.",
#                                    name="log")
#         """
#         自定义trans_primitives:
#         判断是否为正数
#         """
#         import numpy as np

#         def is_positive(vals):
#             return vals > 0

#         # def generate_name(self, base_feature_names):
#         #     return "-(%s)" % (base_feature_names[0])
#         is_positive = make_trans_primitive(function=is_positive,
#                                            input_types=[ft.variable_types.Numeric],
#                                            return_type=ft.variable_types.Boolean,
#                                            # uses_calc_time=True,
#                                            description="Calculates if the value positive.",
#                                            name="is_positive")

#         # 模型相关的参数
#         max_depth = request.POST['max_depth']
#         agg_pri = request.POST.getlist('agg_pri')
#         agg_pri_customer = request.POST.getlist('agg_pri_customer')
#         trans_pri_customer = request.POST.getlist('trans_pri_customer')
#         trans_pri = request.POST.getlist('trans_pri')
#         context = {'max_depth': max_depth, 'agg_pri': agg_pri, 'trans_pri': trans_pri}

#         pd.set_option('display.max_columns', 20)

#         # 将前端页面的提交参数，保存为agg_pri列表
#         agg_pri = context['agg_pri']
#         trans_pri = context['trans_pri']
#         print(trans_pri_customer)
#         # 如果勾选了参数，加上自定义的Time_since_last_by_hour
#         if 'Time_since_last_by_hour' in agg_pri_customer:
#             agg_pri.append(Time_since_last_by_hour)
#         if 'log_e' in trans_pri_customer:
#             trans_pri.append(log)
#         if 'is_positive' in trans_pri_customer:
#             trans_pri.append(is_positive)
#         print("+++++++++++++++++++++++++++++")

#         print(trans_pri)
#         print("+++++++++++++++++++++++++++++")
#         # 生成新的特征融合矩阵
#         feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity=target,
#                                               agg_primitives=agg_pri,
#                                               trans_primitives=trans_pri,
#                                               max_depth=int(context['max_depth']))

#         # 将索引作为第一列插入数据矩阵
#         feature_matrix = feature_matrix.reset_index()
#         new_columns = feature_matrix.columns

#         # 保存数据矩阵,注意在特征选择界面，没有 customer_id 作为选项，因为这只是索引
#         # nlp 数组是将primitives替换为中文后的表头，一并显示在第二行
#         import os
#         if not os.path.isdir(os.getcwd() + "/demo_data/result"):
#             os.mkdir(os.getcwd() + "/demo_data/result")
#         feature_matrix.to_csv("./demo_data/result/all_features.csv", index=False)
#         # print(feature_matrix.head(5))
#         from .columns2NLP import columns2NLP
#         res = []
#         nlp = []
#         for i in new_columns:
#             res.append(str(i))
#             nlp.append(columns2NLP(str(i)))
#         # print(res[0])
#         # print("======================")
#         # print(res)
#         # print(nlp)
#         # print("======================")
#         # 将所有的浮点数精度调整到小数点后两位
#         sample_data1 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix.iloc[0]]
#         sample_data2 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix.iloc[1]]
#         sample_data3 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix.iloc[2]]
#         sample_data4 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix.iloc[3]]
#         sample_data5 = [round(i, 2) if isinstance(i, float) else i for i in feature_matrix.iloc[4]]
#         response = render(request, 'get_results.html', {'res': res,
#                                                         'nlp': nlp,
#                                                         'sample_data1': sample_data1,
#                                                         'sample_data2': sample_data2,
#                                                         'sample_data3': sample_data3,
#                                                         'sample_data4': sample_data4,
#                                                         'sample_data5': sample_data5})
#         response.set_cookie('target_id', res[0])
#         return response

#     except Exception as e:
#         response = render(request, 'erro.html', {'erro': e})
#         return response


# # 函数selected_features用来处理特征选择提交后服务器响应的结果
# def selected_features(request):
#     import re
#     selected = request.POST.getlist('selected')
#     target_id = request.COOKIES['target_id']
#     columns = list(selected)
#     columns.insert(0, target_id)
#     import pandas as pd
#     df = pd.read_csv("./demo_data/result/all_features.csv")
#     # print(columns)
#     new_df = df[columns]
#     # print(new_df)
#     new_df.to_csv("./demo_data/result/selected_features.csv", index=False)

#     # print(new_df.iloc[0])

#     # 显示的时候由于pandas中全部统一处理成float，导致ID之类的整形数变成带小数点的，目前没有找到更好的解决办法，
#     # 只好使用正则表达式进行区分打印，注意，这只是打印，与存储无关。

#     # print("+_++__+_+_+_+_+_+_+_+_+_")
#     # print(new_df.iloc[0])
#     # print("+_++__+_+_+_+_+_+_+_+_+_")

#     def transfer(data_sample):
#         return_list = []
#         for i in data_sample:
#             if i is None:
#                 return_list.append(None)
#             if isinstance(i, str):
#                 return_list.append(i)
#             elif re.search(r"\.0\b", str(i)):
#                 return_list.append(int(i))
#             else:
#                 return_list.append(round(i, 2))
#         return return_list

#     sample_data1 = transfer(new_df.iloc[0])
#     sample_data2 = transfer(new_df.iloc[1])
#     sample_data3 = transfer(new_df.iloc[2])
#     sample_data4 = transfer(new_df.iloc[3])
#     sample_data5 = transfer(new_df.iloc[4])

#     # print(sample_data1)
#     return render(request, "selected_features.html",
#                   {"columns": columns, 'sample_data1': sample_data1, 'sample_data2': sample_data2,
#                    'sample_data3': sample_data3, 'sample_data4': sample_data4,
#                    'sample_data5': sample_data5})