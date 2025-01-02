# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 20:53:14 2024

@author: htht
"""

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise
from docxtpl import DocxTemplate
from docx.shared import Mm
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def Min_Max(data):  # 对连续变量进行标准化处理
    """
    Min_Max归一化
    """
    min_val = np.min(data)
    max_val = np.max(data)
    # 极差
    ranges = max_val - min_val
    norm_data = (data - min_val) / ranges
    return norm_data

# 设置参数
params1 = {'n_estimators': 500,
          'max_depth': 8, 
          'min_child_weight': 1, 
          'gamma': 0,
          'subsample': 0.8, 
          'colsample_bytree': 0.8, 
          'reg_alpha': 0, 
          'reg_lambda': 1, 
          'learning_rate': 0.1,
          'silent':True,
          'objective':'multi:softmax'}
params2 = {'objective': 'multi:softprob',  # 多分类问题
           'num_class': 10,  # 类别数量
           'eval_metric': 'mlogloss',
           'max_depth': 3,
           'eta': 0.1,
           'subsample': 0.8,
           'colsample_bytree': 0.8}
mapping = {0:'养殖',
           1:'食品加工',
           2:'饮品加工',
           3:'纺织品',
           4:'化工材料',
           5:'药品制造',
           6:'建筑材料',
           7:'金属加工',
           8:'电池-电子产品',
           9:'热力生产'}

def Data_Time(input_path,Site_Name):
    # 读取 Excel 文件
    input_file = input_path + '/' + Site_Name + '.xlsx'
    print(input_file)
    data = pd.read_excel(input_file).round(2)

    # 假设时间列的名称为 'Date'
    data['DataTime'] = pd.to_datetime(data['监测时间'])  # 将日期列转换为 datetime 对象
    data['DataTime'] = data['DataTime'].dt.strftime('%Y%m%d')  # 格式化为 'YYYYMMDD'

    # 保存修改后的 Excel 文件
    # data.to_excel('modified_file.xlsx', index=False)
    
    return data

def Anomaly_Detection(data,WQParams,output_path):
    # data = pd.read_csv(input_path,encoding='gbk')
    
    # WQParams = '总氮'
    # 提取水质列
    ammonia_nitrogen = data[[WQParams]].round(2)

    # 使用孤立森林算法
    model = IsolationForest(contamination=0.01)
    # model = model_name1(contamination=0.01)
    data['anomaly'] = model.fit_predict(ammonia_nitrogen.values)

    # 将异常值标记为1，正常值标记为0
    data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})

    # 计算平均值
    mean_value = ammonia_nitrogen.mean().values[0]
    
    if WQParams == "溶解氧":
        # 筛选出异常值且小于平均值的行
        anomalies = data[(data['anomaly'] == 1) & (data[WQParams] < mean_value)]
    else:
        # 筛选出异常值且大于平均值的行
        anomalies = data[(data['anomaly'] == 1) & (data[WQParams] > mean_value)]
    
    
    row_count = len(anomalies) # 获取异常值个数

    # 导出包含异常值的行到Excel
    anomalies.to_excel(output_path + '/' + Site_Name + '_' + WQParams + '_异常值.xlsx', index=False)

    # 绘制散点图
    plt.figure(figsize=(10, 6))
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 正常值
    plt.scatter(data[data['anomaly'] == 0].index, data[data['anomaly'] == 0][WQParams], color='black', label='Normal')
    # 异常值
    plt.scatter(anomalies.index, anomalies[WQParams], color='red', label='Anomaly')

    plt.xlabel('Annual date')
    plt.ylabel('Monitored value')
    plt.title('水质异常值检测--' + WQParams)
    plt.legend()
    plt.grid()

    # 导出散点图
    plt.savefig(output_path + '/' + Site_Name + '_' + WQParams + '_异常检测.png', dpi = 300)
    plt.show()
    return anomalies,row_count

def Water_quality_trace(monitoring_data,monitoring_time):
    # print("run...")
    # print(monitoring_time)
    # 读取数据
    # monitoring_data = anomalies  # 水质监测数据
    # pollution_data = pd.read_csv(input_path + '/' + Site_Name + '_Discharge.csv',encoding='gbk')  # 企业排污数据
    pollution_data = pd.read_excel(input_path + '/' + Site_Name + '_Discharge.xlsx').round(2)  # 企业排污数据
    pollution_count = len(pollution_data)
    # print(f"行数:{pollution_count}")
    # 选择水质特征参数
    monitoring_values = monitoring_data[['经度','纬度','溶解氧', '氨氮', '总氮', '总磷']]
    pollution_values = pollution_data[['经度','纬度','溶解氧', '氨氮', '总氮', '总磷']]
    
    # 加载XGBoost模型进行水质行业溯源
    load_model = xgb.XGBClassifier(params1)
    # load_model.load_model(input_path + '/' + 'xgboost_model.json')
    # prediction = load_model.predict(monitoring_values)
    # print('水质异常疑似行业：',mapping.get(prediction[0]))
    
    # 计算相似度 得到每个企业与监测样本的的距离 距离越小，相似度越高
    distances = pairwise.euclidean_distances(pollution_values, monitoring_values) #（欧氏距离）
    # distances = cosine_similarity(pollution_values, monitoring_values) #余弦相似度
    # distances = Min_Max(distances)

    # 将距离转换为疑似度
    suspected_probabilities = 1 / (0.01 + distances)  # 加1避免除以0的情况
    # suspected_probabilities_df = pd.DataFrame(suspected_probabilities, columns=monitoring_data['SampleID'])

    # 假设pollution_data中有企业名称
    results = pollution_data.copy()
    results['疑似概率'] = suspected_probabilities.max(axis=1)  # 取每个企业最大的疑似概率Suspected_Probability
    # 按照疑似度大小排序
    results = results.sort_values(by = '疑似概率',ascending = False)
    results['疑似概率'] = (results['疑似概率'] * 100).round(2).astype(str) + '%'
    Column_name = pollution_data.columns.tolist() + ['疑似概率'] # 定义Excel文件的列名称
    monitoring_time = str(monitoring_time.iloc[0,0])
    file_name = output_path + '/'  + Site_Name + '_' + WQParams + '_' + monitoring_time + '_企业疑似概率.xlsx'
    results[Column_name].to_excel(file_name, index=False, engine = 'openpyxl')
    first_column = results.iloc[0,:] # 获取疑似概率最高的企业
    return results, first_column, pollution_count

def Enterprise_trace(time_list, data_list):
    time_list = pd.concat(time_list,ignore_index=True)
    data_list = pd.DataFrame(data_list)
    data_list.reset_index(inplace=True)
    # print(time_list)
    # print(data_list)
    # Enterprise_trace_list = data_list.insert(0, 'DataTime', time_list)
    Enterprise_trace_list = pd.concat([time_list, data_list], axis=1)
    Enterprise_trace_list = Enterprise_trace_list.drop('index',axis=1)
    # print(Enterprise_trace_list)
    file_name = output_path + '/'  + Site_Name + '_' + WQParams + '_水质溯源结果.xlsx'
    Enterprise_trace_list.to_excel(file_name, index=False, engine = 'openpyxl')
    return

def Report_Generation():
    # 加载模板
    template_path = input_path + '/' + Site_Name + '_' + WQParams + '_水质污染溯源报告模板.docx'  # 文件路径
    doc = DocxTemplate(template_path)
    # 加载图片
    image_path = output_path + '/' + Site_Name + '_' + WQParams + '_异常检测.png'
    image = InlineImage(doc, image_path, width=Mm(140))

    data = {
    'Province': Province,
    'City': City,
    'Institution': Institution,
    'TIME': datetime.now().strftime('%Y年%m月%d日%H时'),
    'Site_Name': Site_Name,
    'WQParams': WQParams,
    'model_name1': model_name1,
    'model_name2': model_name2,
    'row_count': row_count,
    'pollution_count': pollution_count,
    'image': image,
    }

    # 渲染模板
    doc.render(data)

    # 输出新的文件
    file_name = output_path + '/' + Site_Name + '_' + WQParams + '_水质溯源分析报告.docx'
    # 保存新的 Word 文件
    doc.save(file_name)
    print(f"生成的新文件: {file_name}")
    return

def parse_args():
    parser = argparse.ArgumentParser(description='Run Task with specified options.')
    parser.add_argument('--input_path', default='Water_pollution_tracing/Input', type=str)
    parser.add_argument('--output_path', default='Water_pollution_tracing/Output', type=str)
    parser.add_argument('--Province', default='山东省', type=str)
    parser.add_argument('--City', default='济宁市', type=str)
    parser.add_argument('--Institution', default='山东大学', type=str)
    parser.add_argument('--Site_Name', default='105公路桥', type=str)
    parser.add_argument('--WQParams', default='溶解氧', type=str)
    parser.add_argument('--model_name1', default='IsolationForest', type=str)
    parser.add_argument('--model_name2', default='XGBoost', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print("run...")
    # 读取数据
    # input_path = r'E:\XM\20240806HTHT\Input'
    # output_path = r'E:\XM\20240806HTHT\Output'
    # Province = '山东省'
    # City = '济宁市'
    # Institution = '山东大学'
    # Site_Name = '105公路桥'
    # WQParams = '溶解氧'
    # model_name1 = 'IsolationForest'
    # model_name2 = 'XGBoost'
    
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path
    Province = args.Province
    City = args.City
    Institution = args.Institution
    Site_Name = args.Site_Name
    WQParams = args.WQParams
    model_name1 = args.model_name1
    model_name2 = args.model_name2
    try:
        # 数据时间转换
        data = Data_Time(input_path,Site_Name)
        # 异常值检测
        Anomaly_Detection = Anomaly_Detection(data,WQParams,output_path)
        anomalies = Anomaly_Detection[0]
        row_count = Anomaly_Detection[1]
        # 水质污染行业溯源
        data_list = []
        time_list = []
        for i in range(row_count):
            # 获取水质异常值的逐行数据
            monitoring_data = anomalies.iloc[[i]]
            monitoring_time = monitoring_data[['DataTime']]
            time_list.append(monitoring_time)
            # 对异常值进行溯源
            # Water_quality_trace(monitoring_data,monitoring_time)
            first_column = Water_quality_trace(monitoring_data, monitoring_time)[1] # 溯源，并获取疑似概率最高企业
            data_list.append(first_column) #搜集每次溯源疑似概率最高的企业
        Enterprise_trace(time_list, data_list) # 企业溯源分析
        Report_Generation() # 报告生成
        print("业务分析成功，已写入输出文件!")
            
    except FileNotFoundError:
        print("文件未找到，请检查文件名和路径，或上传数据!")
    except Exception as e:
        print(f"发生错误：{e}")
    
    