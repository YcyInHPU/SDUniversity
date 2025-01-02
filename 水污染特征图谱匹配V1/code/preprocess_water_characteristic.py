# -*- coding: UTF-8 -*- #
"""
@filename:preprocess_water_characteristic.py
@author:Chunyu Yuan
@time:2024-11-04
"""
import pandas as pd
import numpy as np
import os
import shutil
import argparse
from docxtpl import DocxTemplate,InlineImage
from docx.shared import Mm
from datetime import datetime
import platform
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def copy_image(src_folder, dest_folder, image_name):
    # 判断源文件夹是否存在
    if not os.path.exists(src_folder):
        print("源文件夹不存在，请检查路径。")
        return

    # 构造源文件的完整路径
    # src_image_path = os.path.join(src_folder, image_name)
    dest_folder_name = os.path.join(dest_folder, image_name)
    # 检查源文件是否存在
    if os.path.exists(src_folder):
        # 复制文件到目标文件夹
        shutil.copy(src_folder, dest_folder_name)
        # print(f"成功复制 {image_name} 到 {dest_folder}")
    else:
        print("图片未找到，请输入有效的文件名。")

def Report_Generation():
    # 加载模板
    template_path = input_path + '/' + '报告模板.docx'  # 文件路径
    doc = DocxTemplate(template_path)
    # 加载图片
    image_path = 'image.png'
    image = InlineImage(doc, image_path, width=Mm(140))
    shutil.copy(image_src_folder, image_path)
    # 替换属性
    data = {
    'Province': Province,
    'City': City,
    'Basin2': Basin2,
    'Basin3': Basin3,
    'WQParams': WQParams,
    'Institution': Institution,
    'TIME': datetime.now().strftime('%Y年%m月%d日%H时'),
    'image': image,
    }
    # 渲染模板
    doc.render(data)
    # 输出新的文件
    file_name = output_path + '/' + Province + City + Basin2 + Basin3 + '_' + WQParams + '_特征图谱分析报告.docx'
    # 保存新的 Word 文件
    doc.save(file_name)
    # 使用 LibreOffice 转换 Word 文档为 PDF
    os.system("libreoffice --headless --convert-to pdf *.docx")
    print(f"生成的新文件: {file_name}")
    return

def word2pdf():
    """
    word转pdf
    :param wordPath: word文件路径
    :param pdfPath: pdf文件路径
    :return: word to pdf
    """
    wordPath = output_path + '/' + Province + City + Basin2 + Basin3 + '_' + WQParams + '_特征图谱分析报告.docx'
    pdfPath = output_path + '/' + Province + City + Basin2 + Basin3 + '_' + WQParams + '_特征图谱分析报告.pdf'
    if platform.system() == 'Windows':
        from win32com.client import constants, gencache
        word = gencache.EnsureDispatch('Word.Application')
        doc = word.Documents.Open(wordPath, ReadOnly=1)
        doc.ExportAsFixedFormat(pdfPath,
                                constants.wdExportFormatPDF,
                                Item=constants.wdExportDocumentWithMarkup,
                                CreateBookmarks=constants.wdExportCreateHeadingBookmarks)
        word.Quit(constants.wdDoNotSaveChanges)
    else:
        # os.system(f'libreoffice --convert-to pdf {wordPath} --outdir {os.path.split(wordPath)[0]}')
        # os.system(f'libreoffice7.1 --convert-to pdf {wordPath} --outdir {os.path.split(wordPath)[0]}')
        os.system(
            f'libreoffice7.1 --convert-to pdf {wordPath} --outdir {os.path.split(wordPath)[0]} > /dev/null 2>&1')

def parse_args():
    parser = argparse.ArgumentParser(description='Run Task with specified options.')
    parser.add_argument('--input_path', default='4水环境大数据分析/1水污染特征图谱匹配/Input', type=str)
    parser.add_argument('--output_path', default='4水环境大数据分析/1水污染特征图谱匹配/Output', type=str)
    parser.add_argument('--Province', default='山东省', type=str)
    parser.add_argument('--City', default='济南市', type=str)
    parser.add_argument('--Basin2', default='沂沭泗流域', type=str)
    parser.add_argument('--Basin3', default='湖西区', type=str)
    parser.add_argument('--WQParams', default='有色可溶性有机物CDOM', type=str)
    parser.add_argument('--Institution', default='山东大学', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print("run...")
    # 读取数据
    # input_path = r'E:\XM\20240806HTHT\水质特征图谱\Input'
    # output_path = r'E:\XM\20240806HTHT\水质特征图谱\Output'
    # Province = '山东省'
    # City = '济南市'
    # Basin2 = '沂沭泗流域'
    # Basin3 = '湖西区'
    # WQParams = '有色可溶性有机物CDOM'
    # Institution = '山东大学'

    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path
    Province = args.Province
    City = args.City
    Basin2 = args.Basin2
    Basin3 = args.Basin3
    WQParams = args.WQParams
    Institution = args.Institution

    try:
        # 输入源文件夹路径和目标文件夹路径
        image_src_folder = input_path + '/' + WQParams + '.jpg'
        # 输入图片文件名
        image_name = Province + City + Basin2 + Basin3 + '_' + WQParams + '_特征图谱.jpg'

        table_src_folder = input_path + '/' + WQParams + '.xlsx'
        # 输入表格文件名
        table_name = Province + City + Basin2 + Basin3 + '_' + WQParams + '_光谱特征.xlsx'
        # 调用函数进行复制
        copy_image(image_src_folder, output_path, image_name)
        copy_image(table_src_folder, output_path, table_name)
        Report_Generation() # 报告生成
        word2pdf()  # PDF生成
        print("业务分析成功，已写入输出文件!")

    except FileNotFoundError:
        print("文件未找到，请检查文件名和路径，或上传数据!")
    except Exception as e:
        print(f"发生错误：{e}")