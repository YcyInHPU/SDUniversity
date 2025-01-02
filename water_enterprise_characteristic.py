# -*- coding: UTF-8 -*- #
"""
@filename:water_pollution_characteristic.py
@author:Chunyu Yuan
@time:2024-11-04
"""

"""
水质污染溯源
参数：
（1）input_path 输入路径
（2）output_path 输出路径
（3）Province 省份
（4）City 城市
（4）Basin2 二级流域
（5）Basin3 三级流域
（6）WQParams 行业类别
（7）Institution 山东大学

"""
from enum import Enum

from common.task.docker_adapter import model_task_submit
from common.task.load_model import LoadModel, wait_all
from common.task.workflow_param import WorkflowParam
from common.utils.define_remote import remote
from common.utils.file_util import find_files_by_pattern_v2
from common.utils.file_util import walkDirFile

class ModelEnum(Enum):
    EnterprisePollutionCharacteristic = '企业排污特征图谱'

class AnalysisParam(WorkflowParam):
    def __init__(self, team_id, account, workflow_id, input_path, output_path, Province, City, Basin2, Basin3,
                 WQParams, Institution):
        super().__init__(account, output_path, workflow_id, team_id)
        self.input_path = input_path
        self.output_path = output_path
        self.Province = Province
        self.City = City
        self.Basin2 = Basin2
        self.Basin3 = Basin3
        self.WQParams = WQParams
        self.Institution = Institution

@remote(max_retries=0)
def calc_tracing(model: LoadModel, param: AnalysisParam):
    """
    """
    user_host_path, user_container_path = param.get_model_output_path(model.model_enum.name)
    input_param = {
        "input_path": param.input_path,
        "output_path": user_container_path,
        "Province": param.Province,
        "City": param.City,
        "Basin2": param.Basin2,
        "Basin3": param.Basin3,
        "WQParams": param.WQParams,
        "Institution": param.Institution,
    }
    # print(f"** out dir:{user_container_path}")
    model.get_model_submit_command(input_param, param.workflow_id)
    model_task_submit(model)
    # prd_files = walkDirFile(user_host_path, ".xlsx")
    return (WorkflowParam._build_model_file_output("全行业排污特征", find_files_by_pattern_v2(f'{user_host_path}', '.*排污特征.*.jpg$')),
            WorkflowParam._build_model_file_output("排污企业数量", find_files_by_pattern_v2(f'{user_host_path}', '.*企业数量.*.jpg$')),
            WorkflowParam._build_model_file_output("排污特征图谱", find_files_by_pattern_v2(f'{user_host_path}', '.*特征图谱.*.jpg$')),
            WorkflowParam._build_model_file_output("排污特征信息", find_files_by_pattern_v2(f'{user_host_path}', '.*.xlsx$')),
            WorkflowParam._build_model_file_output("排污特征图谱分析报告", find_files_by_pattern_v2(f'{user_host_path}', '.*_分析报告.*.docx$')))

def get_workflow(param: AnalysisParam):
    account = param.account
    # 加载模型
    model = LoadModel(account, ModelEnum.EnterprisePollutionCharacteristic)
    #
    prd_file = (calc_tracing
                .options(**model.to_resources_options(),
                         **model.to_workflow_options())
                .bind(model, param))
    return wait_all.bind(prd_file)

