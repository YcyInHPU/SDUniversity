# -*- coding:utf-8 -*-
"""
水质污染溯源
参数：
（1）input_path 输入路径
（2）output_path 输出路径
（3）Province 省份
（4）City 城市
（5）Institution 山东大学
（6）Site_Name 站点名称
（7）WQParams 水质参数
（8）model_name1 模型选择——异常检测
（9）model_name2 模型选择——水质溯源
"""
from enum import Enum

from common.task.docker_adapter import model_task_submit
from common.task.load_model import LoadModel, wait_all
from common.task.workflow_param import WorkflowParam
from common.utils.define_remote import remote
from common.utils.file_util import find_files_by_pattern_v2
from common.utils.file_util import walkDirFile


class ModelEnum(Enum):
    WaterPollutionTracingAnalysis = '水质与污染源关联分析'


class AnalysisParam(WorkflowParam):
    def __init__(self, team_id, account, workflow_id, input_path, output_path, Province, City, Institution,
                 Site_Name, WQParams, model_name1, model_name2):
        super().__init__(account, output_path, workflow_id, team_id)
        self.input_path = input_path
        self.output_path = output_path
        self.Province = Province
        self.City = City
        self.Institution = Institution
        self.Site_Name = Site_Name
        self.WQParams = WQParams
        self.model_name1 = model_name1
        self.model_name2 = model_name2


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
        "Institution": param.Institution,
        "Site_Name": param.Site_Name,
        "WQParams": param.WQParams,
        "model_name1": param.model_name1,
        "model_name2": param.model_name2,
    }
    # print(f"** out dir:{user_container_path}")
    model.get_model_submit_command(input_param, param.workflow_id)
    model_task_submit(model)
    # prd_files = walkDirFile(user_host_path, ".xlsx")
    return (WorkflowParam._build_model_file_output("水质异常监测专题图", find_files_by_pattern_v2(f'{user_host_path}', '.*.png$')),
            WorkflowParam._build_model_file_output("水质溯源-水质监测异常点", find_files_by_pattern_v2(f'{user_host_path}', '.*_异常值.*.xlsx$')),
            WorkflowParam._build_model_file_output("水质溯源-企业疑似概率", find_files_by_pattern_v2(f'{user_host_path}', '.*_企业疑似概率.*.xlsx$')),
            WorkflowParam._build_model_file_output("水质溯源-产汇流迁移分析", find_files_by_pattern_v2(f'{user_host_path}', '.*_水质与污染源关联分析.*.jpg$')),
            WorkflowParam._build_model_file_output("水质溯源-水质溯源结果", find_files_by_pattern_v2(f'{user_host_path}', '.*_水质溯源结果.*.xlsx$')),
            WorkflowParam._build_model_file_output("水质溯源分析报告", find_files_by_pattern_v2(f'{user_host_path}', '.*_水质溯源分析报告.*.docx$')))


def get_workflow(param: AnalysisParam):
    account = param.account
    # 加载模型
    model = LoadModel(account, ModelEnum.WaterPollutionTracingAnalysis)
    #
    prd_file = (calc_tracing
                .options(**model.to_resources_options(),
                         **model.to_workflow_options())
                .bind(model, param))
    return wait_all.bind(prd_file)
