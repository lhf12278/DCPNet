import argparse
# from UDL.Basis.option import panshaprening_cfg, Config, os
from UDL.AutoDL import TaskDispatcher
import os

class parser_args(TaskDispatcher, name='UPNet'):
    def __init__(self, cfg=None):

        if cfg is None:
            from UDL.Basis.option import panshaprening_cfg
            cfg = panshaprening_cfg()

        script_path = os.path.dirname(os.path.dirname(__file__))
        root_dir = script_path.split(cfg.task)[0]

        model_path = f'{root_dir}/results/{cfg.task}/gf2/UPNet/Test/.pth.tar'
        # model_best_.pth
        # model_path = f''
        parser = argparse.ArgumentParser(description='PyTorch Pansharpening Training')
        # * Logger
        parser.add_argument('--out_dir', metavar='DIR', default=f'{root_dir}/results/{cfg.task}',
                            help='path to save model')
        # * Training
        parser.add_argument('--lr', default=2e-4, type=float)  # 1e-4 2e-4 8
        parser.add_argument('--lr_scheduler', default=True, type=bool)
        parser.add_argument('--samples_per_gpu', default=16, type=int,  # 8
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--print-freq', '-p', default=50, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--seed', default=0, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--epochs', default=200, type=int)
        parser.add_argument('--workers_per_gpu', default=0, type=int)
        parser.add_argument('--resume_from',
                            default=model_path,
                            type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')

        # * Model and Dataset
        parser.add_argument('--arch', '-a', metavar='ARCH', default='UPNet', type=str,
                            choices=['UPNet'])
        parser.add_argument('--dataset', default={'train': 'gf2', 'val': 'RR_TestData_gf2'}, type=str,
                            choices=[None, 'wv2_hp', 'wv3_hp', 'wv4_hp', 'qb_hp',
                                     'TestData_qb_hp', 'TestData_wv2_hp', 'TestData_wv3_hp', 'TestData_wv4_hp',
                                     'San_Francisco_QB_RR_hp', 'San_Francisco_QB_FR_hp', 'NY1_WV3_FR_hp',
                                     'NY1_WV3_RR_hp', 'Alice_WV4_FR', 'Alice_WV4_RR_hp', 'Rio_WV2_FR_hp', 'Rio_WV2_RR_hp'],
                            help="training choices: ['wv2', 'wv3', 'wv4', 'qb'],"
                                 "validation choices: ['valid_wv2','valid_wv3', 'valid_wv4', 'valid_qb']"
                                 "test choices is ['TestData_wv2', 'TestData_wv3', 'TestData_wv4', 'TestData_qb'], and others with RR/FR")
        parser.add_argument('--eval', default=False, type=bool,
                            help="performing evalution for patch2entire")

        args = parser.parse_args()
        # 使用best模型进行测试
        args.start_epoch = args.best_epoch = 1
        args.experimental_desc = "Test"

        cfg.merge_args2cfg(args)
        cfg.img_range = 2047.0
        cfg.reg = True
        cfg.workflow = [('train', 5), ('val', 1)]
        # cfg.workflow = [('train', 1)]
        # cfg.workflow = [('val', 1)]
        # cfg.workflow = [('train', 50)]
        print(cfg.pretty_text)

        self._cfg_dict = cfg