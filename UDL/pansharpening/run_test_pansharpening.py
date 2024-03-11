# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:

import sys
sys.path.append('../..')
from UDL.AutoDL import TaskDispatcher
from UDL.AutoDL.trainer import main

import setproctitle
setproctitle.setproctitle("yangxuji")

if __name__ == '__main__':
    cfg = TaskDispatcher.new(task='pansharpening', mode='entrypoint', arch='UPNet')
    cfg.eval = True
    cfg.workflow = [('val', 1)]
    print(TaskDispatcher._task.keys())
    main(cfg)
