#!/bin/bash
clear
echo "开始执行"
# 撤销本地工作区的更改
git restore .

# 清空暂存区的更改
git restore --staged .

# 删除本地新增的未跟踪文件和目录
git clean -fd

# 拉取新的代码
git pull