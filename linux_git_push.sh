#!/bin/bash

# 检查当前目录是否为 Git 仓库
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "当前目录不是一个 Git 仓库，请在 Git 仓库目录下运行此脚本。"
    exit 1
fi

# 添加所有修改的文件到暂存区
git add .

# 提交暂存区的内容到本地仓库
git commit -m "commit from linux"

# 获取当前分支名称
current_branch=$(git branch --show-current)

# 推送到远程仓库对应的分支
git push origin master

echo "修改已成功提交并推送到远程仓库。"