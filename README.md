# Git 学习笔记

## 目录
- [一、Git 下载与安装]
- [二、基础配置]
- [三、本地代码提交]
- [四、推送到远程仓库]
- [五、完整工作流程示例]
- [六、常用命令速查]
- [七、常见问题]

---

## 一、Git 下载与安装 {#一git-下载与安装}

### Windows
1. 访问官网下载：[https://git-scm.com/download/win](https://git-scm.com/download/win)
2. 双击安装包，按默认选项完成安装
3. 验证安装：  
   ```bash
   git --version
   ```

---

## 二、基础配置
```bash
# 配置用户名
git config --global user.name "Your Name"

# 配置邮箱
git config --global user.email "your.email@example.com"

# 查看配置
git config --list
```

---

## 三、本地代码提交 

### 1. 初始化仓库
```bash
git init
```

### 2. 添加文件到暂存区
```bash
# 添加所有文件
git add .

# 添加指定文件
git add index.html style.css
```

### 3. 提交到本地仓库
```bash
git commit -m "提交说明"
```

> **最佳实践**  
> - 提交信息应简洁明确（建议使用英文）  
> - 推荐格式：`类型: 描述`（如 `feat: add user login`）

---

## 四、推送到远程仓库

### 1. 关联远程仓库
```bash
git remote add origin https://github.com/用户名/仓库名.git
```

### 2. 首次推送
```bash
git push -u origin main
```

### 3. 后续推送
```bash
git push
```

> **⚠️ 注意**  
> - 推送前建议先执行 `git pull` 同步最新代码  
> - 谨慎使用 `git push -f`（强制推送）

---

## 五、完整工作流程示例
```bash
# 初始化仓库
git init

# 添加所有文件
git add .

# 首次提交
git commit -m "initial commit"

# 关联远程仓库
git remote add origin https://github.com/user/repo.git

# 推送代码
git push -u origin main

# 后续修改...
git add modified-file.txt
git commit -m "update: modify file"
git push
```

---

## 六、常用命令速查

| 命令                    | 说明                 |
|-------------------------|----------------------|
| `git status`            | 查看仓库状态         |
| `git log`               | 查看提交历史         |
| `git pull`              | 拉取远程更新         |
| `git clone 仓库URL`     | 克隆远程仓库         |
| `git branch`            | 查看分支列表         |
| `git checkout -b 分支名` | 创建并切换新分支     |

---

## 七、常见问题
### Q: 提交时提示 "nothing to commit"
```bash
# 确保文件已添加修改
git add 文件名
# 或添加所有修改
git add .
```

### Q: 推送时提示权限拒绝
1. 检查远程仓库地址：
   ```bash
   git remote -v
   ```
2. 确认 SSH 密钥配置：  
   [GitHub SSH 配置指南](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

### Q: 如何撤销最后一次提交？
```bash
# 保留修改（仅撤销提交）
git reset --soft HEAD~1

# 丢弃修改（完全撤销）
git reset --hard HEAD~1
```
