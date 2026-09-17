# Shopkeeper Agent（电商智能问数）

把中文业务问题转换为安全的只读 SQL，并在本地电商订单数据上执行。

## 当前实现的完整链路

```text
用户问题 → 元数据匹配 → 上下文构建 → SQL 生成 → 安全校验 → SQLite 执行 → 带来源的结果
```

这个版本刻意使用 SQLite 和内置样例数据，以便无需 Docker 就能先跑通与测试。接口边界已经预留：后续可将规则 SQL 生成器替换为 DeepSeek，将 SQLite 替换为 MySQL，将元数据目录替换为向量库与全文检索。

## 快速开始

在 Windows PowerShell 中：

```powershell
.\.venv\Scripts\python.exe main.py "华北地区销售额"
.\.venv\Scripts\python.exe main.py "各地区订单量"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 接入 DeepSeek（可选）

1. 复制 `.env.example` 为 `.env`，在本机填写 `DEEPSEEK_API_KEY`；密钥绝不提交到 Git。
2. 使用下面的命令。模型生成的 SQL 仍必须通过白名单与只读校验，才会执行。

```powershell
.\.venv\Scripts\python.exe main.py "华北地区销售额" --model deepseek
```

## 目录说明

```text
src/shopkeeper_agent/
  catalog.py       # 指标、维度等元数据，以及问题的召回结果
  generators.py    # 规则生成器与 DeepSeek 生成器
  safety.py        # SQL 只读、单语句、表白名单校验
  database.py      # SQLite 样例数仓与执行器
  service.py       # 把整条工作流串起来
tests/             # 可离线运行的单元测试
main.py            # 命令行入口
```

## 当前边界

- 样例元数据和订单数据是教学数据，不是生产数据。
- 规则生成器仅覆盖已声明的指标和维度；未知问题会明确报错，而非猜测 SQL。
- 这不是“任意 SQL 执行器”：安全层只允许访问 `orders` 表的 `SELECT` 语句。

