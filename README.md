# StockFilter V1

一个使用逐项硬条件筛选美股和 A 股的本地研究工具。所有启用条件必须全部通过；系统不计算分数或权重。

## 启动

```bash
uv sync --extra dev
uv run streamlit run app.py
```

首次启动默认选择“免费在线数据”。建议先把“本次在线同步上限”设为 5 验证网络，成功后改为 0 进行全部候选同步。分别点击“同步美股数据 / 同步 A 股数据”，再点击对应的筛选按钮。

免费在线数据使用：

- 美股：Yahoo 公开行情筛选和历史行情 + SEC EDGAR 官方财报；
- A 股：Yahoo 公开行情筛选和历史行情 + 新浪公开财务报表（通过 AKShare）；
- 原始结果缓存到 `data/cache/`，重复同步会更快；
- Yahoo/yfinance 数据仅限个人研究使用，免费网页接口可能限流或调整字段。

建议在 `.env` 或启动终端中设置 SEC 要求的身份标识：

```bash
export STOCKFILTER_SEC_USER_AGENT="你的名字 your-email@example.com"
```

免费数据的历史 PE 使用“报告披露后可得的年度 EPS”构造月度序列，不等同于商业数据商清洗后的逐日 TTM PE；页面会显示这一口径提示。

命令行验证：

```bash
uv run stockfilter sync --market US --source free_online --limit 5
uv run stockfilter screen --market US
uv run pytest
```

## 导入自己的数据

1. 在 UI 中下载规范化 CSV 模板；
2. 填写后保存为 `data/imports/us_metrics.csv` 或 `data/imports/cn_metrics.csv`；
3. 将数据源切换为“规范化 CSV”，点击同步和筛选。

每个数值必须是截至 `as_of_date` 当时已知的数据。空值不会按 0 处理，任何启用条件缺少数据都会失败。详细指标定义见 [系统策划方案](docs/system-planning.md)。

## 当前边界

- V1 已完成筛选引擎、免费在线同步、参数配置、DuckDB 持久化、运行记录、CSV 导入、演示数据、双市场 UI、导出和自动测试。
- 在线同步先用市值、PE、交易所和行业做服务端预筛，再为候选下载十年行情及财报。页面中的“基础候选数”是进入本地详细计算的数量，不是最终入选数，也不是交易所全部证券数。结果表默认只显示通过全部硬条件的股票；需要排查阈值时，可开启“显示未通过股票（阈值诊断）”。
- 本项目用于研究候选池，不构成投资建议，也不执行交易。
