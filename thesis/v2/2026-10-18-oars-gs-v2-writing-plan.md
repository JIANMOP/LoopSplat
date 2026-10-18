# OARS-GS 第二版期刊论文实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保持第一版不变的前提下，完成以 OARS-GS 为核心、证据可追溯、参考文献全部正式发表且引用顺序一致的中英文期刊论文第二版。

**Architecture:** 第二版采用“证据层、文献层、英文母稿、中文等义稿、验证与审阅输出”五层结构。三个只读研究代理并行核查代码公式、实验结果和正式发表文献，主代理负责统一科研问题、OARS-GS 方法叙事和全文措辞，避免多人直接改同一稿件造成术语与结论漂移。现有实验 CSV、表格和图作为共享只读证据，第二版在 `thesis/v2/` 内维护独立稿件、参考文献和审计记录。

**Tech Stack:** Markdown、BibTeX、Python 3、pytest、Pandoc、CSL、SVG、DOCX、Crossref/出版社/CVF/IEEE/ACM/Springer 官方页面、Git。

**Spec:** `thesis/v2/2026-10-18-oars-gs-v2-writing-design.md`

## Global Constraints

- 第一版 `thesis/manuscript-*.md`、`thesis/references.bib`、审阅文件、共享实验 CSV、表格和图保持不变。
- 论文题名固定为 “OARS-GS: Observation-Adaptive Keyframing, Inertial Rotation Regularization, and Coarse-to-Fine RGB-D Mapping for Gaussian SLAM” 及其已批准中文题名。
- LoopSplat 是实现和实验基线；论文按问题发现、原因分析、方法推导、实验验证和边界讨论组织，不以模块融合为创新主线。
- IMU 方法仅描述为陀螺仪预积分导出的弱旋转正则；平移惯性权重为零，不宣称紧耦合视觉惯性状态估计。
- 金字塔方法仅改变 RGB-D 监督分辨率与使用次数调度，不宣称层次化 Gaussian 或 LOD 地图。
- FMDataset 不报告 ATE/RPE；渲染指标是各方法估计轨迹下的固定已观测视角评测；峰值显存是 PyTorch peak allocated memory。
- 当前结果是同一固定随机种子的逐场景结果与跨场景宏平均，不写成多种子平均、方差或统计显著性。
- 保留 TUM A3 失败结果和场景依赖退化，不宣称全面提升、普遍鲁棒、SOTA 或首次使用 IMU。
- Azure Kinect 只作工程示例，不承担定量消融结论。
- 每条参考文献必须是真实的正式期刊、正式会议论文集或正式出版物；未正式发表的 arXiv 预印本不得引用。
- 每条入稿文献优先要求 DOI；若确无 DOI，必须有正式出版编号、稳定官方页面和审计说明，且需在最终审查中单独批准。
- 英文稿与中文稿必须具有相同的 citation key 集合、相同的首次出现顺序；参考文献表按该顺序排列且不得有未引用条目。
- Humanizer 只能修改行文，不得更改事实、数字、公式、引用、变量、图表编号和结论强度。
- 开题报告不修改、不提交，也不作为投稿附件。

## Review Focus

- 文献正式性：存在同名预印本和正式版本时必须引用正式版本，禁止将 arXiv 元数据伪装成会议或期刊记录。
- 引用序列：英文和中文首次引用序列必须逐项相等，BibTeX 条目顺序必须与正文首次出现顺序相等。
- 创新归属：继承的子图、回环、GS-GS 配准和位姿图后端不能写成 OARS-GS 新贡献。
- 结果解释：TUM A3、不同硬件运行时间、FMDataset 无轨迹真值和 observed-view 协议必须得到正确限制。
- 多代理一致性：研究代理只提交证据报告，不直接分别润色正文；主代理统一术语、公式和结论。

---

### Task 1: 建立第二版严格验证接口

**Files:**
- Modify: `thesis/scripts/validate_bibliography.py`
- Modify: `thesis/scripts/validate_manuscripts.py`
- Modify: `thesis/tests/test_validate_bibliography.py`
- Modify: `thesis/tests/test_validate_manuscripts.py`

**Interfaces:**
- Consumes: 现有 BibTeX 解析器、Markdown citation key 解析规则和第一版测试。
- Produces: `citation_sequence(path: Path) -> list[str]`；`validate_bibliography(..., published_only: bool = False, require_doi: bool = False, require_order: bool = False) -> list[str]`；`validate_manuscripts(..., require_citation_order: bool = False) -> list[str]`。默认参数保持第一版行为不变，第二版命令显式启用严格模式。

- [ ] **Step 1: 编写严格参考文献失败测试**

新增测试并断言：严格模式拒绝 `@misc` arXiv、`eprint`/`archiveprefix=arXiv`、缺失 DOI、带 `https://doi.org/` 前缀的 DOI、未被引用条目，以及 BibTeX 条目顺序与正文首次引用顺序不同。

- [ ] **Step 2: 编写双语首次引用顺序失败测试**

新增 `test_reports_bilingual_first_citation_order_mismatch`：中英文 citation key 集合相同但首次出现顺序不同，启用严格模式时必须失败并报告第一个分歧 key。

- [ ] **Step 3: 运行新测试并确认失败**

Run:

```bash
python -m pytest thesis/tests/test_validate_bibliography.py thesis/tests/test_validate_manuscripts.py -q
```

Expected: 新增测试因严格参数和序列检查尚不存在而 FAIL，原有测试仍可收集。

- [ ] **Step 4: 最小实现严格模式**

保留现有默认行为。严格模式按正文首次出现去重得到 citation sequence，拒绝 arXiv 标记，检查 DOI，比较正文序列与 BibTeX 条目序列，并在双语验证中比较英文与中文首次引用序列。

- [ ] **Step 5: 验证第一版兼容性和第二版严格行为**

Run:

```bash
python -m pytest thesis/tests/test_validate_bibliography.py thesis/tests/test_validate_manuscripts.py -q
python thesis/scripts/validate_bibliography.py thesis/references.bib thesis/manuscript-en.md thesis/manuscript-zh.md
```

Expected: 全部测试 PASS；未启用严格模式的第一版验证仍 PASS。

- [ ] **Step 6: Commit**

```bash
git add thesis/scripts/validate_bibliography.py thesis/scripts/validate_manuscripts.py thesis/tests/test_validate_bibliography.py thesis/tests/test_validate_manuscripts.py
git commit -m "test: add strict OARS-GS manuscript validation"
```

### Task 2: 并行建立方法、实验和正式文献证据包

**Files:**
- Create: `thesis/v2/internal/method-code-audit.md`
- Create: `thesis/v2/internal/experiment-evidence.md`
- Create: `thesis/v2/internal/reference-audit.md`
- Create: `thesis/v2/internal/claim-evidence-matrix.md`

**Interfaces:**
- Consumes: 当前代码、`paper/` 正式论文、第一版审计、共享实验 CSV/表格/图，以及官方网络出版记录。
- Produces: 三份互不重叠的只读研究报告和一份由主代理整合的 claim-evidence matrix。

- [ ] **Step 1: 启动三个只读研究代理**

代理 A 只核查方法代码与公式，逐项记录 IMU 预积分、SO(3) 残差、Huber、GI-KF 判定、临时支撑更新、金字塔调度和损失函数的文件、符号及真实执行顺序。代理 B 只分析 70 条实验记录，给出可复算的宏平均变化、代表场景、失败案例和禁止外推项。代理 C 使用官方来源核查候选参考文献的正式发表状态、DOI、作者、题名、会刊、年份、卷期页码或文章号。

- [ ] **Step 2: 主代理复核代理 A 的代码证据**

对每个公式候选回读对应实现。`method-code-audit.md` 每行必须包含 `method_item | code_location | executed_behavior | equation_candidate | prohibited_claim`，不得根据配置名推断未执行逻辑。

- [ ] **Step 3: 主代理复核代理 B 的实验数字**

从 `thesis/tables/result-source.csv` 和 `dataset-summary.csv` 独立重算摘要拟用百分比。`experiment-evidence.md` 记录公式、分母、场景范围和硬件兼容条件，并单列 TUM A3。

- [ ] **Step 4: 主代理复核代理 C 的出版信息**

每条记录至少包含 `citation_key | formal_title | authors | venue | year | DOI | official_url | publication_status | intended_claim`。任何只有 arXiv 的候选条目标记为 `REJECTED`，不得进入后续 BibTeX。

- [ ] **Step 5: 建立 claim-evidence matrix**

使用列：

```text
claim_id | section | allowed_claim_en | allowed_claim_zh | code_or_result_source | literature_source | numeric_derivation | prohibited_extension
```

覆盖继承后端、三项核心方法、临时支撑机制、三个数据集协议、资源指标和失败边界。

- [ ] **Step 6: 检查证据包无占位符**

Run:

```bash
rg -n "TODO|TBD|PLACEHOLDER|待核验|可能是|大概" thesis/v2/internal
```

Expected: 无匹配；被拒绝文献有明确原因和正式状态记录。

- [ ] **Step 7: Commit**

```bash
git add thesis/v2/internal
git commit -m "docs: audit OARS-GS paper evidence"
```

### Task 3: 建立仅含正式出版物的第二版参考文献库

**Files:**
- Create: `thesis/v2/references.bib`
- Modify: `thesis/v2/internal/reference-audit.md`

**Interfaces:**
- Consumes: Task 2 已核验且 publication status 为正式发表的记录。
- Produces: 仅含拟引用正式出版物的 BibTeX 候选库；最终条目顺序在 Task 6 按英文首次引用序列重排。

- [ ] **Step 1: 确定必要文献范围**

覆盖 Gaussian splatting、dense/RGB-D SLAM、Gaussian SLAM、视觉惯性估计、关键帧与观测管理、coarse-to-fine/多尺度优化、数据集与评测指标。优先使用原始方法论文和数据集论文，不用综述替代关键技术来源。

- [ ] **Step 2: 从官方元数据生成 BibTeX**

每条保留 `author`、`title`、正式 `booktitle`/`journal`、`year`、`pages` 或 `articleno`、纯 DOI 值和官方 URL。删除 `eprint`、`archiveprefix`、预印本 URL 和“submitted”字段。

- [ ] **Step 3: 双重核验核心论文**

LoopSplat、SplaTAM、Photo-SLAM、3DGS、ORB-SLAM3 及直接支撑三个创新点的文献必须由 DOI 注册信息和出版社/正式论文集页面双重核验。题名、作者顺序和年份有冲突时以正式出版记录为准，并在 audit 中说明。

- [ ] **Step 4: 运行严格书目结构验证**

Run:

```bash
python thesis/scripts/validate_bibliography.py thesis/v2/references.bib --published-only --require-doi
rg -ni "arxiv|eprint|archiveprefix|submitted|preprint" thesis/v2/references.bib
```

Expected: 验证 exit 0；`rg` 无匹配。

- [ ] **Step 5: Commit**

```bash
git add thesis/v2/references.bib thesis/v2/internal/reference-audit.md
git commit -m "docs: add formally published OARS-GS references"
```

### Task 4: 撰写 OARS-GS 方法、公式与算法图

**Files:**
- Create: `thesis/v2/sections/method-en.md`
- Create: `thesis/v2/figures/oars-gs-overview.svg`
- Create: `thesis/v2/figures/observation-scheduling.svg`
- Modify: `thesis/v2/internal/claim-evidence-matrix.md`

**Interfaces:**
- Consumes: method-code audit、正式文献库和设计稿第 3 节结构。
- Produces: 统一英文方法章节、连续编号公式、伪代码和两张可编辑矢量图。

- [ ] **Step 1: 写 3.1 系统定义**

定义 RGB-D/IMU 输入、相机位姿、Gaussian 参数、渲染量、跟踪与建图目标。明确标注共享渲染器和继承的 LoopSplat 后端，避免扩展贡献边界。

- [ ] **Step 2: 写 3.2 惯性旋转正则**

从相邻帧陀螺仪中点积分推导旋转增量、坐标系转换、SO(3) 对数残差、尺度归一化和 Huber 代价。每个符号必须定义，并明确何时因时间区间无效而拒绝惯性项。

- [ ] **Step 3: 写 3.3 与 3.4 观测调度**

按代码顺序定义可见集合、新颖度、归一化基线、运动硬门控和间隔约束，再定义临时 tracking-support update 的触发、有限优化预算与生命周期。公式不得包含代码中不可达的评分项。

- [ ] **Step 4: 写 3.5 粗到细 RGB-D 建图**

定义 RGB 平滑下采样、有效深度加权下采样、内参缩放、关键帧使用计数和层级函数。说明该方法改变监督分辨率而不改变 Gaussian 表示层级。

- [ ] **Step 5: 写 3.6 在线算法并绘图**

伪代码按“跟踪、惯性正则、持久关键帧判断、临时支撑、建图层级、子图后端”给出真实执行顺序。两张 SVG 的节点名称与正文术语一致。

- [ ] **Step 6: 公式和创新归属审计**

Run:

```bash
rg -n "tightly coupled|hierarchical Gaussian|state-of-the-art|first.*IMU|module integration" thesis/v2/sections/method-en.md
rg -o '\\tag\{[0-9]+\}' thesis/v2/sections/method-en.md
```

Expected: 禁止主张无匹配；公式编号唯一且连续。

- [ ] **Step 7: Commit**

```bash
git add thesis/v2/sections/method-en.md thesis/v2/figures thesis/v2/internal/claim-evidence-matrix.md
git commit -m "docs: draft OARS-GS method and algorithms"
```

### Task 5: 撰写相关工作与实验论证

**Files:**
- Create: `thesis/v2/sections/related-work-en.md`
- Create: `thesis/v2/sections/experiments-en.md`
- Create: `thesis/v2/sections/results-en.md`
- Modify: `thesis/v2/internal/claim-evidence-matrix.md`

**Interfaces:**
- Consumes: 正式文献 audit、共享表格/图、experiment evidence 和 OARS-GS 方法术语。
- Produces: 围绕研究缺口组织的相关工作、可复现实验协议和分问题结果分析。

- [ ] **Step 1: 按技术问题写 Related Work**

依次撰写 Dense and Gaussian SLAM、Visual-inertial tracking、Keyframe and observation management、Coarse-to-fine optimization。每段先归纳已有技术选择，再指出 OARS-GS 处理的具体空白，避免逐篇论文罗列和无文献的优先权表述。

- [ ] **Step 2: 写 Experimental setup**

报告 TUM RGB-D、Replica、FMDataset 的场景与策略矩阵、固定种子协议、硬件、指标、轨迹对齐、observed-view 渲染和 peak allocated memory 定义。FMDataset 轨迹列必须保持缺失，不把 Azure 写成消融。

- [ ] **Step 3: 写 Tracking accuracy**

分别分析 TUM 和 Replica。TUM 先报告正常场景，再单列 A3 failure case；不得用包含 A3 发散值的宏平均制造概括性提升或退化结论。

- [ ] **Step 4: 写 Rendering/depth 与 Resource efficiency**

所有百分比使用同场景 Baseline 为分母。跨 GPU 的绝对时间和显存只作环境说明；策略效率结论仅使用同场景兼容硬件配对。

- [ ] **Step 5: 写 Ablation 与 failure analysis**

IMU、观测自适应关键帧和粗到细监督分别承担单项创新证据，组合只验证兼容性。区分持久关键帧和 support updates，报告 Replica 的 budget-matched 行为及 TUM A3 边界。

- [ ] **Step 6: 逐句登记定量证据**

含 `%`、`dB`、`cm`、`GiB`、小时或小数指标的句子必须在 claim-evidence matrix 中登记精确 CSV 行和计算式。

- [ ] **Step 7: 验证引用和统计边界**

Run:

```bash
rg -n "standard deviation|statistically significant|novel-view|world-coordinate accuracy|标准差|统计显著" thesis/v2/sections
```

Expected: 无不可用统计或错误评测协议主张；必要边界仅以明确限制出现。

- [ ] **Step 8: Commit**

```bash
git add thesis/v2/sections thesis/v2/internal/claim-evidence-matrix.md
git commit -m "docs: draft OARS-GS literature and experiments"
```

### Task 6: 组装并定稿英文母稿与引用顺序

**Files:**
- Create: `thesis/v2/manuscript-en.md`
- Create: `thesis/v2/README.md`
- Modify: `thesis/v2/references.bib`
- Modify: `thesis/v2/internal/reference-audit.md`
- Modify: `thesis/v2/internal/claim-evidence-matrix.md`

**Interfaces:**
- Consumes: Tasks 2–5 的证据、方法、相关工作和实验章节。
- Produces: 结构完整的英文投稿母稿，以及按英文首次引用排序的最终 BibTeX。

- [ ] **Step 1: 写 Introduction**

按“在线 Gaussian SLAM 观测处理问题、原因、统一研究问题、OARS-GS 方法、实验回答、贡献”推进。贡献描述对应三个算法设计和一项实验验证，不以“we integrate three modules”起笔。

- [ ] **Step 2: 组装 Sections 2–4**

统一标题层级、符号、缩写、公式编号、图表编号和 citation key。正文结构固定为 Introduction、Related Work、OARS-GS、Experiments、Conclusion。

- [ ] **Step 3: 依据结果写 Abstract 和 Conclusion**

摘要在全文完成后撰写，只使用可从共享 CSV 重算的代表性结论。结论回答引言提出的问题并保留 A3、固定种子、observed-view 和 rotation-only IMU 限制。

- [ ] **Step 4: 按英文首次出现顺序重排 references.bib**

提取首次 citation sequence，删除未引用候选，按序重排 BibTeX。同步更新 audit 的首次引用位置，不增补仅用于“凑数量”的文献。

- [ ] **Step 5: 写第二版 README**

记录母稿、共享表格/图、内部审计、严格验证和审阅版生成命令。明确第一版、开题报告和 `internal/` 不属于第二版投稿附件。

- [ ] **Step 6: 运行英文严格验证**

Run:

```bash
python thesis/scripts/validate_bibliography.py thesis/v2/references.bib thesis/v2/manuscript-en.md --published-only --require-doi --require-order
wc -w thesis/v2/manuscript-en.md
```

Expected: 验证 exit 0；英文稿约 5,500–7,500 words，无未引用条目和未解析 citation key。

- [ ] **Step 7: Commit**

```bash
git add thesis/v2/manuscript-en.md thesis/v2/references.bib thesis/v2/README.md thesis/v2/internal
git commit -m "docs: assemble OARS-GS English manuscript"
```

### Task 7: 形成中文等义稿并验证双语一致性

**Files:**
- Create: `thesis/v2/manuscript-zh.md`
- Create: `thesis/v2/sections/method-zh.md`
- Create: `thesis/v2/sections/related-work-zh.md`
- Create: `thesis/v2/sections/experiments-zh.md`
- Create: `thesis/v2/sections/results-zh.md`

**Interfaces:**
- Consumes: 已通过严格验证的英文母稿和最终 BibTeX。
- Produces: 与英文结构、证据和引用顺序逐项一致的中文稿。

- [ ] **Step 1: 按段落功能进行中文重构**

按英文段落的技术判断重写中文，不逐句直译。术语固定为观测自适应关键帧、临时跟踪支撑更新、惯性旋转正则、按使用次数推进的粗到细 RGB-D 建图。

- [ ] **Step 2: 保持公式与证据 token 一致**

复制并核对公式、变量、公式编号、citation key、数字、单位、表格链接和图片链接。中文首次引用顺序必须与英文完全一致。

- [ ] **Step 3: 运行双语严格验证**

Run:

```bash
python thesis/scripts/validate_bibliography.py thesis/v2/references.bib thesis/v2/manuscript-en.md thesis/v2/manuscript-zh.md --published-only --require-doi --require-order
python thesis/scripts/validate_manuscripts.py thesis/v2/manuscript-en.md thesis/v2/manuscript-zh.md --bib thesis/v2/references.bib --root thesis/v2 --require-citation-order
```

Expected: 两个命令 exit 0；citation key、首次出现顺序、公式编号、指标数字和图表编号一致。

- [ ] **Step 4: Commit**

```bash
git add thesis/v2/manuscript-zh.md thesis/v2/sections
git commit -m "docs: add OARS-GS Chinese manuscript"
```

### Task 8: 使用 Humanizer 做事实保持型语言复核

**Files:**
- Modify: `thesis/v2/manuscript-en.md`
- Modify: `thesis/v2/manuscript-zh.md`
- Create: `thesis/v2/internal/manuscript-audit.md`

**Interfaces:**
- Consumes: 已通过技术和引用验证的双语母稿。
- Produces: 去除 AI 写作痕迹且科学事实完全不变的最终母稿，以及事实 token 前后对比记录。

- [ ] **Step 1: 保存编辑前事实 token 快照**

提取 citation sequence、公式块与编号、数字、百分比、单位、数据集名、策略名、图表编号和本地链接，记录到临时文件；`manuscript-audit.md` 记录快照哈希与数量。

- [ ] **Step 2: Humanizer file mode 复核英文稿**

删除 staged opener、not-X-but-Y、one-line closer、机械三段式、过量破折号、宣传词和空泛意义拔高。技术限定和真正的对比关系保留。

- [ ] **Step 3: Humanizer file mode 复核中文稿**

删除“值得注意的是”“不仅……而且……”等模板化铺垫、空泛“综上所述”、机械排比和宣传性判断。使用自然中文科技论文句法，不改变证据范围。

- [ ] **Step 4: 对比编辑前后事实 token**

任何 citation、公式、数字、单位、链接或编号变化均视为失败，恢复并重写相邻句子。仅允许文字表述发生变化。

- [ ] **Step 5: 扫描残留模式并重跑严格验证**

Run:

```bash
rg -n "—|–|not only|not merely|it is worth noting|delve|pivotal|crucial|landscape|值得注意的是|不仅.*而且|具有重要意义|综上所述" thesis/v2/manuscript-en.md thesis/v2/manuscript-zh.md
python thesis/scripts/validate_bibliography.py thesis/v2/references.bib thesis/v2/manuscript-en.md thesis/v2/manuscript-zh.md --published-only --require-doi --require-order
python thesis/scripts/validate_manuscripts.py thesis/v2/manuscript-en.md thesis/v2/manuscript-zh.md --bib thesis/v2/references.bib --root thesis/v2 --require-citation-order
```

Expected: 无无理由残留；两个验证命令 exit 0。

- [ ] **Step 6: Commit**

```bash
git add thesis/v2/manuscript-en.md thesis/v2/manuscript-zh.md thesis/v2/internal/manuscript-audit.md
git commit -m "docs: humanize OARS-GS bilingual manuscripts"
```

### Task 9: 生成审阅版并完成独立终审

**Files:**
- Modify: `thesis/scripts/build_review_documents.py`
- Modify: `thesis/tests/test_build_review_documents.py`
- Create: `thesis/v2/review/manuscript-en-review.docx`
- Create: `thesis/v2/review/manuscript-en-review.pdf`
- Create: `thesis/v2/review/manuscript-zh-review.docx`
- Create: `thesis/v2/review/manuscript-zh-review.pdf`
- Modify: `thesis/v2/README.md`

**Interfaces:**
- Consumes: 双语母稿、v2 bibliography、共享结果表、v2 图和 Pandoc。
- Produces: 可阅读审阅文件、零错误验证结果和独立审稿记录。

- [ ] **Step 1: 编写 v2 审阅导出失败测试**

测试构建器能接收显式 manuscript 目录、bibliography 路径和共享 table root，能展开 `../tables/*.md`，且不会修改第一版 review 文件。

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
python -m pytest thesis/tests/test_build_review_documents.py -q
```

Expected: 新增 v2 参数测试 FAIL，原有第一版测试仍 PASS。

- [ ] **Step 3: 最小扩展审阅构建器**

保留第一版默认值，增加显式输入路径和 table root 参数。输出目录固定为 `thesis/v2/review/`，不得覆盖 `thesis/review/`。

- [ ] **Step 4: 生成 DOCX 和 PDF**

使用 v2 bibliography 和数字引用样式生成双语 DOCX；通过现有可用的办公转换链生成 PDF。检查公式、宽表、图题、分页、参考文献编号和字体。

- [ ] **Step 5: 运行全部自动验证**

Run:

```bash
python -m pytest thesis/tests -q
python thesis/scripts/validate_bibliography.py thesis/v2/references.bib thesis/v2/manuscript-en.md thesis/v2/manuscript-zh.md --published-only --require-doi --require-order
python thesis/scripts/validate_manuscripts.py thesis/v2/manuscript-en.md thesis/v2/manuscript-zh.md --bib thesis/v2/references.bib --root thesis/v2 --require-citation-order
git diff --check
```

Expected: 全部命令 exit 0。

- [ ] **Step 6: 请求独立全文审查**

使用新的 reviewer 代理只读检查六类问题：科研问题到结论的闭环、算法创新归属、公式与代码一致性、全部数字与实验一致性、正式文献与 DOI、双语引用顺序。主代理逐项处理 critical 和 important findings。

- [ ] **Step 7: 确认第一版未变化**

以第二版设计提交 `81ee80d` 为基准检查 `thesis/manuscript-en.md`、`thesis/manuscript-zh.md`、`thesis/references.bib` 和 `thesis/review/` 无差异。

- [ ] **Step 8: 更新 README 并提交**

```bash
git add thesis/scripts/build_review_documents.py thesis/tests/test_build_review_documents.py thesis/v2
git commit -m "docs: finalize OARS-GS journal paper v2"
```
