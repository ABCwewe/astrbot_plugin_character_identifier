# AGENTS.md — astrbot_plugin_character_identifier

> 面向 AI 编码代理与开发者的项目说明。动手前通读本文；与本文冲突的实现需先改文档再改代码。
> 分类器契约以模型仓库 `ABCwewe/wuwa_playable_character_identifier` 的 README 与 `examples/predict.py` 为准；§13 列出的待定项由作者后续提供，代理不得自行编造（阈值、样图集等）。

---

## 1. 项目目标

AstrBot 插件：在 LLM 请求发出前，对消息里的图片做 **二次元角色检测 → 角色识别 → 标注**，并把结果注入 LLM 上下文：

1. 检测：ONNX Runtime 运行 `deepghs/anime_person_detection` 的 `person_detect_v1.1_n`（YOLOv8-nano，单类 `person`）。
2. 识别：ONNX Runtime 运行作者自训练的 `ABCwewe/wuwa_playable_character_identifier`（鸣潮 57 个可操控角色，MobileNetV4，双输出 embedding+logits，配套原型库与三道拒识闸），对每个检测框裁剪后分类。**范围说明**：分类器只认这 57 个角色，其他作品/角色一律判 `unknown`；检测器检出的是“任意二次元人物”，因此图中出现大量 `unknown` 属正常。
3. 标注：OpenCV 画彩色框 + 编号（可选角色名），让 LLM 在图里就能区分人物。
4. 注入：用标注图替换 `req.image_urls` 中的原图，并追加文字说明（角色名 ↔ 框颜色/编号）。

## 2. 硬性约束（不可违反）

| 约束 | 说明 |
|---|---|
| 纯 CPU | 只用 `CPUExecutionProvider`；禁止引入 torch / ultralytics / onnxruntime-gpu |
| 轻量依赖 | 仅 `onnxruntime`、`opencv-python-headless`、`numpy`；网络用 AstrBot 已带的 `aiohttp`/`httpx`；禁止 `requests`（官方开发原则） |
| 不自带模型 | 仓库与发布包不含任何 `.onnx`；全部运行时从 HuggingFace 下载 |
| 低内存 | 空闲可卸载；加载后单会话峰值有预算（见 §10） |
| 不阻塞事件循环 | 推理/解码/绘制全部放线程池，主协程只 `await` |
| 失败即放行 | 任何异常、超时、模型未就绪 → 原样放行 LLM 请求，绝不让用户消息丢失 |
| 数据不落插件目录 | 模型、缓存写 `data/plugin_data/{plugin_name}/`，否则更新/重装会被覆盖 |
| 不用 `@register` | 按最新文档：`main.py` 内继承 `Star`，元数据来自 `metadata.yaml` |

## 3. AstrBot 接口要点（来自官方文档，实现时以此为准）

- **入口**：`main.py` 中 `class CharAnnotatePlugin(Star)`；`__init__(self, context: Context, config: AstrBotConfig)`，存在 `_conf_schema.json` 时配置会自动注入；`async def terminate(self)` 在卸载/停用时调用，必须在此释放模型、取消后台任务。
- **配置**：`_conf_schema.json` 支持 `string/text/int/float/bool/object/list/dict/template_list/file`，字段 `description/hint/default/options/secret/invisible/items`；`AstrBotConfig` 是 dict 子类，可 `save_config()`。Schema 升级时 AstrBot 自动补默认值、删除多余项。
- **钩子**：`@filter.on_llm_request()` 签名 `(self, event: AstrMessageEvent, req: ProviderRequest)`，**三个参数**；钩子内**不能 `yield` 发消息**，需要发送时用 `await event.send(...)`。钩子不能与 `command/event_message_type` 等过滤器叠加。
- **ProviderRequest**：`prompt`、`system_prompt`、`image_urls`（base64 / 网络链接 / 本地路径均可）、`extra_user_content_parts`（`TextPart`，`>=4.24.0` 支持 `.mark_as_temp()` 仅本轮生效）。
- **提示词缓存**：官方警告每轮变化的内容**不要**追加到 `system_prompt`（破坏服务端缓存、成本暴增）。本插件只用 `extra_user_content_parts` 追加说明。
- **存储**：大文件放 `Path(get_astrbot_data_path()) / "plugin_data" / self.name`；小状态可用 `put_kv_data/get_kv_data`（`>=4.9.2`）。
- **依赖**：第三方库必须写进插件目录 `requirements.txt`。
- **版本声明**：`metadata.yaml` 设 `astrbot_version: ">=4.24,<5"`（`mark_as_temp` 需要 4.24.0）。
- **指令**：指令名不能含空格；多子命令用 `command_group`。管理类指令加 `@filter.permission_type(filter.PermissionType.ADMIN)`。
- **调试**：WebUI「插件」页点刷新图标即热重载；加载失败的插件在「加载失败插件」列表重载。
- **已知坑**：近期核心版本对 provider 的 `modalities` 为空/未配置时处理有变动（空列表曾导致 `image_urls` 被静默清空，已有修复 PR）。若用户反馈“图片没传到模型”，先排查所选聊天模型的模态配置，而不是本插件。

## 4. 目录结构

```
astrbot_plugin_char_annotate/
├── AGENTS.md
├── README.md
├── metadata.yaml
├── _conf_schema.json            # WebUI 配置页定义（见 §8）
├── requirements.txt             # onnxruntime / opencv-python-headless / numpy
├── logo.png                     # 256x256，可选
├── main.py                      # 仅做：钩子/指令注册 + 组装各模块，<150 行
├── core/
│   ├── config.py                # AstrBotConfig → 强类型 Settings（dataclass），含校验与默认值
│   ├── registry.py              # 读取 registry.json，模型条目 → ModelSpec
│   ├── downloader.py            # HF 异步下载、断点/原子落盘、镜像、校验
│   ├── runtime.py               # ORT 会话工厂 + 生命周期管理（常驻/空闲卸载）
│   ├── detector.py              # YOLOv8 前/后处理 + 推理
│   ├── classifier.py            # 裁剪 → letterbox 预处理 → embedding+logits → 原型余弦 + 三道闸拒识
│   ├── annotator.py             # 配色、画框、标签、缩放、编码
│   ├── image_io.py              # 从 req/event 取图、解码、体积/尺寸限制
│   ├── pipeline.py              # 串联：解码→检测→识别→标注→文本，含结果缓存
│   └── injector.py              # 改写 ProviderRequest（替换图 + 追加文本）
├── data/
│   └── registry.json            # 可选模型列表（见 §6.2 初始条目与中文名映射文件）
├── scripts/
│   ├── bench.py                 # 基准：延迟、峰值 RSS、加载/卸载耗时
│   └── parity_check.py          # OpenCV 预处理 vs 参考实现（Pillow）一致性验证
└── tests/                       # pytest，不依赖真实模型（用假 ORT 会话与合成图）
```

约定：`core/` 内**不 import astrbot**（便于脱离 AstrBot 单测）；与 AstrBot 耦合只在 `main.py` 与 `injector.py`。

## 5. 数据流

```
on_llm_request(event, req)
  ├─ 开关/名单/会话过滤        → 不命中：return
  ├─ image_io.collect(req)     → 无图或超限：return
  ├─ pipeline.run(images)      （线程池，受信号量与超时约束）
  │    ├─ cache 命中？ → 直接返回上次结果
  │    ├─ runtime.acquire()    惰性加载检测/分类会话
  │    ├─ detector.detect()    → boxes
  │    ├─ 无框 → 返回 None（原图原样放行，不加任何文字）
  │    ├─ classifier.predict(crops) → labels + scores
  │    ├─ 全部 unknown → 返回 None（跳过注入，且不渲染标注图）
  │    └─ annotator.render()   → 标注图 jpg 路径 + 结构化结果（至少 1 个已识别角色才会执行）
  └─ injector.apply(req, results)
       ├─ req.image_urls[i] = 标注图本地路径
       └─ req.extra_user_content_parts.append(TextPart(说明文本))
```

- 多图：按 `max_images`（默认 3）取前 N 张，其余保持原样。
- 同一张图重复出现（引用/转发）：以 `sha1(图像字节) + 模型版本 + 关键参数` 作键走 LRU 结果缓存。
- 无框、**所有框均为 `unknown`**、模型未就绪、超时、异常：**不改动 req**（不替换图、不加文字）。
- 跳过注入的判定在分类之后、渲染之前：全 `unknown` 时不画框、不编码、不写缓存图，省掉这部分开销；该“空结果”仍写入 LRU 结果缓存，同一张图再次出现时直接放行。
- 只要有 **≥1 个**已识别角色，就照常标注并注入：此时 `unknown` 框也画（灰色）并在文字里列出，保持编号与图一致。

## 6. 模块规格

### 6.1 `config.py`
- 把 `AstrBotConfig` 一次性转为 `Settings` dataclass（frozen），其余模块只读 `Settings`，不直接碰 dict。
- 校验并钳制：`idle_timeout_sec ≥ 30`、`threads ≥ 1`、`max_images ≥ 1` 等；非法值回落默认并 `logger.warning`。
- 配置改动后重建 Settings 并通知 `runtime` 重新加载（模型/线程变化需重建会话）。

### 6.2 `registry.py` + `downloader.py`（模型获取）
- `data/registry.json` 每条：`key`、`kind`(`detector`|`classifier`)、`repo_id`、`ref`（`main`/tag/commit；见下）、`files`（`model` 必填；分类器另含配套 `prototypes`）、`preprocess`（预处理方案 id，见 §6.4.1）、可选 `sha256`、`display_names`（类名→中文名映射文件，位于仓库根；见 §6.4.1，缺失不影响就绪）、`scope_note`。
- 初始条目（已核对仓库实际路径）：

| key | kind | 文件（相对仓库根） | 说明 |
|---|---|---|---|
| `person_detect_v1.1_n` | detector | `person_detect_v1.1_n/model.onnx` | deepghs/anime_person_detection，MIT；单类 `person`，F1=0.85 对应阈值 0.327 |
| `halfbody_detect_v1.0_n` | detector | `halfbody_detect_v1.0_n/model.onnx` | deepghs/anime_halfbody_detection，**OpenRAIL**；单类 `halfbody`，F1=0.94 对应阈值 0.512（`threshold.json`）；YOLOv8n 同规格（3.01M 参数），检测上半身区域，适合半身/胸像输入 |
| `wuwa_mnv4l_448_int8`（默认） | classifier | `latest/mnv4l_448/wuwa_playable_character_latest_mnv4l_448_int8.onnx` + `…/prototypes_int8.npz` | 精度优先，33.1 MB，≈100 ms/图（1 线程） |
| `wuwa_mnv4s_384_fp32` | classifier | `latest/mnv4s_384/wuwa_playable_character_latest_mnv4s_384.onnx` + `…/prototypes.npz` | 速度/体积优先，11.3 MB，≈11 ms/图（1 线程） |

  两个分类器条目的 `display_names` 都指向仓库根的 `class_names_zh.yaml`（2.38 kB，所有变体共用）。
  不收录 fp16 变体（x86 CPU 无原生 fp16，不提速）。`versions/3.7/...` 为同结构的版本存档，可作固定版本条目（如 `wuwa_3.7_mnv4l_448_int8`）。
- **onnx 与 npz 必须成对下载、成对使用**：原型库内含该精度专用的拒识阈值 `tau`/`p_min`，跨精度/跨版本混用会让拒识失效。配对规则同仓库约定：`*_int8.onnx → prototypes_int8.npz`，无后缀 → `prototypes.npz`。两文件都下载并校验成功后才算“就绪”，任一失败整对丢弃。
- `latest/` 下文件路径永远不变、更新时同名覆盖，所以 `ref=main` 不可复现。处理方式：下载时先请求 `{endpoint}/api/models/{repo_id}/revision/main` 取得 commit sha，再用该 sha 拼 `resolve` URL；落盘目录以 sha 命名，保证一对文件来自同一提交。不在请求热路径上检查更新；仅在 `/charann download` 或配置 `check_update=true` 时于后台检查。需要稳定行为的用户应选 `versions/` 固定版本条目。
- 下载 URL：`{endpoint}/{repo_id}/resolve/{sha或tag}/{path}`，`endpoint` 默认 `https://huggingface.co`，可配置镜像（镜像可能不支持 `api/models` 接口，此时回落到直接使用 `ref`）；私有库用 `hf_token`（`secret: true`，不得写日志）。
- 落盘：`data/plugin_data/{name}/models/{repo_id替换/为--}/{sha或tag}/…`；先写 `*.part`，校验后 `os.replace` 原子改名；已存在且校验通过则跳过。
- 并发：每个 `key` 一把 `asyncio.Lock`，避免重复下载；失败指数退避重试 3 次。
- **不得在 `__init__` 或钩子热路径同步下载。** 首次需要而模型未就绪时：后台任务启动下载，本次请求放行；下载完成后日志提示，下次生效。支持 `offline_mode`（仅用本地文件，缺失即放行）。
- 管理指令 `/charann download` 可预下载；`/charann status` 报告各模型就绪状态、占用磁盘。

### 6.3 `runtime.py`（会话与生命周期）
- **惰性导入**：`onnxruntime`、`cv2` 在首次加载模型时才 import，插件空闲时不占其内存。
- 会话工厂统一 `SessionOptions`：
  - `intra_op_num_threads = cfg.threads`（默认 2，避免与 Bot 其他任务抢核）、`inter_op_num_threads = 1`、`execution_mode = ORT_SEQUENTIAL`
  - `graph_optimization_level = ORT_ENABLE_ALL`
  - `enable_cpu_mem_arena = False`、`enable_mem_pattern = False`（以少许速度换更低且可回收的内存；`low_memory` 配置关闭时才开启）
  - `add_session_config_entry("session.intra_op.allow_spinning", "0")`：避免空闲线程忙等耗 CPU
  - `providers=["CPUExecutionProvider"]`
- 生命周期：
  - `resident=True`：加载后常驻，不启动看门狗。
  - `resident=False`：每次 `acquire()` 刷新 `last_used`；后台 `asyncio` 看门狗每 `min(idle_timeout/4, 30s)` 检查，空闲超 `idle_timeout_sec` 则卸载（`del session` + `gc.collect()`），下次使用再加载。
  - 引用计数保护：存在在途推理时**禁止**卸载；卸载与加载用同一把锁串行化。
  - 检测器与分类器**各自独立**计时与卸载（分类器通常更大）。
  - `terminate()`：取消看门狗、等待在途任务结束（带超时）、释放全部会话。
- 推理并发：全局 `asyncio.Semaphore(max_concurrency)`（默认 1），推理在专用单线程/小线程池（`run_in_executor`）执行；ORT 推理释放 GIL，不会卡住事件循环。
- 不全局调用 `cv2.setNumThreads`（进程级，会影响其他插件）；预处理尺寸小，默认即可。

### 6.4 `detector.py`（`person_detect_v1.1_n` / `halfbody_detect_v1.0_n`）
- 模型：Ultralytics YOLOv8 导出 ONNX，单类（`person` / `halfbody`）。**加载时读取输入名/形状与输出形状并断言**，不要硬编码：
  - 输入 `NCHW float32`，RGB，`/255`；若输入为动态尺寸，使用配置 `det_imgsz`（默认 640，可降到 480/416 换速度）。
  - 常见输出 `[1, 4+nc, N]`（`nc=1` 时 `[1,5,N]`，需转置为 `[N,5]`，前 4 列为中心点 `xywh`）；若检测到 end2end `[1,N,6]` 形态则走另一分支。
- 预处理：`cv2` letterbox（保持比例、灰边 114、记录 `scale/pad`）→ `cv2.dnn.blobFromImage` 生成 blob（`swapRB=True`），避免多余拷贝。
- 后处理：置信度过滤 → `xywh→xyxy` → 去 letterbox 还原到原图坐标并裁到图内 → `cv2.dnn.NMSBoxes`（IoU 默认 0.5）→ 按置信度取前 `max_persons`（默认 8）。
- 默认 `det_conf=0.327`（已核实：HF `deepghs/anime_person_detection` 仓库 `person_detect_v1.1_n/threshold.json` 中 F1=0.85 对应阈值 0.327，2026-10-02 与作者确认采用）。**切换为 `halfbody_detect_v1.0_n` 时应同步把 `det_conf` 调到 0.512**（该模型 `threshold.json` 的 F1=0.94 最优值）；`det_conf` 是全局配置，不做按模型联动。
- 过滤过小框（短边 < `min_box_px`，默认 24）和极端长宽比，减少无意义识别。

### 6.4.1 `classifier.py`（`wuwa_playable_character_identifier`）
契约来自模型仓库 README 与 `examples/predict.py`（权威参考实现，`Predictor` 类）。本插件**不得改变其判定语义**，只把预处理从 Pillow 移植到 OpenCV。

**模型与原型库**
- ONNX 单输入 `NCHW float32`，边长从 `get_inputs()[0].shape[2]` 读取（448 或 384，不要写死）；两个输出：`embedding`（256 维，图内已 L2 归一化）、`logits`（58 类 = 57 角色 + `_negative`）。
- 配套 `prototypes*.npz`（`np.load(..., allow_pickle=False)`）键：`protos (57,256)`、`class_names (58)`、`proto_class_idx`、`tau`、`p_min`、`version`。**阈值全部来自 npz，不在插件配置里暴露**。
- 若模型输入 batch 维是固定整数 1，则逐个裁剪推理；若为动态，则同图多框合批一次推理（加载时判断）。

**预处理（逐位对齐训练口径，OpenCV 实现）**
1. 解码阶段（`image_io`）已完成：EXIF 转正（`cv2.imdecode` 默认应用 EXIF 方向）、透明像素合成白底、统一 RGB。
2. `letterbox_square(512)`：长边缩放到 512（保宽高比），居中放入 512×512；填充色 = 缩放后图像四边像素的均值色：`border = concat(arr[0,:], arr[-1,:], arr[:,0], arr[:,-1])`，`fill = clip(round(border.mean(axis=0)), 0, 255)`（角点按原实现重复计入，不要“优化”掉）。
3. 若模型输入尺寸 ≠ 512，再整体 resize 到模型输入尺寸（二次缩放，与训练一致，不要合并成一次）。
4. `/255` → ImageNet `mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]` → `float32 NCHW` 连续内存。
- **插值差异风险**：参考实现用 Pillow `BICUBIC`（缩小时带抗锯齿），OpenCV `INTER_CUBIC` 缩小不抗锯齿、核参数也不同。推荐：放大/等比小幅缩放用 `INTER_CUBIC`，明显缩小用 `INTER_AREA`。**必须用 `scripts/parity_check.py` 验证**（见 §11）；未通过前不得宣称与训练口径一致。
- 配置项 `preprocess_backend`：`opencv`（默认）/`pillow`（惰性导入 Pillow，逐位复刻参考实现，作为兜底；Pillow 为 AstrBot 自带依赖）。

**判定（三道闸，任一命中 → `unknown`，与参考实现一致）**
```
sims      = protos @ embedding            # 与 57 个角色原型的余弦
top1_cos  = sims.max()
head_conf = max(softmax(logits))
head_top  = class_names[argmax(logits)]
unknown   = top1_cos < tau or head_conf < p_min or head_top == "_negative"
pred      = None if unknown else head_top   # 非 unknown 时取分类头 argmax
```
- 对外结果结构 `CharResult(pred, unknown, head_conf, top1_cos, top5)`，`top5` 为按类去重的原型余弦。
- **不要把余弦当“置信度”展示**：正确识别时 `top1_cos` 也可能只有 0.2 左右（`tau` 约 0.12–0.14）。给 LLM 的置信度用 `head_conf`。
- `unknown=true` 时模型**不给角色名**；默认不向 LLM 暴露 top5 近邻名（余弦低于 `tau` 的近邻是噪声，会诱导 LLM 误报）。仅当配置 `show_top5_for_unknown=true`（调试用）才附带。
- **展示名（中文名映射）**：模型仓库根目录的 `class_names_zh.yaml`（键 = 模型输出的精确类名，如 `jinhsi_(wuthering_waves)`；值 = 中文名，如 `"今汐"`；含 `_negative: "未知角色"`，该项不会作为识别结果输出，忽略即可）。
  - 随分类器一起下载，**与 onnx/npz 解析到同一 commit sha**（见 §6.2）；它不分精度、不参与“成对”校验，下载失败只影响展示名，不影响模型就绪。
  - 解析：文件格式只有 `key: "value"` 行与 `#` 注释。优先 `yaml.safe_load`（`import yaml` 失败则回落到按行解析），不新增依赖；统一 UTF-8 读取。
  - 取名顺序：映射表命中 → 中文名；未命中 → 去掉 `_(wuthering_waves)` 后缀并把下划线换空格（如 `luuk herssen`），不自行翻译。
  - 映射表随模型版本更新（同一 commit 下载），插件不做额外校验；当前版本训练时有个别 tag 重复属已知问题，后续版本会修复，插件无需处理。
  - 展示名只在渲染/文本阶段套用；结果缓存与内部判定一律使用精确类名。映射文件的 sha 并入标注图缓存键（`label_mode=name` 时图上文字依赖它）。
  - 映射表中缺失的类名只记 `debug` 日志，不报错。

**适用范围与已知限制（来自模型卡，决定检测→识别的衔接策略）**
- 针对**单人全身立绘/插画**训练：多角色同框、Q 版、仅面部特写、极端裁剪、强风格化可能误判或被拒识。因此裁剪用检测框（全身框）外扩 `crop_pad`（默认 5%），不要对半身/头部小框强行识别：`min_box_px` 之下或长宽比异常的框直接标 `unknown`（或跳过识别）。
- 从**原图**裁剪（非检测用的缩放图），以保留分辨率。
- 负类拒识率约 86%–90%，陌生输入并非 100% 被拒；文本说明里的措辞要留余地（见 §6.7）。
- 仅覆盖训练时的 57 个角色，新角色需发新版本（`latest` 更新后用户需重新下载，见 §6.2）。
- 模型权重 **CC BY-NC 4.0（禁止商用）**，README 必须声明（见 §12）。
- 分类器缺失/未就绪：没有任何识别结果，按“全部 unknown”处理——跳过注入，原样放行（不再画“只检测”的框）。
- 量化模型（INT8 QDQ）直接用 ORT CPU 加载；不在运行时做任何量化或转换。

### 6.5 `annotator.py`
- **注意**：`cv2.putText` 不能渲染中日文。因此：
  - 默认 `label_mode="index"`：图上只画 ASCII 的 `#1 #2 …`（OpenCV 即可），角色名放在文字说明里与编号/颜色一一对应。
  - 可选 `label_mode="name"`：用 Pillow 的 `ImageFont`（**仅此路径惰性 import**，AstrBot 本身已带 Pillow）渲染 CJK 名称，需字体文件；字体缺失自动回落 `index`。
- 配色：固定 10 色高对比调色板（BGR，避开红绿难分组合），按编号循环；框线宽 `max(2, round(min(h,w)/300))`；标签带实色底 + 对比色字，位置在框上沿内侧，越界则移入框内。
- 未知角色统一用灰色框，避免与已识别角色混淆。
- 输出：长边限制 `out_max_side`（默认 1280，`cv2.INTER_AREA`），`cv2.imencode('.jpg', quality=85)`（减小体积与 LLM 视觉 token）；写入 `plugin_data/{name}/cache/{sha1[:16]}.jpg`。
- 缓存目录定期清理：保留 `cache_keep_days`（默认 7）。**不要在请求结束后立刻删除**：标注图路径可能被写入会话历史，后续轮次还会引用。

### 6.6 `image_io.py`
- 处理 `req.image_urls` 的各种形态：`http(s)://`、本地路径 / `file://`、`base64://`/data URI。
- 网络下载：`aiohttp`，超时（默认 10s）、体积上限（默认 10 MB），仅接受 `image/*`。
- 解码一律 `cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)`（`cv2.imread` 在 Windows 非 ASCII 路径下会失败）。
- 动图（GIF/APNG/WebP 动画）：取首帧；超出 `max_pixels`（默认 4096×4096）直接放行不处理。
- 解码后若最长边 > `det_pre_max_side`（默认 2048）先整体缩小再进检测，裁剪分类仍从该图取。

### 6.7 `injector.py`
- 替换：只替换**至少含 1 个已识别角色**的那几张，`req.image_urls[i] = 标注图路径`，其余原样（含“全部 unknown”的图）。
- 多图时逐张判定：只有被替换的图才进入文字说明，图序号沿用原图序号。若所有图都被跳过，则整条请求不追加任何文字。
- 文本：`req.extra_user_content_parts.append(TextPart(text=...))`；默认写入会话历史（便于后续轮次记得“谁是谁”），`temp_text=True` 时使用 `.mark_as_temp()`（需 `>=4.24.0`，低版本自动忽略该选项）。
- **禁止修改 `system_prompt`**（缓存问题，见 §3）。
- 文本模板（可在配置里覆盖语言，默认中文）：

```
[角色识别结果] 第1张图检测到 3 个人物，已用彩色框与编号标注：
#1 红色框：<角色名>（置信度 0.93）
#2 蓝色框：<角色名>（置信度 0.88）
#3 灰色框：未识别（不在识别库内或把握不足）
说明：识别库仅覆盖《鸣潮》可操控角色，其他作品的人物会显示“未识别”；结果来自自动识别，可能有误。
```

- 置信度取 `head_conf`（分类头 softmax 最大值），**不是**原型余弦。
- 文本要短：每个角色一行，不附任何调试信息；`max_persons` 限制总长度。
- “识别库范围”一句由 registry 条目的 `scope_note` 提供（`wuwa_*` 条目写“《鸣潮》可操控角色”），更换分类器时随之变化，不要写死在代码里。

### 6.8 `main.py`
- 注册 `on_llm_request` 钩子与管理指令组 `charann`（`status`、`download`、`unload`、`clear_cache`，均限管理员）。
- 钩子内全包 `try/except Exception`，记录 `logger.exception` 后放行；总耗时受 `infer_timeout`（默认 15s）约束，`asyncio.wait_for` 超时即放行。
- 钩子优先级保持默认；若与其他修改 `image_urls` 的插件冲突，通过 `priority` 调整并在 README 说明。

## 7. 管理指令

| 指令 | 作用 |
|---|---|
| `/charann status` | 模型就绪/已加载状态、线程数、常驻/超时设置、缓存大小 |
| `/charann download` | 预下载当前所选检测/分类模型 |
| `/charann unload` | 立即卸载会话释放内存 |
| `/charann clear_cache` | 清理标注图与结果缓存 |

## 8. 配置页（`_conf_schema.json`）

使用 AstrBot 原生配置页。模型选择用 `options` 下拉，选项来自 `data/registry.json` 的 `key`（后续新增模型由作者补充）；增删模型时必须同步 schema，由 `tests/test_schema_registry_sync.py` 保证一致。

```json
{
  "enabled": {"description": "启用插件", "type": "bool", "default": true},

  "models": {
    "description": "模型",
    "type": "object",
    "items": {
      "detector": {"description": "检测模型", "type": "string",
                   "options": ["person_detect_v1.1_n", "halfbody_detect_v1.0_n"],
                   "default": "person_detect_v1.1_n",
                   "hint": "halfbody 检测上半身区域，适合半身/胸像输入；切换后建议把 detect.det_conf 调为 0.512"},
      "classifier": {"description": "角色识别模型", "type": "string",
                     "options": ["wuwa_mnv4l_448_int8", "wuwa_mnv4s_384_fp32"],
                     "default": "wuwa_mnv4l_448_int8",
                     "hint": "l_448_int8 精度优先；s_384_fp32 更快更小"},
      "preprocess_backend": {"description": "分类器预处理后端", "type": "string",
                             "options": ["opencv", "pillow"], "default": "opencv",
                             "hint": "pillow 与训练口径逐位一致，作为兜底"},
      "check_update": {"description": "后台检查 latest 模型更新", "type": "bool", "default": false},
      "hf_endpoint": {"description": "HuggingFace 地址（可填镜像）", "type": "string",
                      "default": "https://huggingface.co"},
      "hf_token": {"description": "HF Token（私有库才需要）", "type": "string",
                   "default": "", "secret": true},
      "offline_mode": {"description": "离线模式（不下载，仅用本地模型）", "type": "bool", "default": false}
    }
  },

  "runtime": {
    "description": "运行时与内存",
    "type": "object",
    "items": {
      "resident": {"description": "模型常驻内存", "type": "bool", "default": false,
                   "hint": "关闭时空闲超时后自动卸载，下次请求重新加载"},
      "idle_timeout_sec": {"description": "空闲卸载超时（秒）", "type": "int", "default": 300,
                           "hint": "仅在不常驻时生效，最小 30"},
      "threads": {"description": "推理线程数", "type": "int", "default": 2},
      "max_concurrency": {"description": "最大并发推理数", "type": "int", "default": 1},
      "low_memory": {"description": "低内存模式", "type": "bool", "default": true,
                     "hint": "关闭 ORT 内存池，内存更低、略慢"},
      "infer_timeout": {"description": "单次处理超时（秒）", "type": "int", "default": 15}
    }
  },

  "detect": {
    "description": "检测",
    "type": "object",
    "items": {
      "det_imgsz": {"description": "检测输入尺寸", "type": "int", "default": 640},
      "det_conf": {"description": "检测置信度阈值", "type": "float", "default": 0.327},
      "nms_iou": {"description": "NMS IoU 阈值", "type": "float", "default": 0.5},
      "max_persons": {"description": "每图最多处理角色数", "type": "int", "default": 8},
      "min_box_px": {"description": "最小框边长（像素）", "type": "int", "default": 24}
    }
  },

  "annotate": {
    "description": "标注与注入",
    "type": "object",
    "items": {
      "label_mode": {"description": "图上标签", "type": "string",
                     "options": ["index", "name"], "default": "index",
                     "hint": "name 需要字体文件以显示中日文"},
      "font_path": {"description": "字体文件路径（label_mode=name）", "type": "string", "default": ""},
      "out_max_side": {"description": "标注图长边上限", "type": "int", "default": 1280},
      "max_images": {"description": "每条消息最多处理图片数", "type": "int", "default": 3},
      "temp_text": {"description": "说明文字仅本轮有效（不入历史）", "type": "bool", "default": false},
      "show_top5_for_unknown": {"description": "未识别时附带近邻候选（调试）", "type": "bool", "default": false},
      "cache_keep_days": {"description": "标注图保留天数", "type": "int", "default": 7}
    }
  },

  "scope": {
    "description": "作用范围",
    "type": "object",
    "items": {
      "session_whitelist": {"description": "仅这些会话生效（留空=全部）", "type": "list", "default": []},
      "session_blacklist": {"description": "这些会话不生效", "type": "list", "default": []}
    }
  }
}
```

## 9. 常见实现陷阱（务必避开）

1. 钩子里 `yield` 发消息 → 无效。要提示用户用 `await event.send(event.plain_result(...))`，且尽量不打扰用户。
2. `on_llm_request` 不能与 `@filter.command` 等同装饰同一函数。
3. 往 `system_prompt` 追加每轮不同内容 → 破坏提示词缓存。
4. 在 `__init__` 里加载模型/下载 → 拖慢 AstrBot 启动且阻塞事件循环。
5. `cv2.imread` 读非 ASCII 路径失败 → 一律 `imdecode`。
6. `cv2.putText` 画中日文 → 乱码（`?`）。见 §6.5。
7. 检测框坐标忘了去 letterbox 的 `pad/scale` → 框整体偏移。
8. 用缩放后的图裁剪分类 → 分类输入分辨率损失；必须从原图裁。
9. 看门狗卸载时有在途推理 → 崩溃；必须引用计数 + 同一把锁。
10. 立即删除标注图 → 会话历史后续轮次引用失效；用按天清理。
11. 把 HF Token 写进日志或回显到配置页以外 → 泄露；`secret` 只遮 UI，不加密，插件自己不得输出。
12. `opencv-python` 与 `opencv-python-headless` 同时安装会互相覆盖；本插件只声明 headless。
13. onnx 与 `prototypes*.npz` 精度/版本错配 → `tau`/`p_min` 失效，拒识乱套。必须成对下载、成对加载。
14. 把原型余弦当置信度展示给 LLM（正确识别也只有 ~0.2）→ 应使用 `head_conf`。
15. 对 `unknown` 的近邻名做展示 → 噪声，会诱导 LLM 误报角色。
16. 用 `latest/` 的 `main` 直连下载 → 更新时 onnx/npz 可能来自不同提交；先解析 commit sha。
17. 把 letterbox 的 512→模型尺寸两步合并成一步、或省略填充色“边缘均值”规则 → 偏离训练口径，指标下降。

## 10. 性能与内存预算（目标值，以 `scripts/bench.py` 实测为准）

| 项 | 目标 |
|---|---|
| 空闲且已卸载时插件额外占用 | 仅 Python 对象与配置，不含 ORT/cv2 会话（接近 0） |
| 检测器常驻（`person_detect_v1.1_n`，≈3M 参数） | 峰值额外 RSS 数十 MB 量级 |
| 分类器常驻（`mnv4l_448` int8，文件 33.1 MB；或 `mnv4s_384` fp32，11.3 MB） | 峰值额外 RSS 数十至一百余 MB 量级（large）/ 数十 MB（small） |
| 分类器单裁剪延迟（模型卡实测，ORT CPU） | large int8 ≈100 ms（1 线程）/ ≈35 ms（4 线程）；small fp32 ≈11 ms / ≈3.8 ms |
| 单图端到端（≤1280 长边，≤8 框） | 检测 + N×分类；large int8 在 2 线程下 8 框约需秒级，需靠 `max_persons`、small 模型或线程数权衡；超过 `infer_timeout` 即放行 |
| 冷启动加载（含 ORT 初始化） | 数秒内；日志输出实际耗时 |

优化清单（按收益排序）：
1. 惰性导入 + 空闲卸载（内存收益最大）。
2. 检测输入尺寸可配（640 → 480/416）；检测前整体降采样超大图。
3. 分类只对前 `max_persons` 个、且 ≥ `min_box_px` 的框做；同图多框在模型支持动态 batch 时合批；位置/内容几乎相同的裁剪（hash）复用结果。
4. 结果 LRU 缓存（键含模型版本与参数），容量默认 64 条；标注图按内容哈希命名天然去重。
5. 关闭 ORT 内存池与 spinning；线程数默认 2，不抢 Bot 主业务。
6. 预处理用 `cv2.dnn.blobFromImage` / `cv2.resize(INTER_AREA|INTER_LINEAR)`，避免 numpy 多次拷贝与 float64 中间量；全程 float32、`np.ascontiguousarray`。
7. 输出 JPEG q85 + 长边 1280，减少 IO 与 LLM 视觉 token。
8. 日志只在 `info` 级输出一行摘要（张数/框数/耗时），详细信息放 `debug`。

## 11. 测试与质量

- `pytest`：`tests/` 不依赖真实模型。用假会话对象（返回构造好的 YOLO 输出张量）测检测后处理；用合成图测 letterbox 还原、NMS、标注渲染、`image_io` 各输入形态、下载器（本地 HTTP 假服务：续传/校验失败/并发锁）、运行时（常驻 vs 空闲卸载、在途保护、`terminate`）。
- **预处理一致性（硬性验收）**：`scripts/parity_check.py` 对一批样图同时跑 OpenCV 实现与参考 `predict.py`（Pillow）的 `preprocess`，比较 ① 输入张量差异 ② embedding 余弦 ③ `unknown`/`pred` 一致率 ④ 两个模型变体各自结果。建议通过线：embedding 余弦均值 ≥ 0.995、`pred`/`unknown` 一致率 ≥ 99%（阈值可由作者调整）。未过线则默认后端改为 `pillow` 或调整插值策略后重测。样图需含：大图缩小、小图放大、非方形、带透明通道、边缘有渐变背景。
- 必测边界：全 unknown（不注入、不产生缓存图）、部分 unknown（灰框保留并列入说明）、多图中仅部分命中、无框、单框、框贴边、超大图、动图、损坏图、模型缺失、下载中、超时、分类器缺失降级。
- 代码风格：提交前 `ruff format` / `ruff check`；全量类型标注；公共函数写简洁中文注释；`core/` 不依赖 astrbot。
- 日志用 `from astrbot.api import logger`，不用 `print`。
- 完成定义（DoD）：测试通过；`parity_check.py` 达标；`bench.py` 结果记录进 README；无 `.onnx` 入库；`metadata.yaml`/schema/registry 三者一致。

## 12. 发布清单

- `metadata.yaml`：`name: astrbot_plugin_char_annotate`、`display_name`、`desc`、`short_desc`、`version`、`author`、`repo`、`astrbot_version: ">=4.24,<5"`。插件名全小写、无空格、以 `astrbot_plugin_` 开头。
- `requirements.txt`：`onnxruntime`、`opencv-python-headless`、`numpy`（给出下限版本，避免 numpy 2 与旧 ORT 不兼容）。
- README：功能、截图（标注图示例）、配置说明、资源占用实测、致谢，并**明确模型许可**：检测模型 `deepghs/anime_person_detection` 为 MIT，`deepghs/anime_halfbody_detection` 为 **OpenRAIL**；分类模型 `ABCwewe/wuwa_playable_character_identifier` 权重为 **CC BY-NC 4.0（仅限研究与个人用途，禁止商用）**；《鸣潮》及角色版权归 KURO GAMES，模型为非官方粉丝项目。插件不分发权重（运行时下载）。
- 向 AstrBot 插件市场提交（plugins.astrbot.app 的 `+` 按钮 → 提交到 GitHub Issue）。

## 13. 待作者补充

- [x] 检测默认阈值 `det_conf`：已核实 `person_detect_v1.1_n/threshold.json` → **0.327**（2026-10-02 作者确认采用）
- [ ] `scope_note` 文案确认（暂用 §6.7 约定：`wuwa_*` 条目写"《鸣潮》可操控角色"）
- [x] `parity_check.py` 的验收线与样图集：样图集**只提供 `--images` 样图目录参数**，不内置样图；验收线暂用 §11 建议值（余弦均值 ≥0.995、一致率 ≥99%）作为默认，作者可改
- [ ] 是否需要 `label_mode=name` 的默认字体方案（当前实现：字体缺失自动回落 `index`）
- [ ] 目标平台范围（若只支持部分适配器，在 `metadata.yaml` 声明 `support_platforms`；当前不声明=全平台）
- [ ] 后续新增分类器版本/变体时，在 `registry.json` 与 schema 的 `options` 中同步添加

> 注：插件唯一名以现有仓库/目录为准 `astrbot_plugin_character_identifier`（§12 的 `astrbot_plugin_char_annotate` 为笔误）；`metadata.yaml` 需补 `short_desc` 与 `astrbot_version: ">=4.24,<5"`。
