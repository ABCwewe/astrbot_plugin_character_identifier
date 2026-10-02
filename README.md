# astrbot_plugin_character_identifier

AstrBot 插件：在 LLM 请求发出前，对消息中的图片做**二次元人物检测 → 《鸣潮》可操控角色识别 → 彩色框标注**，并把标注图与文字说明注入模型上下文，让 LLM 能"看清"图里有谁、在哪。

## 功能

1. **检测**：可选两种 YOLOv8n 检测器（默认全身）：`deepghs/anime_person_detection` 的 `person_detect_v1.1_n`（全身，F1 0.85 @ 0.327，MIT）与 `deepghs/anime_halfbody_detection` 的 `halfbody_detect_v1.0_n`（上半身，F1 0.94 @ 0.512，OpenRAIL，适合半身/胸像输入）。
2. **识别**：作者自训练的 `ABCwewe/wuwa_playable_character_identifier`（鸣潮 57 个可操控角色，MobileNetV4，双输出 embedding+logits，配套原型库与三道拒识闸），对每个检测框裁剪分类。
3. **标注**：OpenCV 画彩色框 + 编号（可选中文名），已识别角色彩色框，未识别灰色框。
4. **注入**：用标注图替换原图（仅替换含已识别角色的图），并追加文字说明（角色名 ↔ 框颜色/编号/置信度）。

- 分类器只认训练时的 57 个角色，其他作品/角色一律判 `unknown`（图中大量 `unknown` 属正常）。
- 无检测框、全部 `unknown`、模型未就绪、超时、任何异常：**原样放行请求**，不改图不加字。
- 多图默认处理前 3 张；同一张图按内容哈希走结果缓存。

## 模型许可与声明

- 检测模型 [`deepghs/anime_person_detection`](https://huggingface.co/deepghs/anime_person_detection)：**MIT License**；[`deepghs/anime_halfbody_detection`](https://huggingface.co/deepghs/anime_halfbody_detection)：**OpenRAIL**。
- 分类模型 [`ABCwewe/wuwa_playable_character_identifier`](https://huggingface.co/ABCwewe/wuwa_playable_character_identifier)：权重以 **CC BY-NC 4.0** 提供，**仅供研究与个人用途，禁止商业使用**。
- 《鸣潮》（Wuthering Waves）及相关角色版权归 **KURO GAMES** 所有；本插件为非官方粉丝项目。
- 本插件**不分发任何模型权重**：全部在运行时从 HuggingFace 下载到 `data/plugin_data/astrbot_plugin_character_identifier/models/`。

## 安装

从 AstrBot 插件市场安装，或克隆本仓库到 `data/plugins/` 后在 WebUI 重载。依赖（`onnxruntime`、`opencv-python-headless`、`numpy`）会自动安装。

首次使用建议先让管理员执行 `/charann download` 预下载模型（检测器约 12 MB，分类器 large int8 约 33 MB）；未预下载时首次识别请求会触发后台下载，当次请求原样放行。

## 指令（管理员）

| 指令 | 作用 |
|---|---|
| `/charann status` | 模型就绪/已加载状态、线程数、常驻/超时设置、缓存占用 |
| `/charann download` | 后台预下载当前所选检测/分类模型 |
| `/charann unload` | 立即卸载会话释放内存 |
| `/charann clear_cache` | 清理标注图与结果缓存 |

## 配置

WebUI 插件配置页（`_conf_schema.json`），常用项：

- **models**：检测/分类模型选择（下拉）、预处理后端（`opencv` 默认 / `pillow` 兜底，与训练口径逐位一致）、HuggingFace 地址（可填镜像 `hf-mirror.com`）、HF Token（私有库才需要，仅 UI 遮显）、离线模式。
- **runtime**：常驻内存（默认关，空闲 300s 自动卸载）、推理线程数（默认 2）、最大并发（默认 1）、低内存模式、单次处理超时（默认 15s）。
- **detect**：输入尺寸（默认 640）、置信度阈值（默认 0.327，模型卡 F1 最优实测值）、NMS IoU、每图最多人数（默认 8）、最小框边长。
- **annotate**：图上标签（`index` 默认 / `name` 需字体文件渲染中文）、标注图长边上限（默认 1280）、每条消息最多处理图数、说明文字是否仅本轮有效、未识别时是否附带近邻候选（调试）、缓存保留天数。
- **scope**：会话白/黑名单。

## 资源占用

实测数据（`scripts/bench.py`，AMD Ryzen 7 5700X，2 推理线程，INT8 large 分类器）：

| 场景 | 数值 |
|---|---|
| 空闲（已卸载）额外内存 | cv2/ORT 未导入，仅 Python 对象 |
| 检测器会话加载 | ≈155 ms，RSS +45 MB |
| 分类器会话加载（mnv4l_448 int8） | ≈120 ms，RSS +48 MB |
| 检测延迟（imgsz=640） | 均值 64 ms |
| 分类单裁剪（S=448） | 均值 61 ms |
| 端到端（检测 + 8 框分类） | 均值 554 ms（默认 `infer_timeout=15s` 内余量充足） |
| 卸载释放 | ≈15 ms，RSS 回落 ≈55 MB |

预处理一致性（`scripts/parity_check.py`，OpenCV 后端 vs 训练参考实现，两变体）：embedding 余弦均值 ≥0.9957、pred/unknown 一致率 100%，pillow 后端与参考实现逐位一致；官方样图集待作者提供后回填。

## 已知限制

- 分类器针对**单人全身立绘/插画**训练：多角色同框、Q 版、仅面部特写、极端裁剪、强风格化可能误判或被拒识。
- 负类拒识率约 86%–90%：陌生输入并非 100% 被拒，文字说明已留余地。
- 仅覆盖训练时的 57 个角色；新角色需等模型更新（`latest` 变更后重新 `/charann download`）。
- `cv2.putText` 无法渲染中日文，`label_mode=name` 需要提供字体文件（缺失自动回落 `index`）。
- 若反馈"图片没传到模型"，请先检查所选聊天模型的**模态（vision）配置**是否正确，再排查本插件。

## 开发与测试

```bash
# 单元测试（不依赖真实模型）
.venv/Scripts/python -m pytest data/plugins/astrbot_plugin_character_identifier/tests -q

# 预处理一致性验证（需真实模型，作者提供样图目录后）
python scripts/parity_check.py --images <样图目录>

# 基准测试
python scripts/bench.py
```

## 致谢

- [deepghs/anime_person_detection](https://huggingface.co/deepghs/anime_person_detection)（MIT）
- [AstrBot](https://github.com/AstrBotDevs/AstrBot)
