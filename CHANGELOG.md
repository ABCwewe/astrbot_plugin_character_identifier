# 更新日志

本项目的所有显著变更都将记录在本文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循[语义化版本 2.0.0](https://semver.org/lang/zh-CN/)。

## [1.1.0] - 2026-10-02

### Added

- 新增半身检测模型选项 `halfbody_detect_v1.0_n`（[deepghs/anime_halfbody_detection](https://huggingface.co/deepghs/anime_halfbody_detection)，OpenRAIL；单类 `halfbody`，F1=0.94 对应阈值 0.512），适合半身/胸像输入；默认检测模型仍为 `person_detect_v1.1_n`。
- 配置页提示：切换半身检测后建议将「检测置信度阈值」同步调整为 0.512（`det_conf` 为全局配置，不做按模型联动）。

### Fixed

- `tests/test_registry.py` 对注册表条目数的写死断言改为按 key/kind 的结构断言，新增模型条目不再破坏测试。

## [1.0.0] - 2026-10-02

### Added

- `on_llm_request` 钩子：对消息图片做二次元人物检测 → 《鸣潮》57 个可操控角色识别（embedding 原型余弦 + 分类头置信度 + 背景负类三道拒识闸）→ OpenCV 彩色框与编号标注 → 用标注图替换原图并追加文字说明（角色名 ↔ 框颜色/编号/置信度）。
- 检测/分类模型经 `data/registry.json` 注册可选；模型运行时从 HuggingFace 下载：`latest` 条目先解析 commit sha 再按同一提交成对下载 onnx 与原型库（拒识阈值随精度配套），`*.part` 校验后原子改名，支持镜像端点与 HF Token；任一文件失败整对丢弃。
- 会话生命周期：惰性导入 onnxruntime/cv2、空闲自动卸载（看门狗）、引用计数保护在途推理、可选常驻、推理线程池与全局并发信号量；任何异常/超时/模型未就绪一律原样放行 LLM 请求。
- 预处理双后端：`opencv`（默认）与 `pillow`（与训练参考实现逐位一致）；`scripts/parity_check.py` 验证两变体 embedding 余弦均值与判定一致率。
- 管理指令组 `charann`：`status`（模型/缓存状态）、`download`（后台预下载）、`unload`（立即卸载）、`clear_cache`（清理缓存），均限管理员。
- WebUI 原生配置页：模型选择、运行时与内存、检测参数、标注与注入、会话黑白名单。
- 识别库范围文案由 registry 条目 `scope_note` 提供；中文展示名随模型仓库 `class_names_zh.yaml` 同 commit 下载。
- `scripts/bench.py` 基准脚本；94 项单元测试（假 ORT 会话与本机回环假 HF 服务，不依赖真实模型）。

### Security

- `hf_token` 仅用于请求头，不写入日志；配置页以 `secret` 遮显，插件自身不回显。

[1.1.0]: https://github.com/ABCwewe/astrbot_plugin_character_identifier/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ABCwewe/astrbot_plugin_character_identifier/releases/tag/v1.0.0
