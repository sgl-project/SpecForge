# CI 优化修改说明

本文档记录本次 CI 优化涉及的每个文件、修改目的、运行时行为和维护要求。

## 保持不变的约束

- GPU 容器继续使用 `--gpus all`，一次 CI 可以访问全部 GPU。
- GPU job 继续使用 `specforge-gpu-ci` concurrency group；`cancel-in-progress: false` 和
  `queue: max` 保证不同 PR 的 GPU job 串行排队。
- workflow 级 `pr-test-${{ github.ref }}` 继续取消同一 PR 的旧运行。
- SGLang 数据再生成测试继续使用 `--mem-frac=0.8`。
- 用户将共享内存调整为 16 GiB 的修改被保留。

## 文件说明

### `.github/workflows/test.yaml`

这个文件负责自托管 GPU CI。

主要修改如下：

1. `pull_request` 增加 `ready_for_review` 事件。Draft PR 转为 Ready 时会启动测试，不需要再推送一次提交。
2. 自托管 GPU job 只接收上游仓库的同仓 PR，避免公开 fork 的任意代码直接运行在长期存在的自托管机器上。维护者仍可通过 `workflow_dispatch` 手动运行上游分支。
3. 保留 `--gpus all`、16 GiB shared memory 和全局串行 concurrency。
4. 移除 `--privileged` 和 `--pid=host`。当前测试只需要 CUDA 和共享内存，不需要主机 PID namespace 或特权容器。
5. 增加全卡空闲检查。只要任意 GPU 上存在 compute process，job 会输出进程列表并立即失败；它不会自动选择部分 GPU，也不会终止未知进程。
6. 不再复制 `/github/home/sf`。每次 job 都创建干净的 `sf` venv，避免旧依赖和不同 PR 相互污染；uv 自身的下载缓存仍然可以复用。
7. 安装 `.[data]`，显式保证在线数据再生成 gate 所需的 OpenAI client 可用；同时移除无条件允许 prerelease 依赖的选项。
8. SGLang capture patch 在所有测试前统一应用，保持原有完整测试环境。
9. 数据再生成测试随默认 unittest discovery 一起运行；live server-capture 仍保留为需要专用环境变量的独立 gate。
10. 无论 job 成功、失败或被取消，最后都会输出 GPU 状态，便于排查显存占用和遗留进程。
11. `actions/checkout` 固定到完整 commit SHA，避免可变 tag 带来的供应链风险。

外部 fork PR 当前会运行 GitHub-hosted lint/package 检查，但不会自动获得自托管 GPU。若以后希望外部 fork 运行 GPU 测试，应优先提供一次性 Runner 或维护者审批后的隔离执行流程。

### `.github/workflows/lint.yaml`

这个文件负责快速的 GitHub-hosted 检查。

主要修改如下：

1. 除 PR 外，`main` push 也会运行，确保合并后的分支继续可构建。
2. 增加同 ref 自动取消旧运行、只读权限和 10 分钟超时。
3. 删除 CI 中无意义的 `pre-commit install`；CI 只需直接执行 `pre-commit run --all-files`。
4. GitHub Actions 固定到完整 commit SHA。
5. 增加独立 `package` job，在干净 GitHub-hosted Runner 上构建 wheel 和 sdist、运行 `twine check`，并检查 wheel 中只能出现 `specforge/` 和对应 dist-info 内容。

`package` job 可以在发布前发现错误的 setuptools 包发现配置，避免将 `build/`、测试或其他仓库目录发布到 PyPI。

### `pyproject.toml`

将 setuptools 自动包发现范围从排除少数目录改为明确包含 `specforge*`：

```toml
[tool.setuptools.packages.find]
include = ["specforge*"]
```

原配置会在存在历史 `build/` 目录时，把 `build/lib`、`benchmarks` 等内容再次装入 wheel，甚至形成递归的 `build/lib/build/lib`。明确 allowlist 后，发布包只包含 SpecForge Python package。

### `.github/workflows/publish_docs.yaml`

这个文件负责文档构建和部署。

主要修改如下：

1. 文档相关 PR 现在会构建文档，但不会部署，因此损坏的文档可以在合并前发现。
2. 将 build 和 deploy 拆成两个 job。build job 只有 `contents: read`；deploy job 才拥有 `contents: write`。
3. build 产物通过短期 GitHub artifact 传递给 deploy job，部署 job 不再重新 checkout 或执行源码。
4. 只有 `main` 上的非 PR 事件可以部署，避免从手动选择的任意分支发布文档。
5. 增加 `assets/logo.png` 路径触发、job timeout，并固定所有 Action 的完整 SHA。
6. 保留现有 `peaceiris/actions-gh-pages` 部署方式，因此不要求立即修改 GitHub Pages 的仓库设置。

### `.github/workflows/publish_pypi.yaml`

这个文件负责 PyPI 发布。

主要修改如下：

1. 只允许上游仓库的 `main` 分支执行发布。
2. 发布 workflow 使用独立 concurrency group 串行排队，避免重复发布并发执行。
3. 修正 PyPI project URL：`specforgeee` 改为 `specforge`。
4. 同时构建 wheel 和 sdist，并在上传前运行 `twine check`。
5. 在无依赖临时 venv 中安装 wheel 并执行 `import specforge`，验证 wheel 至少可以被正常安装和导入。
6. 使用 PyPI Trusted Publishing；删除长期 `PYPI_TOKEN`，仅保留 `id-token: write`。
7. 更新并固定 checkout、setup-python 和 PyPI publish Action 的完整 SHA。

仓库管理员需要在 PyPI 项目中配置 Trusted Publisher，并确保 repository、workflow filename、`pypi` environment 与当前 workflow 一致。完成配置前不要尝试正式发布。

### `.github/dependabot.yml`

新增每周 GitHub Actions 更新检查。因为 workflows 使用完整 SHA，Dependabot 负责创建升级 PR，兼顾不可变引用和后续安全更新。

### `.github/CI_MAINTENANCE.md`

即本文档。它用于向后续维护者说明本次修改的边界、原因、验证方法和外部配置要求。

## 已完成的验证

- 所有 GitHub workflow 和 Dependabot 文件通过 YAML 解析。
- 干净源码快照成功构建 wheel 和 sdist；wheel 共 135 个条目，非 `specforge` 条目数量为 0。
- `git diff --check` 通过。

本次 PR 不修改 `tests/`；现有测试套件仍由 GPU workflow 的普通 unittest discovery 执行。

没有执行正式 PyPI 发布。远程 H20 的 Node.js 是 v12，而文档 CI 明确使用 Node.js 20，因此文档的完整 npm/Sphinx 构建应由 GitHub-hosted workflow 验证。
