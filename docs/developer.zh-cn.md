# 开发者指南

作者：[SwordJack](https://github.com/SwordJack)

[TOC]

## 多语言

- 汉语 [developer.zh-cn.md](developer.zh-cn.md)
- English [developer.md](developer.md)

## 源代码

### 关于文件夹

本项目中涉及操作系统、文件系统、日志打印的方法，均存放于 [`core`](/ark/core/) 目录中，涉及此类内容的操作，请从中引用，而非重新“造轮子”。

### 代码规范

- 本项目的 [ark](/ark/) 目录为源代码目录，代码规范参照 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) 执行，对于每一个类、方法、函数等，都应当清晰标注其参数类型、返回值类型，并在注释中加以有效说明。
- 项目中涉及字符串值的代码，请使用双引号作为主要引号。
- 在每一个 Python 文件的开头，请加注文档信息，例如在 [logger.py](/ark/core/logger.py) 中，本人加注了如下格式信息。后续的文档信息格式请亦参考此例。

```python
#! python3
# -*- encoding: utf-8 -*-
"""
Format logger output using python logging and colorlog module.

Reference: https://github.com/jyesselm/dreem/blob/main/dreem/logger.py

@File   :   logger.py
@Created:   2025/04/01 16:16
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""
```

## 单元测试

- 本项目的 [test](/test/) 目录为单元测试目录，单元测试目录中的目录结构应与源代码目录中的目录结构相一致，对于某个源代码文件的单元测试代码文件，其文件名应为源代码文件名前加上 `test_*` 前缀，如 [`logger.py`](/ark/core/logger.py)，其单元测试代码文件为 [`test_logger.py`](/test/core/test_logger.py)。
- 通过 Pytest 进行单元测试，其详细使用方法可参考 [Pytest Documentation](https://docs.pytest.org/en/stable/)。
- 一般情况下，每一个源代码文件（除配置文件外），都应当有一个与之相对应的单元测试文件，单元测试应当测试源代码的一般运行、异常处理、边缘案例等，以保证本项目代码的稳健性（robustness）。

## 版本控制

- 本项目通过 Git 进行版本控制。
- 项目主分支为 [`main`](https://github.com/SwordJack/ARK/tree/main) 分支，为本项目对外发布或供外部项目应用的代码版本，目前由本项目的所有者控制。
- 项目的开发一般情况下围绕 [`develop`](https://github.com/SwordJack/ARK/tree/develop) 分支进行，开发者请基于此分支创建新分支进行开发，并提交 [Pull Request](https://github.com/SwordJack/ARK/pulls) 以申请将所创建的分支合并入 `develop` 分支中。
- 开发者新创建的分支，其命名规范为：`<名字>/<类型>_<描述>`，例如：**Tom** 想要给项目添加一个**吃西瓜（eat watermelon）**的功能，那么，他就可以如此命名他的分支：`tom/feature_eat_watermelon`。
- 分支的类型有以下若干种：

| 前缀        | 全名          | 含义                                                        |
| ----------- | ------------- | ----------------------------------------------------------- |
| `feature_*` | feature       | 添加新功能                                                  |
| `fix_*`     | fix           | 修复某个 bug （低优先级，合并到 `develop` 中）              |
| `hotfix_*`  | hot fix       | 紧急修复某个严重 bug （高优先级，需要直接合并到 `main` 中） |
| `docs_*`    | documentation | 给项目添加或完善文档内容、多语言内容等                      |

- 当一个分支被合并后，其会被重命名：如 `tom/feature_eat_watermelon` 分支被合并后，它会被重命名为 `zarchive/tom/feature_eat_watermelon`，以表示此分支已经完成其使命，不应再被启用。