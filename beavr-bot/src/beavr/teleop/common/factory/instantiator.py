from importlib import import_module
from typing import Any


class DotDict(dict):
    """Dictionary with attribute-style access.

    Example
    -------
    >>> cfg = DotDict({"a": 1, "b": {"c": 2}})
    >>> cfg.a
    1
    >>> cfg.b.c
    2
    """

    def __getattr__(self, item: str) -> Any:  # noqa: D401
        try:
            value = self[item]
            # Automatically convert nested dicts/lists to DotDict where relevant
            if isinstance(value, dict) and not isinstance(value, DotDict):
                value = DotDict(value)
                self[item] = value
            elif isinstance(value, list):
                value = [DotDict(v) if isinstance(v, dict) else v for v in value]
                self[item] = value
            return value
        except KeyError as exc:
            raise AttributeError(item) from exc

    # Attribute-style assignment/deletion delegate to dict methods
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def copy(self):  # noqa: D401
        return DotDict(dict(self.items()))


def _convert_to_dotdict(obj: Any) -> Any:
    """Recursively convert dicts/lists into DotDict instances."""
    if isinstance(obj, dict) and not isinstance(obj, DotDict):
        return DotDict({k: _convert_to_dotdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_convert_to_dotdict(v) for v in obj]
    return obj


def instantiate_from_target(cfg: Any):
    """从一个类似 Hydra 风格的 ``_target_`` 字典中实例化对象。

    这是一个对 ``hydra.utils.instantiate`` 的轻量化替代实现，它支持：
    - 嵌套的字典/列表（会进行递归递归实例化）
    - 对于已经是建好的对象，直接原样返回
    - 对于带有 ``build`` 方法的数据类/对象（会立即调用该方法进行构建）
    """
    # 1. 如果传进来的配置对象自己知道怎么“造”自己（有 build 方法），就直接调用它的 build() 方法。
    if hasattr(cfg, "build") and callable(cfg.build):
        return cfg.build()

    # 2. Lists/tuples → recurse 处理列表和元组：如果配置是个列表，就遍历里面的每一个元素，并且对自己调用自己（递归），把列表里的东西也全部实例化
    if isinstance(cfg, list):
        return [instantiate_from_target(c) for c in cfg]
    if isinstance(cfg, tuple):
        return tuple(instantiate_from_target(c) for c in cfg)

    # 3. 基础出口：如果传进来的既不是字典，也不是上面处理过的类型比如只是个数字 10，或者字符串 "hello"），说明它不需要被实例化，直接原样返回。
    if not isinstance(cfg, dict):
        return cfg

    # 4. 核心字典处理逻辑开始
    cfg_dict = dict(cfg)  # 做一个浅拷贝，这样我们在 pop 删除元素时不会破坏原始的 cfg 数据
    # 尝试把 `_target_` 这个特殊的键拿出来。
    # pop 的作用是：如果找到了，就把它拿出来并从字典里删掉；找不到就返回 None。
    target = cfg_dict.pop("_target_", None)
    if target is None:
        # 如果字典里根本没有 `_target_`（说明这只是个普通的参数字典，不是用来造类的），
        # 尝试把它转换成一种可以通过点号访问属性的字典（dotdict），并原样返回。
        return _convert_to_dotdict(cfg_dict)

    # 5. 动态导入与实例化
    # 假设 target 是 "beavr.teleop.components.Operator"
    # rsplit(".", 1) 会把它从右边第一个点切开，变成 ["beavr.teleop.components", "Operator"]
    module_path, cls_name = target.rsplit(".", 1)
    # 动态执行类似 `import beavr.teleop.components` 的操作
    module = import_module(module_path)
    # 从刚刚导入的模块中，把 `Operator` 这个类（Class）抓出来
    cls = getattr(module, cls_name)

    # 6. 递归处理参数：因为造这个类可能需要传入参数，而参数本身可能也是需要实例化的类
    # 所以遍历字典里剩下的所有键值对，把值（v）再次扔进 instantiate_from_target 递归处理
    params = {k: instantiate_from_target(v) for k, v in cfg_dict.items()}
    
    # 7. 字典解包实例化：拿着刚刚抓出来的类（cls），和处理好的参数字典（params），使用 ** 进行字典解包，完成对象的创建并返回。
    return cls(**params)
