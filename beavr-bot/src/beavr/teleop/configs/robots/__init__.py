import abc

import draccus

# 通过 draccus.ChoiceRegistry 实现的“注册表模式”，
class TeleopRobotConfig(draccus.ChoiceRegistry, abc.ABC):
    """Base class for teleoperation *combo* robot configurations.
        遥操作*组合*机器人配置的基类。
        
    Any dataclass inheriting from this class can be registered with the
    ``@TeleopRobotConfig.register_subclass("my_name")`` decorator and later
    instantiated from that short name via
    ``TeleopRobotConfig.get_choice_class("my_name")``.
    
    任何继承自此类的数据类都可以使用``@TeleopRobotConfig.register_subclass("my_name")``装饰器进行注册，
    之后可以通过``TeleopRobotConfig.get_choice_class("my_name")``从该简短名称实例化
    """
    

    pass
