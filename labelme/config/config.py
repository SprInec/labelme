import os
import yaml
import logging

logger = logging.getLogger(__name__)


def save_config(config):
    """
    保存配置到用户配置文件

    Args:
        config: 要保存的配置
    """
    try:
        # 保存到~/.labelmerc
        user_config_file = os.path.join(os.path.expanduser("~"), ".labelmerc")

        with open(user_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"配置已保存到: {user_config_file}")
        return True
    except Exception as e:
        logger.error(f"保存配置失败: {e}")
        return False
