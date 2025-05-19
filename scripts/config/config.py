import yaml
import os
def get_config(keys):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config.yaml')
    try:
        with open(config_path, 'r',encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件 {config_path} 不存在！")

    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            print(f'配置文件中找不到{keys}')
            return None
    return value

