import yaml
from easydict import EasyDict


def config_from_yaml_file(cfg_file, config):
    with open(cfg_file, mode='r', encoding="utf-8") as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


def config_from_list(cfg_list, config):
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0  # ensure cfg_list is key value pairs
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, f"key not found: {subkey}"
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, f"key not found: {subkey}"
        try:
            value = literal_eval(v)
        except ValueError:
            value = v

        if not isinstance(value, type(d[subkey])) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif not isinstance(value, type(d[subkey])) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for i, x in enumerate(val_list):
                val_list[i] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert (
                isinstance(value, type(d[subkey]))
            ), f"type {type(value)} does not match original type {type(d[subkey])}."
            d[subkey] = value


def merge_new_config(config, new_config):
    if "basic_config" in new_config:
        with open(new_config["basic_config"], 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config
