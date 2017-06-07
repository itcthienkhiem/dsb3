#!/usr/bin/env python
# encoding: utf-8
import importlib

_config = None


def set_configuration(config_dir, config_name):
    global _config #global声明全局变量
    _config = importlib.import_module("%s.%s" % (config_dir, config_name))#动态导入模块
    print "Loaded", _config


def config():
    return _config
