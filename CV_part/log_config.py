import logging
import sys
from logging import handlers
from logging import config

#定义一个子进程日志的配置，这个日志只用于子进程将LogRecord对象写入队列中
def work_config(LogQueue):
    logging.config.dictConfig(
            {
                'version':1,
                'disable_existing_loggers':False,
                'handlers':{
                    'queue':{
                        'class':'logging.handlers.QueueHandler',
                        'queue':LogQueue,
                    }
                },
                'root':{
                    'level':'DEBUG',
                    'handlers':['queue']
                }
            }
            
            )

#定义一个监听者函数，用于配置监听handlers的日志，并启动监听
def listener_config(LogQueue):
    #配置真正干活的handlers

    #consile这个日志处理器，用于将日志信息输出到控制台，设置参数sys.stdout来防止正常程序运行状态被标红
    console_handler=logging.StreamHandler(sys.stdout)
    #file这个handle用于将日志输出到指定文件路径
    file_handler=logging.handlers.RotatingFileHandler(filename='video_log.log',mode='a',encoding='utf-8',maxBytes=10*1024*1024,backupCount=5)

    #设置日志处理器的过滤器
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    #设置终端和文件的输出格式
    formatter=logging.Formatter('%(asctime)s - PID:%(process)d - %(levelname)s - %(name)s[line:%(lineno)d] - %(message)s')
    #将格式化器挂到日志处理器上
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    #创建并启动监听handlers
    listener=logging.handlers.QueueListener(LogQueue,console_handler,file_handler,respect_handler_level=True)
    listener.start()

    return listener





