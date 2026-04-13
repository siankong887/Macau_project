import multiprocessing
import time
import requests
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import m3u8
import av
from fractions import Fraction
import os
import shutil
import tempfile
from log_config import work_config,listener_config
import logging
import pandas as pd


def ts_process(url,queue,temp_dir,log_queue,video_name):
    """
    该进程用于动态获取m3u8中的ts文件，存储队列，做简单ts去重检验后发送至frame_process进程
    进行具体的处理
    """
    #调用函数，将日志记录器挂载到进程中
    work_config(log_queue)
    logger=logging.getLogger(__name__)
    #进程开始，打印进程相关信息
    logger.info(f"{video_name}的抓取并下载ts视频段进程开始")

    #获取m3u8地址
    m3u8_url=url

    #计时，并通过elapsed_time>threshold 进行进程的关闭
    start_time=time.time()
    #这个参数用于控制爬取时间，如果超过这个时间，则关闭进程,默认设置为30小时
    threshold=60*60

    """
    这部分用于创建会话，定义一个Retry（）的重试策略，并将这个策略作为配置器挂载到会话上
    """
    #创建一个会话，用于维护一个一定时间的tcp连接，降低计算开销
    session=requests.Session()
    # 添加一个常见的浏览器User-Agent，伪装成浏览器，防止被服务器拒绝
    headers = {
        #添加agent，伪装乘浏览器，而不是暴露这是一个python脚本，暴露是python脚本很容易被服务器拒绝
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        #Accept参数用于告诉服务器可以接受哪一些参数，这里设置为*/*，表示接受所有参数
        'Accept': '*/*',
        #Accept-Language参数用于告诉服务器可以接受哪一种语言，这里设置为zh-CN,zh;q=0.9,en;q=0.8，表示接受中文和英文
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        #Connection参数用于告诉服务器是否保持连接，这里设置为keep-alive，显示的告诉客户端要表示保持连接
        'Connection': 'keep-alive'
    }
    session.headers.update(headers)
    #创建一个重连策略
    retries=Retry(
        total=20,
        connect=10,
        read=10,
        backoff_factor=1, # 指数退避：1s,2s,4s,8s...
        status_forcelist=[408,429,444,499,500,502,503,504],
        allowed_methods=frozenset(["GET"]) # 允许在GET上重试
    )

    #将重连策略挂载到会话上
    http_adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", http_adapter)
    session.mount("http://", http_adapter)

    #定义一个已经见过的ts集合，用于去重
    seen_ts=set()


    #无限循环获取m3u8地址，并获取ts文件
    while True:
        try:
            #因为配置器已经挂载到会话中，所以每一次请求如果发生错误都会按照重连策略进行重试
            response=session.get(m3u8_url,timeout=5)# 设置较长的超时时间，降低Read timed out概率
            #如果请求成功，则解析m3u8文件
            playlist=m3u8.loads(response.text,uri=m3u8_url)
        except requests.exceptions.RequestException as e:
            logger.critical(f"加载M3U8或ts视频段失败，即使重试了16次共九个小时等待: {e}")
            queue.put("exit")
            return
        #列表推导式获取m3u8中的ts视频段，并过滤掉已经见过的ts视频段
        new_ts_list=[segment.absolute_uri for segment in playlist.segments if segment.absolute_uri not in seen_ts]
        if new_ts_list:
            for ts_url in new_ts_list:
                try:
                    #下载ts文件
                    ts_response=session.get(ts_url,timeout=12)
                    #检查下载状态，确保请求成功，不成功抛出异常
                    ts_response.raise_for_status()
                    # 创建临时文件并写入内容
                    fd, temp_path = tempfile.mkstemp(suffix=".ts", dir=temp_dir)
                    with os.fdopen(fd, 'wb') as temp_file:
                        temp_file.write(ts_response.content)
                    # 将临时文件路径放入队列
                    queue.put(temp_path)
                    # 将ts_url添加到已经见过的ts集合中
                    seen_ts.add(ts_url)
                except requests.exceptions.RequestException as e:
                    logger.error(f"下载TS文件失败 {ts_url}: {e}")
                    continue # 跳过下载失败的TS文件
            #清空new_ts_list列表，防止下一个循环得到一个有数据的列表
            new_ts_list=None
        #定义一个退出机制
        elapsed_time=time.time()-start_time
        if elapsed_time>threshold:
            queue.put("exit")
            logger.info(f"{video_name}的抓取并下载ts视频段进程结束")
            return

        time.sleep(1)





def frame_process(queue,log_queue,video_name):
    """
    消费者进程：只负责将队列中的TS片段顺序追加为一个大型TS文件；编码/解码与时间戳修正在离线流程完成。
    """
    work_config(log_queue)
    logger = logging.getLogger(__name__)
    logger.info(f"{video_name}的TS拼接进程开始")
    #这里的作用是如果传入的视频是以.mp4结尾的，就将其替换为.ts结尾，否则保持不变
    #如果传入的视频是以.ts结尾的，就保持不变，否则添加.ts结尾
    # output_name = (video_name[:-4] + '.ts') if video_name.endswith('.mp4') else (video_name if video_name.endswith('.ts') else f"{video_name}.ts")
    output_address = f"videos/{video_name}"
    #这里是确保输出目录存在，不存在就创建
    # os.makedirs(os.path.dirname(output_address), exist_ok=True)
    
    #定义写入缓冲区的大小
    BUFFER_SIZE = 1024 * 1024
    #定义一个计数器，用于记录拼接的ts视频段数
    segment_index = 0
    #定义一个计数器，用于记录拼接的ts视频段的字节数
    bytes_written = 0

    with open(output_address, 'ab') as out_fp:
        while True:
            try:
                ts_url = queue.get(timeout=7200)
            except Exception as e:
                logger.error(f"尝试获取ts视频段失败，即使等待两小时，生产者进程可能出现问题,开始尝试关闭视频写入:{e}")
                break

            if ts_url == "exit":
                break

            if not os.path.exists(ts_url):
                logger.warning(f"未找到TS临时文件: {ts_url}")
                continue

            try:
                #二进制模式打开视频段
                with open(ts_url, 'rb') as in_fp:
                    while True:
                        #一次只读一定数量的二进制文件，避免大量的内存占用
                        chunk = in_fp.read(BUFFER_SIZE)
                        if not chunk:
                            break
                        #rb--->ab 没有编码解码开销速度受限于磁盘的io速度，但是这是消费者进程，不影响生产者的爬取操作
                        out_fp.write(chunk)
                segment_index += 1
                bytes_written += os.path.getsize(ts_url)
                if segment_index % 100 == 0:
                    logger.info(f"已追加{segment_index}个TS段，累计{bytes_written}字节")
            except Exception as e:
                logger.error(f"追加TS段失败 {ts_url}: {e}")
            finally:
                try:
                    os.remove(ts_url)
                except Exception:
                    pass

        out_fp.flush()
        os.fsync(out_fp.fileno())

    logger.info(f"{video_name}的TS拼接进程结束，累计段数: {segment_index}, 总字节: {bytes_written}")




if __name__ == "__main__":


    #以下是需要帮忙配置的一些参数
    csv_address='camera_location.csv'

    #创建输出的视频文件夹
    if not os.path.exists('videos'):
       os.mkdir('videos')

    #显示的设置进程的启动方式，因为linux会默认用fork()模式copy主进程以启动进程，会与日志的配置产生冲突
    # try:
    #     multiprocessing.set_start_method('spawn')
    # except RuntimeError:
    #     # 如果fork模式已经被设置，则跳过
    #     pass
    if sys.platform.startswith('linux'):
        multiprocessing.set_start_method('fork')
    else:
        multiprocessing.set_start_method('spawn')

    #创建一个用于日志通信的队列
    log_queue=multiprocessing.Queue(maxsize=9999)

    #调用函数异步的执行监听
    listener=listener_config(log_queue)

    #进程间日志记录器不互通，所以如果主进程想要记录信息也要调用work_config
    work_config(log_queue)

    logger=logging.getLogger(__name__)


    TEMP_DIR = "ts_temp_storage"
    # 程序启动时，如果临时目录已经存在，就递归的从目录树的叶子节点开始删除目录并最终删除根目录
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    #不管目录存不存在最后都会创建一个临时目录
    os.makedirs(TEMP_DIR)

    #读取csv文件，并获取相关信息
    df=pd.read_csv(csv_address,encoding='utf-8')
    m3u8_data=df['playlist_id']
    m3u8_list=m3u8_data.tolist()
    #拼接一个vedio_name的列表
    vedio_name=[]
    region=df['region_id']
    region_list=region.tolist()
    location=df['location_id']
    location_list=location.tolist()
    for i in range(len(m3u8_list)):
        vedio_name.append(f"{region_list[i]}_{location_list[i]}.ts")


    #循环开始进程，并等待进程结束
    process_list=[]
    for i in range(len(m3u8_list)):
        url=f"https://streaming1.dsatmacau.com/traffic/{m3u8_list[i]}.m3u8"
        
        # 为每个视频任务创建一个独立的临时子目录
        video_temp_dir = os.path.join(TEMP_DIR, vedio_name[i].replace('.ts', ''))
        os.makedirs(video_temp_dir, exist_ok=True)

        queue=multiprocessing.Queue(maxsize=9999)
        p_ts = multiprocessing.Process(target=ts_process,args=(url,queue,video_temp_dir,log_queue,vedio_name[i]))
        p_frame = multiprocessing.Process(target=frame_process,args=(queue,log_queue,vedio_name[i]))
        p_ts.start()
        p_frame.start()
        process_list.append(p_ts)
        process_list.append(p_frame)

    #等待进程结束，先将主进程挂起
    for p in process_list:
        p.join()
    


    # 程序结束后，清理临时目录
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    logger.info("进程皆以结束，临时文件清理成功")
    #发送哨兵值，告诉日志停止监听，等待线程处理完后 关闭
    listener.stop()