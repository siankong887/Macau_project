import logging
import multiprocessing
import os
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import shutil
import sys
import tempfile
import time

import av
from fractions import Fraction
import m3u8
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
CV_PART_DIR = CURRENT_DIR.parent
if str(CV_PART_DIR) not in sys.path:
    sys.path.insert(0, str(CV_PART_DIR))

from cv_paths import CVPaths
try:
    from .log_config import listener_config, work_config
except ImportError:
    from log_config import listener_config, work_config


PATHS = CVPaths.from_file(__file__)


def ts_process(url, queue, temp_dir, log_queue, video_name):
    """
    该进程用于动态获取m3u8中的ts文件，存储队列，做简单ts去重检验后发送至frame_process进程
    进行具体的处理
    """
    work_config(log_queue)
    logger = logging.getLogger(__name__)
    logger.info(f"{video_name}的抓取并下载ts视频段进程开始")

    m3u8_url = url
    start_time = time.time()
    threshold = 60 * 60

    session = requests.Session()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    }
    session.headers.update(headers)
    retries = Retry(
        total=20,
        connect=10,
        read=10,
        backoff_factor=1,
        status_forcelist=[408, 429, 444, 499, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )

    http_adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", http_adapter)
    session.mount("http://", http_adapter)

    seen_ts = set()

    while True:
        try:
            response = session.get(m3u8_url, timeout=5)
            playlist = m3u8.loads(response.text, uri=m3u8_url)
        except requests.exceptions.RequestException as exc:
            logger.critical(f"加载M3U8或ts视频段失败，即使重试了16次共九个小时等待: {exc}")
            queue.put("exit")
            return

        new_ts_list = [segment.absolute_uri for segment in playlist.segments if segment.absolute_uri not in seen_ts]
        if new_ts_list:
            for ts_url in new_ts_list:
                try:
                    ts_response = session.get(ts_url, timeout=12)
                    ts_response.raise_for_status()
                    fd, temp_path = tempfile.mkstemp(suffix=".ts", dir=temp_dir)
                    with os.fdopen(fd, "wb") as temp_file:
                        temp_file.write(ts_response.content)
                    queue.put(temp_path)
                    seen_ts.add(ts_url)
                except requests.exceptions.RequestException as exc:
                    logger.error(f"下载TS文件失败 {ts_url}: {exc}")
                    continue
            new_ts_list = None

        if time.time() - start_time > threshold:
            queue.put("exit")
            logger.info(f"{video_name}的抓取并下载ts视频段进程结束")
            return

        time.sleep(1)


def frame_process(queue, log_queue, video_name):
    """
    消费者进程：只负责将队列中的TS片段顺序追加为一个大型TS文件；编码/解码与时间戳修正在离线流程完成。
    """
    work_config(log_queue)
    logger = logging.getLogger(__name__)
    logger.info(f"{video_name}的TS拼接进程开始")

    output_dir = PATHS.crawler_videos_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_address = output_dir / video_name

    buffer_size = 1024 * 1024
    segment_index = 0
    bytes_written = 0

    with open(output_address, "ab") as out_fp:
        while True:
            try:
                ts_url = queue.get(timeout=7200)
            except Exception as exc:
                logger.error(f"尝试获取ts视频段失败，即使等待两小时，生产者进程可能出现问题,开始尝试关闭视频写入:{exc}")
                break

            if ts_url == "exit":
                break

            if not os.path.exists(ts_url):
                logger.warning(f"未找到TS临时文件: {ts_url}")
                continue

            try:
                with open(ts_url, "rb") as in_fp:
                    while True:
                        chunk = in_fp.read(buffer_size)
                        if not chunk:
                            break
                        out_fp.write(chunk)
                segment_index += 1
                bytes_written += os.path.getsize(ts_url)
                if segment_index % 100 == 0:
                    logger.info(f"已追加{segment_index}个TS段，累计{bytes_written}字节")
            except Exception as exc:
                logger.error(f"追加TS段失败 {ts_url}: {exc}")
            finally:
                try:
                    os.remove(ts_url)
                except Exception:
                    pass

        out_fp.flush()
        os.fsync(out_fp.fileno())

    logger.info(f"{video_name}的TS拼接进程结束，累计段数: {segment_index}, 总字节: {bytes_written}")


if __name__ == "__main__":
    csv_address = PATHS.crawler_camera_csv_path

    PATHS.crawler_workspace_dir.mkdir(parents=True, exist_ok=True)
    PATHS.crawler_videos_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform.startswith("linux"):
        multiprocessing.set_start_method("fork")
    else:
        multiprocessing.set_start_method("spawn")

    log_queue = multiprocessing.Queue(maxsize=9999)
    listener = listener_config(log_queue, PATHS.crawler_log_path)
    work_config(log_queue)
    logger = logging.getLogger(__name__)

    temp_dir = PATHS.crawler_temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_address, encoding="utf-8")
    m3u8_data = df["playlist_id"]
    m3u8_list = m3u8_data.tolist()
    video_names = []
    region = df["region_id"]
    region_list = region.tolist()
    location = df["location_id"]
    location_list = location.tolist()
    for idx in range(len(m3u8_list)):
        video_names.append(f"{region_list[idx]}_{location_list[idx]}.ts")

    process_list = []
    for idx, playlist_id in enumerate(m3u8_list):
        url = f"https://streaming1.dsatmacau.com/traffic/{playlist_id}.m3u8"
        video_temp_dir = temp_dir / video_names[idx].replace(".ts", "")
        video_temp_dir.mkdir(parents=True, exist_ok=True)

        queue = multiprocessing.Queue(maxsize=9999)
        p_ts = multiprocessing.Process(target=ts_process, args=(url, queue, str(video_temp_dir), log_queue, video_names[idx]))
        p_frame = multiprocessing.Process(target=frame_process, args=(queue, log_queue, video_names[idx]))
        p_ts.start()
        p_frame.start()
        process_list.append(p_ts)
        process_list.append(p_frame)

    for process in process_list:
        process.join()

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    logger.info("进程皆以结束，临时文件清理成功")
    listener.stop()
