"""
持续爬取澳门治安警察局关闸边境摄像头图片
数据源: https://www.fsm.gov.mo/psp/pspmonitor/mobile/PortasdoCerco.aspx

可用摄像头:
  image1  - 离境车辆通道入口
  image5  - 入境车辆通道
  image7  - 入境大厅
  image11 - 离境大厅
  image12 - 离境通道（近孙逸仙公园）
  image13 - 离境大厅入口
  image14 - 离境通道出口（往拱北）
"""

import requests
import time
import os
import argparse
from datetime import datetime


BASE_URL = "https://www.fsm.gov.mo/psp/pspmonitor/CamCapture"
DEFAULT_IMAGES = ["image14"]
DEFAULT_INTERVAL = 3  # 秒


def scrape_once(image_ids: list[str], output_dir: str) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for img_id in image_ids:
        url = f"{BASE_URL}/{img_id}.jpg"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()

            img_dir = os.path.join(output_dir, img_id)
            os.makedirs(img_dir, exist_ok=True)

            filepath = os.path.join(img_dir, f"{img_id}_{ts}.jpg")
            with open(filepath, "wb") as f:
                f.write(resp.content)

            size_kb = len(resp.content) / 1024
            print(f"[{ts}] {img_id} saved ({size_kb:.1f} KB)")

        except requests.RequestException as e:
            print(f"[{ts}] {img_id} FAILED: {e}")


def main():
    parser = argparse.ArgumentParser(description="爬取澳门关闸边境摄像头图片")
    parser.add_argument(
        "--images", nargs="+", default=DEFAULT_IMAGES,
        help="要爬取的图片ID，如 image14 image7（默认: image14）"
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL,
        help="爬取间隔秒数（默认: 30）"
    )
    parser.add_argument(
        "--output", default="border_cam_data",
        help="输出目录（默认: border_cam_data）"
    )
    parser.add_argument(
        "--duration", type=int, default=0,
        help="总运行时长（秒），0=无限运行（默认: 0）"
    )
    args = parser.parse_args()

    print(f"开始爬取: {args.images}")
    print(f"间隔: {args.interval}s | 输出: {args.output}/")
    print("Ctrl+C 停止\n")

    start = time.time()
    try:
        while True:
            scrape_once(args.images, args.output)
            if args.duration > 0 and (time.time() - start) >= args.duration:
                print("达到设定时长，停止。")
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
