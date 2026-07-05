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

from __future__ import annotations

import requests
import time
import os
import argparse
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


BASE_URL = "https://www.fsm.gov.mo/psp/pspmonitor/CamCapture"
REFERER_URL = "https://www.fsm.gov.mo/psp/pspmonitor/mobile/PortasdoCerco.aspx"
DEFAULT_IMAGES = ["image14"]
DEFAULT_INTERVAL = 3  # 秒
DEFAULT_TIMEOUT = 15
MIN_IMAGE_BYTES = 100
DEFAULT_DUPLICATE_WARN_ROUNDS = 20
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Referer": REFERER_URL,
    "Cache-Control": "no-cache, no-store, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


class ScrapeResult:
    def __init__(
        self,
        image_id: str,
        ok: bool,
        saved: bool = False,
        content_hash: str | None = None,
        duplicate: bool = False,
    ):
        self.image_id = image_id
        self.ok = ok
        self.saved = saved
        self.content_hash = content_hash
        self.duplicate = duplicate


def build_cache_buster(img_id: str) -> str:
    return f"{img_id}_{time.time_ns()}"


def scrape_one(
    img_id: str,
    ts: str,
    output_dir: str,
    timeout: int,
    cache_buster: bool,
    previous_hash: str | None,
    skip_duplicates: bool,
) -> ScrapeResult:
    url = f"{BASE_URL}/{img_id}.jpg"
    params = {"_": build_cache_buster(img_id)} if cache_buster else None
    try:
        resp = requests.get(
            url,
            params=params,
            headers=REQUEST_HEADERS,
            timeout=timeout,
        )
        resp.raise_for_status()

        content = resp.content
        content_type = resp.headers.get("Content-Type", "").lower()
        if content_type and "image" not in content_type:
            raise ValueError(f"unexpected content type: {content_type}")
        if len(content) < MIN_IMAGE_BYTES:
            raise ValueError(f"image too small: {len(content)} bytes")
        if not content.startswith(b"\xff\xd8"):
            raise ValueError("response is not a JPEG image")

        content_hash = hashlib.md5(content).hexdigest()
        duplicate = bool(previous_hash and content_hash == previous_hash)
        if duplicate and skip_duplicates:
            print(f"[{ts}] {img_id} duplicate skipped (md5={content_hash[:8]})")
            return ScrapeResult(
                image_id=img_id,
                ok=True,
                saved=False,
                content_hash=content_hash,
                duplicate=True,
            )

        img_dir = os.path.join(output_dir, img_id)
        os.makedirs(img_dir, exist_ok=True)

        filepath = os.path.join(img_dir, f"{img_id}_{ts}.jpg")
        tmp_filepath = f"{filepath}.tmp"
        try:
            with open(tmp_filepath, "wb") as f:
                f.write(content)
            os.replace(tmp_filepath, filepath)
        except OSError:
            if os.path.exists(tmp_filepath):
                os.remove(tmp_filepath)
            raise

        size_kb = len(content) / 1024
        suffix = " duplicate" if duplicate else ""
        print(
            f"[{ts}] {img_id} saved ({size_kb:.1f} KB, "
            f"md5={content_hash[:8]}{suffix})"
        )
        return ScrapeResult(
            image_id=img_id,
            ok=True,
            saved=True,
            content_hash=content_hash,
            duplicate=duplicate,
        )

    except requests.RequestException as e:
        print(f"[{ts}] {img_id} FAILED: {e}")
        return ScrapeResult(image_id=img_id, ok=False)
    except (ValueError, OSError) as e:
        print(f"[{ts}] {img_id} FAILED: {e}")
        return ScrapeResult(image_id=img_id, ok=False)


def scrape_once(
    image_ids: list[str],
    output_dir: str,
    timeout: int,
    parallel: bool,
    cache_buster: bool,
    previous_hashes: dict[str, str],
    skip_duplicates: bool,
) -> tuple[int, int, list[ScrapeResult]]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    started = time.time()

    if parallel and len(image_ids) > 1:
        results = []
        with ThreadPoolExecutor(max_workers=len(image_ids)) as executor:
            futures = [
                executor.submit(
                    scrape_one,
                    img_id,
                    ts,
                    output_dir,
                    timeout,
                    cache_buster,
                    previous_hashes.get(img_id),
                    skip_duplicates,
                )
                for img_id in image_ids
            ]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        results = [
            scrape_one(
                img_id,
                ts,
                output_dir,
                timeout,
                cache_buster,
                previous_hashes.get(img_id),
                skip_duplicates,
            )
            for img_id in image_ids
        ]

    ok = sum(int(result.ok) for result in results)
    saved = sum(int(result.saved) for result in results)
    elapsed = time.time() - started
    failed = len(image_ids) - ok
    skipped = ok - saved
    print(
        f"[{ts}] round done in {elapsed:.1f}s | "
        f"saved={saved} skipped={skipped} failed={failed}"
    )
    return saved, failed, results


def main():
    parser = argparse.ArgumentParser(description="爬取澳门关闸边境摄像头图片")
    parser.add_argument(
        "--images", nargs="+", default=DEFAULT_IMAGES,
        help="要爬取的图片ID，如 image14 image7（默认: image14）"
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL,
        help=f"爬取间隔秒数（默认: {DEFAULT_INTERVAL}）"
    )
    parser.add_argument(
        "--output", default="border_cam_data",
        help="输出目录（默认: border_cam_data）"
    )
    parser.add_argument(
        "--duration", type=int, default=0,
        help="总运行时长（秒），0=无限运行（默认: 0）"
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"单个请求超时秒数（默认: {DEFAULT_TIMEOUT}）"
    )
    parser.add_argument(
        "--parallel", action=argparse.BooleanOptionalAction, default=True,
        help="并行抓取多个摄像头（默认: true）"
    )
    parser.add_argument(
        "--cache-buster",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="为每次图片请求追加时间戳参数，避免拿到代理/CDN缓存图（默认: true）",
    )
    parser.add_argument(
        "--skip-duplicates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="如果当前图片和同摄像头上一张完全一致，则不保存（默认: true）",
    )
    parser.add_argument(
        "--duplicate-warn-rounds",
        type=int,
        default=DEFAULT_DUPLICATE_WARN_ROUNDS,
        help=(
            "同一摄像头连续多少轮抓到完全相同 JPEG 时输出 WARN "
            f"（默认: {DEFAULT_DUPLICATE_WARN_ROUNDS}）"
        ),
    )
    args = parser.parse_args()

    print(f"开始爬取: {args.images}")
    print(
        f"间隔: {args.interval}s | 超时: {args.timeout}s | "
        f"并行: {args.parallel} | cache_buster: {args.cache_buster} | "
        f"skip_duplicates: {args.skip_duplicates} | 输出: {args.output}/"
    )
    print("Ctrl+C 停止\n")

    start = time.time()
    next_run = start
    total_saved = 0
    total_failed = 0
    last_hashes: dict[str, str] = {}
    duplicate_rounds = {img_id: 0 for img_id in args.images}
    try:
        while True:
            now = time.time()
            if now < next_run:
                time.sleep(next_run - now)

            saved, failed, results = scrape_once(
                args.images,
                args.output,
                timeout=args.timeout,
                parallel=args.parallel,
                cache_buster=args.cache_buster,
                previous_hashes=last_hashes,
                skip_duplicates=args.skip_duplicates,
            )
            total_saved += saved
            total_failed += failed
            for result in results:
                if not result.ok or not result.content_hash:
                    continue

                last_hashes[result.image_id] = result.content_hash
                if result.duplicate:
                    duplicate_rounds[result.image_id] = (
                        duplicate_rounds.get(result.image_id, 0) + 1
                    )
                    warn_every = max(1, args.duplicate_warn_rounds)
                    if duplicate_rounds[result.image_id] % warn_every == 0:
                        repeated_seconds = duplicate_rounds[result.image_id] * args.interval
                        print(
                            f"[WARN] {result.image_id} got the exact same JPEG "
                            f"for {duplicate_rounds[result.image_id]} consecutive "
                            f"rounds (~{repeated_seconds}s). The source may be "
                            "cached or stale."
                        )
                else:
                    duplicate_rounds[result.image_id] = 0

            if args.duration > 0 and (time.time() - start) >= args.duration:
                print(
                    "达到设定时长，停止。 "
                    f"total_saved={total_saved}, total_failed={total_failed}"
                )
                break

            next_run += args.interval
            if next_run < time.time():
                next_run = time.time()
    except KeyboardInterrupt:
        print(f"\n已停止。total_saved={total_saved}, total_failed={total_failed}")


if __name__ == "__main__":
    main()
