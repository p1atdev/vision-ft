import pickle
import click
from typing import Any, Dict, List


def load_snapshot(file_path: str) -> Dict[str, Any]:
    with open(file_path, "rb") as f:
        snapshot = pickle.load(f)
    return snapshot


def format_bytes(size: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def find_peak_allocated_memory(snapshot: Dict[str, Any]) -> int:
    """
    記録されたtraceイベントから、alloc/freeのシミュレーションを行い、
    ピーク時の総アロケーションサイズを計算する。

    ※ここでは、"alloc" イベントでメモリを加算し、
    "free_completed" イベントでメモリを減算しています。
    """
    peak_memory: int = 0
    current_memory: int = 0
    # device_traces は各デバイスのイベントリスト（各イベントは dict として記録）
    device_traces: List[List[Dict[str, Any]]] = snapshot.get("device_traces", [])
    for trace in device_traces:
        for event in trace:
            action: str = event.get("action", "")
            size: int = event.get("size", 0)
            if action == "alloc":
                current_memory += size
            elif action == "free_completed":
                current_memory -= size
            # その他のイベント（例："segment_alloc", "segment_free" など）も必要に応じて処理する
            if current_memory > peak_memory:
                peak_memory = current_memory
    return peak_memory


@click.command()
@click.argument("pickle_path", type=click.Path(exists=True))
def main(pickle_path: str) -> None:
    snapshot = load_snapshot(pickle_path)
    peak_memory = find_peak_allocated_memory(snapshot)
    print(f"ピーク時の合計アロケーションサイズ: {format_bytes(float(peak_memory))}")


if __name__ == "__main__":
    main()
