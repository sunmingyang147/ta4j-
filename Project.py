from abc import ABC, abstractmethod
from queue import Queue
from threading import Thread
from decimal import Decimal
from typing import List, Optional

class CompressedBar:
    """
    使用位运算压缩存储金融时间序列数据的 Bar 对象
    """
    def __init__(self, start_time: int, end_time: int,
                 open_price: float, close_price: float,
                 high_price: float, low_price: float):
        self.start_time = start_time  # Unix 时间戳
        self.end_time = end_time  # Unix 时间戳
        # 压缩价格存储
        self.compressed_prices = self.compress_prices(
            open_price, close_price, high_price, low_price
        )

    @staticmethod
    def compress_prices(open_price: float, close_price: float,
                        high_price: float, low_price: float) -> int:
        def price_to_fixed(price: float) -> int:
            return int(price * 100000)

        compressed = 0
        compressed |= (price_to_fixed(open_price) & 0x3FFFFFFF)
        compressed |= (price_to_fixed(close_price) & 0x3FFFFFFF) << 30
        compressed |= (price_to_fixed(high_price) & 0x3FFFFFFF) << 60
        compressed |= (price_to_fixed(low_price) & 0x3FFFFFFF) << 90
        return compressed

    def decompress_prices(self) -> tuple:
        def fixed_to_price(fixed_price: int) -> float:
            return fixed_price / 100000

        open_price = fixed_to_price(self.compressed_prices & 0x3FFFFFFF)
        close_price = fixed_to_price((self.compressed_prices >> 30) & 0x3FFFFFFF)
        high_price = fixed_to_price((self.compressed_prices >> 60) & 0x3FFFFFFF)
        low_price = fixed_to_price((self.compressed_prices >> 90) & 0x3FFFFFFF)

        return open_price, close_price, high_price, low_price


class CircularBuffer:
    """
    循环数组实现，用于高效管理 Bar 数据
    """
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: List[Optional[CompressedBar]] = [None] * max_size  # 预分配固定大小内存
        self.current_index = 0  # 当前写入位置
        self.size = 0  # 实际存储的元素数量

    def add(self, item: CompressedBar):
        """
        添加新元素，覆盖最旧的数据
        """
        if not isinstance(item, CompressedBar):
            raise ValueError("Item must be of type CompressedBar.")
        self.buffer[self.current_index] = item
        self.current_index = (self.current_index + 1) % self.max_size
        if self.size < self.max_size:
            self.size += 1

    def get(self, index: int) -> CompressedBar:
        """
        根据逻辑索引获取元素
        """
        if index < 0 or index >= self.size:
            raise IndexError("Index out of range")
        actual_index = (self.current_index - self.size + index) % self.max_size
        if self.buffer[actual_index] is None:
            raise ValueError("No CompressedBar found at this index.")
        return self.buffer[actual_index]


class Indicator(ABC):
    """
    指标接口，所有指标类需实现该接口。
    """
    @abstractmethod
    def get_value(self, index: int) -> Decimal:
        pass

    @abstractmethod
    def update(self, bar: CompressedBar):
        pass


class CachedIndicator(Indicator):
    """
    带缓存机制的指标基类。
    """
    def __init__(self, series: 'BarSeries'):
        self.series = series
        self.cached_values: List[Optional[Decimal]] = [None] * series.bars.max_size  # 初始化缓存

    def get_value(self, index: int) -> Decimal:
        if index < 0 or index >= self.series.bars.size:
            raise IndexError("Index out of range.")
        if self.cached_values[index] is None:  # 如果缓存不存在，则计算并存储
            self.cached_values[index] = self.calculate_value(index)
        return self.cached_values[index] if self.cached_values[index] is not None else Decimal(0)  # 确保返回 Decimal 类型

    @abstractmethod
    def calculate_value(self, index: int) -> Decimal:
        pass

    def invalidate_cache(self, start_index: int):
        """
        从指定索引开始，标记缓存为失效。
        """
        for i in range(start_index, len(self.cached_values)):
            self.cached_values[i] = None


class ClosePriceIndicator(CachedIndicator):
    """
    表示收盘价指标。
    """
    def calculate_value(self, index: int) -> Decimal:
        bar = self.series.get_bar(index)
        return Decimal(bar.decompress_prices()[1])  # 收盘价

    def update(self, bar: CompressedBar):
        # 更新逻辑（如果需要）
        pass


class BarSeries:
    """
    表示一个时间序列，存储多个 Bar 对象。
    """
    def __init__(self, max_size: int):
        self.bars = CircularBuffer(max_size)
        self.time_index_map = {}
        self.indicators = []

    def add_bar(self, bar: CompressedBar):
        self.bars.add(bar)
        self.time_index_map[bar.start_time] = self.bars.size - 1
        self.update_indicators(bar)

    def get_bar(self, index: int) -> CompressedBar:
        return self.bars.get(index)

    def update_indicators(self, bar: CompressedBar):
        for indicator in self.indicators:
            indicator.update(bar)

    def add_indicator(self, indicator: Indicator):
        self.indicators.append(indicator)

    def remove_oldest_bar(self):
        # 删除最旧的 Bar 对象
        if self.bars.size > 0:
            oldest_bar = self.bars.get(0)
            del self.time_index_map[oldest_bar.start_time]
            # 不再添加 None
            self.bars.add(oldest_bar)  # 这里可以选择添加一个新的 Bar


class IndicatorWorker(Thread):
    def __init__(self, indicator: Indicator, message_queue: Queue):
        super().__init__()
        self.indicator = indicator
        self.message_queue = message_queue

    def run(self):
        while True:
            bar = self.message_queue.get()
            if bar is None:  # 结束信号
                break
            self.indicator.update(bar)


class BarSeriesWithIndicators(BarSeries):
    def __init__(self, max_size: int):
        super().__init__(max_size)
        self.message_queue = Queue()
        self.workers = []

    def add_indicator(self, indicator: Indicator):
        super().add_indicator(indicator)
        worker = IndicatorWorker(indicator, self.message_queue)
        worker.start()
        self.workers.append(worker)

    def update_series(self, bar: CompressedBar):
        self.add_bar(bar)
        self.message_queue.put(bar)  # 将更新消息放入队列


# 示例使用
if __name__ == "__main__":
    bar_series = BarSeriesWithIndicators(max_size=100)

    # 添加收盘价指标
    close_price_indicator = ClosePriceIndicator(bar_series)
    bar_series.add_indicator(close_price_indicator)

    # 添加一些 Bar 数据
    bar1 = CompressedBar(1, 2, 100.0, 105.0, 110.0, 95.0)
    bar2 = CompressedBar(2, 3, 105.0, 107.0, 112.0, 100.0)
    bar_series.update_series(bar1)
    bar_series.update_series(bar2)

    # 获取收盘价指标值
    print("Close Price of Bar 1:", close_price_indicator.get_value(0))
    print("Close Price of Bar 2:", close_price_indicator.get_value(1))