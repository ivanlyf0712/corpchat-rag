"""
CorpChat Search — Temporal Query Parsing
==========================================
Detects time windows in user queries for time-sensitive retrieval,
aligned with the Hindsight temporal-retrieval component.

Design (grilled + spec .scratch/hindsight-temporal):
  - Rule-first (<1ms): 最近N天/周/月/年, 最近, 前天/昨天/今天,
    本周/上周, 本月/这个月/上月/上个月, 今年/去年, absolute dates.
  - LLM fallback only fires when a time keyword is present but no rule
    matched; returns None gracefully when the LLM is unavailable.
  - Returns None when no time intent, so non-temporal retrieval behavior
    is completely unchanged.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

from .config import LITELLM_MODEL, TEMPORAL_DEFAULT_WINDOW_DAYS, logger
from .litellm_client import LiteLLMClient


@dataclass
class TimeWindow:
    """Detected time window (ISO date strings; None = open bound)."""

    start: Optional[str]
    end: Optional[str]
    matched: str


# 时间关键词启发: 命中才考虑时序意图, 避免无意义 LLM 回退
_TIME_CHARS = set("天周月年日時时")

# 相对单位 -> 天数
_UNIT_DAYS = {
    "天": 1,
    "日": 1,
    "周": 7,
    "个星期": 7,
    "星期": 7,
    "个月": 30,
    "月": 30,
    "年": 365,
}


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _day_iso(d) -> str:
    return _iso(datetime.combine(d, datetime.min.time()))


class TimeExpressionParser:
    """从查询中解析时间窗口。规则优先 (<1ms)，LLM 回退。"""

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = LITELLM_MODEL,
        allow_llm: bool = True,
    ):
        self._client = LiteLLMClient(api_base=api_base, api_key=api_key, model=model)
        self._cache: Dict[str, Optional[TimeWindow]] = {}
        # allow_llm=False → 纯规则解析 (确定性, 无 API 调用)。用于 eval/answer path
        # 等需要可复现结果的场景; 规则未命中的查询直接返回 None。
        self.allow_llm = allow_llm

    def parse(self, query: str, now: Optional[datetime] = None) -> Optional[TimeWindow]:
        """返回查询中的时间窗口; 无时间意图返回 None (不改变检索行为)。"""
        if not query:
            return None
        cache_key = query[:100]
        if cache_key in self._cache:
            return self._cache[cache_key]
        now = now or datetime.now()

        window = self._parse_rules(query, now)
        if window is None and self.allow_llm and self._needs_llm(query):
            window = self._parse_llm(query, now)

        self._cache[cache_key] = window
        return window

    @staticmethod
    def _needs_llm(query: str) -> bool:
        """仅当查询含时间词且有数字或范围连接词时才尝试 LLM 回退。

        避免对规则已覆盖的裸时间词 (上周/最近等) 及无关查询发起 API 调用。
        """
        has_time_char = any(c in query for c in _TIME_CHARS)
        has_range = any(w in query for w in ("之前", "以前", "之后", "以来", "期间", "从", "到", "至"))
        has_digit = any(c.isdigit() for c in query)
        return has_time_char and (has_range or has_digit)

    # ── 规则优先 (<1ms) ───────────────────────────────────────
    def _parse_rules(self, query: str, now: datetime) -> Optional[TimeWindow]:
        today = now.date()

        # 最近/近/前 N 单位 (最具体优先)
        m = re.search(r"(最近|近|前)\s*(\d+)\s*(个星期|星期|个月|天|日|周|月|年)", query)
        if m:
            days = int(m.group(2)) * _UNIT_DAYS.get(m.group(3), 1)
            return TimeWindow(
                start=_iso(now - timedelta(days=days)),
                end=_iso(now),
                matched=m.group(0),
            )

        # 最近 / 近 (bare)
        m = re.search(r"(最近|近)", query)
        if m:
            return TimeWindow(
                start=_iso(now - timedelta(days=TEMPORAL_DEFAULT_WINDOW_DAYS)),
                end=_iso(now),
                matched=m.group(0),
            )

        # 前天 / 昨天 / 今天
        m = re.search(r"(前天|昨天|今天)", query)
        if m:
            w = m.group(0)
            if w == "前天":
                start, end = today - timedelta(days=2), today - timedelta(days=1)
            elif w == "昨天":
                start, end = today - timedelta(days=1), today
            else:  # 今天
                start = end = today
            return TimeWindow(start=_day_iso(start), end=_day_iso(end), matched=w)

        # 上周 / 本周 / 这周
        m = re.search(r"(上周|本周|这周)", query)
        if m:
            monday = today - timedelta(days=today.weekday())
            if m.group(0) == "上周":
                start, end = monday - timedelta(days=7), monday - timedelta(days=1)
            else:
                start, end = monday, today
            return TimeWindow(start=_day_iso(start), end=_day_iso(end), matched=m.group(0))

        # 上个月 / 这个月 / 上月 / 本月
        m = re.search(r"(上个月|这个月|上月|本月)", query)
        if m:
            first = today.replace(day=1)
            if m.group(0) in ("上个月", "上月"):
                end = first - timedelta(days=1)
                start = end.replace(day=1)
            else:  # 这个月 / 本月
                start, end = first, today
            return TimeWindow(start=_day_iso(start), end=_day_iso(end), matched=m.group(0))

        # 去年 / 今年
        m = re.search(r"(去年|今年)", query)
        if m:
            jan1 = today.replace(month=1, day=1)
            if m.group(0) == "去年":
                start = jan1.replace(year=jan1.year - 1)
                end = jan1 - timedelta(days=1)
            else:  # 今年
                start, end = jan1, today
            return TimeWindow(start=_day_iso(start), end=_day_iso(end), matched=m.group(0))


        # 绝对日期 (最具体优先)
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", query)
        if m:
            d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return TimeWindow(start=_iso(d), end=_iso(d), matched=m.group(0))

        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", query)
        if m:
            d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return TimeWindow(start=_iso(d), end=_iso(d), matched=m.group(0))

        # 裸 YYYY-MM (2026-07): 整月窗口。必须放在 YYYY-MM-DD 之后 (更具体的规则优先),
        # 否则 "2026-07-01" 会被裸月规则误吞成整月窗口。
        m = re.search(r"(\d{4})-(\d{1,2})", query)
        if m:
            d = datetime(int(m.group(1)), int(m.group(2)), 1)
            end_d = (d.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            return TimeWindow(start=_iso(d), end=_iso(end_d), matched=m.group(0))

        m = re.search(r"(\d{4})年(\d{1,2})月", query)
        if m:
            d = datetime(int(m.group(1)), int(m.group(2)), 1)
            end_d = (d.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            return TimeWindow(start=_iso(d), end=_iso(end_d), matched=m.group(0))

        m = re.search(r"(\d{4})年", query)
        if m:
            d = datetime(int(m.group(1)), 1, 1)
            end_d = datetime(int(m.group(1)) + 1, 1, 1) - timedelta(days=1)
            return TimeWindow(start=_iso(d), end=_iso(end_d), matched=m.group(0))

        # M月D日 (无年份: 若未来则今年, 否则去年)
        m = re.search(r"(\d{1,2})月(\d{1,2})日", query)
        if m:
            d = datetime(now.year, int(m.group(1)), int(m.group(2)))
            if d.date() < today:
                d = d.replace(year=now.year - 1)
            return TimeWindow(start=_iso(d), end=_iso(d), matched=m.group(0))

        return None

    # ── LLM 回退 (仅时间关键词存在但规则未命中时) ──────────────
    def _parse_llm(self, query: str, now: datetime) -> Optional[TimeWindow]:
        try:
            result = self._client.chat(
                [
                    {
                        "role": "user",
                        "content": (
                            "Extract the time range from this query. Reply ONLY a JSON object "
                            '{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} with the narrower '
                            "bound if only one side is given; reply {} if there is no time range. "
                            f"Query: {query}"
                        ),
                    }
                ],
                temperature=0,
                max_tokens=80,
                timeout=8,
            )
            if not result:
                return None
            data = json.loads(result)
            start = data.get("start") or None
            end = data.get("end") or None
            if not start and not end:
                return None
            return TimeWindow(start=start, end=end, matched=query)
        except Exception as e:
            logger.debug(f"时序 LLM 解析失败: {e}")
            return None

