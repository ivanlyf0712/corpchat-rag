"""
CorpChat Search — Persona (CARA-style disposition layer)
==========================================================
Tunable disposition traits (skepticism / literality / empathy / style) that
condition answer generation via system-prompt injection at the answer points.

Prompt conditioning only — no fine-tuning, no new runtime services. The neutral
default (0.5, balanced) appends no substantive instructions, so existing answers
are unchanged unless a profile is tuned.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class DispositionProfile:
    """CARA disposition profile: 0..1 traits + output style.

    - skepticism: 对检索证据不足的结论标注不确定性, 避免臆断
    - literality: 严格依据检索原文回答, 不添加推测
    - empathy:    先回应情绪/语气再给信息, 温和体谅
    - style:      concise / balanced / detailed
    """

    skepticism: float = 0.5
    literality: float = 0.5
    empathy: float = 0.5
    style: str = "balanced"

    # 低/中/高 阈值: 低于 LOW 或高于 HIGH 才追加对应指令
    LOW = 0.35
    HIGH = 0.65

    def _instructions(self) -> str:
        lines = []
        if self.skepticism >= self.HIGH:
            lines.append("- 若检索证据不足, 明确标注不確定性, 避免臆斷結論")
        elif self.skepticism <= self.LOW:
            lines.append("- 基於檢索到的證據直接給出結論, 不必反覆標註不確定性")
        if self.literality >= self.HIGH:
            lines.append("- 嚴格依據檢索到的原文回答, 不添加推測或引申")
        elif self.literality <= self.LOW:
            lines.append("- 可在原文基礎上做合理的概括與引申, 無需逐字引用")
        if self.empathy >= self.HIGH:
            lines.append("- 以溫和、體諒的語氣回答, 先回應對方情緒再給信息")
        elif self.empathy <= self.LOW:
            lines.append("- 直接、簡潔地給出信息, 不過度修飾語氣")
        if self.style == "concise":
            lines.append("- 回答精簡: 用最短篇幅給出結論與要點")
        elif self.style == "detailed":
            lines.append("- 回答詳細: 展開說明背景、要點與後續建議")
        return "\n".join(lines)

    def build_system_prompt(self, base_prompt: str) -> str:
        """在基础 system prompt 后追加性格指令 (中性默认无实质追加)。"""
        if not base_prompt:
            return base_prompt
        instructions = self._instructions()
        if not instructions:
            return base_prompt
        return f"{base_prompt}\n\n回答風格 (依設定的人格):\n{instructions}"

    def to_dict(self) -> Dict:
        return {
            "skepticism": self.skepticism,
            "literality": self.literality,
            "empathy": self.empathy,
            "style": self.style,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DispositionProfile":
        return cls(
            skepticism=float(d.get("skepticism", 0.5)),
            literality=float(d.get("literality", 0.5)),
            empathy=float(d.get("empathy", 0.5)),
            style=str(d.get("style", "balanced")),
        )
