from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReplyDraft:
    text: str
    locale: str


class ReplyComposer:
    def compose(
        self,
        intent: str,
        output_data: dict[str, object],
        locale: str,
    ) -> ReplyDraft | None:
        normalized_locale = "zh-CN" if locale == "zh-CN" else "en"
        if intent == "product_info":
            name = _text(output_data, "name")
            price = _text(output_data, "price")
            description = _text(output_data, "description")
            if not all((name, price, description, _text(output_data, "product_id"))):
                return None
            text = (
                f"{name} 售价 {price}。{description}"
                if normalized_locale == "zh-CN"
                else f"{name} costs {price}. {description}"
            )
            return ReplyDraft(text=text, locale=normalized_locale)

        if intent == "inventory":
            name = _text(output_data, "name")
            product_id = _text(output_data, "product_id")
            available = output_data.get("available")
            quantity = output_data.get("quantity")
            if not name or not product_id or not isinstance(available, bool):
                return None
            if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 0:
                return None
            if normalized_locale == "zh-CN":
                text = (
                    f"{name} 目前有库存，可售 {quantity} 件。"
                    if available
                    else f"{name} 目前缺货。"
                )
            else:
                text = (
                    f"{name} is in stock ({quantity} available)."
                    if available
                    else f"{name} is currently out of stock."
                )
            return ReplyDraft(text=text, locale=normalized_locale)

        if intent == "order_status":
            order_id = _text(output_data, "order_id")
            status = _text(output_data, "status")
            updated_at = _text(output_data, "updated_at")
            if not all((order_id, status, updated_at)):
                return None
            text = (
                f"订单 {order_id} 当前状态为 {status}，更新时间 {updated_at}。"
                if normalized_locale == "zh-CN"
                else f"Order {order_id} is {status}. Last updated: {updated_at}."
            )
            return ReplyDraft(text=text, locale=normalized_locale)

        if intent == "shipping":
            values = tuple(
                _text(output_data, key)
                for key in ("order_id", "carrier", "tracking_number", "status")
            )
            if not all(values):
                return None
            order_id, carrier, tracking_number, status = values
            text = (
                f"订单 {order_id} 由 {carrier} 配送，运单号 {tracking_number}，当前状态 {status}。"
                if normalized_locale == "zh-CN"
                else f"Order {order_id} ships with {carrier}. Tracking {tracking_number}; status: {status}."
            )
            return ReplyDraft(text=text, locale=normalized_locale)

        if intent == "return_eligibility":
            order_id = _text(output_data, "order_id")
            eligible = output_data.get("eligible")
            reason = _text(output_data, "reason")
            if not order_id or not isinstance(eligible, bool) or not reason:
                return None
            if normalized_locale == "zh-CN":
                text = f"订单 {order_id}{'符合' if eligible else '不符合'}退货条件：{reason}"
            else:
                text = f"Order {order_id} is {'eligible' if eligible else 'not eligible'} for return: {reason}"
            return ReplyDraft(text=text, locale=normalized_locale)
        return None


def _text(output_data: dict[str, object], key: str) -> str | None:
    value = output_data.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()
