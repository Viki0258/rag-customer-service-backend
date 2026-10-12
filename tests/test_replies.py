from dataclasses import FrozenInstanceError

import pytest

from whatsapp_support.replies import ReplyComposer, ReplyDraft


def test_reply_draft_is_immutable_and_keeps_locale() -> None:
    draft = ReplyDraft(text="In stock", locale="en")

    assert draft.text == "In stock"
    assert draft.locale == "en"
    with pytest.raises(FrozenInstanceError):
        draft.text = "changed"  # type: ignore[misc]


def test_reply_composer_rejects_missing_required_tool_fields() -> None:
    assert ReplyComposer().compose(
        "inventory", {"product_id": "sku-1", "available": True}, "en"
    ) is None


def test_reply_composer_formats_inventory_in_both_supported_locales() -> None:
    data = {
        "product_id": "sku-1",
        "name": "Everyday Backpack",
        "available": True,
        "quantity": 7,
    }

    assert ReplyComposer().compose("inventory", data, "en").text == (
        "Everyday Backpack is in stock (7 available)."
    )
    assert ReplyComposer().compose("inventory", data, "zh-CN").text == (
        "Everyday Backpack 目前有库存，可售 7 件。"
    )


def test_reply_composer_formats_product_and_protected_results() -> None:
    composer = ReplyComposer()

    product = composer.compose(
        "product_info",
        {
            "product_id": "sku-1",
            "name": "Everyday Backpack",
            "price": "49.00 USD",
            "description": "A durable everyday backpack.",
        },
        "en",
    )
    order = composer.compose(
        "order_status",
        {"order_id": "order-1", "status": "shipped", "updated_at": "today"},
        "en",
    )

    assert product is not None
    assert product.text == (
        "Everyday Backpack costs 49.00 USD. A durable everyday backpack."
    )
    assert order is not None
    assert "order-1" in order.text
    assert "shipped" in order.text


def test_reply_text_does_not_expose_internal_tool_fields() -> None:
    draft = ReplyComposer().compose(
        "product_info",
        {
            "product_id": "sku-1",
            "name": "Item",
            "price": "1 USD",
            "description": "Safe",
            "internal_cost": "0.1",
            "token": "secret",
        },
        "en",
    )

    assert draft is not None
    assert "internal_cost" not in draft.text
    assert "secret" not in draft.text


def test_unknown_locale_falls_back_to_english_copy() -> None:
    draft = ReplyComposer().compose(
        "inventory",
        {
            "product_id": "sku-1",
            "name": "Everyday Backpack",
            "available": True,
            "quantity": 7,
        },
        "fr",
    )

    assert draft is not None
    assert draft.locale == "en"
    assert draft.text == "Everyday Backpack is in stock (7 available)."
