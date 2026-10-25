"""Offline structured-data access for the ZentroMart support agent.

This module owns CSV loading, relational joins and date/serviceability
calculations.  It deliberately does not perform identity verification; the
offline ``tools`` package is responsible for releasing private order details
after verification.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


REFERENCE_DATE = date(2026, 10, 20)
"""Fixed "today" from the synthetic capstone dataset."""

BUSINESS_WEEKDAYS = frozenset(range(6))
"""Monday through Saturday; Sunday is not a business day."""
TABLE_NAMES = (
    "customers",
    "orders",
    "order_items",
    "products",
    "pincodes",
    "holidays",
    "support_logs",
)
_TRUE_VALUES = {"true", "1", "yes", "y"}


def _default_data_dir() -> Path:
    """Return the supplied capstone dataset's data directory."""

    return (
        Path(__file__).resolve().parents[2]
        / "Capstone_Project_Dataset_FTTT_2026"
        / "data"
    )


def _parse_date(value: date | datetime | str) -> date:
    """Parse a date or ISO timestamp, raising a useful error for bad input."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("A non-empty date value is required.")
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"Invalid date value: {value!r}.") from exc


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a frame to JSON-friendly records without changing CSV values."""

    return frame.to_dict(orient="records")


class DataStore:
    """Load and query all structured capstone data.

    All source columns are loaded as strings and empty CSV cells remain empty
    strings.  Numeric conversion is performed only for calculations, so IDs,
    pincodes, SKUs and refund references cannot lose leading zeroes.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        reference_date: date | datetime | str = REFERENCE_DATE,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else _default_data_dir()
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Structured data directory does not exist: {self.data_dir}"
            )
        self.reference_date = _parse_date(reference_date)
        self.tables = {
            name: self._read_table(name)
            for name in TABLE_NAMES
        }
        self.customers = self.tables["customers"]
        self.orders = self.tables["orders"]
        self.order_items = self.tables["order_items"]
        self.products = self.tables["products"]
        self.pincodes = self.tables["pincodes"]
        self.holidays = self.tables["holidays"]
        self.support_logs = self.tables["support_logs"]
        self._holiday_dates = {
            _parse_date(value)
            for value in self.holidays["date"].tolist()
            if str(value).strip()
        }

    def _read_table(self, name: str) -> pd.DataFrame:
        path = self.data_dir / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required structured-data file is missing: {path}")
        return pd.read_csv(path, dtype=str, keep_default_na=False)

    @staticmethod
    def _first(frame: pd.DataFrame) -> dict[str, Any] | None:
        if frame.empty:
            return None
        return _records(frame.iloc[[0]])[0]

    def find_customer(
        self,
        customer_id: str | None = None,
        *,
        phone: str | None = None,
        email: str | None = None,
    ) -> dict[str, Any] | None:
        """Find one customer by ID, phone or email."""

        filters: list[pd.Series] = []
        if customer_id is not None:
            filters.append(self.customers["customer_id"] == str(customer_id).strip())
        if phone is not None:
            filters.append(self.customers["phone"] == str(phone).strip())
        if email is not None:
            filters.append(
                self.customers["email"].str.casefold() == str(email).strip().casefold()
            )
        if not filters:
            raise ValueError("Provide customer_id, phone or email.")
        matches = self.customers
        for condition in filters:
            matches = matches[condition]
        return self._first(matches)

    def find_product(
        self,
        sku: str | None = None,
        *,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find products by exact SKU or case-insensitive catalog text."""

        if sku is not None:
            matches = self.products[
                self.products["sku"].str.casefold() == str(sku).strip().casefold()
            ]
        elif query is not None and str(query).strip():
            terms = str(query).strip().casefold().split()
            searchable = (
                self.products["sku"] + " " + self.products["title"] + " "
                + self.products["brand"] + " " + self.products["description"]
            ).str.casefold()
            mask = pd.Series(True, index=self.products.index)
            for term in terms:
                mask &= searchable.str.contains(term, regex=False)
            matches = self.products[mask]
        else:
            raise ValueError("Provide sku or a non-empty query.")
        return _records(matches)

    def get_order_items(
        self,
        order_id: str,
        *,
        include_product: bool = True,
    ) -> list[dict[str, Any]]:
        """Return order lines, optionally enriched with catalog fields."""

        items = self.order_items[
            self.order_items["order_id"] == str(order_id).strip()
        ].copy()
        if include_product:
            product_columns = [
                "sku", "brand", "fulfilled_by", "assured",
                "warranty_months", "description",
            ]
            items = items.merge(
                self.products[product_columns],
                on="sku",
                how="left",
                suffixes=("", "_product"),
            )
        return _records(items)

    def get_order(self, order_id: str, *, include_items: bool = True) -> dict[str, Any] | None:
        """Return an order enriched with its customer, pincode and item rows."""

        order = self._first(
            self.orders[self.orders["order_id"] == str(order_id).strip()]
        )
        if order is None:
            return None

        customer = self.find_customer(order["customer_id"])
        pincode = self.check_pincode(order["delivery_pincode"])
        result: dict[str, Any] = {
            "order": order,
            "customer": customer,
            "delivery": pincode,
        }
        if include_items:
            result["items"] = self.get_order_items(order["order_id"])
        return result

    def check_pincode(self, pincode: str) -> dict[str, Any] | None:
        """Return delivery, COD and Express flags for a pincode."""

        match = self.pincodes[
            self.pincodes["pincode"] == str(pincode).strip()
        ]
        return self._first(match)

    def is_serviceable(self, pincode: str) -> bool:
        record = self.check_pincode(pincode)
        return bool(record and record["serviceable"].casefold() in _TRUE_VALUES)

    def cod_available(self, pincode: str) -> bool:
        record = self.check_pincode(pincode)
        return bool(record and record["cod_available"].casefold() in _TRUE_VALUES)

    def express_available(self, pincode: str) -> bool:
        record = self.check_pincode(pincode)
        return bool(record and record["express_available"].casefold() in _TRUE_VALUES)

    def calendar_days_between(
        self,
        start: date | datetime | str,
        end: date | datetime | str,
    ) -> int:
        """Return elapsed calendar days from ``start`` to ``end``."""

        return (_parse_date(end) - _parse_date(start)).days

    def business_days_between(
        self,
        start: date | datetime | str,
        end: date | datetime | str,
    ) -> int:
        """Count Monday-Saturday, non-holiday days in ``(start, end]``.

        The interval is signed, matching ``calendar_days_between``.  This is
        useful for refund and delivery deadlines where the starting event day
        is not counted as an elapsed business day.
        """

        start_date = _parse_date(start)
        end_date = _parse_date(end)
        if start_date == end_date:
            return 0
        sign = 1 if end_date > start_date else -1
        lower, upper = sorted((start_date, end_date))
        count = sum(
            1
            for offset in range(1, (upper - lower).days + 1)
            if (candidate := lower + timedelta(days=offset)).weekday() != 6
            and candidate not in self._holiday_dates
        )
        return sign * count

    def is_business_day(self, value: date | datetime | str) -> bool:
        """Return whether a date is Monday-Saturday and not a listed holiday."""

        candidate = _parse_date(value)
        return (
            candidate.weekday() in BUSINESS_WEEKDAYS
            and candidate not in self._holiday_dates
        )

    def return_deadline(
        self,
        delivered_on: date | datetime | str,
        window_days: int,
    ) -> date:
        """Calculate a return deadline using calendar days, including day one."""

        if window_days < 0:
            raise ValueError("window_days must not be negative.")
        return _parse_date(delivered_on) + timedelta(days=window_days)

    def business_day_deadline(
        self,
        started_on: date | datetime | str,
        business_days: int,
    ) -> date:
        """Calculate a deadline by adding business days after a start date."""

        if business_days < 0:
            raise ValueError("business_days must not be negative.")
        candidate = _parse_date(started_on)
        remaining = business_days
        while remaining:
            candidate += timedelta(days=1)
            if self.is_business_day(candidate):
                remaining -= 1
        return candidate

    def days_since(self, value: date | datetime | str) -> int:
        """Return calendar days elapsed from a value until the dataset date."""

        return self.calendar_days_between(value, self.reference_date)

    def business_days_since(self, value: date | datetime | str) -> int:
        """Return business days elapsed from a value until the dataset date."""

        return self.business_days_between(value, self.reference_date)

    def orders_for_customer(self, customer_id: str) -> list[dict[str, Any]]:
        """Return all orders for a customer, newest first by placed timestamp."""

        matches = self.orders[
            self.orders["customer_id"] == str(customer_id).strip()
        ].sort_values("placed_at", ascending=False)
        return _records(matches)

    def support_logs_for(
        self,
        *,
        customer_id: str | None = None,
        order_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return historical support logs filtered by customer and/or order."""

        if customer_id is None and order_id is None:
            raise ValueError("Provide customer_id or order_id.")
        matches = self.support_logs
        if customer_id is not None:
            matches = matches[matches["customer_id"] == str(customer_id).strip()]
        if order_id is not None:
            matches = matches[matches["order_id"] == str(order_id).strip()]
        return _records(matches)


__all__ = [
    "BUSINESS_WEEKDAYS",
    "DataStore",
    "REFERENCE_DATE",
    "TABLE_NAMES",
]


if __name__ == "__main__":
    store = DataStore()
    print(
        f"Loaded {len(store.customers)} customers, "
        f"{len(store.orders)} orders and {len(store.products)} products."
    )
