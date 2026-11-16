"""Allow-list of joins the system will ever perform. Anything else is refused (JOIN_NOT_ALLOWED).
All declared joins are many-to-one from the 'fact' side to a 'dimension' side, so they cannot multiply rows
as long as the dimension key is unique (which safe_left_join verifies on every call)."""
from dataclasses import dataclass

from app.errors import PipelineError


@dataclass(frozen=True)
class Relationship:
    left: str
    right: str
    left_keys: tuple
    right_keys: tuple
    cardinality: str = "many_to_one"


RELATIONSHIPS = [
    Relationship("orders", "products", ("sku",), ("sku",)),
    Relationship("orders", "stores", ("store_id",), ("store_id",)),
    Relationship("orders", "customers", ("customer_id",), ("customer_id",)),
    Relationship("inventory", "products", ("sku",), ("sku",)),
    Relationship("inventory", "stores", ("store_id",), ("store_id",)),
]

# default columns brought in from each dimension dataset (explicit list = no accidental column explosion)
DEFAULT_RIGHT_COLUMNS = {
    "products": ["product_name", "category", "brand", "cost"],
    "stores": ["store_name", "city", "region"],
    "customers": ["city"],
}


def get_relationship(left: str, right: str) -> Relationship:
    for r in RELATIONSHIPS:
        if r.left == left and r.right == right:
            return r
    raise PipelineError("JOIN_NOT_ALLOWED", f"Join {left} -> {right} is not a declared relationship.",
                        {"allowed": [f"{r.left}->{r.right}" for r in RELATIONSHIPS]})
