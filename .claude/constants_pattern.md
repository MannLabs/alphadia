# String Constants Pattern - Instruction Set for Coding Agents

## Purpose

This document provides instructions for introducing string constants for access of
- DataFrame (specifically pd.DataFrame) columns and
- dictionary keys
using the `string constants` pattern.

---

## Motivation

**Problem**: Raw string literals for DataFrame column or dictionary keys access are error-prone and hard to maintain.

```python
# BAD: Magic strings scattered throughout codebase: Typo risk, no IDE support, hard to refactor
some_df["column_name"]
some_dict["key_name"]
```

**Solution**: Centralized, immutable string constants with IDE support.

```python
# GOOD: Centralized constants: IDE autocomplete and navigation, compile-time checking, easy refactoring
some_df[Cols.COLUMN_NAME]
some_dict[Keys.KEY_NAME]
```

**Benefits**:
1. **Single source of truth** - Column names defined once in a constants module
2. **Immutability** - Constants cannot be modified at runtime
3. **IDE support** - Autocomplete, "find usages", safe renaming
4. **Runtime validation** - `get_values()` method for validating against allowed values
5. **Discoverability** - Related constants grouped in semantic classes

---

## Pattern Implementation

The `ConstantsClass` metaclass is required to enforce immutability and provide utility methods.
Add it if it does not already exist in your codebase.

```python
class ConstantsClass(type):
    """A metaclass for classes that should only contain string constants."""

    def __setattr__(self, name, value):
        raise TypeError("Constants class cannot be modified")

    def get_values(cls):
        """Get all user-defined string values of the class."""
        return [
            value
            for key, value in cls.__dict__.items()
            if not key.startswith("__") and isinstance(value, str)
        ]
```

---

## How to Apply This Pattern

### Step 1: Identify Magic Strings

Search for raw string DataFrame access patterns:
```python
# Patterns to find and replace:
df["column_name"]           # Bracket access with string literal
df.column_name              # Attribute access (also problematic)
some_dict["key_name"]    # Dict access for column configs
```

### Step 2: Group Related Constants

Create a new class for semantically related columns. Naming convention: `<Domain>Cols` or `<Domain>Keys`.

```python
class UserCols(metaclass=ConstantsClass):
    """String constants for user table columns."""

    ID = "id"
    NAME = "name"
    EMAIL = "email"
```

### Step 3: Add to Constants Module

Place the new class in the project's constants module (e.g., `constants/keys.py`), grouped with related classes.

### Step 4: Replace Magic Strings

```python
# BEFORE
df["id"]
df["name"]

if "name" in df.columns:
    ...


# AFTER
from constants.keys import UserCols

df[UserCols.ID]
df[UserCols.NAME]

# make sure to also capture occurrences like this:
if UserCols.NAME in df.columns:
    ...
```

Important note:
The constants have dataframe-type scope, so don't blindly replace all strings across dataframes. Only replace those relevant to the specific dataframe.
There can well be cases where two dataframes share the same column name, but you should only replace the string literal in the context of the specific dataframe.

Example:
```python

# BEFORE
user_df = user_df[
    user_df["id"].isin(some_other_df["id"])
]


# AFTER: WRONG - DO NOT DO THIS
user_df = user_df[
    user_df[UserCols.ID].isin(some_other_df[UserCols.ID])
]

# AFTER: RIGHT - DO THIS - OPTION 1/2
user_df = user_df[
    user_df[UserCols.ID].isin(some_other_df["id"])
]

# AFTER: RIGHT - DO THIS - OPTION 2/2
user_df = user_df[
    user_df[UserCols.ID].isin(some_other_df[SomeOtherCols.ID])
]
```

### Step 5: Use for Validation (optional)

```python
if column not in UserCols.get_values():
    raise ValueError(f"Invalid column: {column}. Valid: {UserCols.get_values()}")
```

---

## Rules and Conventions

1. **Location**: All constant classes go in a dedicated constants module
2. **Naming**:
   - Class: `<Domain>Cols` for DataFrame columns, `<Domain>Keys` for config/dict keys
   - Constants: `UPPER_SNAKE_CASE`
   - Values: `lower_snake_case` strings
3. **Docstring**: Every class must have a docstring explaining its purpose
4. **Grouping**: Group related constants logically, e.g. for a specific type of DataFrame. Duplications (e.g. `Cols.ID` in case several dataframes share the same column) are allowed.
5. **Bracket notation**: Always use `df[ConstantClass.COLUMN]`, never attribute access `df.column`

---

## Examples

### Example 1: DataFrame Columns

```python
class OrderCols(metaclass=ConstantsClass):
    """String constants for order table columns."""

    ORDER_ID = "order_id"
    CUSTOMER_ID = "customer_id"
    TOTAL = "total"
    STATUS = "status"
```

**Usage**:
```python
from constants.keys import OrderCols

# DataFrame access
orders_df[OrderCols.ORDER_ID].values
orders_df[OrderCols.TOTAL].sum()

# Configuration
config = {
    "group_by": [OrderCols.CUSTOMER_ID],
    "aggregate": [OrderCols.TOTAL],
}
```

### Example 2: Allowed Values with Validation

```python
class OrderStatus(metaclass=ConstantsClass):
    """String constants for order status values."""

    PENDING = "pending"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
```

**Usage with validation**:
```python
from constants.keys import OrderStatus

if status not in OrderStatus.get_values():
    raise ValueError(
        f"Unknown status: {status}. "
        f"Valid options are {OrderStatus.get_values()}"
    )
```


---

## Checklist for Adding New Constants

- [ ] Identified all magic strings to replace
- [ ] Created class with `metaclass=ConstantsClass`
- [ ] Added descriptive docstring
- [ ] Used `UPPER_SNAKE_CASE` for constant names
- [ ] Used `lower_snake_case` for string values
- [ ] Added class to constants module
- [ ] Updated imports in affected files
- [ ] Replaced all magic strings with constant references
- [ ] Used bracket notation `df[Const.COL]` not attribute access
