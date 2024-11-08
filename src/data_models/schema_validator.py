from collections import Counter
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ValidationError, validator, field_validator


class ID(BaseModel):
    """
    A model representing the ID field of the dataset.
    """

    name: str
    description: str


class Target(BaseModel):
    """
    A model representing the target field of a regression problem.
    """

    name: str
    description: str
    example: float


class DataType(str, Enum):
    """Enum for the data type of feature"""

    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"


class Feature(BaseModel):
    """
    A model representing the predictor fields in the dataset. Validates the
    presence and type of the 'example' field based on the 'dataType' field
    for NUMERIC dataType and presence and contents of the 'categories' field
    for CATEGORICAL dataType.
    """

    name: str
    description: str
    dataType: DataType
    nullable: bool
    example: Optional[float] = None
    categories: Optional[List[str]] = None

    @field_validator("example")
    def example_is_present_with_data_type_is_numeric(cls, v, values):
        data_type = values.data.get("dataType")
        if data_type == "NUMERIC" and v is None:
            raise ValueError(
                f"`example` must be present and a float or an integer "
                f"when dataType is NUMERIC. Check field: {values}"
            )
        return v

    @field_validator("categories")
    def categories_are_present_with_data_type_is_categorical(cls, v, values):
        data_type = values.data.get("dataType")
        if data_type == "CATEGORICAL" and v is None:
            raise ValueError(
                "`categories` must be present when dataType is CATEGORICAL. "
                f"Check field: {values}"
            )
        return v

    @field_validator("categories")
    def categories_are_non_empty_strings(cls, v, values):
        categories = values.data.get("categories")
        if categories is not None:
            if len(categories) == 0:
                raise ValueError(
                    f"`categories` must not be empty. Check field: {values}"
                )
            for category in categories:
                if str(category) == "" or not isinstance(category, str):
                    raise ValueError(
                        f"`categories` must be a list of strings. Check field: {values}"
                    )
        return v


class SchemaModel(BaseModel):
    """
    A schema validator for regression problems. Validates the
    problem category, version, and predictor fields of the input schema.
    """

    title: str
    description: str = None
    modelCategory: str
    schemaVersion: float
    inputDataFormat: str
    encoding: str
    id: ID
    target: Target
    features: List[Feature]

    @field_validator("modelCategory")
    def valid_problem_category(cls, v):
        if v != "regression":
            raise ValueError(f"modelCategory must be 'regression'. Given {v}")
        return v

    @field_validator("schemaVersion")
    def valid_version(cls, v):
        if v != 1.0:
            raise ValueError(f"schemaVersion must be set to 1.0. Given {v}")
        return v

    @field_validator("features")
    def at_least_one_predictor_field(cls, v):
        if len(v) < 1:
            raise ValueError(
                f"features must have at least one field defined. Given {v}"
            )
        return v

    @field_validator("features")
    def unique_feature_names(cls, v):
        """
        Check that the feature names are unique.
        """
        feature_names = [feature.name for feature in v]
        duplicates = [
            item for item, count in Counter(feature_names).items() if count > 1
        ]

        if duplicates:
            raise ValueError(
                "Duplicate feature names found in schema: " f"`{', '.join(duplicates)}`"
            )

        return v


def validate_schema_dict(schema_dict: dict) -> dict:
    """
    Validate the schema
    Args:
        schema_dict: dict
            data schema as a python dictionary

    Raises:
        ValueError: if the schema is invalid

    Returns:
        dict: validated schema as a python dictionary
    """
    try:
        schema_dict = SchemaModel.model_validate(schema_dict).model_dump()
        return schema_dict
    except ValidationError as exc:
        raise ValueError(f"Invalid schema: {exc}") from exc
