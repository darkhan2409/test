"""Декларативные схемы препроцессинга — §4, §5, §6, §11.

Модели проверяют инварианты регламента на границе: если объект собрался,
он соответствует контракту, и дальше по цепочке проверять это заново не нужно.
"""

from .constants import (
    DELTA_FIELD,
    EVENT_TYPE_FIELD,
    MISSING,
    RESERVED_FIELD_NAMES,
)
from .event import CalendarTimeFeatures, CanonicalEvent, FieldValue, SourceMeta
from .feature_schema import (
    EventFeatureSchema,
    FeatureSchema,
    FieldSpec,
    FieldType,
    HighCardinalityPolicy,
    ProfileFeatureSchema,
    VocabularyPolicy,
)
from .profile import CanonicalProfile
from .source_contract import (
    ColumnSpec,
    ColumnType,
    CorrectionReversalSpec,
    DeleteRule,
    LateArrivingPolicy,
    PiiClass,
    SourceContract,
    SourceContractRegistry,
    SourceKind,
    TimeFieldSpec,
    TimestampKind,
    TimezonePolicy,
    TimezoneSpec,
    UnknownFieldPolicy,
    UpdateRule,
    load_source_contracts,
)

__all__ = [
    "CalendarTimeFeatures",
    "CanonicalEvent",
    "CanonicalProfile",
    "ColumnSpec",
    "ColumnType",
    "CorrectionReversalSpec",
    "DELTA_FIELD",
    "DeleteRule",
    "EVENT_TYPE_FIELD",
    "EventFeatureSchema",
    "FeatureSchema",
    "FieldSpec",
    "FieldType",
    "FieldValue",
    "HighCardinalityPolicy",
    "LateArrivingPolicy",
    "MISSING",
    "PiiClass",
    "ProfileFeatureSchema",
    "RESERVED_FIELD_NAMES",
    "SourceContract",
    "SourceContractRegistry",
    "SourceKind",
    "SourceMeta",
    "TimeFieldSpec",
    "TimestampKind",
    "TimezonePolicy",
    "TimezoneSpec",
    "UnknownFieldPolicy",
    "UpdateRule",
    "VocabularyPolicy",
    "load_source_contracts",
]
