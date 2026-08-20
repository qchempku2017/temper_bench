import json
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Self
from uuid import UUID, uuid5

from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    model_serializer,
    model_validator,
)
from monty.json import MSONable


class MSONableModel(BaseModel, MSONable):
    """Pydantic model subclass to allow saving and loading from `json` files."""

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """Use Pydantic's schema generation instead of Monty's legacy hook."""
        return handler(core_schema)

    def as_dict(self):
        """Return the model as a dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, d):
        """Create a model from a dictionary."""
        return cls.model_validate(d)


def canonical_uuid(namespace: UUID, payload: Any) -> UUID:
    """Return a deterministic UUIDv5 for a JSON-compatible payload.

    Compact, key-sorted JSON is used as the UUID name so identity does not
    depend on dictionary insertion order or presentation-only whitespace.
    Callers are responsible for normalizing collections whose order is not
    semantically significant before passing them here.
    """
    canonical_payload = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return uuid5(namespace, canonical_payload)


class ManagedIdentityModel(MSONableModel):
    """MSONable model with a stored, content-derived identity lifecycle.

    Subclasses declaratively specify the stored identity field, UUIDv5
    namespace, payload schema, and identity-source attribute paths. Dotted
    paths select nested attributes. The base class constructs the canonical
    payload, verifies or initializes the ID, serializes it as a UUID string,
    and regenerates it after validated source-field reassignment.

    Model-wide consistency checks that must precede identity verification
    belong in :meth:`_validate_before_identity`. Managed-identity subclasses
    must not add independent ``mode="after"`` or ``mode="wrap"`` model
    validators, because their ordering relative to this inherited finalizer
    would be unclear.
    """

    model_config = ConfigDict(validate_assignment=True)

    _identity_initialized: bool = PrivateAttr(default=False)
    _IDENTITY_FIELD_NAME: ClassVar[str]
    _IDENTITY_SOURCE_FIELDS: ClassVar[tuple[str, ...]]
    _IDENTITY_SOURCE_NORMALIZERS: ClassVar[
        dict[str, Callable[[Any], Any]]
    ] = {}
    _IDENTITY_NAMESPACE: ClassVar[UUID]
    _IDENTITY_SCHEMA: ClassVar[str]
    _IDENTITY_LABEL: ClassVar[str]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Validate a subclass's declarative identity configuration."""
        super().__pydantic_init_subclass__(**kwargs)

        if "_finalize_managed_identity" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override _finalize_managed_identity; "
                "override _validate_before_identity instead."
            )
        incompatible_validators = [
            name
            for name, decorator in cls.__pydantic_decorators__.model_validators.items()
            if name != "_finalize_managed_identity"
            and decorator.info.mode in {"after", "wrap"}
        ]
        if incompatible_validators:
            raise TypeError(
                f"{cls.__name__} declares model validators that may run after "
                "managed identity finalization: "
                f"{sorted(incompatible_validators)!r}. Move those checks to "
                "_validate_before_identity."
            )

        identity_field = cls._IDENTITY_FIELD_NAME
        if identity_field not in cls.model_fields:
            raise TypeError(
                f"{cls.__name__}._IDENTITY_FIELD_NAME names missing model "
                f"field {identity_field!r}."
            )
        if not isinstance(cls._IDENTITY_NAMESPACE, UUID):
            raise TypeError(f"{cls.__name__}._IDENTITY_NAMESPACE must be a UUID.")
        if not isinstance(cls._IDENTITY_SCHEMA, str) or not cls._IDENTITY_SCHEMA:
            raise TypeError(
                f"{cls.__name__}._IDENTITY_SCHEMA must be a non-empty string."
            )

        sources = cls._IDENTITY_SOURCE_FIELDS
        if not isinstance(sources, tuple) or not sources:
            raise TypeError(
                f"{cls.__name__}._IDENTITY_SOURCE_FIELDS must be a non-empty tuple."
            )
        if len(set(sources)) != len(sources):
            raise TypeError(
                f"{cls.__name__}._IDENTITY_SOURCE_FIELDS contains duplicates."
            )

        for source in sources:
            if (
                not isinstance(source, str)
                or not source
                or any(not segment.isidentifier() for segment in source.split("."))
            ):
                raise TypeError(
                    f"Invalid identity source path {source!r} on {cls.__name__}."
                )
            root = source.split(".", 1)[0]
            if root not in cls.model_fields:
                raise TypeError(
                    f"Identity source {source!r} on {cls.__name__} starts with "
                    f"missing model field {root!r}."
                )
            if root == identity_field:
                raise TypeError(
                    f"Managed identity field {identity_field!r} cannot be one of "
                    f"{cls.__name__}._IDENTITY_SOURCE_FIELDS."
                )

        for source in sources:
            prefix = f"{source}."
            if any(other.startswith(prefix) for other in sources):
                raise TypeError(
                    f"Identity source {source!r} on {cls.__name__} overlaps a "
                    "more specific source path."
                )

        undeclared_normalizers = (
            set(cls._IDENTITY_SOURCE_NORMALIZERS) - set(sources)
        )
        if undeclared_normalizers:
            raise TypeError(
                f"{cls.__name__} declares normalizers for non-source paths: "
                f"{sorted(undeclared_normalizers)!r}."
            )
        if not all(
            callable(value)
            for value in cls._IDENTITY_SOURCE_NORMALIZERS.values()
        ):
            raise TypeError(
                f"Every {cls.__name__} identity-source normalizer must be callable."
            )

    @staticmethod
    def _identity_json_value(value: Any) -> Any:
        """Convert a selected source value to deterministic JSON-compatible data."""
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, (UUID, Path)):
            return str(value)
        if isinstance(value, Enum):
            return ManagedIdentityModel._identity_json_value(value.value)
        if isinstance(value, dict):
            return {
                key: ManagedIdentityModel._identity_json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                ManagedIdentityModel._identity_json_value(item)
                for item in value
            ]
        if isinstance(value, (set, frozenset)):
            raise TypeError(
                "Unordered identity sources require an explicit normalizer."
            )
        return value

    def _resolve_identity_source(self, source: str) -> Any:
        """Resolve one declared dotted attribute path on this model."""
        value: Any = self
        traversed: list[str] = []
        for segment in source.split("."):
            traversed.append(segment)
            try:
                value = getattr(value, segment)
            except AttributeError as error:
                raise TypeError(
                    f"Identity source {source!r} on {type(self).__name__} cannot "
                    f"resolve {'.'.join(traversed)!r}."
                ) from error
        return value

    def _identity_payload(self) -> dict[str, Any]:
        """Build the canonical payload from declared identity-source paths."""
        payload: dict[str, Any] = {"identity_schema": self._IDENTITY_SCHEMA}
        for source in self._IDENTITY_SOURCE_FIELDS:
            value = self._resolve_identity_source(source)
            normalizer = self._IDENTITY_SOURCE_NORMALIZERS.get(source)
            if normalizer is not None:
                value = normalizer(value)
            value = self._identity_json_value(value)

            # Unpacks segmented payload into nested dict.
            target = payload
            segments = source.split(".")
            for segment in segments[:-1]:
                nested = target.setdefault(segment, {})
                if not isinstance(nested, dict):
                    raise TypeError(
                        f"Identity source {source!r} conflicts with another "
                        f"source on {type(self).__name__}."
                    )
                target = nested
            target[segments[-1]] = value

        return payload

    def _validate_before_identity(self) -> None:
        """Run subclass-specific model checks before finalizing identity.

        Subclasses may override this ordinary method instead of declaring a
        ``mode="after"`` model validator. They should raise ``ValueError`` for
        invalid model-wide state and must not initialize the identity directly.
        """

    def _compute_identity(self) -> UUID:
        """Compute this model's deterministic UUIDv5 identity."""
        return canonical_uuid(self._IDENTITY_NAMESPACE, self._identity_payload())

    def _initialize_or_verify_identity(self) -> Self:
        """Initialize a missing identity or verify a serialized one once."""
        identity = getattr(self, self._IDENTITY_FIELD_NAME)
        if self._identity_initialized and identity is not None:
            return self

        expected = self._compute_identity()
        if identity is not None and identity != expected:
            raise ValueError(
                f"{self._IDENTITY_FIELD_NAME} {identity} does not match "
                f"{self._IDENTITY_LABEL} contents; expected {expected}."
            )
        object.__setattr__(self, self._IDENTITY_FIELD_NAME, expected)
        object.__setattr__(self, "_identity_initialized", True)
        return self

    @classmethod
    def _identity_source_roots(cls) -> frozenset[str]:
        """Return top-level fields whose reassignment invalidates identity."""
        return frozenset(
            source.split(".", 1)[0]
            for source in cls._IDENTITY_SOURCE_FIELDS
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Regenerate the stored identity after validated field reassignment."""
        initialized = getattr(self, "_identity_initialized", False)
        if initialized and name == self._IDENTITY_FIELD_NAME:
            raise AttributeError(
                f"{self._IDENTITY_FIELD_NAME} is system-managed and cannot be assigned."
            )
        if not initialized or name not in self._identity_source_roots():
            super().__setattr__(name, value)
            return

        old_value = getattr(self, name)
        old_identity = getattr(self, self._IDENTITY_FIELD_NAME)
        object.__setattr__(self, self._IDENTITY_FIELD_NAME, None)
        try:
            super().__setattr__(name, value)
        except Exception:
            object.__setattr__(self, name, old_value)
            object.__setattr__(self, self._IDENTITY_FIELD_NAME, old_identity)
            raise

    def model_copy(
        self,
        *,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Return a validated copy with identity regenerated from its contents."""
        data = self.model_dump(
            mode="python",
            exclude={self._IDENTITY_FIELD_NAME},
        )
        if update:
            data.update({
                key: value
                for key, value in update.items()
                if key != self._IDENTITY_FIELD_NAME
            })
        return type(self).model_validate(data)

    @model_serializer(mode="wrap")
    def serialize_managed_identity(self, handler: Any) -> dict[str, Any]:
        """Serialize the managed identity as a standard UUID string."""
        data = handler(self)
        identity = data.get(self._IDENTITY_FIELD_NAME)
        if identity is not None:
            data[self._IDENTITY_FIELD_NAME] = str(identity)
        return data

    @model_validator(mode="before")
    @classmethod
    def load_monty_identity(cls, data: Any) -> Any:
        """Accept the managed UUID in Monty's tagged-dictionary format."""
        if not isinstance(data, dict):
            return data
        value = data.get(cls._IDENTITY_FIELD_NAME)
        if isinstance(value, dict) and value.get("@module") == "uuid":
            data = dict(data)
            data[cls._IDENTITY_FIELD_NAME] = value.get("string", value)
        return data

    @model_validator(mode="after")
    def _finalize_managed_identity(self) -> Self:
        """Run model-wide checks, then initialize or verify the managed ID."""
        self._validate_before_identity()
        return self._initialize_or_verify_identity()

