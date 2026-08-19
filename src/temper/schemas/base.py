from pathlib import Path

from pydantic import BaseModel
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
