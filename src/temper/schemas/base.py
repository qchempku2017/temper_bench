from pathlib import Path

from pydantic import BaseModel


class JsonIOModel(BaseModel):
    """Pydantic model subclass to allow saving and loading from `json` files."""

    def save_json(self, path: str | Path) -> None:
        """Save the model to a JSON file."""
        Path(path).write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "JsonIOModel":
        """Load the model from a JSON file."""
        return cls.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )