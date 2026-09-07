"""Shared JSON-serializable errors for engine CLI commands."""

from __future__ import annotations

ENGINE_COMMAND_CONTRACT_VERSION = "1"


class EngineCommandError(Exception):
	"""Base error that preserves the engine command JSON error contract."""

	contract_version = ENGINE_COMMAND_CONTRACT_VERSION

	def __init__(self, code: str, message: str) -> None:
		super().__init__(message)
		self.code = code
		self.message = message

	def to_dict(self) -> dict[str, object]:
		return {
			"version": self.contract_version,
			"error": {
				"code": self.code,
				"message": self.message,
			},
		}
