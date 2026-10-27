"""Paper routes use 1-based inclusive endpoints; execution indices are 0-based."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    start: int | None
    end: int | None

    def __post_init__(self) -> None:
        if (self.start is None) != (self.end is None):
            raise ValueError("Both route endpoints must be specified, or both absent")
        if self.start is not None and (self.start < 1 or self.end < self.start):
            raise ValueError("Expected 1 <= start <= end, inclusive")

    @classmethod
    def parse(cls, value: str) -> "Route":
        if value == "standard":
            return cls(None, None)
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError("Route must be 'standard' or 'start:end' (1-based inclusive)")
        return cls(int(parts[0]), int(parts[1]))

    @property
    def id(self) -> str:
        return "standard" if self.start is None else f"{self.start}:{self.end}"

    def indices(self, num_layers: int) -> tuple[int, ...]:
        if num_layers < 1:
            raise ValueError("Model must have at least one layer")
        base = tuple(range(num_layers))
        if self.start is None:
            return base
        if self.end > num_layers:
            raise ValueError(f"Route {self.id} exceeds model depth {num_layers}")
        return base[: self.end] + base[self.start - 1 : self.end] + base[self.end :]


def enumerate_routes(num_layers: int, candidates: list[str] | None) -> list[Route]:
    if candidates is None:
        loops = [Route(i, j) for i in range(1, num_layers + 1) for j in range(i, num_layers + 1)]
    else:
        loops = [Route.parse(value) for value in candidates]
        if len({route.id for route in loops}) != len(loops):
            raise ValueError("Duplicate candidate routes")
        loops = [route for route in loops if route.start is not None]
        if not loops:
            raise ValueError("Specify at least one nonstandard candidate")
    for route in loops:
        route.indices(num_layers)
    return [Route(None, None), *loops]
