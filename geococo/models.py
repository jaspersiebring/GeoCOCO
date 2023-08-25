from pydantic.dataclasses import dataclass
from pydantic import Field

@dataclass
class WindowSource:
    width_window_pixels: int = Field(gt=0)
    width_overlap_pixels: int = Field(ge=0)
    height_window_pixels: int = Field(gt=0)
    height_overlap_pixels: int = Field(ge=0)
    
    def __post_init__(self):
        self.width_step_pixels: int = (
            self.width_window_pixels - self.width_overlap_pixels * 2
        )
        self.height_step_pixels: int = (
            self.height_window_pixels - self.height_overlap_pixels * 2
        )

        if self.width_step_pixels <= 0 and self.height_step_pixels <= 0:
            raise ValueError(
                f"Both width_step_pixels ({self.width_step_pixels}) and height_step_pixels ({self.height_step_pixels}) must be positive. Increase width_window_pixels ({self.width_window_pixels}) and height_window_pixels ({self.height_window_pixels}) or decrease width_overlap_pixels ({self.width_overlap_pixels}) and height_overlap_pixels ({self.height_overlap_pixels})"
            )
        if self.width_step_pixels <= 0:
            raise ValueError(
                f"width_step_pixels ({self.width_step_pixels}) must be positive, increase width_window_pixels ({self.width_window_pixels}) or decrease width_overlap_pixels ({self.width_overlap_pixels})"
            )
        if self.height_step_pixels <= 0:
            raise ValueError(
                f"height_step_pixels ({self.height_step_pixels}) must be positive, increase height_window_pixels ({self.height_overlap_pixels}) or decrease height_overlap_pixels ({self.height_overlap_pixels})"
            )
    