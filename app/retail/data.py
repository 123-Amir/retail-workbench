from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RetailData:
    """Typed, cleaned frames keyed by dataset name, plus the data_version each answer must cite."""
    frames: dict
    versions: dict = field(default_factory=dict)

    @classmethod
    def from_run(cls, service, owner: str, run_id: str):
        run = service.get_run(owner, run_id)
        frames, versions = {}, {}
        for d in run["datasets"]:
            if d["status"] == "succeeded":
                frames[d["name"]] = service.read_cleaned(owner, run_id, d["name"], typed=True)
                versions[d["name"]] = d["data_version"]
        return cls(frames, versions)

    def get(self, name: str) -> pd.DataFrame | None:
        return self.frames.get(name)
