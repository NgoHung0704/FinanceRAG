from typing import Optional

from .BaseTask import BaseTask
from .TaskMetadata import TaskMetadata


class FinDER(BaseTask):
    def __init__(self, use_local_data=True):
        # Use local data by default if available
        if use_local_data:
            import os
            data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
            dataset_config = {
                "path": "local",  # Required by TaskMetadata validation
                "data_folder": data_folder,
                "subset": "finder",
            }
        else:
            dataset_config = {
                "path": "Linq-AI-Research/FinanceRAG",
                "subset": "FinDER",
            }
        
        self.metadata: TaskMetadata = TaskMetadata(
        name="FinDER",
        description="Prepared for competition from Linq",
        reference=None,
        dataset=dataset_config,
        type="RAG",
        category="s2p",
        modalities=["text"],
        date=None,
        domains=["Report"],
        task_subtypes=[
            "Financial retrieval",
            "Question answering",
        ],
        license=None,
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="human-generated",
        bibtex_citation=None,
    )
        super().__init__(self.metadata)

