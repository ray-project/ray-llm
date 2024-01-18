import uuid
from datetime import datetime, timezone
from typing import Any, List

from gradio import FlaggingCallback
from pymongo import MongoClient

from rayllm.common.constants import COLLECTION_NAME, DB_NAME, NUM_LLM_OPTIONS
from rayllm.common.llm_event import LlmEvent, LlmResponse, Vote


class MongoLogger(FlaggingCallback):
    """Logs flagged events to Mongo DB."""

    def __init__(self, url, project_name) -> None:
        self.url = url
        self.client = MongoClient(url)
        self.project_name = project_name
        self.components = None
        self.db = None
        try:
            self.client.admin.command("ping")
            print("Pinged MongoDB. Correctly set up")
        except Exception as e:
            print(e)

    def setup(self, components: list, flagging_dir: str = None):
        """FlaggingCallback-compliant setup method."""
        self.components = components
        # Check if the database exists
        if DB_NAME in self.client.list_database_names():
            self.db = self.client[DB_NAME]
            print(f"Database '{DB_NAME}' already exists.")
        else:
            # The database doesn't exist, so create it
            self.db = self.client[DB_NAME]
            print(f"Database '{DB_NAME}' created.")

        # OK, now we create a collection.
        # Check if the collection exists
        if COLLECTION_NAME in self.db.list_collection_names():
            # The collection exists
            print(
                f"Collection '{COLLECTION_NAME}' "
                f"already exists in database '{DB_NAME}'."
            )
        else:
            # The collection doesn't exist, so create it
            self.db.create_collection(COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' created in database '{DB_NAME}'.")

    def flag(self, flag_data: List[Any], flag_option: str = "", username: str = ""):
        print(f"last value is: {flag_data}")
        event = LlmEvent(
            project_name=self.project_name,
            created_at=datetime.now(timezone.utc),
            instance_id=str(uuid.uuid4()),
            user_prompt=flag_data[0],
            responses=[
                LlmResponse(
                    model_id=flag_data[i],  # 1, 2, 3
                    text=flag_data[i + NUM_LLM_OPTIONS],  # 4, 5, 6
                    gen_stats=flag_data[-2][i - 1],  # 0, 1, 2
                )
                for i in range(1, NUM_LLM_OPTIONS + 1)
            ],
            session_id=flag_data[-1],
        )
        if flag_data[-3]:
            vote_number = int(flag_data[-3][-1])
            event.votes = Vote(llm=flag_data[vote_number], score=1)

        print(f"Event is {event.json()}")
        result = self.client[DB_NAME][COLLECTION_NAME].insert_one(event.dict())
        print(f"Mongo result {result}")
