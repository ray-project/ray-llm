import uuid
from datetime import datetime, timezone
from typing import Any

from gradio import FlaggingCallback
from pymongo import MongoClient

from aviary.common.constants import COLLECTION_NAME, DB_NAME
from aviary.common.llm_event import LlmEvent, LlmResponse, Vote


class MongoLogger(FlaggingCallback):
    """Logs flagged events to Mongo DB."""

    def __init__(self, url, project_name) -> None:
        self.url = url
        self.client = MongoClient(url)
        self.project_name = project_name
        self.components = None
        try:
            self.client.admin.command("ping")
            print("Pinged MongoDB. Correctly set up")
        except Exception as e:
            print(e)

    def setup(self, components):
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

    def flag(self, flag_data: list[Any], flag_option: str = "", username: str = ""):
        print(f"last value is: {flag_data}")
        event = LlmEvent(
            project_name=self.project_name,
            created_at=datetime.now(timezone.utc),
            instance_id=str(uuid.uuid4()),
            user_prompt=flag_data[0],
            # TODO(mwk): Work out how to generalize this to _n_ inputs
            responses=[
                LlmResponse(
                    model_id=flag_data[1], text=flag_data[4], gen_stats=flag_data[8][0]
                ),
                LlmResponse(
                    model_id=flag_data[2], text=flag_data[5], gen_stats=flag_data[8][1]
                ),
                LlmResponse(
                    model_id=flag_data[3], text=flag_data[6], gen_stats=flag_data[8][2]
                ),
            ],
            session_id=flag_data[9],
        )
        if flag_data[7]:
            vote_number = int(flag_data[7][-1])
            event.votes = Vote(llm=flag_data[vote_number], score=1)

        print(f"Event is {event.json()}")
        result = self.client[DB_NAME][COLLECTION_NAME].insert_one(event.dict())
        print(f"Mongo result {result}")
