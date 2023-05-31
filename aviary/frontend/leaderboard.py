import pandas as pd
from pymongo import DESCENDING, MongoClient

from aviary.common.constants import COLLECTION_NAME, DB_NAME, G5_COST_PER_S_IN_DOLLARS


class Leaderboard:
    def __init__(self, url: str, project_name: str):
        self.url = url
        self.client = MongoClient(url)
        self.db = self.client[DB_NAME]
        self.coll = self.db[COLLECTION_NAME]
        self.project_name = project_name

    def generate_votes_leaderboard(self) -> pd.DataFrame:
        pipeline_votes = [
            {"$match": {"votes": {"$ne": None}}},
            {
                "$group": {
                    "_id": {"llm": "$votes.llm"},
                    "Votes": {"$sum": "$votes.score"},
                }
            },
            {"$sort": {"count": DESCENDING}},
            {
                "$project": {
                    "LLM": "$_id.llm",
                    "_id": 0,
                    "Votes": 1,
                }
            },
        ]

        pipeline_contentions = [
            {"$match": {"votes": {"$ne": None}}},
            {"$unwind": {"path": "$responses"}},
            {
                "$group": {
                    "_id": {"llm": "$responses.model_id"},
                    "In Contention": {"$sum": 1.0},
                }
            },
            {
                "$project": {
                    "LLM": "$_id.llm",
                    "_id": 0,
                    "In Contention": 1,
                }
            },
        ]

        df_contentions = pd.DataFrame(
            list(self.coll.aggregate(pipeline_contentions)),
            columns=["LLM", "In Contention"],
        )
        df_votes = pd.DataFrame(
            list(self.coll.aggregate(pipeline_votes)), columns=["LLM", "Votes"]
        )
        df = pd.merge(df_votes, df_contentions, on="LLM", how="right").fillna(0)
        # Use m-estimate correction with prior of 1/3
        df["Win Ratio"] = (df["Votes"] + 1) / (df["In Contention"] + 3) * 3 * 1000
        df["Win Ratio"] = df["Win Ratio"].astype(int)
        df = df.sort_values(by="Win Ratio", ascending=False)
        return df

    def generate_perf_leaderboard(self) -> pd.DataFrame:
        pipeline = [
            {"$match": {"votes": {"$ne": None}}},
            {"$unwind": {"path": "$responses"}},
            {"$match": {"responses": {"$ne": None}}},
            {
                "$group": {
                    "_id": {"llm": "$responses.model_id"},
                    "avg_latency": {"$avg": "$responses.gen_stats.total_time"},
                    "avg_length": {"$avg": "$responses.gen_stats.num_total_tokens"},
                }
            },
            {
                "$project": {
                    "LLM": "$_id.llm",
                    "_id": 0,
                    "Lat (s)": "$avg_latency",
                    "Tokens (i/o)": "$avg_length",
                }
            },
        ]

        df = pd.DataFrame(
            list(self.coll.aggregate(pipeline)),
            columns=["LLM", "Lat (s)", "Tokens (i/o)"],
        )
        print(f"Raw DF \n{df}")
        df["Tokens/s"] = df["Tokens (i/o)"] / df["Lat (s)"]
        df["Cost per answer"] = df["Lat (s)"] * G5_COST_PER_S_IN_DOLLARS
        df["CP 1k tokens $"] = 1000 / df["Tokens/s"] * G5_COST_PER_S_IN_DOLLARS
        df = df.sort_values(by="Tokens/s", ascending=False)
        df = df.round(
            {
                "Lat (s)": 1,
                "Tokens (i/o)": 1,
                "Tokens/s": 1,
                "Cost per answer": 4,
                "CP 1k tokens $": 4,
            }
        )
        print(df)
        return df


class DummyLeaderboard(Leaderboard):
    def __init__(self, url: str = None, project_name: str = None):
        pass

    def generate_votes_leaderboard(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["LLM", "In Contention", "Win Ratio"],
        )

    def generate_perf_leaderboard(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "LLM",
                "Lat (s)",
                "Tokens (i/o)",
                "Tokens/s",
                "Cost per answer",
                "CP 1k tokens $",
            ]
        )
