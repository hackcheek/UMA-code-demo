import aiohttp
import asyncio
import requests
from pprint import pprint
from dataclasses import dataclass
from typing import TypedDict, Literal


NOTION_SECRET = 'secret_WilkE9n1LOtVX9EiawYLFniy2UtMwknCvfBJ0fhXrCv'
NOTION_DATABASE_ID = '9edad184b8b44b638e3cea33842c5da3'


@dataclass
class Record:
    ip: str
    network: Literal['devnet/anvil', 'testnet/optimism', 'testnet/sepolia']
    code: str


class AsyncIter:
    def __init__(self, fn):
        self.fn = fn

    def __aiter__(self):
        return self
    
    async def __anext__(self):
        return await self.fn
    

class NotionDB:
    def __init__(self):
        self.id = NOTION_DATABASE_ID
        self.secret = NOTION_SECRET
        self.headers = {
            "Authorization": "Bearer " + self.secret,
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

    async def make_request(self, url, *args, **kwargs) -> dict:
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(url, *args, **kwargs) as resp:
                return await resp.json()

    def build_record(self, result: dict) -> Record:
        return Record(
            ip=result['properties']['ip']['rich_text'][0]['plain_text'],
            network=result['properties']['chain']['select']['name'],
            code=result['properties']['code']['rich_text'][0]['plain_text'],
        )

    async def get_records(self, num_pages=None) -> list[Record]:
        url = f"https://api.notion.com/v1/databases/{self.id}/query"
        get_all = num_pages is None
        page_size = 100 if get_all else num_pages
        payload = {"page_size": page_size}
        data = await self.make_request(url, json=payload)
        records = list(map(self.build_record, data["results"]))

        if data["has_more"] and get_all:
            async def fn():
                if data["has_more"] and get_all:
                    payload = {"page_size": page_size, "start_cursor": data["next_cursor"]}
                    url = f"https://api.notion.com/v1/databases/{self.id}/query"
                    return await self.make_request(url, json=payload)

            iterator = AsyncIter(fn)
            async for resp in iterator:
                records.extend(list(map(self.build_record, data["results"])))
                
        return records

    def _make_notion_object(self, record: Record) -> dict:
        return {
            "ip": {
                "rich_text": [{
                    "type": "text",
                    "text": {
                        "content": record.ip
                    }
                }]
            },
            "chain": {
                "select": {
                    "name": record.network
                }
            },
            "code": {
                "rich_text": [{
                    "type": "text",
                    "text": {
                        "content": record.code
                    }
                }]
            }
        }

    async def add_record(self, record: Record):
        create_url = "https://api.notion.com/v1/pages"
        obj = self._make_notion_object(record)
        payload = {"parent": {"database_id": self.id}, "properties": obj}
        res = await self.make_request(create_url, headers=self.headers, json=payload)
        print(res)
        return res

    async def show_records(self):
        records = await self.get_records()
        for record in records:
            props = record["properties"]
            print(">", props)


if __name__ == "__main__":
    db = NotionDB()
    asyncio.run(db.show_records())
    # db.add_record(Record(id=0, ip='0.0.0.0', code='print("hello-world")', network='devnet/anvil'))
