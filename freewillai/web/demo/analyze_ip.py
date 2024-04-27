import aiohttp
import os
import json
import asyncio
import folium
from web.demo.db import NotionDB, Record


db = NotionDB()
CACHE_PATH = 'web/demo/ipinfo.json'

if not os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, 'w') as f:
        json.dump([], f)


async def get_ip_info(ip):
    async with aiohttp.ClientSession() as session:
        async with session.get("https://ipinfo.io/" + ip + "/json") as resp:
            return await resp.json()


async def get_set_of_ips(records: list[Record] | None = None):
    records = records or await db.get_records()
    return tuple(set([record.ip for record in records]))


async def get_all_info(records: list[Record] | None = None):
    ips = await get_set_of_ips(records)
    with open(CACHE_PATH, 'r') as f:
        cache = json.load(f)

    cached_ips = [el['ip'] for el in cache]
    for ip in ips:
        if ip not in cached_ips:
            print('[+] Getting info for', ip)
            cache.append(await get_ip_info(ip))
    
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f)

    return cache


def display_on_map(data):
    """Display the data on a map using folium"""
    m = folium.Map()
    for i in data:
        popup = '<br>'.join([f"{key} = {value}" for key, value in i.items() if key != 'loc'])
        folium.Marker(
            location=i['loc'].split(','),
            popup=popup,
            icon=folium.Icon(color='red', icon='info-sign'),
        ).add_to(m)

    m.save('web/demo/templates/map.html')
    os.system('open web/demo/templates/map.html')


async def main():
    records = await db.get_records()
    info = await get_all_info(records)
    print(json.dumps(info, indent=4, sort_keys=True))
    data = []
    for record in records:
        for info_record in info:
            if record.ip == info_record['ip']:
                info_record.update({'code': record.code, 'network': record.network})
                data.append(info_record)

    print('1->', len(data), '2->', len(info))
    display_on_map(data)


if __name__ == '__main__':
    asyncio.run(main())
