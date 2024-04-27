import asyncio
import sys

_, filename = sys.argv

async def aexec(code):
    # Make an async function with the code and `exec` it
    exec(
        f'async def __FWAI_EXECUTION(): ' +
        ''.join(f'\n {l}' for l in code.split('\n'))
    )

    # Get `__ex` from local variables, call it and return the result
    return await locals()['__FWAI_EXECUTION']()

def main():
    with open(filename, 'r') as code:
        content = code.read()
        exec(
            'import asyncio\n'
            f'async def __FWAI_EXECUTION():' +
            ''.join(f'\n {l}' for l in content.split('\n')) +
            '\nasyncio.run(__FWAI_EXECUTION())'
        )

main()
