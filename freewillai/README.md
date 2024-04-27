# freewillai
Free Will AI

# Requirements to test in testnet
1- We have to deploy the contracts on testnet and get their addresses.

```bash
export FREEWILLAI_TOKEN_ADDRESS=<token-address>
export FREEWILLAI_TASK_RUNNER_ADDRESS=<task_runner-address>
```

2- Each node needs a private key so we have to create an account for each node on this testnet

```bash
python -m freewillai.node -s 200 -p <node-private-key>
```

3- The user/client also needs a private key
```bash
export PRIVATE_KEY=<user-private-key>

# or put it in .env file
# .env
PRIVATE_KEY=<user-private-key>
```
4- Set the endpoint provider to environment
```bash
export FREEWILLAI_RPC=http://<host>:<port>...
```


# Commands to deploy demo (Digital Ocean server command line format):
## Spin up anvil and ipfs for workers and client interaction
```bash
sudo ./docker-manager setup
```
## Spin up workers. There is 9 workers availables
### one worker (attached)
```bash
sudo ./docker-manager up worker 1
:'                             ^^^ 
                          worker_id (1-9)'
```
### group of workers (detached)
```bash
sudo ./docker-manager up workers 5
:'                              ^^^
                   workers amount to get up (1-9) 
                   Empty will run all of workers (9)'
```
## Spin up repl on demo.freewillai.org
```bash
sudo ./docker-manager up repl

# To make it tolerant of out-of-memory issues
sudo OOM_TOLERANCE=1 ./docker-manager up repl
```

## To stop containers replace up to down. 
### Example:
```bash
sudo ./docker-manager down repl
sudo ./docker-manager down workers 5
sudo ./docker-manager down worker 1
```

## Also we can get up all (anvil, ipfs, 9 workers and repl)
```bash
sudo ./docker-manager up all
```
## Get down all docker containers even non-freewilai containers <br>(maybe change it to get down just freewillai containers)
```bash
sudo ./docker-manager down all 
```

## View worker logs
```bash
sudo ./docker-manager logs worker 2
:'                               ^^^ 
                            worker_id (1-9)'
```

# Troubleshooting
### Suddenly repl stopped and demo.freewillai.org page has 404 error <br>It often is due to out of memory issue. To address it you just need to restart the repl
```bash
sudo ./docker-manager down repl
sudo ./docker-manager up repl
```
Now we can run this command to restart repl every time it dies
```bash
sudo OOM_TOLERANCE=1 ./docker-manager up repl
```
(C) Copyright FreeWillAI
